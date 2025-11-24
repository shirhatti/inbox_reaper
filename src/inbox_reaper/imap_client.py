"""IMAP client module for email operations.

This module provides a comprehensive IMAP client with:
- OAuth authentication support
- Connection pooling and automatic reconnection
- Batch operations for fetching and deleting emails
- Proper MIME message parsing
- Context manager support for resource cleanup
"""

import email
import imaplib
import logging
import time
from datetime import datetime
from email.header import decode_header
from email.utils import parsedate_to_datetime
from typing import Any

from .credential_helper import get_credentials
from .oauth_config import detect_provider
from .oauth_flow import generate_xoauth2_string, refresh_access_token

logger = logging.getLogger(__name__)


class IMAPConnectionError(Exception):
    """Raised when IMAP connection fails."""

    pass


class IMAPAuthenticationError(Exception):
    """Raised when IMAP authentication fails."""

    pass


class IMAPClient:
    """IMAP client with OAuth support and connection management.

    Features:
    - OAuth authentication with automatic token refresh
    - Connection pooling and automatic reconnection
    - Batch operations for headers, bodies, and deletion
    - Context manager support for proper cleanup
    - Retry logic for transient failures

    Example:
        with IMAPClient(email="user@gmail.com") as client:
            headers = client.batch_fetch_headers(limit=100)
            bodies = client.batch_fetch_bodies(list(headers.keys())[:10])
            client.batch_delete(list(headers.keys())[:5], dry_run=True)
    """

    def __init__(
        self,
        email: str,
        provider: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize IMAP client.

        Args:
            email: User's email address
            provider: Email provider ('gmail' or 'outlook'), auto-detected if None
            max_retries: Maximum number of retry attempts for transient failures
            retry_delay: Delay in seconds between retry attempts
        """
        self.email = email
        self.provider = provider or detect_provider(email)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._imap: imaplib.IMAP4_SSL | None = None
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._connected = False

        # Load credentials
        self._load_credentials()

        # Determine IMAP host based on provider
        if self.provider == "gmail":
            self.imap_host = "imap.gmail.com"
        elif self.provider == "outlook":
            self.imap_host = "outlook.office365.com"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        logger.info(f"Initialized IMAP client for {email} ({self.provider})")

    def _load_credentials(self) -> None:
        """Load OAuth credentials from keyring."""
        credentials = get_credentials(self.email)
        if not credentials:
            raise IMAPAuthenticationError(
                f"No credentials found for {self.email}. "
                "Please run 'inbox-reaper setup' first."
            )

        self._access_token = credentials.get("access_token")
        self._refresh_token = credentials.get("refresh_token")

        if not self._access_token:
            raise IMAPAuthenticationError("Missing access token in credentials")

    def _refresh_access_token(self) -> None:
        """Refresh the OAuth access token."""
        if not self._refresh_token:
            raise IMAPAuthenticationError("No refresh token available")

        logger.info("Refreshing access token...")
        try:
            tokens = refresh_access_token(self._refresh_token, self.provider)
            self._access_token = tokens.get("access_token")

            # Update stored credentials
            from .credential_helper import store_credentials

            credentials = get_credentials(self.email) or {}
            credentials.update(tokens)
            store_credentials(self.email, credentials)

            logger.info("Access token refreshed successfully")
        except Exception as e:
            raise IMAPAuthenticationError(f"Failed to refresh access token: {e}") from e

    def connect(self) -> None:
        """Establish connection to IMAP server with OAuth authentication."""
        if self._connected and self._imap:
            return

        logger.info(f"Connecting to {self.imap_host}...")

        try:
            # Create SSL connection
            self._imap = imaplib.IMAP4_SSL(self.imap_host, 993)

            # Authenticate using XOAUTH2
            auth_string = generate_xoauth2_string(self.email, self._access_token or "")

            try:
                self._imap.authenticate("XOAUTH2", lambda x: auth_string)  # type: ignore[arg-type,return-value]
            except imaplib.IMAP4.error as e:
                # Try refreshing token if authentication fails
                if "AUTHENTICATIONFAILED" in str(e).upper():
                    logger.info("Authentication failed, attempting token refresh...")
                    self._refresh_access_token()
                    auth_string = generate_xoauth2_string(
                        self.email, self._access_token or ""
                    )
                    self._imap.authenticate("XOAUTH2", lambda x: auth_string)  # type: ignore[arg-type,return-value]
                else:
                    raise

            # Select INBOX folder
            self._imap.select("INBOX")
            self._connected = True

            logger.info("Connected successfully to IMAP server")

        except Exception as e:
            self._connected = False
            self._imap = None
            raise IMAPConnectionError(f"Failed to connect to IMAP server: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from IMAP server."""
        if self._imap and self._connected:
            try:
                self._imap.logout()
                logger.info("Disconnected from IMAP server")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._imap = None
                self._connected = False

    def _ensure_connected(self) -> imaplib.IMAP4_SSL:
        """Ensure connection is established and return IMAP instance.

        Returns:
            Active IMAP connection

        Raises:
            IMAPConnectionError: If connection cannot be established
        """
        if not self._connected or not self._imap:
            self.connect()

        if not self._imap:
            raise IMAPConnectionError("Failed to establish IMAP connection")

        return self._imap

    def _retry_operation(self, operation, *args, **kwargs) -> Any:
        """Retry an operation with exponential backoff.

        Args:
            operation: Function to retry
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (imaplib.IMAP4.abort, imaplib.IMAP4.error, ConnectionError) as e:
                last_exception = e
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                # Reconnect on connection errors
                if attempt < self.max_retries - 1:
                    self._connected = False
                    time.sleep(self.retry_delay * (2**attempt))
                    try:
                        self.connect()
                    except Exception as conn_error:
                        logger.error(f"Reconnection failed: {conn_error}")

        raise last_exception or Exception("Operation failed after all retries")

    def _decode_header(self, header_value: str | bytes | None) -> str:
        """Decode email header value.

        Args:
            header_value: Raw header value

        Returns:
            Decoded header string
        """
        if not header_value:
            return ""

        if isinstance(header_value, bytes):
            header_value = header_value.decode("utf-8", errors="replace")

        decoded_parts = decode_header(header_value)
        result = []

        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                if encoding:
                    try:
                        result.append(part.decode(encoding))
                    except (LookupError, UnicodeDecodeError):
                        result.append(part.decode("utf-8", errors="replace"))
                else:
                    result.append(part.decode("utf-8", errors="replace"))
            else:
                result.append(str(part))

        return " ".join(result)

    def _parse_email_date(self, date_str: str | None) -> datetime:
        """Parse email date string to datetime.

        Args:
            date_str: Date string from email header

        Returns:
            Parsed datetime object
        """
        if not date_str:
            return datetime.now()

        try:
            return parsedate_to_datetime(date_str)
        except (TypeError, ValueError):
            logger.warning(f"Failed to parse date: {date_str}")
            return datetime.now()

    def _extract_attachments(self, msg: email.message.Message) -> list[str]:
        """Extract attachment filenames from email message.

        Args:
            msg: Email message object

        Returns:
            List of attachment filenames
        """
        attachments = []

        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    decoded_filename = self._decode_header(filename)
                    attachments.append(decoded_filename)

        return attachments

    def _extract_body(self, msg: email.message.Message) -> str:
        """Extract body text from email message.

        Args:
            msg: Email message object

        Returns:
            Body text (prefer plain text, fallback to HTML)
        """
        body = ""

        if msg.is_multipart():
            # Look for text/plain first, then text/html
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get_content_disposition()

                # Skip attachments
                if content_disposition == "attachment":
                    continue

                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            body = payload.decode(charset, errors="replace")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to decode text/plain part: {e}")
                elif content_type == "text/html" and not body:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            body = payload.decode(charset, errors="replace")
                    except Exception as e:
                        logger.warning(f"Failed to decode text/html part: {e}")
        else:
            # Simple message
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
            except Exception as e:
                logger.warning(f"Failed to decode message body: {e}")

        return body.strip()

    def batch_fetch_headers(self, limit: int = 100) -> dict[str, dict]:
        """Fetch email headers in batch using IMAP FETCH command.

        Args:
            limit: Maximum number of emails to fetch

        Returns:
            Dictionary mapping UID to header dict with keys:
            - uid: str
            - subject: str
            - sender: str
            - date: datetime
            - attachments: list[str]
        """
        imap = self._ensure_connected()

        def _fetch_headers():
            # Search for all messages
            typ, data = imap.search(None, "ALL")
            if typ != "OK":
                raise IMAPConnectionError("Failed to search messages")

            message_ids = data[0].split()
            if not message_ids:
                logger.info("No messages found in INBOX")
                return {}

            # Limit to most recent messages
            message_ids = (
                message_ids[-limit:] if len(message_ids) > limit else message_ids
            )

            # Fetch headers for all messages
            message_set = b",".join(message_ids)
            typ, data = imap.fetch(
                message_set, "(UID BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])"
            )

            if typ != "OK":
                raise IMAPConnectionError("Failed to fetch headers")

            headers = {}
            current_uid = None

            for item in data:
                if isinstance(item, tuple):
                    # Parse UID from response
                    header_data = item[0]
                    if isinstance(header_data, bytes):
                        header_str = header_data.decode("utf-8", errors="replace")
                        # Extract UID from response like "123 (UID 456 BODY..."
                        if "UID" in header_str:
                            uid_part = header_str.split("UID")[1].split()[0]
                            current_uid = uid_part.strip(")")

                    # Parse email headers
                    msg_data = item[1]
                    if isinstance(msg_data, bytes) and current_uid:
                        msg = email.message_from_bytes(msg_data)

                        subject = self._decode_header(msg.get("Subject"))
                        sender = self._decode_header(msg.get("From"))
                        date_str = msg.get("Date")
                        date = self._parse_email_date(date_str)

                        headers[current_uid] = {
                            "uid": current_uid,
                            "subject": subject,
                            "sender": sender,
                            "date": date,
                            "attachments": [],  # Will be populated in full fetch
                        }

            logger.info(f"Fetched {len(headers)} email headers")
            return headers

        return self._retry_operation(_fetch_headers)

    def batch_fetch_bodies(self, uids: list[str]) -> dict[str, dict]:
        """Fetch full email bodies for specific UIDs.

        Args:
            uids: List of email UIDs to fetch

        Returns:
            Dictionary mapping UID to complete Email dict with keys:
            - uid: str
            - subject: str
            - sender: str
            - body: str
            - date: datetime
            - attachments: list[str]
        """
        if not uids:
            return {}

        imap = self._ensure_connected()

        def _fetch_bodies():
            # Fetch complete messages for UIDs
            uid_set = ",".join(uids)
            typ, data = imap.uid("FETCH", uid_set, "(BODY.PEEK[])")

            if typ != "OK":
                raise IMAPConnectionError("Failed to fetch message bodies")

            emails = {}

            for item in data:
                if isinstance(item, tuple):
                    # Parse UID from response
                    header_data = item[0]
                    current_uid = None

                    if isinstance(header_data, bytes):
                        header_str = header_data.decode("utf-8", errors="replace")
                        # Extract UID from response like "123 (UID 456 BODY..."
                        if "UID" in header_str:
                            uid_part = header_str.split("UID")[1].split()[0]
                            current_uid = uid_part.strip(")")

                    # Parse full message
                    msg_data = item[1]
                    if isinstance(msg_data, bytes) and current_uid:
                        msg = email.message_from_bytes(msg_data)

                        subject = self._decode_header(msg.get("Subject"))
                        sender = self._decode_header(msg.get("From"))
                        date_str = msg.get("Date")
                        date = self._parse_email_date(date_str)
                        body = self._extract_body(msg)
                        attachments = self._extract_attachments(msg)

                        emails[current_uid] = {
                            "uid": current_uid,
                            "subject": subject,
                            "sender": sender,
                            "body": body,
                            "date": date,
                            "attachments": attachments,
                        }

            logger.info(f"Fetched {len(emails)} complete email bodies")
            return emails

        return self._retry_operation(_fetch_bodies)

    def batch_delete(self, uids: list[str], dry_run: bool = True) -> dict[str, bool]:
        """Mark emails for deletion and optionally expunge.

        Args:
            uids: List of email UIDs to delete
            dry_run: If True, don't actually delete (just log)

        Returns:
            Dictionary mapping UID to success status (True/False)
        """
        if not uids:
            return {}

        if dry_run:
            logger.info(f"DRY RUN: Would delete {len(uids)} emails")
            return dict.fromkeys(uids, True)

        imap = self._ensure_connected()

        def _delete_messages():
            results = {}

            # Mark messages as deleted
            uid_set = ",".join(uids)
            typ, data = imap.uid("STORE", uid_set, "+FLAGS", r"(\Deleted)")

            if typ != "OK":
                logger.error(f"Failed to mark messages as deleted: {data}")
                return dict.fromkeys(uids, False)

            # Expunge to permanently delete
            typ, data = imap.expunge()

            if typ == "OK":
                logger.info(f"Successfully deleted {len(uids)} emails")
                results = dict.fromkeys(uids, True)
            else:
                logger.error(f"Failed to expunge messages: {data}")
                results = dict.fromkeys(uids, False)

            return results

        return self._retry_operation(_delete_messages)

    def __enter__(self) -> "IMAPClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.disconnect()
