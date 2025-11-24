"""Async IMAP client using aioimaplib for better async support.

This module provides native async IMAP operations with:
- OAuth2 authentication support
- UID-based email fetching with windowing
- Structured error handling
- Automatic token refresh detection
"""

import email
import logging
from datetime import datetime
from email.header import decode_header
from typing import Any

import aioimaplib

from .oauth_flow import generate_xoauth2_string
from .state import Email

logger = logging.getLogger(__name__)


class IMAPAuthError(Exception):
    """Raised when IMAP authentication fails."""

    pass


class IMAPConnectionError(Exception):
    """Raised when IMAP connection fails."""

    pass


class AsyncIMAPClient:
    """Async IMAP client with OAuth2 support and UID-based windowing.

    This client provides native async operations for:
    - Connecting with OAuth2 authentication
    - Fetching emails with UID-based pagination
    - Batch email operations
    - Automatic auth error detection for token refresh
    """

    def __init__(
        self,
        email_address: str,
        access_token: str,
        provider: str,
        timeout: int = 30,
    ):
        """Initialize async IMAP client.

        Args:
            email_address: User's email address
            access_token: OAuth access token
            provider: Email provider ('gmail' or 'outlook')
            timeout: Connection timeout in seconds
        """
        self.email_address = email_address
        self.access_token = access_token
        self.provider = provider
        self.timeout = timeout
        self.client: aioimaplib.IMAP4_SSL | None = None

        # Determine IMAP host
        if provider == "gmail":
            self.host = "imap.gmail.com"
        elif provider == "outlook":
            self.host = "outlook.office365.com"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def connect(self) -> None:
        """Connect to IMAP server and authenticate with OAuth2.

        Raises:
            IMAPConnectionError: If connection fails
            IMAPAuthError: If authentication fails (token may need refresh)
        """
        try:
            # Create connection
            self.client = aioimaplib.IMAP4_SSL(
                host=self.host, port=993, timeout=self.timeout
            )

            # Wait for server greeting
            await self.client.wait_hello_from_server()

            # Authenticate with XOAUTH2
            auth_string = generate_xoauth2_string(self.email_address, self.access_token)
            response = await self.client.authenticate("XOAUTH2", auth_string)

            # Check authentication response
            if response.result != "OK":
                # Check for common auth failure patterns
                error_msg = " ".join(response.lines)
                if any(
                    pattern in error_msg.lower()
                    for pattern in [
                        "authentication failed",
                        "invalid credentials",
                        "authenticationfailed",
                    ]
                ):
                    raise IMAPAuthError(f"OAuth authentication failed: {error_msg}")
                raise IMAPConnectionError(f"IMAP connection failed: {error_msg}")

            logger.info(f"Connected to {self.host} as {self.email_address}")

        except IMAPAuthError:
            # Re-raise auth errors for token refresh handling
            raise
        except Exception as e:
            raise IMAPConnectionError(f"Failed to connect to IMAP server: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from IMAP server."""
        if self.client:
            try:
                await self.client.logout()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.client = None

    async def select_mailbox(self, mailbox: str = "INBOX") -> dict[str, Any]:
        """Select a mailbox.

        Args:
            mailbox: Mailbox name (default: INBOX)

        Returns:
            Dictionary with mailbox info (exists, recent, etc.)

        Raises:
            IMAPConnectionError: If not connected or select fails
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        response = await self.client.select(mailbox)

        if response.result != "OK":
            raise IMAPConnectionError(f"Failed to select {mailbox}: {response.lines}")

        # Parse mailbox info from response
        # Response format: ['flags', 'exists', 'recent', 'uidvalidity', 'etc']
        info: dict[str, Any] = {}
        for line in response.lines:
            line_str = line.decode() if isinstance(line, bytes) else str(line)
            if "EXISTS" in line_str:
                info["exists"] = int(line_str.split()[0])
            elif "RECENT" in line_str:
                info["recent"] = int(line_str.split()[0])

        return info

    async def fetch_emails(
        self,
        limit: int = 100,
        max_uid: str | None = None,
        mark_seen: bool = False,
    ) -> list[Email]:
        """Fetch emails with UID-based windowing.

        This implements the UID windowing pattern from DESIGN.md for pagination.
        Fetches emails in reverse order (newest first) with optional UID filtering.

        Args:
            limit: Maximum number of emails to fetch
            max_uid: Maximum UID to fetch (exclusive). If provided, fetches
                    emails with UID < max_uid. This enables pagination.
            mark_seen: Whether to mark emails as seen (default: False)

        Returns:
            List of Email objects

        Raises:
            IMAPConnectionError: If not connected or fetch fails
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        try:
            # Search for all UIDs (we'll filter by max_uid afterwards)
            response = await self.client.uid("search", None, "ALL")
            if response.result != "OK":
                raise IMAPConnectionError(f"UID search failed: {response.lines}")

            # Parse UIDs from response
            uids_data = response.lines[0]
            if isinstance(uids_data, bytes):
                uids_data = uids_data.decode()

            if not uids_data or uids_data == "":
                return []

            all_uids = uids_data.split()

            # Filter by max_uid if specified
            if max_uid:
                all_uids = [uid for uid in all_uids if int(uid) < int(max_uid)]

            # Sort in reverse (newest first) and limit
            all_uids = sorted(all_uids, key=int, reverse=True)[:limit]

            if not all_uids:
                return []

            # Fetch emails
            emails = []
            for uid in all_uids:
                try:
                    email_obj = await self._fetch_single_email(uid, mark_seen)
                    if email_obj:
                        emails.append(email_obj)
                except Exception as e:
                    logger.warning(f"Failed to fetch email UID {uid}: {e}")
                    continue

            return emails

        except Exception as e:
            raise IMAPConnectionError(f"Failed to fetch emails: {e}") from e

    async def _fetch_single_email(
        self, uid: str, mark_seen: bool = False
    ) -> Email | None:
        """Fetch a single email by UID.

        Args:
            uid: Email UID
            mark_seen: Whether to mark as seen

        Returns:
            Email object or None if fetch fails
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        # Use BODY.PEEK to avoid marking as seen, or BODY to mark as seen
        fetch_command = "BODY[]" if mark_seen else "BODY.PEEK[]"

        response = await self.client.uid("fetch", uid, f"({fetch_command})")

        if response.result != "OK":
            logger.warning(f"Failed to fetch UID {uid}: {response.lines}")
            return None

        # Parse email message
        # Response format: [b'UID FETCH (... BODY[] {size}', b'raw email data', b')']
        raw_email = None
        for line in response.lines:
            if isinstance(line, bytes) and b"From:" in line:
                raw_email = line
                break

        if not raw_email:
            logger.warning(f"No email data found for UID {uid}")
            return None

        # Parse with email library
        msg = email.message_from_bytes(raw_email)

        # Extract fields
        subject = self._decode_header(msg.get("Subject", ""))
        sender = self._decode_header(msg.get("From", ""))
        date_str = msg.get("Date", "")

        # Parse date
        try:
            from email.utils import parsedate_to_datetime

            email_date = parsedate_to_datetime(date_str)
        except Exception:
            email_date = datetime.now()

        # Extract body
        body = self._extract_body(msg)

        # Extract attachments
        attachments = self._extract_attachments(msg)

        return Email(
            uid=uid,
            subject=subject,
            sender=sender,
            body=body,
            date=email_date,
            attachments=attachments,
        )

    def _decode_header(self, header: str | None) -> str:
        """Decode email header that may contain encoded words.

        Args:
            header: Raw header string

        Returns:
            Decoded header string
        """
        if not header:
            return ""

        decoded_parts = []
        for part, encoding in decode_header(header):
            if isinstance(part, bytes):
                try:
                    decoded_parts.append(
                        part.decode(encoding or "utf-8", errors="replace")
                    )
                except Exception:
                    decoded_parts.append(part.decode("utf-8", errors="replace"))
            else:
                decoded_parts.append(str(part))

        return "".join(decoded_parts)

    def _extract_body(self, msg: email.message.Message) -> str:
        """Extract email body from message.

        Prefers text/plain, falls back to text/html.

        Args:
            msg: Email message

        Returns:
            Email body text
        """
        body = ""

        if msg.is_multipart():
            # Try to find text/plain first
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="replace")
                            break
                    except Exception:
                        continue

            # Fall back to text/html if no text/plain found
            if not body:
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/html":
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode("utf-8", errors="replace")
                                break
                        except Exception:
                            continue
        else:
            # Not multipart - get payload directly
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="replace")
            except Exception:
                body = str(msg.get_payload())

        # Truncate to first 1000 chars as per DESIGN.md
        return body[:1000]

    def _extract_attachments(self, msg: email.message.Message) -> list[str]:
        """Extract attachment filenames from message.

        Args:
            msg: Email message

        Returns:
            List of attachment filenames
        """
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                # Check if part is an attachment
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    if filename:
                        attachments.append(self._decode_header(filename))

        return attachments

    async def delete_emails(self, uids: list[str]) -> int:
        """Delete emails by UID.

        Marks emails as deleted and expunges them from the mailbox.

        Args:
            uids: List of email UIDs to delete

        Returns:
            Number of emails successfully deleted

        Raises:
            IMAPConnectionError: If not connected or delete fails
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        if not uids:
            return 0

        deleted_count = 0

        try:
            # Mark emails as deleted
            uid_set = ",".join(uids)
            response = await self.client.uid("store", uid_set, "+FLAGS", r"(\Deleted)")

            if response.result != "OK":
                raise IMAPConnectionError(
                    f"Failed to mark emails as deleted: {response.lines}"
                )

            # Expunge to permanently delete
            response = await self.client.expunge()

            if response.result != "OK":
                raise IMAPConnectionError(
                    f"Failed to expunge deleted emails: {response.lines}"
                )

            deleted_count = len(uids)
            logger.info(f"Deleted {deleted_count} emails")

        except Exception as e:
            raise IMAPConnectionError(f"Failed to delete emails: {e}") from e

        return deleted_count

    async def __aenter__(self) -> "AsyncIMAPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
