"""Async IMAP client using aioimaplib for better async support.

This module provides native async IMAP operations with:
- OAuth2 authentication support
- UID-based email fetching with windowing
- Structured error handling
- Automatic token refresh detection
"""

import asyncio
import email
import logging
import re
from datetime import datetime
from email.header import decode_header
from email.utils import parsedate_to_datetime
from typing import Any, Self

import aioimaplib

from .state import Email
from .text_utils import truncate_symmetric

logger = logging.getLogger(__name__)

_FETCH_UID_RE = re.compile(r"\bUID (\d+)")

# Plain-text body length kept per email. Symmetric truncation keeps both the
# opening and the footer (where unsubscribe links live).
MAX_BODY_CHARS = 4000


def parse_fetch_literals(lines: list[Any]) -> dict[str, bytes]:
    """Map UID -> literal payload from a (possibly multi-message) UID FETCH response.

    aioimaplib returns literals as ``bytearray`` and every other line as ``bytes``.
    Servers may report the UID before the literal (Gmail) or after it in the
    trailing line (some Exchange servers), so both orders are handled.
    """
    results: dict[str, bytes] = {}
    uid: str | None = None
    literal: bytes | None = None

    for line in lines:
        if isinstance(line, bytearray):
            literal = bytes(line)
            if uid is not None:
                results[uid] = literal
                uid, literal = None, None
            continue

        text = line.decode(errors="replace") if isinstance(line, bytes) else str(line)
        match = _FETCH_UID_RE.search(text)
        if " FETCH " in f" {text} ":
            # Start of a new message's response
            uid = match.group(1) if match else None
            literal = None
        elif match and literal is not None:
            results[match.group(1)] = literal
            uid, literal = None, None

    return results


_AUTH_FAILURE_PATTERNS = (
    "authentication failed",
    "invalid credentials",
    "authenticationfailed",
)


def _require_ok(response: Any, action: str) -> None:
    """Raise IMAPConnectionError unless an IMAP response completed OK."""
    if response.result != "OK":
        raise IMAPConnectionError(f"Failed to {action}: {response.lines}")


def _parse_date(date_str: str) -> datetime:
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return datetime.now()


def generate_xoauth2_string(email: str, access_token: str) -> str:
    """Generate XOAUTH2 authentication string for IMAP/SMTP.

    Args:
        email: User's email address
        access_token: OAuth access token

    Returns:
        Raw XOAUTH2 string (not base64-encoded, as aioimaplib will encode it)
    """
    return f"user={email}\x01auth=Bearer {access_token}\x01\x01"


class IMAPAuthError(Exception):
    """Raised when IMAP authentication fails."""


class IMAPConnectionError(Exception):
    """Raised when IMAP connection fails."""


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
        timeout: int = 120,
    ):
        """Initialize async IMAP client.

        Args:
            email_address: User's email address
            access_token: OAuth access token
            provider: Email provider ('gmail' or 'outlook')
            timeout: Connection timeout in seconds (default: 120s for batch operations)
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
            logger.info("Connecting to %s:993...", self.host)
            self.client = aioimaplib.IMAP4_SSL(
                host=self.host, port=993, timeout=self.timeout
            )

            # Wait for server greeting with timeout
            logger.info("Waiting for server greeting...")
            await asyncio.wait_for(
                self.client.wait_hello_from_server(), timeout=self.timeout
            )

            # Authenticate with XOAUTH2 with timeout
            logger.info("Authenticating with OAuth2...")
            response = await asyncio.wait_for(
                self.client.xoauth2(self.email_address, self.access_token),
                timeout=self.timeout,
            )
        except TimeoutError as e:
            raise IMAPConnectionError(
                f"Connection timeout after {self.timeout}s connecting to {self.host}"
            ) from e
        except Exception as e:
            raise IMAPConnectionError(f"Failed to connect to IMAP server: {e}") from e

        if response.result != "OK":
            error_msg = " ".join(map(str, response.lines))
            # Auth errors are distinct so callers can refresh the token
            if any(p in error_msg.lower() for p in _AUTH_FAILURE_PATTERNS):
                raise IMAPAuthError(f"OAuth authentication failed: {error_msg}")
            raise IMAPConnectionError(f"IMAP connection failed: {error_msg}")

        logger.info("Connected to %s as %s", self.host, self.email_address)

    async def disconnect(self) -> None:
        """Disconnect from IMAP server."""
        if self.client:
            try:
                await self.client.logout()
            except Exception as e:
                logger.warning("Error during disconnect: %s", e)
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
            # Search for all UIDs using search_uids method
            all_uids = await self.search_uids("ALL")

            # Filter by max_uid if specified
            if max_uid:
                all_uids = [uid for uid in all_uids if int(uid) < int(max_uid)]

            # Sort in reverse (newest first) and limit
            all_uids = sorted(all_uids, key=int, reverse=True)[:limit]

            if not all_uids:
                return []

            messages = await self.fetch_messages(all_uids, mark_seen=mark_seen)
            return [messages[uid] for uid in all_uids if uid in messages]

        except Exception as e:
            raise IMAPConnectionError(f"Failed to fetch emails: {e}") from e

    async def fetch_messages(
        self, uids: list[str], mark_seen: bool = False
    ) -> dict[str, Email]:
        """Fetch and parse full messages for many UIDs in a single round trip.

        Falls back to one FETCH per UID if the server rejects the batched command.

        Args:
            uids: Email UIDs to fetch
            mark_seen: Whether to mark as seen

        Returns:
            Dictionary mapping UID to Email (UIDs that failed are omitted)

        Raises:
            IMAPConnectionError: If not connected or the batch fetch times out
                (the connection is likely dead, so per-UID retries would stall)
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        if not uids:
            return {}

        # Use BODY.PEEK to avoid marking as seen, or BODY to mark as seen
        fetch_command = "BODY[]" if mark_seen else "BODY.PEEK[]"

        try:
            response = await asyncio.wait_for(
                self.client.uid("fetch", ",".join(uids), f"(UID {fetch_command})"),
                timeout=self.timeout,
            )
        except TimeoutError as e:
            raise IMAPConnectionError(
                f"Timed out after {self.timeout}s fetching {len(uids)} messages"
            ) from e

        if response.result != "OK":
            logger.warning("Batch message fetch failed, fetching individually")
            return await self._fetch_messages_individually(uids, fetch_command)

        return self._parse_messages(parse_fetch_literals(response.lines))

    async def _fetch_messages_individually(
        self, uids: list[str], fetch_command: str
    ) -> dict[str, Email]:
        """Fetch messages one UID at a time as a fallback."""
        assert self.client is not None
        messages: dict[str, Email] = {}

        for uid in uids:
            try:
                response = await asyncio.wait_for(
                    self.client.uid("fetch", uid, f"(UID {fetch_command})"),
                    timeout=self.timeout,
                )
                if response.result != "OK":
                    logger.warning("Failed to fetch UID %s: %s", uid, response.lines)
                    continue
                literals = parse_fetch_literals(response.lines)
                if uid not in literals and len(literals) == 1:
                    # Single-message response without a parseable UID
                    literals = {uid: next(iter(literals.values()))}
                messages.update(self._parse_messages(literals))
            except Exception as e:
                logger.warning("Failed to fetch email UID %s: %s", uid, e)

        return messages

    def _parse_messages(self, raw_messages: dict[str, bytes]) -> dict[str, Email]:
        messages: dict[str, Email] = {}
        for uid, raw in raw_messages.items():
            try:
                messages[uid] = self._parse_message(uid, raw)
            except Exception as e:
                logger.warning("Error parsing email UID %s: %s", uid, e)
        return messages

    def _parse_message(self, uid: str, raw_email: bytes) -> Email:
        msg = email.message_from_bytes(raw_email)
        return Email(
            uid=uid,
            subject=self._decode_header(msg.get("Subject", "")),
            sender=self._decode_header(msg.get("From", "")),
            body=self._extract_body(msg),
            date=_parse_date(msg.get("Date", "")),
            attachments=self._extract_attachments(msg),
            has_list_unsubscribe="List-Unsubscribe" in msg,
        )

    def _parse_header_fields(self, raw_header: bytes) -> dict[str, Any]:
        msg = email.message_from_bytes(raw_header)
        return {
            "subject": self._decode_header(msg.get("Subject", "")),
            "sender": self._decode_header(msg.get("From", "")),
            "date": _parse_date(msg.get("Date", "")),
        }

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

    @staticmethod
    def _decode_part(part: email.message.Message) -> str:
        payload = part.get_payload(decode=True)
        if not payload or not isinstance(payload, bytes):
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            # Unknown charset label
            return payload.decode("utf-8", errors="replace")

    def _extract_body(self, msg: email.message.Message) -> str:
        """Extract email body from message as plain text.

        Prefers text/plain, falls back to text/html (stripped to text), and
        truncates symmetrically so both the start and the footer are kept.

        Args:
            msg: Email message

        Returns:
            Email body text
        """
        body = ""

        if msg.is_multipart():
            # Try text/plain first, then fall back to text/html
            for wanted in ("text/plain", "text/html"):
                for part in msg.walk():
                    if part.get_content_type() != wanted:
                        continue
                    if part.get_content_disposition() == "attachment":
                        continue
                    try:
                        body = self._decode_part(part)
                    except Exception:
                        continue
                    if body:
                        break
                if body:
                    break
        else:
            # Not multipart - get payload directly
            try:
                body = self._decode_part(msg)
            except Exception:
                body = str(msg.get_payload())

        return truncate_symmetric(body, MAX_BODY_CHARS)

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

    async def search_uids(self, criteria: str = "ALL") -> list[str]:
        """Search for email UIDs matching criteria.

        Args:
            criteria: IMAP search criteria (default: "ALL")

        Returns:
            List of UIDs as strings

        Raises:
            IMAPConnectionError: If not connected or search fails
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        try:
            # Use uid_search to directly get UIDs (native aioimaplib method)
            # Pass charset=None to avoid BADCHARSET errors on Exchange/Outlook servers
            response = await self.client.uid_search(criteria, charset=None)
        except Exception as e:
            raise IMAPConnectionError(f"Failed to search UIDs: {e}") from e
        _require_ok(response, "search UIDs")

        # Parse UIDs from response
        uids_data = response.lines[0]
        if isinstance(uids_data, bytes):
            uids_data = uids_data.decode()

        return str(uids_data).split() if uids_data else []

    async def fetch_headers(self, uids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch email headers for a list of UIDs.

        Args:
            uids: List of email UIDs to fetch headers for

        Returns:
            Dictionary mapping UID to header dict with fields:
            - subject: Email subject
            - sender: Sender email address
            - date: Email date as datetime object

        Raises:
            IMAPConnectionError: If not connected or fetch fails
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        if not uids:
            return {}

        headers = {}

        try:
            # Batch fetch all headers in a single command to reduce round trips
            # This prevents connection timeouts from multiple sequential requests
            uid_set = ",".join(uids)

            response = await asyncio.wait_for(
                self.client.uid("fetch", uid_set, "(UID BODY.PEEK[HEADER])"),
                timeout=self.timeout,
            )
        except TimeoutError:
            logger.warning("Batch fetch timeout, falling back to individual fetches")
            return await self._fetch_headers_individually(uids)
        except Exception as e:
            raise IMAPConnectionError(f"Failed to fetch headers: {e}") from e

        if response.result != "OK":
            logger.warning("Batch fetch failed: %s", response.result)
            # Fall back to individual fetches on batch failure
            return await self._fetch_headers_individually(uids)

        for uid, raw_header in parse_fetch_literals(response.lines).items():
            try:
                headers[uid] = self._parse_header_fields(raw_header)
            except Exception as e:
                logger.warning("Error parsing header for UID %s: %s", uid, e)

        return headers

    async def _fetch_headers_individually(
        self, uids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch headers one at a time as a fallback.

        Args:
            uids: List of email UIDs to fetch headers for

        Returns:
            Dictionary mapping UID to header dict
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        headers = {}

        for uid in uids:
            try:
                response = await asyncio.wait_for(
                    self.client.uid("fetch", uid, "(BODY.PEEK[HEADER])"),
                    timeout=self.timeout,
                )

                if response.result != "OK":
                    logger.warning(
                        "Failed to fetch headers for UID %s: %s", uid, response.result
                    )
                    continue

                literals = parse_fetch_literals(response.lines)
                raw_header = literals.get(uid) or next(iter(literals.values()), None)
                if not raw_header:
                    logger.warning("No raw_header found for UID %s", uid)
                    continue

                headers[uid] = self._parse_header_fields(raw_header)

            except TimeoutError:
                logger.warning("Timeout fetching headers for UID %s", uid)
                continue
            except Exception as e:
                logger.warning("Error fetching headers for UID %s: %s", uid, e)
                continue

        return headers

    async def fetch_bodies(self, uids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch email bodies for a list of UIDs.

        Args:
            uids: List of email UIDs to fetch bodies for

        Returns:
            Dictionary mapping UID to body dict with fields:
            - subject, sender, date, body, attachments

        Raises:
            IMAPConnectionError: If not connected or fetch fails
        """
        if not self.client:
            raise IMAPConnectionError("Not connected to IMAP server")

        if not uids:
            return {}

        try:
            messages = await self.fetch_messages(uids, mark_seen=False)
        except Exception as e:
            raise IMAPConnectionError(f"Failed to fetch bodies: {e}") from e

        return {
            uid: {
                "subject": msg.subject,
                "sender": msg.sender,
                "date": msg.date,
                "body": msg.body,
                "attachments": msg.attachments,
            }
            for uid, msg in messages.items()
        }

    async def delete_emails(self, uids: list[str]) -> int:
        """Delete emails by UID.

        For Gmail: Moves emails to [Gmail]/Trash before deletion to ensure they're
        actually deleted rather than just archived. For other providers: Marks
        emails as deleted and expunges them from the mailbox.

        Note: Gmail's default IMAP behavior archives emails when deleted from INBOX.
        To permanently delete, we first copy to Trash, then expunge from INBOX.

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

        uid_set = ",".join(uids)

        try:
            # For Gmail, copy to Trash first to ensure proper deletion
            if self.host == "imap.gmail.com":
                logger.debug("Gmail detected: Moving %s emails to Trash", len(uids))
                response = await self.client.uid("copy", uid_set, '"[Gmail]/Trash"')
                if response.result != "OK":
                    logger.warning(
                        "Failed to copy emails to Trash: %s. "
                        "Proceeding with standard deletion (will archive instead).",
                        response.lines,
                    )

            # Mark emails as deleted
            response = await self.client.uid("store", uid_set, "+FLAGS", r"(\Deleted)")
            _require_ok(response, "mark emails as deleted")

            # Expunge to remove from current mailbox
            # Note: Gmail auto-expunges, so this may be redundant but harmless
            response = await self.client.expunge()
            _require_ok(response, "expunge deleted emails")
        except IMAPConnectionError:
            raise
        except Exception as e:
            raise IMAPConnectionError(f"Failed to delete emails: {e}") from e

        logger.info("Deleted %s emails from mailbox", len(uids))
        return len(uids)

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
