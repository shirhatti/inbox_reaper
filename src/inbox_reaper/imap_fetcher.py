"""IMAP email fetching with OAuth support.

Fetches emails from IMAP servers using OAuth2 authentication.
"""

import email
import imaplib
from datetime import datetime
from email.header import decode_header
from email.message import Message as EmailMessage

from .oauth_config import PROVIDER_CONFIGS
from .state import Email


def get_imap_server(provider: str) -> str:
    """Get IMAP server for the provider."""
    return PROVIDER_CONFIGS[provider]["imap_server"]


def create_auth_string(email_address: str, access_token: str) -> str:
    """Create XOAUTH2 authentication string.

    Args:
        email_address: User's email address
        access_token: OAuth2 access token

    Returns:
        XOAUTH2 authentication string
    """
    return f"user={email_address}\x01auth=Bearer {access_token}\x01\x01"


def decode_mime_header(header_value: str | None) -> str:
    """Decode MIME encoded header value.

    Args:
        header_value: Raw header value (may be MIME encoded)

    Returns:
        Decoded string
    """
    if not header_value:
        return ""

    decoded_parts = []
    for part, encoding in decode_header(header_value):
        if isinstance(part, bytes):
            decoded_parts.append(part.decode(encoding or "utf-8", errors="replace"))
        else:
            decoded_parts.append(part)

    return "".join(decoded_parts)


def extract_body(msg: EmailMessage) -> str:
    """Extract email body from message.

    Handles multipart messages and various encodings.

    Args:
        msg: Email message object

    Returns:
        Email body as string (prefers HTML, falls back to plain text)
    """
    body_html = None
    body_plain = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain" and not body_plain:
                try:
                    body_plain = part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    )
                except Exception:
                    continue

            elif content_type == "text/html" and not body_html:
                try:
                    body_html = part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    )
                except Exception:
                    continue
    else:
        # Single part message
        try:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                body_plain = payload.decode(
                    msg.get_content_charset() or "utf-8", errors="replace"
                )
            else:
                body_plain = str(payload)
        except Exception:
            body_plain = ""

    # Prefer HTML (contains more info for marketing detection), fall back to plain
    return body_html or body_plain or ""


def extract_attachments(msg: EmailMessage) -> list[str]:
    """Extract attachment filenames from message.

    Args:
        msg: Email message object

    Returns:
        List of attachment filenames
    """
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    # Decode filename if needed
                    filename = decode_mime_header(filename)
                    attachments.append(filename)

    return attachments


def parse_imap_message(msg_data: bytes, uid: str) -> Email:
    """Parse IMAP message data into Email object.

    Args:
        msg_data: Raw email message data from IMAP
        uid: Email UID

    Returns:
        Email object
    """
    msg = email.message_from_bytes(msg_data)

    # Extract headers
    subject = decode_mime_header(msg.get("Subject", ""))
    sender = decode_mime_header(msg.get("From", ""))
    date_str = msg.get("Date", "")

    # Parse date
    try:
        date = email.utils.parsedate_to_datetime(date_str)
        if date.tzinfo is None:
            # Make timezone-aware (assume UTC if not specified)
            date = date.replace(tzinfo=datetime.now().astimezone().tzinfo)
    except Exception:
        date = datetime.now()

    # Extract body and attachments
    body = extract_body(msg)
    attachments = extract_attachments(msg)

    return Email(
        uid=uid,
        subject=subject,
        sender=sender,
        body=body,
        date=date,
        attachments=attachments,
    )


def fetch_emails(
    email_address: str, access_token: str, provider: str, limit: int = 50
) -> list[Email]:
    """Fetch emails from IMAP server using OAuth2.

    Args:
        email_address: User's email address
        access_token: OAuth2 access token
        provider: Email provider (gmail, outlook)
        limit: Maximum number of emails to fetch

    Returns:
        List of Email objects

    Raises:
        Exception: If IMAP connection or fetch fails
    """
    imap_server = get_imap_server(provider)

    # Connect to IMAP server
    mail = imaplib.IMAP4_SSL(imap_server)

    try:
        # Authenticate using XOAUTH2
        auth_string = create_auth_string(email_address, access_token)
        mail.authenticate("XOAUTH2", lambda x: auth_string.encode())

        # Select INBOX
        mail.select("INBOX")

        # Search for all emails (or recent ones)
        _, message_numbers = mail.search(None, "ALL")

        if not message_numbers[0]:
            return []

        # Get list of message IDs
        msg_ids = message_numbers[0].split()

        # Fetch only the most recent emails (up to limit)
        msg_ids_to_fetch = msg_ids[-limit:] if len(msg_ids) > limit else msg_ids

        emails = []
        for msg_id in msg_ids_to_fetch:
            # Fetch email
            _, msg_data = mail.fetch(msg_id, "(RFC822)")

            if msg_data and msg_data[0]:
                email_obj = parse_imap_message(msg_data[0][1], msg_id.decode())
                emails.append(email_obj)

        return emails

    finally:
        try:
            mail.close()
            mail.logout()
        except Exception:
            pass
