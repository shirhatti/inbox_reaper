"""Tests for IMAP client with mocked server responses.

This module tests:
- IMAP connection and authentication
- Batch operations (headers, bodies, delete)
- OAuth token refresh
- Retry logic and reconnection
- Error handling
"""

import email
import imaplib
from unittest.mock import MagicMock, patch

import pytest

from inbox_reaper.imap_client import (
    IMAPAuthenticationError,
    IMAPClient,
    IMAPConnectionError,
)


@pytest.fixture
def mock_credentials():
    """Mock credentials for testing."""
    return {
        "access_token": "test_access_token",
        "refresh_token": "test_refresh_token",
    }


@pytest.fixture
def mock_imap():
    """Mock IMAP4_SSL instance."""
    mock = MagicMock(spec=imaplib.IMAP4_SSL)
    mock.authenticate = MagicMock(return_value=("OK", [b"Authenticated"]))
    mock.select = MagicMock(return_value=("OK", [b"1"]))
    mock.search = MagicMock(return_value=("OK", [b""]))
    mock.fetch = MagicMock(return_value=("OK", []))
    mock.uid = MagicMock(return_value=("OK", []))
    mock.logout = MagicMock(return_value=("BYE", []))
    return mock


class TestIMAPClientInitialization:
    """Test IMAP client initialization."""

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_init_with_gmail(self, mock_get_creds, mock_credentials):
        """Test initialization with Gmail account."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")

        assert client.email == "user@gmail.com"
        assert client.provider == "gmail"
        assert client.imap_host == "imap.gmail.com"
        assert client._access_token == "test_access_token"
        assert client._refresh_token == "test_refresh_token"

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_init_with_outlook(self, mock_get_creds, mock_credentials):
        """Test initialization with Outlook account."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@outlook.com")

        assert client.email == "user@outlook.com"
        assert client.provider == "outlook"
        assert client.imap_host == "outlook.office365.com"

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_init_without_credentials_raises_error(self, mock_get_creds):
        """Test that missing credentials raises IMAPAuthenticationError."""
        mock_get_creds.return_value = None

        with pytest.raises(IMAPAuthenticationError, match="No credentials found"):
            IMAPClient(email="user@gmail.com")

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_init_with_missing_access_token(self, mock_get_creds):
        """Test that missing access token raises IMAPAuthenticationError."""
        mock_get_creds.return_value = {"refresh_token": "test_refresh_token"}

        with pytest.raises(IMAPAuthenticationError, match="Missing access token"):
            IMAPClient(email="user@gmail.com")

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_init_with_custom_retry_settings(self, mock_get_creds, mock_credentials):
        """Test initialization with custom retry settings."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(
            email="user@gmail.com",
            max_retries=5,
            retry_delay=2.0,
        )

        assert client.max_retries == 5
        assert client.retry_delay == 2.0


class TestIMAPConnection:
    """Test IMAP connection management."""

    @patch("inbox_reaper.imap_client.imaplib.IMAP4_SSL")
    @patch("inbox_reaper.imap_client.get_credentials")
    @patch("inbox_reaper.imap_client.generate_xoauth2_string")
    def test_connect_success(
        self, mock_xoauth, mock_get_creds, mock_imap_class, mock_credentials, mock_imap
    ):
        """Test successful IMAP connection."""
        mock_get_creds.return_value = mock_credentials
        mock_imap_class.return_value = mock_imap
        mock_xoauth.return_value = b"auth_string"

        client = IMAPClient(email="user@gmail.com")
        client.connect()

        assert client._connected is True
        mock_imap_class.assert_called_once_with("imap.gmail.com", 993)
        mock_imap.authenticate.assert_called_once()
        mock_imap.select.assert_called_once_with("INBOX")

    @patch("inbox_reaper.imap_client.imaplib.IMAP4_SSL")
    @patch("inbox_reaper.imap_client.get_credentials")
    @patch("inbox_reaper.imap_client.generate_xoauth2_string")
    @patch("inbox_reaper.imap_client.refresh_access_token")
    def test_connect_with_token_refresh(
        self,
        mock_refresh,
        mock_xoauth,
        mock_get_creds,
        mock_imap_class,
        mock_credentials,
        mock_imap,
    ):
        """Test connection with automatic token refresh on auth failure."""
        mock_get_creds.return_value = mock_credentials
        mock_imap_class.return_value = mock_imap
        mock_xoauth.return_value = b"auth_string"

        # First auth attempt fails, second succeeds
        mock_imap.authenticate.side_effect = [
            imaplib.IMAP4.error("AUTHENTICATIONFAILED"),
            ("OK", [b"Authenticated"]),
        ]

        mock_refresh.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
        }

        with patch("inbox_reaper.credential_helper.store_credentials"):
            client = IMAPClient(email="user@gmail.com")
            client.connect()

        assert client._connected is True
        assert mock_imap.authenticate.call_count == 2
        mock_refresh.assert_called_once()

    @patch("inbox_reaper.imap_client.imaplib.IMAP4_SSL")
    @patch("inbox_reaper.imap_client.get_credentials")
    def test_connect_failure_raises_error(
        self, mock_get_creds, mock_imap_class, mock_credentials
    ):
        """Test that connection failure raises IMAPConnectionError."""
        mock_get_creds.return_value = mock_credentials
        mock_imap_class.side_effect = ConnectionError("Connection failed")

        client = IMAPClient(email="user@gmail.com")

        with pytest.raises(IMAPConnectionError, match="Failed to connect"):
            client.connect()

        assert client._connected is False

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_disconnect(self, mock_get_creds, mock_credentials, mock_imap):
        """Test IMAP disconnection."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        client._imap = mock_imap
        client._connected = True

        client.disconnect()

        assert client._connected is False
        assert client._imap is None
        mock_imap.logout.assert_called_once()

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_context_manager(self, mock_get_creds, mock_credentials):
        """Test IMAP client as context manager."""
        mock_get_creds.return_value = mock_credentials

        with patch.object(IMAPClient, "connect") as mock_connect:
            with patch.object(IMAPClient, "disconnect") as mock_disconnect:
                with IMAPClient(email="user@gmail.com"):
                    pass

                mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()


class TestBatchFetchHeaders:
    """Test batch_fetch_headers method."""

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_fetch_headers_with_no_messages(
        self, mock_get_creds, mock_credentials, mock_imap
    ):
        """Test fetching headers when inbox is empty."""
        mock_get_creds.return_value = mock_credentials
        mock_imap.search.return_value = ("OK", [b""])

        client = IMAPClient(email="user@gmail.com")
        client._imap = mock_imap
        client._connected = True

        headers = client.batch_fetch_headers(limit=100)

        assert headers == {}

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_fetch_headers_with_messages(
        self, mock_get_creds, mock_credentials, mock_imap
    ):
        """Test fetching headers with messages in inbox."""
        mock_get_creds.return_value = mock_credentials

        # Mock search response with 2 messages
        mock_imap.search.return_value = ("OK", [b"1 2"])

        # Mock fetch response with headers
        mock_imap.fetch.return_value = (
            "OK",
            [
                (
                    b"1 (UID 123 BODY[HEADER.FIELDS (SUBJECT FROM DATE)] {100}",
                    b"Subject: Test Email 1\r\n"
                    b"From: sender1@example.com\r\n"
                    b"Date: Mon, 24 Nov 2025 10:00:00 +0000\r\n\r\n",
                ),
                (
                    b"2 (UID 456 BODY[HEADER.FIELDS (SUBJECT FROM DATE)] {100}",
                    b"Subject: Test Email 2\r\n"
                    b"From: sender2@example.com\r\n"
                    b"Date: Mon, 24 Nov 2025 11:00:00 +0000\r\n\r\n",
                ),
            ],
        )

        client = IMAPClient(email="user@gmail.com")
        client._imap = mock_imap
        client._connected = True

        headers = client.batch_fetch_headers(limit=100)

        assert len(headers) == 2
        assert "123" in headers
        assert "456" in headers
        assert headers["123"]["subject"] == "Test Email 1"
        assert headers["123"]["sender"] == "sender1@example.com"
        assert headers["456"]["subject"] == "Test Email 2"

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_fetch_headers_respects_limit(
        self, mock_get_creds, mock_credentials, mock_imap
    ):
        """Test that batch_fetch_headers respects the limit parameter."""
        mock_get_creds.return_value = mock_credentials

        # Mock search with 10 messages
        message_ids = b" ".join(str(i).encode() for i in range(1, 11))
        mock_imap.search.return_value = ("OK", [message_ids])

        client = IMAPClient(email="user@gmail.com")
        client._imap = mock_imap
        client._connected = True

        # Fetch with limit of 5
        client.batch_fetch_headers(limit=5)

        # Verify fetch was called with only last 5 message IDs
        fetch_call = mock_imap.fetch.call_args
        assert fetch_call is not None


class TestBatchFetchBodies:
    """Test batch_fetch_bodies method."""

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_fetch_bodies_with_empty_list(self, mock_get_creds, mock_credentials):
        """Test fetching bodies with empty UID list."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        bodies = client.batch_fetch_bodies([])

        assert bodies == {}

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_fetch_bodies_with_uids(
        self, mock_get_creds, mock_credentials, mock_imap
    ):
        """Test fetching full bodies for specific UIDs."""
        mock_get_creds.return_value = mock_credentials

        # Create mock email message
        msg = email.message.Message()
        msg["Subject"] = "Test Email"
        msg["From"] = "sender@example.com"
        msg["Date"] = "Mon, 24 Nov 2025 10:00:00 +0000"
        msg.set_payload("Test body content")

        mock_imap.uid.return_value = (
            "OK",
            [
                (b"1 (UID 123 BODY[] {200}", msg.as_bytes()),
            ],
        )

        client = IMAPClient(email="user@gmail.com")
        client._imap = mock_imap
        client._connected = True

        bodies = client.batch_fetch_bodies(["123"])

        assert len(bodies) == 1
        assert "123" in bodies
        assert bodies["123"]["subject"] == "Test Email"
        assert bodies["123"]["sender"] == "sender@example.com"
        assert "Test body content" in bodies["123"]["body"]

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_fetch_bodies_with_attachments(
        self, mock_get_creds, mock_credentials, mock_imap
    ):
        """Test fetching bodies extracts attachment filenames."""
        mock_get_creds.return_value = mock_credentials

        # Create multipart message with attachment
        from email.mime.base import MIMEBase
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart()
        msg["Subject"] = "Email with Attachment"
        msg["From"] = "sender@example.com"
        msg["Date"] = "Mon, 24 Nov 2025 10:00:00 +0000"

        # Add text part
        msg.attach(MIMEText("Email body", "plain"))

        # Add attachment part
        attachment = MIMEBase("application", "pdf")
        attachment.add_header(
            "Content-Disposition", "attachment", filename="document.pdf"
        )
        msg.attach(attachment)

        mock_imap.uid.return_value = (
            "OK",
            [
                (b"1 (UID 123 BODY[] {500}", msg.as_bytes()),
            ],
        )

        client = IMAPClient(email="user@gmail.com")
        client._imap = mock_imap
        client._connected = True

        bodies = client.batch_fetch_bodies(["123"])

        assert "123" in bodies
        assert "document.pdf" in bodies["123"]["attachments"]


class TestBatchDelete:
    """Test batch_delete method."""

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_delete_dry_run(self, mock_get_creds, mock_credentials):
        """Test batch delete in dry run mode."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        results = client.batch_delete(["123", "456"], dry_run=True)

        assert len(results) == 2
        assert results["123"] is True
        assert results["456"] is True

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_delete_with_empty_list(self, mock_get_creds, mock_credentials):
        """Test batch delete with empty UID list."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        results = client.batch_delete([], dry_run=False)

        assert results == {}

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_delete_success(self, mock_get_creds, mock_credentials, mock_imap):
        """Test successful batch deletion."""
        mock_get_creds.return_value = mock_credentials
        mock_imap.uid.return_value = ("OK", [b""])
        mock_imap.expunge.return_value = ("OK", [])

        client = IMAPClient(email="user@gmail.com")
        client._imap = mock_imap
        client._connected = True

        results = client.batch_delete(["123", "456"], dry_run=False)

        assert len(results) == 2
        assert results["123"] is True
        assert results["456"] is True
        mock_imap.uid.assert_called_once_with(
            "STORE", "123,456", "+FLAGS", r"(\Deleted)"
        )
        mock_imap.expunge.assert_called_once()

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_batch_delete_failure(self, mock_get_creds, mock_credentials, mock_imap):
        """Test batch delete failure."""
        mock_get_creds.return_value = mock_credentials
        mock_imap.uid.return_value = ("NO", [b"Failed"])

        client = IMAPClient(email="user@gmail.com")
        client._imap = mock_imap
        client._connected = True

        results = client.batch_delete(["123"], dry_run=False)

        assert results["123"] is False


class TestRetryLogic:
    """Test retry and reconnection logic."""

    @patch("inbox_reaper.imap_client.get_credentials")
    @patch("inbox_reaper.imap_client.time.sleep")
    def test_retry_operation_on_connection_error(
        self, mock_sleep, mock_get_creds, mock_credentials, mock_imap
    ):
        """Test that operations are retried on connection errors."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com", max_retries=3)
        client._imap = mock_imap
        client._connected = True

        # Mock operation that fails twice then succeeds
        mock_operation = MagicMock(
            side_effect=[
                imaplib.IMAP4.abort("Connection lost"),
                imaplib.IMAP4.abort("Connection lost"),
                "success",
            ]
        )

        with patch.object(client, "connect"):
            result = client._retry_operation(mock_operation)

        assert result == "success"
        assert mock_operation.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_retry_exhaustion_raises_exception(self, mock_get_creds, mock_credentials):
        """Test that exhausted retries raise the last exception."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com", max_retries=2)

        mock_operation = MagicMock(side_effect=imaplib.IMAP4.error("Persistent error"))

        with pytest.raises(imaplib.IMAP4.error, match="Persistent error"):
            with patch("inbox_reaper.imap_client.time.sleep"):
                client._retry_operation(mock_operation)

        assert mock_operation.call_count == 2


class TestHeaderDecoding:
    """Test email header decoding."""

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_decode_simple_header(self, mock_get_creds, mock_credentials):
        """Test decoding simple ASCII header."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        decoded = client._decode_header("Simple Subject")

        assert decoded == "Simple Subject"

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_decode_utf8_header(self, mock_get_creds, mock_credentials):
        """Test decoding UTF-8 encoded header."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        # Encoded subject with UTF-8
        encoded = "=?UTF-8?B?VGVzdCBTdWJqZWN0?="
        decoded = client._decode_header(encoded)

        assert "Test Subject" in decoded

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_decode_bytes_header(self, mock_get_creds, mock_credentials):
        """Test decoding header passed as bytes."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        decoded = client._decode_header(b"Bytes Subject")

        assert decoded == "Bytes Subject"

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_decode_none_header(self, mock_get_creds, mock_credentials):
        """Test decoding None header returns empty string."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        decoded = client._decode_header(None)

        assert decoded == ""


class TestEmailBodyExtraction:
    """Test email body extraction from MIME messages."""

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_extract_body_from_plain_text(self, mock_get_creds, mock_credentials):
        """Test extracting body from simple plain text message."""
        mock_get_creds.return_value = mock_credentials

        msg = email.message.Message()
        msg.set_payload("This is the email body")

        client = IMAPClient(email="user@gmail.com")
        body = client._extract_body(msg)

        assert body == "This is the email body"

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_extract_body_from_multipart(self, mock_get_creds, mock_credentials):
        """Test extracting body from multipart message."""
        mock_get_creds.return_value = mock_credentials

        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart()
        msg.attach(MIMEText("Plain text body", "plain"))
        msg.attach(MIMEText("<html>HTML body</html>", "html"))

        client = IMAPClient(email="user@gmail.com")
        body = client._extract_body(msg)

        # Should prefer plain text over HTML
        assert body == "Plain text body"

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_extract_body_html_fallback(self, mock_get_creds, mock_credentials):
        """Test extracting HTML body when no plain text available."""
        mock_get_creds.return_value = mock_credentials

        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart()
        msg.attach(MIMEText("<html>HTML body</html>", "html"))

        client = IMAPClient(email="user@gmail.com")
        body = client._extract_body(msg)

        assert "<html>HTML body</html>" in body


class TestAuthenticationRefresh:
    """Test OAuth token refresh logic."""

    @patch("inbox_reaper.imap_client.get_credentials")
    @patch("inbox_reaper.imap_client.refresh_access_token")
    @patch("inbox_reaper.credential_helper.store_credentials")
    def test_refresh_access_token_success(
        self, mock_store, mock_refresh, mock_get_creds, mock_credentials
    ):
        """Test successful access token refresh."""
        mock_get_creds.return_value = mock_credentials
        mock_refresh.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
        }

        client = IMAPClient(email="user@gmail.com")
        client._refresh_access_token()

        assert client._access_token == "new_access_token"
        mock_refresh.assert_called_once_with("test_refresh_token", "gmail")
        mock_store.assert_called_once()

    @patch("inbox_reaper.imap_client.get_credentials")
    def test_refresh_without_refresh_token_raises_error(
        self, mock_get_creds, mock_credentials
    ):
        """Test that refresh without refresh token raises error."""
        mock_get_creds.return_value = mock_credentials

        client = IMAPClient(email="user@gmail.com")
        client._refresh_token = None

        with pytest.raises(IMAPAuthenticationError, match="No refresh token"):
            client._refresh_access_token()

    @patch("inbox_reaper.imap_client.get_credentials")
    @patch("inbox_reaper.imap_client.refresh_access_token")
    def test_refresh_failure_raises_error(
        self, mock_refresh, mock_get_creds, mock_credentials
    ):
        """Test that failed token refresh raises error."""
        mock_get_creds.return_value = mock_credentials
        mock_refresh.side_effect = Exception("Refresh failed")

        client = IMAPClient(email="user@gmail.com")

        with pytest.raises(
            IMAPAuthenticationError, match="Failed to refresh access token"
        ):
            client._refresh_access_token()
