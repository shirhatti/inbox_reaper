"""Tests for OAuth authentication flow."""

import base64
from unittest.mock import Mock, patch

from inbox_reaper.oauth_flow import (
    generate_xoauth2_string,
    refresh_access_token,
    verify_imap_connection,
)


class TestGenerateXOAuth2String:
    """Tests for XOAUTH2 string generation."""

    def test_basic_format(self):
        """Test that XOAUTH2 string has correct format."""
        email = "test@example.com"
        token = "test_access_token"

        result = generate_xoauth2_string(email, token)

        # Result should be raw string (not base64-encoded)
        expected = f"user={email}\x01auth=Bearer {token}\x01\x01"
        assert result == expected

    def test_returns_string(self):
        """Test that result is a string."""
        result = generate_xoauth2_string("test@example.com", "token123")

        # Should be a string
        assert isinstance(result, str)

        # Should contain expected components
        assert "user=test@example.com" in result
        assert "auth=Bearer token123" in result

    def test_different_emails_produce_different_strings(self):
        """Test that different emails produce different auth strings."""
        token = "same_token"
        result1 = generate_xoauth2_string("user1@example.com", token)
        result2 = generate_xoauth2_string("user2@example.com", token)

        assert result1 != result2

    def test_different_tokens_produce_different_strings(self):
        """Test that different tokens produce different auth strings."""
        email = "test@example.com"
        result1 = generate_xoauth2_string(email, "token1")
        result2 = generate_xoauth2_string(email, "token2")

        assert result1 != result2


class TestVerifyImapConnection:
    """Tests for IMAP connection verification."""

    @patch("imaplib.IMAP4_SSL")
    def test_successful_gmail_connection(self, mock_imap_class):
        """Test successful Gmail IMAP connection."""
        # Setup mock
        mock_imap = Mock()
        mock_imap_class.return_value = mock_imap
        mock_imap.authenticate.return_value = ("OK", [b"Success"])
        mock_imap.select.return_value = ("OK", [b"10"])
        mock_imap.search.return_value = ("OK", [b"1 2 3 4 5 6 7 8 9 10"])

        # Test connection
        success, message = verify_imap_connection(
            "test@gmail.com", "access_token_123", "gmail"
        )

        # Verify results
        assert success is True
        assert "Connected successfully" in message
        assert "10 messages" in message

        # Verify IMAP calls
        mock_imap_class.assert_called_once_with("imap.gmail.com", 993)
        mock_imap.authenticate.assert_called_once()
        mock_imap.select.assert_called_once_with("INBOX")
        mock_imap.logout.assert_called_once()

    @patch("imaplib.IMAP4_SSL")
    def test_successful_outlook_connection(self, mock_imap_class):
        """Test successful Outlook IMAP connection."""
        # Setup mock
        mock_imap = Mock()
        mock_imap_class.return_value = mock_imap
        mock_imap.authenticate.return_value = ("OK", [b"Success"])
        mock_imap.select.return_value = ("OK", [b"5"])
        mock_imap.search.return_value = ("OK", [b"1 2 3 4 5"])

        # Test connection
        success, message = verify_imap_connection(
            "test@outlook.com", "access_token_456", "outlook"
        )

        # Verify results
        assert success is True
        assert "Connected successfully" in message
        assert "5 messages" in message

        # Verify correct IMAP host
        mock_imap_class.assert_called_once_with("outlook.office365.com", 993)

    @patch("imaplib.IMAP4_SSL")
    def test_authentication_callback_returns_bytes(self, mock_imap_class):
        """Test that authenticate callback returns bytes, not string."""
        # Setup mock to capture the callback
        mock_imap = Mock()
        mock_imap_class.return_value = mock_imap
        mock_imap.search.return_value = ("OK", [b""])

        callback = None

        def capture_callback(mechanism, cb):
            nonlocal callback
            callback = cb
            return ("OK", [b"Success"])

        mock_imap.authenticate.side_effect = capture_callback

        # Test connection
        verify_imap_connection("test@gmail.com", "token", "gmail")

        # Verify callback returns bytes
        assert callback is not None
        result = callback(b"")
        assert isinstance(result, bytes)

    @patch("imaplib.IMAP4_SSL")
    def test_unknown_provider(self, mock_imap_class):
        """Test handling of unknown provider."""
        success, message = verify_imap_connection(
            "test@example.com", "token", "unknown_provider"
        )

        assert success is False
        assert "Unknown provider" in message
        mock_imap_class.assert_not_called()

    @patch("imaplib.IMAP4_SSL")
    def test_authentication_failure(self, mock_imap_class):
        """Test handling of authentication failure."""
        # Setup mock to raise exception
        mock_imap = Mock()
        mock_imap_class.return_value = mock_imap
        mock_imap.authenticate.side_effect = Exception(
            "AUTHENTICATE command error: BAD"
        )

        # Test connection
        success, message = verify_imap_connection(
            "test@gmail.com", "invalid_token", "gmail"
        )

        # Verify failure
        assert success is False
        assert "Connection failed" in message
        assert "AUTHENTICATE command error" in message

    @patch("imaplib.IMAP4_SSL")
    def test_select_inbox_failure(self, mock_imap_class):
        """Test handling of SELECT INBOX failure."""
        # Setup mock
        mock_imap = Mock()
        mock_imap_class.return_value = mock_imap
        mock_imap.authenticate.return_value = ("OK", [b"Success"])
        mock_imap.select.return_value = ("OK", [b"0"])
        mock_imap.search.return_value = ("NO", [])

        # Test connection
        success, message = verify_imap_connection(
            "test@gmail.com", "access_token", "gmail"
        )

        # Verify failure
        assert success is False
        assert "Failed to select INBOX" in message
        mock_imap.logout.assert_called_once()

    @patch("imaplib.IMAP4_SSL")
    def test_empty_inbox(self, mock_imap_class):
        """Test connection with empty inbox."""
        # Setup mock
        mock_imap = Mock()
        mock_imap_class.return_value = mock_imap
        mock_imap.authenticate.return_value = ("OK", [b"Success"])
        mock_imap.select.return_value = ("OK", [b"0"])
        mock_imap.search.return_value = ("OK", [b""])

        # Test connection
        success, message = verify_imap_connection(
            "test@gmail.com", "access_token", "gmail"
        )

        # Verify success with 0 messages
        assert success is True
        assert "0 messages" in message


class TestRefreshAccessToken:
    """Tests for token refresh functionality."""

    @patch("inbox_reaper.oauth_flow.OAuth2Session")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_successful_token_refresh(self, mock_get_config, mock_session_class):
        """Test successful token refresh."""
        # Setup mock config
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "client_secret": "test_secret",
            "token_uri": "https://oauth.example.com/token",
        }

        # Setup mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        expected_tokens = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_session.refresh_token.return_value = expected_tokens

        # Test token refresh
        result = refresh_access_token("old_refresh_token", "gmail")

        # Verify result
        assert result == expected_tokens

        # Verify session creation
        mock_session_class.assert_called_once_with(
            client_id="test_client_id",
            client_secret="test_secret",
            token={"refresh_token": "old_refresh_token"},
        )

        # Verify refresh call
        mock_session.refresh_token.assert_called_once_with(
            "https://oauth.example.com/token", refresh_token="old_refresh_token"
        )

    @patch("inbox_reaper.oauth_flow.OAuth2Session")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_token_refresh_without_client_secret(
        self, mock_get_config, mock_session_class
    ):
        """Test token refresh for provider without client secret (public client)."""
        # Setup mock config without client_secret
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "token_uri": "https://oauth.example.com/token",
        }

        # Setup mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.refresh_token.return_value = {"access_token": "new_token"}

        # Test token refresh
        refresh_access_token("refresh_token", "outlook")

        # Verify session created with None for client_secret
        mock_session_class.assert_called_once_with(
            client_id="test_client_id",
            client_secret=None,
            token={"refresh_token": "refresh_token"},
        )
