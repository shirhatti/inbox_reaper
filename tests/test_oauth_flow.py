"""Tests for OAuth authentication flow."""

import asyncio
from unittest.mock import AsyncMock, Mock, call, patch

from inbox_reaper.imap_client import generate_xoauth2_string
from inbox_reaper.oauth_flow import (
    perform_oauth_flow,
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


class TestPerformOAuthFlow:
    """Tests for OAuth flow (both device code and redirect flows)."""

    @patch("inbox_reaper.oauth_flow.time.sleep")
    @patch("inbox_reaper.oauth_flow.requests.post")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_successful_device_flow(self, mock_get_config, mock_post, mock_sleep):
        """Test successful device code flow (Outlook)."""
        # Setup mock config
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "device_code_uri": "https://oauth.example.com/device/code",
            "token_uri": "https://oauth.example.com/token",
            "scope": "test_scope",
            "flow_type": "device_code",
        }

        # Mock device code response
        device_response = Mock()
        device_response.json.return_value = {
            "device_code": "test_device_code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://example.com/device",
            "interval": 1,
            "expires_in": 900,
        }

        # Mock token responses (first pending, then success)
        pending_response = Mock()
        pending_response.json.return_value = {"error": "authorization_pending"}

        success_response = Mock()
        success_response.json.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
        }

        # Set up post to return different responses
        mock_post.side_effect = [device_response, pending_response, success_response]

        # Run the flow
        result = perform_oauth_flow("test@example.com", "outlook")

        # Verify result
        assert result["access_token"] == "test_access_token"
        assert result["refresh_token"] == "test_refresh_token"

        # Verify calls
        assert mock_post.call_count == 3
        # First call: device code request
        assert mock_post.call_args_list[0] == call(
            "https://oauth.example.com/device/code",
            data={"client_id": "test_client_id", "scope": "test_scope"},
        )
        # Second call: first token poll (pending)
        assert mock_post.call_args_list[1] == call(
            "https://oauth.example.com/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": "test_client_id",
                "device_code": "test_device_code",
            },
        )

    @patch("inbox_reaper.oauth_flow.time.sleep")
    @patch("inbox_reaper.oauth_flow.requests.post")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_device_flow_with_client_secret(
        self, mock_get_config, mock_post, mock_sleep
    ):
        """Test device flow includes client_secret when available."""
        # Setup mock config with client_secret
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "client_secret": "test_secret",
            "device_code_uri": "https://oauth.example.com/device/code",
            "token_uri": "https://oauth.example.com/token",
            "scope": "test_scope",
            "flow_type": "device_code",
        }

        # Mock responses
        device_response = Mock()
        device_response.json.return_value = {
            "device_code": "test_device_code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://example.com/device",
            "interval": 1,
            "expires_in": 900,
        }

        success_response = Mock()
        success_response.json.return_value = {"access_token": "test_token"}

        mock_post.side_effect = [device_response, success_response]

        # Run the flow
        perform_oauth_flow("test@example.com", "outlook")

        # Verify token request includes client_secret
        token_call = mock_post.call_args_list[1]
        assert token_call[1]["data"]["client_secret"] == "test_secret"

    @patch("inbox_reaper.oauth_flow.time.time")
    @patch("inbox_reaper.oauth_flow.time.sleep")
    @patch("inbox_reaper.oauth_flow.requests.post")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_device_flow_expiration(
        self, mock_get_config, mock_post, mock_sleep, mock_time
    ):
        """Test device code flow handles expiration."""
        # Setup mock config
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "device_code_uri": "https://oauth.example.com/device/code",
            "token_uri": "https://oauth.example.com/token",
            "scope": "test_scope",
            "flow_type": "device_code",
        }

        # Mock device code response with short expiration
        device_response = Mock()
        device_response.json.return_value = {
            "device_code": "test_device_code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://example.com/device",
            "interval": 1,
            "expires_in": 10,  # 10 seconds
        }

        # Mock time to simulate expiration
        mock_time.side_effect = [0, 15]  # Start at 0, then jump to 15 seconds

        mock_post.return_value = device_response

        # Test that expiration raises error
        try:
            perform_oauth_flow("test@example.com", "outlook")
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError as e:
            assert "expired" in str(e).lower()

    @patch("inbox_reaper.oauth_flow.time.sleep")
    @patch("inbox_reaper.oauth_flow.requests.post")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_device_flow_user_declined(self, mock_get_config, mock_post, mock_sleep):
        """Test device flow handles user declining authorization."""
        # Setup mock config
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "device_code_uri": "https://oauth.example.com/device/code",
            "token_uri": "https://oauth.example.com/token",
            "scope": "test_scope",
            "flow_type": "device_code",
        }

        # Mock responses
        device_response = Mock()
        device_response.json.return_value = {
            "device_code": "test_device_code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://example.com/device",
            "interval": 1,
            "expires_in": 900,
        }

        declined_response = Mock()
        declined_response.json.return_value = {"error": "authorization_declined"}

        mock_post.side_effect = [device_response, declined_response]

        # Test that decline raises error
        try:
            perform_oauth_flow("test@example.com", "outlook")
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError as e:
            assert "declined" in str(e).lower()

    @patch("inbox_reaper.oauth_flow.time.sleep")
    @patch("inbox_reaper.oauth_flow.requests.post")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_device_flow_slow_down(self, mock_get_config, mock_post, mock_sleep):
        """Test device flow handles slow_down error by increasing interval."""
        # Setup mock config
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "device_code_uri": "https://oauth.example.com/device/code",
            "token_uri": "https://oauth.example.com/token",
            "scope": "test_scope",
            "flow_type": "device_code",
        }

        # Mock responses
        device_response = Mock()
        device_response.json.return_value = {
            "device_code": "test_device_code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://example.com/device",
            "interval": 5,
            "expires_in": 900,
        }

        slow_down_response = Mock()
        slow_down_response.json.return_value = {"error": "slow_down"}

        success_response = Mock()
        success_response.json.return_value = {"access_token": "test_token"}

        mock_post.side_effect = [
            device_response,
            slow_down_response,
            success_response,
        ]

        # Run the flow
        result = perform_oauth_flow("test@example.com", "outlook")

        # Verify success
        assert result["access_token"] == "test_token"

        # Verify sleep was called with increased interval (5 initial + 5 added = 10)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[1][0][0] == 10


class TestVerifyImapConnection:
    """Tests for IMAP connection verification."""

    @patch("inbox_reaper.oauth_flow.AsyncIMAPClient")
    def test_successful_gmail_connection(self, mock_client_class):
        """Test successful Gmail IMAP connection."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.select_mailbox.return_value = {"exists": 10, "recent": 2}
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Test connection (run async function)
        success, message = asyncio.run(
            verify_imap_connection("test@gmail.com", "access_token_123", "gmail")
        )

        # Verify results
        assert success is True
        assert "Connected successfully" in message
        assert "10 messages" in message

        # Verify client was created correctly
        mock_client_class.assert_called_once_with(
            "test@gmail.com", "access_token_123", "gmail"
        )
        mock_client.select_mailbox.assert_called_once_with("INBOX")

    @patch("inbox_reaper.oauth_flow.AsyncIMAPClient")
    def test_successful_outlook_connection(self, mock_client_class):
        """Test successful Outlook IMAP connection."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.select_mailbox.return_value = {"exists": 5, "recent": 1}
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Test connection
        success, message = asyncio.run(
            verify_imap_connection("test@outlook.com", "access_token_456", "outlook")
        )

        # Verify results
        assert success is True
        assert "Connected successfully" in message
        assert "5 messages" in message

        # Verify correct provider passed
        mock_client_class.assert_called_once_with(
            "test@outlook.com", "access_token_456", "outlook"
        )

    @patch("inbox_reaper.oauth_flow.AsyncIMAPClient")
    def test_unknown_provider(self, mock_client_class):
        """Test handling of unknown provider."""
        # Setup mock to raise ValueError for unknown provider
        mock_client_class.side_effect = ValueError("Unknown provider: unknown_provider")

        success, message = asyncio.run(
            verify_imap_connection("test@example.com", "token", "unknown_provider")
        )

        assert success is False
        assert "Unexpected error" in message or "Unknown provider" in message

    @patch("inbox_reaper.oauth_flow.AsyncIMAPClient")
    def test_authentication_failure(self, mock_client_class):
        """Test handling of authentication failure."""
        # Import the exception class
        from inbox_reaper.imap_client import IMAPAuthError

        # Setup mock to raise authentication error
        mock_client = AsyncMock()
        mock_client.__aenter__.side_effect = IMAPAuthError(
            "OAuth authentication failed"
        )
        mock_client_class.return_value = mock_client

        # Test connection
        success, message = asyncio.run(
            verify_imap_connection("test@gmail.com", "invalid_token", "gmail")
        )

        # Verify failure
        assert success is False
        assert "Authentication failed" in message
        assert "token may need refresh" in message

    @patch("inbox_reaper.oauth_flow.AsyncIMAPClient")
    def test_connection_failure(self, mock_client_class):
        """Test handling of connection failure."""
        # Import the exception class
        from inbox_reaper.imap_client import IMAPConnectionError

        # Setup mock to raise connection error
        mock_client = AsyncMock()
        mock_client.__aenter__.side_effect = IMAPConnectionError(
            "Failed to connect to IMAP server"
        )
        mock_client_class.return_value = mock_client

        # Test connection
        success, message = asyncio.run(
            verify_imap_connection("test@gmail.com", "token", "gmail")
        )

        # Verify failure
        assert success is False
        assert "Connection failed" in message

    @patch("inbox_reaper.oauth_flow.AsyncIMAPClient")
    def test_empty_inbox(self, mock_client_class):
        """Test connection with empty inbox."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.select_mailbox.return_value = {"exists": 0}
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Test connection
        success, message = asyncio.run(
            verify_imap_connection("test@gmail.com", "access_token", "gmail")
        )

        # Verify success with 0 messages
        assert success is True
        assert "0 messages" in message


class TestRefreshAccessToken:
    """Tests for token refresh functionality."""

    @patch("inbox_reaper.oauth_flow.OAuth2Session")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_successful_token_refresh_with_secret(
        self, mock_get_config, mock_session_class
    ):
        """Test successful token refresh with client secret (Gmail)."""
        # Setup mock config
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "client_secret": "test_secret",
            "token_uri": "https://oauth.example.com/token",
            "flow_type": "redirect",
        }

        # Setup mock OAuth2Session
        expected_tokens = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_session = Mock()
        mock_session.refresh_token.return_value = expected_tokens
        mock_session_class.return_value = mock_session

        # Test token refresh
        result = refresh_access_token("old_refresh_token", "gmail")

        # Verify result
        assert result == expected_tokens

        # Verify OAuth2Session was called with correct parameters
        mock_session_class.assert_called_once_with(
            client_id="test_client_id",
            client_secret="test_secret",
            token={"refresh_token": "old_refresh_token"},
        )

        # Verify refresh_token was called
        mock_session.refresh_token.assert_called_once_with(
            "https://oauth.example.com/token",
            refresh_token="old_refresh_token",
        )

    @patch("inbox_reaper.oauth_flow.requests.post")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_token_refresh_without_client_secret(self, mock_get_config, mock_post):
        """Test token refresh for provider without client secret (Outlook)."""
        # Setup mock config without client_secret
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "token_uri": "https://oauth.example.com/token",
            "flow_type": "device_code",
        }

        # Setup mock response
        expected_tokens = {"access_token": "new_token", "expires_in": 3600}
        mock_response = Mock()
        mock_response.json.return_value = expected_tokens
        mock_post.return_value = mock_response

        # Test token refresh
        result = refresh_access_token("refresh_token", "outlook")

        # Verify result
        assert result == expected_tokens

        # Verify POST request doesn't include client_secret
        mock_post.assert_called_once_with(
            "https://oauth.example.com/token",
            data={
                "grant_type": "refresh_token",
                "client_id": "test_client_id",
                "refresh_token": "refresh_token",
            },
        )

    @patch("inbox_reaper.oauth_flow.requests.post")
    @patch("inbox_reaper.oauth_flow.get_oauth_config")
    def test_token_refresh_http_error(self, mock_get_config, mock_post):
        """Test token refresh handles HTTP errors (device code flow)."""
        # Setup mock config
        mock_get_config.return_value = {
            "client_id": "test_client_id",
            "token_uri": "https://oauth.example.com/token",
            "flow_type": "device_code",
        }

        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_post.return_value = mock_response

        # Test that exception is raised
        try:
            refresh_access_token("invalid_token", "gmail")
            raise AssertionError("Should have raised exception")
        except Exception as e:
            assert "401 Unauthorized" in str(e)
