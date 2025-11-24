"""Tests for credential storage using keyring."""

import json
from unittest.mock import patch

from inbox_reaper.credential_helper import (
    SERVICE_NAME,
    erase_credentials,
    get_credentials,
    list_accounts,
    store_credentials,
)


class TestGetCredentials:
    """Tests for retrieving credentials."""

    @patch("inbox_reaper.credential_helper.keyring.get_password")
    def test_get_existing_credentials(self, mock_get_password):
        """Test retrieving existing credentials."""
        # Setup mock
        test_creds = {
            "email": "test@example.com",
            "provider": "gmail",
            "access_token": "token123",
            "refresh_token": "refresh123",
            "expires_at": "2025-11-24T12:00:00",
            "token_type": "Bearer",
        }
        mock_get_password.return_value = json.dumps(test_creds)

        # Test retrieval
        result = get_credentials("test@example.com")

        # Verify result
        assert result == test_creds

        # Verify keyring call
        mock_get_password.assert_called_once_with(SERVICE_NAME, "test@example.com")

    @patch("inbox_reaper.credential_helper.keyring.get_password")
    def test_get_nonexistent_credentials(self, mock_get_password):
        """Test retrieving credentials that don't exist."""
        # Setup mock to return None
        mock_get_password.return_value = None

        # Test retrieval
        result = get_credentials("nonexistent@example.com")

        # Verify result is None
        assert result is None

        # Verify keyring call
        mock_get_password.assert_called_once_with(
            SERVICE_NAME, "nonexistent@example.com"
        )

    @patch("inbox_reaper.credential_helper.keyring.get_password")
    def test_get_credentials_handles_json_parsing(self, mock_get_password):
        """Test that JSON parsing works correctly."""
        # Setup mock with complex JSON
        test_creds = {
            "email": "test@example.com",
            "provider": "outlook",
            "access_token": "token_with_special_chars_!@#$%",
            "refresh_token": "refresh_with_unicode_\u00e9",
            "expires_at": "2025-12-31T23:59:59",
            "token_type": "Bearer",
            "extra_field": {"nested": "value"},
        }
        mock_get_password.return_value = json.dumps(test_creds)

        # Test retrieval
        result = get_credentials("test@example.com")

        # Verify all fields parsed correctly
        assert result == test_creds


class TestStoreCredentials:
    """Tests for storing credentials."""

    @patch("inbox_reaper.credential_helper.keyring.set_password")
    def test_store_credentials(self, mock_set_password):
        """Test storing credentials."""
        # Test data
        test_email = "test@example.com"
        test_creds = {
            "email": test_email,
            "provider": "gmail",
            "access_token": "token123",
            "refresh_token": "refresh123",
            "expires_at": "2025-11-24T12:00:00",
            "token_type": "Bearer",
        }

        # Store credentials
        store_credentials(test_email, test_creds)

        # Verify keyring call
        mock_set_password.assert_called_once()
        call_args = mock_set_password.call_args
        assert call_args[0][0] == SERVICE_NAME
        assert call_args[0][1] == test_email

        # Verify JSON encoding
        stored_json = call_args[0][2]
        assert json.loads(stored_json) == test_creds

    @patch("inbox_reaper.credential_helper.keyring.set_password")
    def test_store_credentials_overwrites_existing(self, mock_set_password):
        """Test that storing credentials overwrites existing ones."""
        # Store credentials twice
        email = "test@example.com"
        creds1 = {"access_token": "old_token"}
        creds2 = {"access_token": "new_token"}

        store_credentials(email, creds1)
        store_credentials(email, creds2)

        # Verify called twice
        assert mock_set_password.call_count == 2

        # Verify second call has new credentials
        last_call = mock_set_password.call_args
        stored_json = last_call[0][2]
        assert json.loads(stored_json) == creds2

    @patch("inbox_reaper.credential_helper.keyring.set_password")
    def test_store_credentials_handles_special_characters(self, mock_set_password):
        """Test storing credentials with special characters."""
        test_creds = {
            "email": "test+alias@example.com",
            "access_token": "token_with_!@#$%^&*()",
            "unicode_field": "value_with_émojis_🎉",
        }

        store_credentials("test+alias@example.com", test_creds)

        # Verify JSON encoding handles special characters
        stored_json = mock_set_password.call_args[0][2]
        assert json.loads(stored_json) == test_creds


class TestEraseCredentials:
    """Tests for erasing credentials."""

    @patch("inbox_reaper.credential_helper.keyring.delete_password")
    def test_erase_existing_credentials(self, mock_delete_password):
        """Test erasing existing credentials."""
        # Erase credentials
        erase_credentials("test@example.com")

        # Verify keyring call
        mock_delete_password.assert_called_once_with(SERVICE_NAME, "test@example.com")

    @patch("inbox_reaper.credential_helper.keyring.delete_password")
    def test_erase_nonexistent_credentials(self, mock_delete_password):
        """Test erasing credentials that don't exist (should not raise error)."""
        # Setup mock to raise PasswordDeleteError
        import keyring.errors

        mock_delete_password.side_effect = keyring.errors.PasswordDeleteError()

        # Should not raise exception
        erase_credentials("nonexistent@example.com")

        # Verify keyring call was made
        mock_delete_password.assert_called_once_with(
            SERVICE_NAME, "nonexistent@example.com"
        )

    @patch("inbox_reaper.credential_helper.keyring.delete_password")
    def test_erase_credentials_multiple_accounts(self, mock_delete_password):
        """Test erasing credentials for multiple accounts."""
        # Erase multiple accounts
        emails = ["user1@example.com", "user2@example.com", "user3@example.com"]

        for email in emails:
            erase_credentials(email)

        # Verify all were deleted
        assert mock_delete_password.call_count == 3
        for i, email in enumerate(emails):
            assert mock_delete_password.call_args_list[i][0] == (SERVICE_NAME, email)


class TestListAccounts:
    """Tests for listing accounts."""

    def test_list_accounts_returns_empty_list(self):
        """Test that list_accounts returns empty list due to keyring limitations."""
        # Should return empty list
        result = list_accounts()

        assert result == []
        assert isinstance(result, list)


class TestServiceName:
    """Tests for service name constant."""

    def test_service_name_constant(self):
        """Test that SERVICE_NAME is correctly defined."""
        assert SERVICE_NAME == "inbox-reaper-oauth"
        assert isinstance(SERVICE_NAME, str)
