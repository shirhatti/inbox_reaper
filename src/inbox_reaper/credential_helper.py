"""Credential storage using the keyring library.

This module provides a simple wrapper around the keyring library for
secure credential storage across all platforms.
"""

import json
from datetime import datetime

import keyring

SERVICE_NAME = "inbox-reaper-oauth"


def _normalize_credentials(creds: dict) -> dict:
    """Normalize credentials to ensure consistent format.

    Handles legacy credentials that may have:
    - expires_at as Unix timestamp (int) instead of ISO string
    - Extra OAuth response fields (expires_in, scope)

    Args:
        creds: Raw credentials dictionary

    Returns:
        Normalized credentials dictionary
    """
    if not creds:
        return creds

    # Handle expires_at format conversion
    expires_at = creds.get("expires_at")
    if expires_at is not None and not isinstance(expires_at, str):
        # Convert Unix timestamp to ISO string
        if isinstance(expires_at, (int, float)):
            creds["expires_at"] = datetime.fromtimestamp(expires_at).isoformat()

    # Remove extra OAuth response fields that shouldn't be stored
    # Keep only the essential credential fields
    essential_fields = {
        "email",
        "provider",
        "access_token",
        "refresh_token",
        "expires_at",
        "token_type",
    }
    normalized = {k: v for k, v in creds.items() if k in essential_fields}

    return normalized


def get_credentials(account: str) -> dict | None:
    """Retrieve credentials for an account.

    Args:
        account: Email account identifier

    Returns:
        Dictionary containing credentials, or None if not found
    """
    credentials_json = keyring.get_password(SERVICE_NAME, account)
    if credentials_json:
        creds = json.loads(credentials_json)
        return _normalize_credentials(creds)
    return None


def store_credentials(account: str, credentials: dict) -> None:
    """Store credentials for an account.

    Args:
        account: Email account identifier
        credentials: Dictionary containing credentials to store
    """
    # Normalize credentials before storing to ensure consistent format
    normalized = _normalize_credentials(credentials)
    credentials_json = json.dumps(normalized)
    keyring.set_password(SERVICE_NAME, account, credentials_json)


def erase_credentials(account: str) -> None:
    """Remove credentials for an account.

    Args:
        account: Email account identifier
    """
    try:
        keyring.delete_password(SERVICE_NAME, account)
    except keyring.errors.PasswordDeleteError:
        # Account doesn't exist, that's fine
        pass


def list_accounts() -> list[str]:
    """List all stored account identifiers.

    Returns:
        List of account email addresses

    Note:
        This function has limited support as keyring doesn't provide
        a standard way to list all stored credentials. It will return
        an empty list, and accounts are discovered through get operations.
    """
    # keyring doesn't provide a standard way to list credentials
    # This limitation is acceptable as credentials are typically accessed directly
    return []
