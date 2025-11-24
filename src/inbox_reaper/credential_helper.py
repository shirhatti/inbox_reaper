"""Credential storage using the keyring library.

This module provides a simple wrapper around the keyring library for
secure credential storage across all platforms.
"""

import json

import keyring

SERVICE_NAME = "inbox-reaper-oauth"


def get_credentials(account: str) -> dict | None:
    """Retrieve credentials for an account.

    Args:
        account: Email account identifier

    Returns:
        Dictionary containing credentials, or None if not found
    """
    credentials_json = keyring.get_password(SERVICE_NAME, account)
    if credentials_json:
        return json.loads(credentials_json)  # type: ignore[no-any-return]
    return None


def store_credentials(account: str, credentials: dict) -> None:
    """Store credentials for an account.

    Args:
        account: Email account identifier
        credentials: Dictionary containing credentials to store
    """
    credentials_json = json.dumps(credentials)
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
