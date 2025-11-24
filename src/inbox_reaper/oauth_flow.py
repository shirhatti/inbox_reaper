"""OAuth authentication flow for email providers.

This module handles the OAuth 2.0 authentication flow using device code flow:
- Device authorization request
- User code display
- Token polling and retrieval
- Token refresh
"""

import base64
import imaplib
import time

import requests

from .oauth_config import get_oauth_config


def perform_oauth_flow(email: str, provider: str) -> dict:
    """Perform OAuth 2.0 authentication using device code flow.

    Args:
        email: User's email address
        provider: Email provider ('gmail' or 'outlook')

    Returns:
        Dictionary containing OAuth tokens

    Raises:
        ValueError: If authentication fails
        RuntimeError: If OAuth flow encounters an error
    """
    config = get_oauth_config(provider)

    # Step 1: Request device and user codes
    device_code_data = {
        "client_id": config["client_id"],
        "scope": config["scope"],
    }

    response = requests.post(config["device_code_uri"], data=device_code_data)
    response.raise_for_status()
    device_response = response.json()

    # Extract values from response
    device_code = device_response["device_code"]
    user_code = device_response["user_code"]
    verification_uri = device_response["verification_uri"]
    interval = device_response.get("interval", 5)  # Default to 5 seconds
    expires_in = device_response.get("expires_in", 900)  # Default to 15 minutes

    # Step 2: Display instructions to user
    print("\n" + "=" * 60)
    print("DEVICE AUTHENTICATION REQUIRED")
    print("=" * 60)
    print(f"\n1. Visit: {verification_uri}")
    print(f"2. Enter code: {user_code}")
    print(f"\nCode expires in {expires_in // 60} minutes.")
    print("Waiting for you to complete authentication...\n")

    # Step 3: Poll for token
    token_data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "client_id": config["client_id"],
        "device_code": device_code,
    }

    # Add client_secret if available (for Gmail)
    if config.get("client_secret"):
        token_data["client_secret"] = config["client_secret"]

    start_time = time.time()
    while True:
        # Check if we've exceeded the expiration time
        if time.time() - start_time > expires_in:
            raise RuntimeError("Device code expired. Please try again.")

        time.sleep(interval)

        token_response = requests.post(config["token_uri"], data=token_data)
        token_json = token_response.json()

        # Check for errors
        if "error" in token_json:
            error = token_json["error"]

            if error == "authorization_pending":
                # User hasn't completed authentication yet, keep polling
                print(".", end="", flush=True)
                continue
            elif error == "slow_down":
                # We're polling too fast, increase interval
                interval += 5
                continue
            elif error == "authorization_declined":
                raise RuntimeError("User declined authorization")
            elif error == "expired_token":
                raise RuntimeError("Device code expired")
            else:
                error_desc = token_json.get("error_description", "")
                raise RuntimeError(f"OAuth error: {error} - {error_desc}")

        # Success! We have tokens
        if "access_token" in token_json:
            print("\n\n✓ Authentication successful!")
            return token_json  # type: ignore[no-any-return]

        # Unexpected response
        raise RuntimeError(f"Unexpected response: {token_json}")


def refresh_access_token(refresh_token: str, provider: str) -> dict:
    """Refresh an expired access token.

    Args:
        refresh_token: OAuth refresh token
        provider: Email provider ('gmail' or 'outlook')

    Returns:
        Dictionary containing new OAuth tokens

    Raises:
        RuntimeError: If token refresh fails
    """
    config = get_oauth_config(provider)

    # Prepare token refresh request
    token_data = {
        "grant_type": "refresh_token",
        "client_id": config["client_id"],
        "refresh_token": refresh_token,
    }

    # Add client_secret if available (for Gmail)
    if config.get("client_secret"):
        token_data["client_secret"] = config["client_secret"]

    # Request new tokens
    response = requests.post(config["token_uri"], data=token_data)
    response.raise_for_status()

    return response.json()  # type: ignore[no-any-return]


def generate_xoauth2_string(email: str, access_token: str) -> str:
    """Generate XOAUTH2 authentication string for IMAP/SMTP.

    Args:
        email: User's email address
        access_token: OAuth access token

    Returns:
        Base64-encoded XOAUTH2 string
    """
    auth_string = f"user={email}\x01auth=Bearer {access_token}\x01\x01"
    return base64.b64encode(auth_string.encode()).decode()


def verify_imap_connection(
    email: str, access_token: str, provider: str
) -> tuple[bool, str]:
    """Verify IMAP connection with OAuth credentials.

    Args:
        email: User's email address
        access_token: OAuth access token
        provider: Email provider ('gmail' or 'outlook')

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Determine IMAP host
        if provider == "gmail":
            imap_host = "imap.gmail.com"
        elif provider == "outlook":
            imap_host = "outlook.office365.com"
        else:
            return False, f"Unknown provider: {provider}"

        # Connect to IMAP server
        imap = imaplib.IMAP4_SSL(imap_host, 993)

        # Authenticate using XOAUTH2
        auth_string = generate_xoauth2_string(email, access_token)
        imap.authenticate("XOAUTH2", lambda x: auth_string.encode())  # type: ignore[arg-type,return-value]

        # Select INBOX to verify connection
        imap.select("INBOX")
        typ, data = imap.search(None, "ALL")

        if typ == "OK":
            num_messages = len(data[0].split())
            imap.logout()
            return True, f"Connected successfully! {num_messages} messages in INBOX"
        else:
            imap.logout()
            return False, "Failed to select INBOX"

    except Exception as e:
        return False, f"Connection failed: {str(e)}"
