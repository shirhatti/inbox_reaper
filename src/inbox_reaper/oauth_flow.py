"""OAuth authentication flow for email providers.

This module handles OAuth 2.0 authentication flows:
- PKCE redirect flow (for Gmail)
- Device code flow (for Outlook)
- Token refresh
"""

import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import requests
from authlib.common.security import generate_token
from authlib.integrations.requests_client import OAuth2Session

from .imap_client import AsyncIMAPClient, IMAPAuthError, IMAPConnectionError
from .oauth_config import get_oauth_config


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    auth_code: str | None = None
    error: str | None = None

    def do_GET(self):  # noqa: N802
        """Handle GET request from OAuth redirect."""
        # Parse the query parameters
        query_components = parse_qs(urlparse(self.path).query)

        if "code" in query_components:
            # Success - got authorization code
            OAuthCallbackHandler.auth_code = query_components["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
                <html>
                <head><title>Authentication Successful</title></head>
                <body>
                    <h1>Authentication Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """
            )
        elif "error" in query_components:
            # Error occurred
            OAuthCallbackHandler.error = query_components["error"][0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"""
                <html>
                <head><title>Authentication Failed</title></head>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>Error: {OAuthCallbackHandler.error}</p>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """.encode()
            )
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Invalid request")

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


def _perform_redirect_flow(email: str, provider: str, config: dict) -> dict:
    """Perform OAuth 2.0 authentication using PKCE redirect flow.

    Args:
        email: User's email address
        provider: Email provider ('gmail' or 'outlook')
        config: OAuth configuration dictionary

    Returns:
        Dictionary containing OAuth tokens

    Raises:
        RuntimeError: If OAuth flow encounters an error
    """
    # Create OAuth2Session with PKCE support
    session = OAuth2Session(
        client_id=config["client_id"],
        client_secret=config.get("client_secret"),  # None for public clients
        redirect_uri=config["redirect_uri"],
        scope=config["scope"],
        code_challenge_method="S256",  # Use SHA256 for PKCE
    )

    # Generate PKCE code verifier
    code_verifier = generate_token(48)

    # Build authorization URL with PKCE
    auth_params = {}

    # Add email hint for better UX
    if provider == "gmail":
        auth_params["login_hint"] = email

    auth_url, state = session.create_authorization_url(
        config["auth_uri"], code_verifier=code_verifier, **auth_params
    )

    print("\nOpening browser for authentication...")
    print(f"If the browser doesn't open, visit this URL:\n{auth_url}\n")

    # Open browser
    webbrowser.open(auth_url)

    # Parse redirect URI to determine port
    parsed_uri = urlparse(config["redirect_uri"])
    port = parsed_uri.port or 80

    # Start local server to receive callback
    server = HTTPServer(("127.0.0.1", port), OAuthCallbackHandler)

    print(f"Waiting for authentication callback on port {port}...")

    # Handle one request (the OAuth callback)
    server.handle_request()

    if OAuthCallbackHandler.error:
        raise RuntimeError(f"OAuth authentication failed: {OAuthCallbackHandler.error}")

    if not OAuthCallbackHandler.auth_code:
        raise RuntimeError("No authorization code received")

    # Build authorization response URL
    authorization_response = (
        f"{config['redirect_uri']}?code={OAuthCallbackHandler.auth_code}&state={state}"
    )

    # Exchange authorization code for tokens (authlib handles PKCE automatically)
    tokens = session.fetch_token(
        config["token_uri"],
        authorization_response=authorization_response,
        code_verifier=code_verifier,
    )

    return tokens  # type: ignore[no-any-return]


def _perform_device_code_flow(email: str, provider: str, config: dict) -> dict:
    """Perform OAuth 2.0 authentication using device code flow.

    Args:
        email: User's email address
        provider: Email provider ('gmail' or 'outlook')
        config: OAuth configuration dictionary

    Returns:
        Dictionary containing OAuth tokens

    Raises:
        RuntimeError: If OAuth flow encounters an error
    """
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


def perform_oauth_flow(email: str, provider: str) -> dict:
    """Perform OAuth 2.0 authentication flow.

    Uses the appropriate flow based on provider configuration:
    - PKCE redirect flow for Gmail
    - Device code flow for Outlook

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
    flow_type = config.get("flow_type", "device_code")

    if flow_type == "redirect":
        return _perform_redirect_flow(email, provider, config)
    elif flow_type == "device_code":
        return _perform_device_code_flow(email, provider, config)
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")


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
    flow_type = config.get("flow_type", "device_code")

    # Use authlib for redirect flow providers (Gmail)
    if flow_type == "redirect":
        session = OAuth2Session(
            client_id=config["client_id"],
            client_secret=config.get("client_secret"),
            token={"refresh_token": refresh_token},
        )
        tokens = session.refresh_token(config["token_uri"], refresh_token=refresh_token)
        return tokens  # type: ignore[no-any-return]

    # Use simple requests for device code flow providers (Outlook)
    else:
        token_data = {
            "grant_type": "refresh_token",
            "client_id": config["client_id"],
            "refresh_token": refresh_token,
        }

        # Add client_secret if available
        if config.get("client_secret"):
            token_data["client_secret"] = config["client_secret"]

        # Request new tokens
        response = requests.post(config["token_uri"], data=token_data)
        response.raise_for_status()

        return response.json()  # type: ignore[no-any-return]


async def verify_imap_connection(
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
        # Use async IMAP client
        async with AsyncIMAPClient(email, access_token, provider) as client:
            # Select INBOX to verify connection
            mailbox_info = await client.select_mailbox("INBOX")
            num_messages = mailbox_info.get("exists", 0)

            return True, f"Connected successfully! {num_messages} messages in INBOX"

    except IMAPAuthError as e:
        # Authentication failed - token may need refresh
        return False, f"Authentication failed (token may need refresh): {str(e)}"
    except IMAPConnectionError as e:
        # Connection failed
        return False, f"Connection failed: {str(e)}"
    except Exception as e:
        # Unexpected error
        return False, f"Unexpected error: {str(e)}"
