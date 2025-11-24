"""OAuth authentication flow for email providers.

This module handles the OAuth 2.0 authentication flow, including:
- Authorization URL generation
- Local HTTP server for redirect handling
- Token exchange
- Token refresh
"""

import base64
import json
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests

from .oauth_config import get_oauth_config


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    auth_code: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):
        """Handle GET request from OAuth redirect."""
        # Parse the query parameters
        query_components = parse_qs(urlparse(self.path).query)

        if 'code' in query_components:
            # Success - got authorization code
            OAuthCallbackHandler.auth_code = query_components['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''
                <html>
                <head><title>Authentication Successful</title></head>
                <body>
                    <h1>Authentication Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            ''')
        elif 'error' in query_components:
            # Error occurred
            OAuthCallbackHandler.error = query_components['error'][0]
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f'''
                <html>
                <head><title>Authentication Failed</title></head>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>Error: {OAuthCallbackHandler.error}</p>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            '''.encode())
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Invalid request')

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


def perform_oauth_flow(email: str, provider: str) -> Dict:
    """Perform OAuth 2.0 authentication flow.

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

    # Build authorization URL
    auth_params = {
        'client_id': config['client_id'],
        'response_type': 'code',
        'redirect_uri': config['redirect_uri'],
        'scope': config['scope'],
    }

    # Add email hint for better UX
    if provider == 'gmail':
        auth_params['login_hint'] = email
    elif provider == 'outlook':
        auth_params['login_hint'] = email

    auth_url = f"{config['auth_uri']}?{urllib.parse.urlencode(auth_params)}"

    print(f"\nOpening browser for authentication...")
    print(f"If the browser doesn't open, visit this URL:\n{auth_url}\n")

    # Open browser
    webbrowser.open(auth_url)

    # Start local server to receive callback
    port = int(config['redirect_uri'].split(':')[-1].rstrip('/'))
    server = HTTPServer(('localhost', port), OAuthCallbackHandler)

    print(f"Waiting for authentication callback on port {port}...")

    # Handle one request (the OAuth callback)
    server.handle_request()

    if OAuthCallbackHandler.error:
        raise RuntimeError(f"OAuth authentication failed: {OAuthCallbackHandler.error}")

    if not OAuthCallbackHandler.auth_code:
        raise RuntimeError("No authorization code received")

    # Exchange authorization code for tokens
    token_params = {
        'client_id': config['client_id'],
        'code': OAuthCallbackHandler.auth_code,
        'redirect_uri': config['redirect_uri'],
        'grant_type': 'authorization_code',
    }

    response = requests.post(config['token_uri'], data=token_params)

    if response.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {response.text}")

    tokens = response.json()

    # Validate required fields
    if 'access_token' not in tokens:
        raise RuntimeError("No access token in response")

    return tokens


def refresh_access_token(refresh_token: str, provider: str) -> Dict:
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

    token_params = {
        'client_id': config['client_id'],
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token',
    }

    response = requests.post(config['token_uri'], data=token_params)

    if response.status_code != 200:
        raise RuntimeError(f"Token refresh failed: {response.text}")

    tokens = response.json()

    if 'access_token' not in tokens:
        raise RuntimeError("No access token in refresh response")

    return tokens


def generate_xoauth2_string(email: str, access_token: str) -> str:
    """Generate XOAUTH2 authentication string for IMAP/SMTP.

    Args:
        email: User's email address
        access_token: OAuth access token

    Returns:
        Base64-encoded XOAUTH2 string
    """
    auth_string = f'user={email}\x01auth=Bearer {access_token}\x01\x01'
    return base64.b64encode(auth_string.encode()).decode()


def test_imap_connection(email: str, access_token: str, provider: str) -> tuple[bool, str]:
    """Test IMAP connection with OAuth credentials.

    Args:
        email: User's email address
        access_token: OAuth access token
        provider: Email provider ('gmail' or 'outlook')

    Returns:
        Tuple of (success: bool, message: str)
    """
    import imaplib

    try:
        # Determine IMAP host
        if provider == 'gmail':
            imap_host = 'imap.gmail.com'
        elif provider == 'outlook':
            imap_host = 'outlook.office365.com'
        else:
            return False, f"Unknown provider: {provider}"

        # Connect to IMAP server
        imap = imaplib.IMAP4_SSL(imap_host, 993)

        # Authenticate using XOAUTH2
        auth_string = generate_xoauth2_string(email, access_token)
        imap.authenticate('XOAUTH2', lambda x: auth_string)

        # Select INBOX to verify connection
        imap.select('INBOX')
        typ, data = imap.search(None, 'ALL')

        if typ == 'OK':
            num_messages = len(data[0].split())
            imap.logout()
            return True, f"Connected successfully! {num_messages} messages in INBOX"
        else:
            imap.logout()
            return False, "Failed to select INBOX"

    except Exception as e:
        return False, f"Connection failed: {str(e)}"
