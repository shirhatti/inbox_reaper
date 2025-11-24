"""OAuth configuration for Gmail and Outlook using Thunderbird's OAuth credentials.

This module provides OAuth configurations for popular email providers,
using Thunderbird's publicly available OAuth client IDs.
"""

OAUTH_CONFIGS = {
    'gmail': {
        'client_id': '406964657835-aq8lmia8j95dhl1a2bvharmfk3t1hgqj.apps.googleusercontent.com',
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'scope': 'https://mail.google.com/',
        'redirect_uri': 'http://127.0.0.1:9004'  # Thunderbird uses various ports
    },
    'outlook': {
        'client_id': '08162f7c-0fd2-4200-a84a-f25a4db0b584',  # Thunderbird's Azure app
        'auth_uri': 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
        'token_uri': 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
        'scope': 'https://outlook.office.com/IMAP.AccessAsUser.All https://outlook.office.com/SMTP.Send offline_access',
        'redirect_uri': 'http://localhost:9004'
    }
}


def get_oauth_config(provider: str) -> dict:
    """Get OAuth configuration for a specific provider.

    Args:
        provider: Email provider name ('gmail' or 'outlook')

    Returns:
        OAuth configuration dictionary

    Raises:
        ValueError: If provider is not supported
    """
    if provider not in OAUTH_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(OAUTH_CONFIGS.keys())}")

    return OAUTH_CONFIGS[provider]


def detect_provider(email: str) -> str:
    """Auto-detect email provider from email address.

    Args:
        email: Email address

    Returns:
        Provider name ('gmail' or 'outlook')

    Raises:
        ValueError: If provider cannot be detected
    """
    email_lower = email.lower()

    if '@gmail.com' in email_lower or '@googlemail.com' in email_lower:
        return 'gmail'
    elif '@outlook.com' in email_lower or '@hotmail.com' in email_lower or '@live.com' in email_lower:
        return 'outlook'
    else:
        raise ValueError(
            f"Cannot auto-detect provider for {email}. "
            "Please specify --provider explicitly (gmail or outlook)"
        )
