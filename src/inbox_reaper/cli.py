"""CLI entry point for inbox-reaper.

Provides a Click-based command-line interface for the email classification system.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import click

from . import credential_helper
from .oauth_config import detect_provider
from .oauth_flow import (
    perform_oauth_flow,
    refresh_access_token,
    verify_imap_connection,
)
from .state import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Inbox Reaper - Email classification and cleaning system.

    A hybrid deterministic + AI email classifier that processes emails through
    a multi-layer decision pipeline to efficiently classify and optionally
    delete marketing emails.
    """
    pass


@cli.command()
@click.option(
    "--model",
    default="gemma2:2b",
    help="Ollama model name for AI classification",
    show_default=True,
)
@click.option(
    "--ollama-url",
    default="http://localhost:11434",
    help="Ollama base URL",
    show_default=True,
)
@click.option(
    "--batch-size",
    default=50,
    type=int,
    help="Number of emails to process in each batch",
    show_default=True,
)
@click.option(
    "--concurrent-limit",
    default=25,
    type=int,
    help="Maximum concurrent AI classification requests",
    show_default=True,
)
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    help="Enable dry-run mode (no actual deletions)",
    show_default=True,
)
@click.option(
    "--checkpoint-path",
    default="checkpoints.db",
    type=click.Path(),
    help="Path to SQLite checkpoint database for resumable processing",
    show_default=True,
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume from checkpoint if available (default: ask user)",
    show_default=True,
)
@click.option(
    "--email",
    required=True,
    help="Email address to process",
)
@click.option(
    "--max-emails",
    type=int,
    help="Maximum number of emails to process (for testing)",
)
@click.option(
    "--keywords",
    multiple=True,
    help="Critical keywords to trigger KEEP decision (can be specified multiple times)",
)
@click.option(
    "--whitelist-domain",
    multiple=True,
    help="Whitelisted domains to trigger KEEP (can specify multiple times)",
)
def process(
    model: str,
    ollama_url: str,
    batch_size: int,
    concurrent_limit: int,
    dry_run: bool,
    checkpoint_path: str,
    resume: bool,
    email: str,
    max_emails: int | None,
    keywords: tuple,
    whitelist_domain: tuple,
):
    """Process emails through the classification pipeline with LangGraph.

    This command runs the email classification pipeline using LangGraph for
    orchestration, with support for checkpointing and resumable processing.

    Example:
        inbox-reaper process --email user@example.com --keywords "important"
        inbox-reaper process --email user@example.com --max-emails 10
    """
    click.echo("Inbox Reaper - Email Classification System (LangGraph)")
    click.echo("=" * 60)

    # Create configuration
    config = Config(
        email=email,
        model_name=model,
        ollama_base_url=ollama_url,
        batch_size=batch_size,
        concurrent_ai_limit=concurrent_limit,
        dry_run=dry_run,
        max_emails=max_emails,
        keywords=list(keywords),
        whitelist_domains=list(whitelist_domain),
    )

    click.echo("\nConfiguration:")
    click.echo(f"  Model: {config.model_name}")
    click.echo(f"  Ollama URL: {config.ollama_base_url}")
    click.echo(f"  Batch size: {config.batch_size}")
    click.echo(f"  Concurrent limit: {config.concurrent_ai_limit}")
    click.echo(f"  Dry run: {config.dry_run}")
    click.echo(f"  Checkpoint: {checkpoint_path}")
    click.echo(f"  Keywords: {config.keywords or 'None'}")
    click.echo(f"  Whitelisted domains: {config.whitelist_domains or 'None'}")

    # Run the LangGraph pipeline
    try:
        click.echo("\nStarting LangGraph pipeline execution (Streaming Mode)...")
        click.echo("-" * 60)

        import asyncio

        from .langgraph_streaming import run_streaming_pipeline

        asyncio.run(run_streaming_pipeline(config))

        click.echo("\n" + "=" * 60)
        click.echo("Processing Complete!")
        click.echo("=" * 60)

    except KeyboardInterrupt:
        click.echo("\n\nProcessing interrupted by user.")
        raise
    except Exception as e:
        click.echo(f"\nError during processing: {e}", err=True)
        raise


@cli.command()
def version():
    """Display version information."""
    click.echo("Inbox Reaper v0.1.0")
    click.echo("Email classification and cleaning system")


@cli.command()
@click.argument("email")
@click.option(
    "--provider",
    type=click.Choice(["gmail", "outlook"]),
    help="Email provider (auto-detected if not specified)",
)
def login(email: str, provider: str | None):
    """Authenticate and store OAuth credentials for an email account.

    This command opens a browser window for OAuth authentication and securely
    stores the credentials using the system's native credential storage.

    Example:
        inbox-reaper login user@gmail.com
        inbox-reaper login user@company.com --provider outlook
    """
    click.echo(f"Authenticating {email}...")

    # Auto-detect provider if not specified
    if not provider:
        try:
            provider = detect_provider(email)
            click.echo(f"Auto-detected provider: {provider}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            return

    # Perform OAuth flow
    try:
        tokens = perform_oauth_flow(email, provider)

        # Store credentials
        credentials = {
            "email": email,
            "provider": provider,
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token"),
            "expires_at": (
                datetime.now() + timedelta(seconds=tokens.get("expires_in", 3600))
            ).isoformat(),
            "token_type": tokens.get("token_type", "Bearer"),
        }

        credential_helper.store_credentials(email, credentials)
        click.echo(f"\n✓ Credentials saved for {email}")

    except Exception as e:
        click.echo(f"\nError during authentication: {e}", err=True)
        return


@cli.command()
@click.argument("email")
def logout(email: str):
    """Remove stored OAuth credentials for an email account.

    Example:
        inbox-reaper logout user@gmail.com
    """
    credential_helper.erase_credentials(email)
    click.echo(f"✓ Credentials removed for {email}")


@cli.command()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information including expiration status",
)
def accounts(verbose: bool):
    """List all stored email accounts.

    Example:
        inbox-reaper accounts
        inbox-reaper accounts --verbose

    Note:
        Due to limitations in the keyring library, this command cannot
        automatically list all stored accounts. Use 'inbox-reaper test <email>'
        to verify if credentials exist for a specific account.
    """
    click.echo("Account listing is not supported by the keyring library.")
    click.echo("\nTo check if credentials exist for a specific account:")
    click.echo("  inbox-reaper test <email>")
    click.echo("\nTo add a new account:")
    click.echo("  inbox-reaper login <email>")


@cli.command()
@click.argument("email")
def test(email: str):
    """Test IMAP connection with stored OAuth credentials.

    This command verifies that the stored credentials work by attempting
    to connect to the IMAP server and list messages in the INBOX.

    Example:
        inbox-reaper test user@gmail.com
    """
    # Get credentials
    creds = credential_helper.get_credentials(email)
    if not creds:
        click.echo(f"No credentials found for {email}", err=True)
        click.echo(f"\nUse 'inbox-reaper login {email}' to authenticate first.")
        return

    # Check if token expired and refresh if needed
    try:
        expires_at_str = creds.get("expires_at")
        if expires_at_str and isinstance(expires_at_str, str):
            expires = datetime.fromisoformat(expires_at_str)
            if expires < datetime.now():
                click.echo("Token expired, refreshing...")
                tokens = refresh_access_token(creds["refresh_token"], creds["provider"])
                creds["access_token"] = tokens["access_token"]
                creds["expires_at"] = (
                    datetime.now() + timedelta(seconds=tokens.get("expires_in", 3600))
                ).isoformat()
                credential_helper.store_credentials(email, creds)
                click.echo("✓ Token refreshed successfully")
        else:
            click.echo("Warning: Token expiration time missing or invalid", err=True)
    except Exception as e:
        click.echo(f"Warning: Could not refresh token: {e}", err=True)

    # Test connection
    click.echo(f"\nTesting IMAP connection for {email}...")
    success, message = asyncio.run(
        verify_imap_connection(email, creds["access_token"], creds["provider"])
    )

    if success:
        click.echo(f"✓ {message}")
    else:
        click.echo(f"✗ {message}", err=True)


@cli.command()
@click.option(
    "--email",
    required=True,
    help="Email address to fetch from",
)
@click.option(
    "--uid",
    required=True,
    help="UID of the email to fetch",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output .eml file path (default: email_<uid>.eml)",
)
def fetch_email(email, uid, output):
    """Fetch a specific email by UID and save it as an .eml file.

    This is a debug command to inspect specific emails.
    """
    from .imap_client import AsyncIMAPClient

    async def _fetch():
        # Get credentials
        creds = credential_helper.get_credentials(email)
        if not creds:
            click.echo(
                f"No credentials found for {email}. "
                "Please run 'inbox-reaper login' first.",
                err=True,
            )
            return False

        # Check if token expired and refresh if needed
        try:
            expires = datetime.fromisoformat(creds.get("expires_at", ""))
            if expires < datetime.now():
                click.echo("Token expired, refreshing...")
                tokens = refresh_access_token(creds["refresh_token"], creds["provider"])
                creds["access_token"] = tokens["access_token"]
                creds["expires_at"] = (
                    datetime.now() + timedelta(seconds=tokens.get("expires_in", 3600))
                ).isoformat()
                credential_helper.store_credentials(email, creds)
                click.echo("✓ Token refreshed successfully")
        except Exception as e:
            click.echo(f"Warning: Could not refresh token: {e}", err=True)

        # Connect to IMAP
        click.echo(f"Connecting to IMAP for {email}...")
        client = AsyncIMAPClient(
            email_address=email,
            access_token=creds["access_token"],
            provider=creds["provider"],
        )

        try:
            async with client:
                await client.select_mailbox("INBOX")
                click.echo(f"Fetching email UID {uid}...")

                # Fetch the email
                bodies = await client.fetch_bodies([uid])

                if uid not in bodies:
                    click.echo(f"Email with UID {uid} not found.", err=True)
                    return False

                email_data = bodies[uid]

                # Convert to .eml format (simple text representation)
                eml_content = f"""From: {email_data["sender"]}
Subject: {email_data["subject"]}
Date: {email_data["date"]}

{email_data["body"]}
"""

                # Determine output path
                output_path = output or f"email_{uid}.eml"

                # Save to file
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(eml_content)

                click.echo(f"✓ Email saved to {output_path}")
                click.echo(f"  Size: {len(eml_content)} bytes")
                click.echo(f"  Attachments: {email_data.get('attachments', [])}")

                return True

        except Exception as e:
            click.echo(f"Error fetching email: {e}", err=True)
            return False

    # Run async function
    asyncio.run(_fetch())


@cli.command()
@click.argument("email")
@click.option("--refresh", is_flag=True, help="Also test token refresh")
def diagnose(email: str, refresh: bool):
    """Diagnose OAuth token and credential issues.

    This command shows detailed information about stored credentials,
    decodes JWT tokens to inspect expiration claims, and can test token refresh.

    Example:
        inbox-reaper diagnose user@gmail.com
        inbox-reaper diagnose user@gmail.com --refresh
    """
    import base64
    import json

    # Get credentials
    creds = credential_helper.get_credentials(email)
    if not creds:
        click.echo(f"No credentials found for {email}", err=True)
        click.echo(f"\nUse 'inbox-reaper login {email}' to authenticate first.")
        return

    click.echo("=" * 60)
    click.echo("CREDENTIAL DIAGNOSTICS")
    click.echo("=" * 60)
    click.echo(f"\nEmail: {email}")
    click.echo(f"Provider: {creds.get('provider', 'UNKNOWN')}")
    click.echo(f"Token Type: {creds.get('token_type', 'UNKNOWN')}")

    # Check expires_at field
    expires_at_str = creds.get("expires_at")
    click.echo(f"\nStored expires_at: {expires_at_str}")
    click.echo(f"expires_at type: {type(expires_at_str).__name__}")

    if expires_at_str:
        if isinstance(expires_at_str, str):
            try:
                expires_at = datetime.fromisoformat(expires_at_str)
                now = datetime.now()
                if expires_at < now:
                    delta = now - expires_at
                    click.echo(f"Status: EXPIRED {delta} ago", err=True)
                else:
                    delta = expires_at - now
                    click.echo(f"Status: Valid for {delta}")
            except Exception as e:
                click.echo(f"Error parsing expires_at: {e}", err=True)
        else:
            click.echo(
                f"WARNING: expires_at is not a string: {expires_at_str}", err=True
            )
    else:
        click.echo("WARNING: expires_at is missing or None", err=True)

    # Try to decode access token as JWT
    access_token = creds.get("access_token", "")
    click.echo("\n--- Access Token ---")
    click.echo(f"Token length: {len(access_token)} characters")
    if len(access_token) > 50:
        click.echo(f"Token preview: {access_token[:50]}...")
    else:
        click.echo(f"Token: {access_token}")

    # Check if it looks like a JWT (has 3 parts separated by dots)
    parts = access_token.split(".")
    if len(parts) == 3:
        click.echo("\nToken appears to be a JWT (3 parts)")
        try:
            # Decode header
            header_data = parts[0]
            # Add padding if needed
            header_data += "=" * (4 - len(header_data) % 4)
            header_json = base64.urlsafe_b64decode(header_data).decode("utf-8")
            header = json.loads(header_json)
            click.echo(f"\nJWT Header: {json.dumps(header, indent=2)}")

            # Decode payload
            payload_data = parts[1]
            # Add padding if needed
            payload_data += "=" * (4 - len(payload_data) % 4)
            payload_json = base64.urlsafe_b64decode(payload_data).decode("utf-8")
            payload = json.loads(payload_json)
            click.echo(f"\nJWT Payload: {json.dumps(payload, indent=2)}")

            # Check for expiration in JWT
            if "exp" in payload:
                exp_timestamp = payload["exp"]
                exp_dt = datetime.fromtimestamp(exp_timestamp)
                now = datetime.now()
                click.echo(f"\nJWT exp claim: {exp_dt.isoformat()}")
                if exp_dt < now:
                    delta = now - exp_dt
                    click.echo(f"JWT Status: EXPIRED {delta} ago", err=True)
                else:
                    delta = exp_dt - now
                    click.echo(f"JWT Status: Valid for {delta}")
            else:
                click.echo("\nNo 'exp' claim found in JWT")

        except Exception as e:
            click.echo(f"\nError decoding JWT: {e}", err=True)
    else:
        click.echo(
            f"\nToken does not appear to be a JWT (has {len(parts)} parts, expected 3)"
        )

    # Test token refresh if requested
    if refresh:
        click.echo("\n" + "=" * 60)
        click.echo("TESTING TOKEN REFRESH")
        click.echo("=" * 60)

        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            click.echo("No refresh token found", err=True)
            return

        try:
            click.echo(f"\nRefreshing token for {creds['provider']}...")
            tokens = refresh_access_token(refresh_token, creds["provider"])

            click.echo("\nRefresh Response:")
            click.echo(f"  Keys in response: {list(tokens.keys())}")

            # Show what we got back
            for key, value in tokens.items():
                if key in ["access_token", "refresh_token"]:
                    # Mask sensitive values
                    if len(value) > 30:
                        preview = f"{value[:20]}...{value[-10:]}"
                    else:
                        preview = value[:30]
                    click.echo(f"  {key}: {preview}")
                else:
                    click.echo(f"  {key}: {value}")

            # Check if expires_in is present
            expires_in = tokens.get("expires_in")
            if expires_in:
                hours = expires_in / 3600
                click.echo(
                    f"\n✓ expires_in present: {expires_in} seconds ({hours:.1f} hours)"
                )
                future_expiry = datetime.now() + timedelta(seconds=expires_in)
                click.echo(f"  Would expire at: {future_expiry.isoformat()}")
            else:
                click.echo(
                    "\n✗ WARNING: expires_in NOT present in refresh response",
                    err=True,
                )

        except Exception as e:
            click.echo(f"\nError during token refresh: {e}", err=True)
            import traceback

            click.echo(traceback.format_exc(), err=True)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
