"""CLI entry point for inbox-reaper.

Provides a Click-based command-line interface for the email classification system.
"""

from datetime import datetime, timedelta
from pathlib import Path

import click

from . import credential_helper
from .checkpoint_manager import CheckpointManager
from .langgraph_dag import run_langgraph_pipeline
from .oauth_config import detect_provider
from .oauth_flow import perform_oauth_flow, refresh_access_token, verify_imap_connection
from .state import Config, Email


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
    keywords: tuple,
    whitelist_domain: tuple,
):
    """Process emails through the classification pipeline with LangGraph.

    This command runs the email classification pipeline using LangGraph for
    orchestration, with support for checkpointing and resumable processing.

    Example:
        inbox-reaper process --keywords "important" --whitelist-domain "gmail.com"
        inbox-reaper process --checkpoint-path my_checkpoint.db --resume
        inbox-reaper process --dry-run --no-resume
    """
    click.echo("Inbox Reaper - Email Classification System (LangGraph)")
    click.echo("=" * 60)

    # Create configuration
    config = Config(
        model_name=model,
        ollama_base_url=ollama_url,
        batch_size=batch_size,
        concurrent_ai_limit=concurrent_limit,
        dry_run=dry_run,
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

    # Initialize checkpoint manager
    try:
        checkpoint_manager = CheckpointManager(checkpoint_path)
        click.echo(f"\nCheckpoint database initialized: {checkpoint_path}")
    except Exception as e:
        click.echo(f"\nError initializing checkpoint: {e}", err=True)
        click.echo("Tip: Use --checkpoint-path to specify a different location")
        if click.confirm("Continue without checkpointing?", default=False):
            click.echo("Warning: Processing without checkpoints - cannot resume if interrupted")
            checkpoint_manager = None
        else:
            click.echo("Aborted.")
            return

    # Check for resumable session
    should_resume = False
    if checkpoint_manager and resume:
        # Verify checkpoint integrity
        is_valid, integrity_msg = checkpoint_manager.verify_checkpoint_integrity()

        if not is_valid:
            click.echo(f"\nCheckpoint integrity check failed: {integrity_msg}")
            if click.confirm("Clear corrupted checkpoint and start fresh?", default=True):
                checkpoint_manager.clear_checkpoint()
                click.echo("Checkpoint cleared. Starting fresh.")
            else:
                click.echo("Aborted.")
                return
        else:
            # Check if there's a resumable session
            resume_info = checkpoint_manager.get_resume_info()
            if resume_info["can_resume"]:
                should_resume = checkpoint_manager.display_resume_prompt()
                if not should_resume:
                    # User chose not to resume, clear checkpoint
                    checkpoint_manager.clear_checkpoint()
                    click.echo("Starting fresh processing session.")

    # Run the LangGraph pipeline
    try:
        click.echo("\nStarting LangGraph pipeline execution...")
        click.echo("-" * 60)

        final_state = run_langgraph_pipeline(config, checkpoint_path)

        click.echo("\n" + "=" * 60)
        click.echo("Processing Complete!")
        click.echo("=" * 60)

        # Display final summary using CheckpointManager
        if checkpoint_manager:
            checkpoint_manager.display_summary(final_state)
            # Clear checkpoint on successful completion
            checkpoint_manager.clear_checkpoint()
        else:
            # Fallback display if no checkpoint manager
            click.echo(f"\nTotal processed: {final_state['total_processed']}")
            click.echo(f"Total deleted:   {final_state['total_deleted']}")
            click.echo(f"Total kept:      {final_state['total_kept']}")
            if final_state['errors']:
                click.echo(f"Errors:          {len(final_state['errors'])}")

    except KeyboardInterrupt:
        click.echo("\n\nProcessing interrupted by user.")
        if checkpoint_manager:
            click.echo("Progress has been saved. Use --resume to continue from checkpoint.")
        raise
    except Exception as e:
        click.echo(f"\nError during processing: {e}", err=True)
        if checkpoint_manager:
            click.echo("Progress has been saved. Use --resume to continue from checkpoint.")
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

    # Test connection
    click.echo(f"\nTesting IMAP connection for {email}...")
    success, message = verify_imap_connection(
        email, creds["access_token"], creds["provider"]
    )

    if success:
        click.echo(f"✓ {message}")
    else:
        click.echo(f"✗ {message}", err=True)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
