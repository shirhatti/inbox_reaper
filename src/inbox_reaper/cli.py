"""CLI entry point for inbox-reaper.

Provides a Click-based command-line interface for the email classification system.
"""

import difflib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import click
from google import genai

from . import credential_helper
from .dag import run_pipeline, run_pipeline_with_adk
from .imap_fetcher import fetch_emails
from .oauth_config import detect_provider
from .oauth_flow import perform_oauth_flow, refresh_access_token, verify_imap_connection
from .sanitizer import PiiSanitizer, dict_to_email, email_to_dict
from .state import Config, Decision, Email, ProcessingState


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
    "--use-adk/--no-adk",
    default=False,
    help="Use Google ADK for agent orchestration (experimental)",
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
    use_adk: bool,
    keywords: tuple,
    whitelist_domain: tuple,
):
    """Process emails through the classification pipeline.

    This command runs the email classification agent on a batch of emails.
    Currently uses mock data for demonstration purposes.

    Example:
        inbox-reaper process --keywords "important" --whitelist-domain "gmail.com"
    """
    click.echo("🚀 Inbox Reaper - Email Classification System")
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

    click.echo("\n📋 Configuration:")
    click.echo(f"  Model: {config.model_name}")
    click.echo(f"  Ollama URL: {config.ollama_base_url}")
    click.echo(f"  Batch size: {config.batch_size}")
    click.echo(f"  Concurrent limit: {config.concurrent_ai_limit}")
    click.echo(f"  Dry run: {config.dry_run}")
    click.echo(f"  Keywords: {config.keywords or 'None'}")
    click.echo(f"  Whitelisted domains: {config.whitelist_domains or 'None'}")

    # Create mock emails for demonstration
    mock_emails = create_mock_emails()

    click.echo(f"\n📧 Processing {len(mock_emails)} mock emails...")

    # Create initial state
    initial_state = ProcessingState(config=config, emails=mock_emails)

    # Run the pipeline
    if use_adk:
        click.echo("\n🤖 Using Google ADK for agent orchestration...")
        # Initialize Google GenAI client
        # Note: Requires GOOGLE_API_KEY environment variable
        try:
            client = genai.Client()
            final_state = run_pipeline_with_adk(initial_state, client)
        except Exception as e:
            click.echo(f"\n⚠️  ADK initialization failed: {e}", err=True)
            click.echo("   Falling back to simple pipeline...\n")
            final_state = run_pipeline(initial_state)
    else:
        click.echo("\n⚙️  Using simple sequential pipeline...")
        final_state = run_pipeline(initial_state)

    # Display final results
    click.echo("\n" + "=" * 60)
    click.echo("✅ Processing Complete!")
    click.echo("=" * 60)

    click.echo("\n📊 Final Statistics:")
    click.echo(f"  Total processed: {final_state.total_processed}")
    click.echo(f"  Total kept: {final_state.total_kept}")
    click.echo(f"  Total deleted: {final_state.total_deleted}")

    if final_state.errors:
        click.echo(f"\n⚠️  Errors encountered: {len(final_state.errors)}")
        for error in final_state.errors:
            click.echo(f"    - {error}")

    # Display decisions
    if final_state.decisions:
        click.echo("\n📝 Decisions:")
        for i, decision in enumerate(final_state.decisions, 1):
            emoji = "🗑️ " if decision.decision.value == "delete" else "📬"
            click.echo(
                f"  {i}. {emoji} [{decision.decision.value.upper()}] "
                f"{decision.email.subject[:50]}... "
                f"(reason: {decision.reason.value})"
            )


def create_mock_emails() -> list[Email]:
    """Create mock emails for demonstration purposes.

    In production, this would be replaced with actual IMAP fetching.
    """
    return [
        Email(
            uid="1001",
            subject="SALE: 50% off everything!",
            sender="marketing@store.com",
            body="Limited time offer! Get 50% off all items in our store...",
            date=datetime.now(),
            attachments=[],
        ),
        Email(
            uid="1002",
            subject="Your bank statement for November",
            sender="notifications@wellsfargo.com",
            body="Your monthly statement is now available...",
            date=datetime.now(),
            attachments=["statement.pdf"],
        ),
        Email(
            uid="1003",
            subject="Meeting notes from Mario Romo",
            sender="mario@company.com",
            body="Hi team, here are the notes from our meeting...",
            date=datetime.now(),
            attachments=[],
        ),
        Email(
            uid="1004",
            subject="Weekly newsletter",
            sender="news@techblog.com",
            body="Here's what happened in tech this week...",
            date=datetime.now(),
            attachments=[],
        ),
        Email(
            uid="1005",
            subject="Important: Password reset required",
            sender="security@gmail.com",
            body="We detected unusual activity on your account...",
            date=datetime.now(),
            attachments=[],
        ),
    ]


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


@cli.command()
@click.argument("email")
@click.option(
    "--count",
    "-n",
    default=10,
    type=int,
    help="Number of emails to fetch",
    show_default=True,
)
@click.option(
    "--output-dir",
    "-o",
    default="test_data/golden",
    help="Output directory for test data",
    show_default=True,
)
def create_test_data(email: str, count: int, output_dir: str):
    """Create golden test dataset from real emails with PII sanitization.

    This command fetches emails from your account, applies automatic PII
    sanitization, and guides you through an interactive review process.
    You'll label each email and review sanitization in your $EDITOR.

    Example:
        inbox-reaper create-test-data user@gmail.com --count 20
    """
    click.echo("🔬 Golden Dataset Creator")
    click.echo("=" * 60)

    # Check for EDITOR
    editor = os.environ.get("EDITOR")
    if not editor:
        click.echo(
            "⚠️  $EDITOR environment variable not set. Using 'vi' as default.",
            err=True,
        )
        editor = "vi"

    # Get credentials
    creds = credential_helper.get_credentials(email)
    if not creds:
        click.echo(f"✗ No credentials found for {email}", err=True)
        click.echo(f"\nUse 'inbox-reaper login {email}' to authenticate first.")
        return

    # Refresh token if expired
    try:
        expires = datetime.fromisoformat(creds.get("expires_at", ""))
        if expires < datetime.now():
            click.echo("🔄 Token expired, refreshing...")
            tokens = refresh_access_token(creds["refresh_token"], creds["provider"])
            creds["access_token"] = tokens["access_token"]
            creds["expires_at"] = (
                datetime.now() + timedelta(seconds=tokens.get("expires_in", 3600))
            ).isoformat()
            credential_helper.store_credentials(email, creds)
    except Exception as e:
        click.echo(f"⚠️  Could not refresh token: {e}", err=True)

    # Fetch emails
    click.echo(f"\n📧 Fetching {count} emails from {email}...")
    try:
        emails = fetch_emails(email, creds["access_token"], creds["provider"], count)
        click.echo(f"✓ Fetched {len(emails)} emails")
    except Exception as e:
        click.echo(f"✗ Failed to fetch emails: {e}", err=True)
        return

    if not emails:
        click.echo("No emails found.")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize sanitizer
    sanitizer = PiiSanitizer()

    # Process each email
    click.echo(f"\n🔍 Processing {len(emails)} emails...")
    click.echo("=" * 60)

    saved_count = 0
    for i, original_email in enumerate(emails, 1):
        click.echo(f"\n📬 Email {i}/{len(emails)}")
        click.echo("-" * 60)
        click.echo(f"From: {original_email.sender}")
        click.echo(f"Subject: {original_email.subject}")
        click.echo(f"Date: {original_email.date}")
        click.echo(
            f"Body preview: {original_email.body[:100]}..."
            if len(original_email.body) > 100
            else f"Body: {original_email.body}"
        )

        # Ask if user wants to include this email
        include = click.confirm("\nInclude this email in test dataset?", default=True)
        if not include:
            continue

        # Sanitize email
        sanitized_email = sanitizer.sanitize_email(original_email)

        # Create diff for review
        original_dict = email_to_dict(original_email)
        sanitized_dict = email_to_dict(sanitized_email)

        original_json = json.dumps(original_dict, indent=2)
        sanitized_json = json.dumps(sanitized_dict, indent=2)

        diff = "\n".join(
            difflib.unified_diff(
                original_json.splitlines(),
                sanitized_json.splitlines(),
                fromfile="original.json",
                tofile="sanitized.json",
                lineterm="",
            )
        )

        # Write diff to temp file and open in editor
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".diff", delete=False
        ) as tmp:
            tmp.write("# Review PII Sanitization\n")
            tmp.write("# Lines starting with - are from original\n")
            tmp.write("# Lines starting with + are sanitized\n")
            tmp.write("# Close the editor when done reviewing\n")
            tmp.write("\n")
            tmp.write(diff)
            tmp_path = tmp.name

        click.echo(f"\n📝 Opening diff in {editor}...")
        click.echo(
            "Review the sanitization. Close the editor when done (changes won't be saved)."
        )

        try:
            subprocess.run([editor, tmp_path], check=True)
        except subprocess.CalledProcessError:
            click.echo("⚠️  Editor exited with error, continuing...")
        except FileNotFoundError:
            click.echo(f"⚠️  Editor '{editor}' not found, skipping review...")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Ask for additional manual edits to sanitized email
        if click.confirm("\nMake manual edits to sanitized email?", default=False):
            # Open sanitized JSON in editor for manual editing
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                tmp.write(sanitized_json)
                tmp_path = tmp.name

            try:
                subprocess.run([editor, tmp_path], check=True)

                # Read back edited content
                with open(tmp_path) as f:
                    edited_json = f.read()
                    sanitized_dict = json.loads(edited_json)
                    sanitized_email = dict_to_email(sanitized_dict)

                click.echo("✓ Manual edits applied")
            except subprocess.CalledProcessError:
                click.echo("⚠️  Editor exited with error, using auto-sanitized version")
            except json.JSONDecodeError as e:
                click.echo(f"⚠️  Invalid JSON after editing: {e}")
                click.echo("Using auto-sanitized version")
            except Exception as e:
                click.echo(f"⚠️  Error reading edits: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # Ask for ground truth labels
        click.echo("\n🏷️  Label this email:")
        decision = click.prompt(
            "Decision",
            type=click.Choice(["keep", "delete"], case_sensitive=False),
            default="delete",
        )

        reason = click.prompt(
            "Why? (e.g., 'marketing newsletter', 'important bill', 'personal email')",
            type=str,
            default="marketing",
        )

        notes = click.prompt(
            "Additional notes (optional)", type=str, default="", show_default=False
        )

        # Create test data entry
        test_entry = {
            "email": email_to_dict(sanitized_email),
            "ground_truth": {
                "decision": decision,
                "reason": reason,
                "notes": notes,
                "created_at": datetime.now().isoformat(),
                "created_from_account": email,
            },
        }

        # Save to file
        filename = f"{decision}_{sanitized_email.uid}_{i:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(test_entry, f, indent=2)

        click.echo(f"✓ Saved to {filepath}")
        saved_count += 1

    # Summary
    click.echo("\n" + "=" * 60)
    click.echo("✅ Test Data Creation Complete!")
    click.echo(f"📊 Saved {saved_count} out of {len(emails)} emails")
    click.echo(f"📁 Output directory: {output_path.absolute()}")

    # Show sanitization statistics
    click.echo("\n🔒 PII Sanitization Statistics:")
    click.echo(f"  Emails sanitized: {len(sanitizer.email_map)}")
    click.echo(f"  Domains mapped: {len(sanitizer.domain_map)}")
    click.echo(f"  Phone numbers redacted: {len(sanitizer.phone_map)}")
    click.echo(f"  Names anonymized: {sanitizer.name_counter}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
