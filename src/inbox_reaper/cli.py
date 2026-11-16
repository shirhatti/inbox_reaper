"""CLI entry point for inbox-reaper.

Provides a Click-based command-line interface for the email classification system.
"""

import asyncio
import base64
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import click

from . import credential_helper
from .config_loader import load_config_file, merge_config_with_cli_args
from .imap_client import AsyncIMAPClient
from .mlx_backend import download_model as download_mlx_model
from .oauth_config import detect_provider
from .oauth_flow import (
    perform_oauth_flow,
    refresh_access_token,
    refresh_if_expiring,
    verify_imap_connection,
)
from .state import Config
from .timeutil import expiry_after, parse_timestamp, utcnow

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


def build_config(
    config_path: str | None, cli_args: dict[str, Any], *, require_email: bool = True
) -> Config:
    """Load the optional config file, overlay CLI args, and validate."""
    file_config = {}
    if config_path:
        try:
            click.echo(f"Loading configuration from: {config_path}")
            file_config = load_config_file(config_path)
        except Exception as e:
            click.echo(f"Error loading config file: {e}", err=True)
            raise click.Abort from e

    # Merge configurations (CLI args take precedence)
    merged_config = merge_config_with_cli_args(file_config, cli_args)

    # Validate required fields
    if require_email and not merged_config.get("email"):
        click.echo("Error: --email is required (or must be in config file)", err=True)
        raise click.Abort

    try:
        return Config(**merged_config)
    except Exception as e:
        click.echo(f"Error creating configuration: {e}", err=True)
        raise click.Abort from e


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file (YAML or JSON)",
)
@click.option(
    "--model",
    default=None,
    help="MLX model name for AI classification (Hugging Face model ID)",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    help="Number of emails to process in each batch",
)
@click.option(
    "--concurrent-limit",
    default=None,
    type=int,
    help="Emails classified per batched AI call (default: 8)",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=None,
    help="Enable dry-run mode (no actual deletions)",
)
@click.option(
    "--email",
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
@click.option(
    "--rules",
    "rules_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Rules file from 'inbox-reaper learn', applied before heuristics and AI",
)
@click.option(
    "--combiner",
    "combiner_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Combiner from 'inbox-reaper train': decides from the feature vector, "
    "sending only contested emails to the AI",
)
@click.option(
    "--history",
    "history_path",
    type=click.Path(dir_okay=False),
    help="Mailbox history file for a combiner trained with one; created if "
    "missing, then updated incrementally on each run",
)
@click.option(
    "--ai/--no-ai",
    "use_ai",
    default=None,
    help="--no-ai: keep whatever the filters and combiner do not confidently "
    "decide, without loading the model",
)
@click.option(
    "--only-uids",
    "only_uids_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Process only these UIDs: one per line, or a --report file (its "
    "delete rows), so a real run can only delete what you reviewed",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False),
    help="Write every decision (uid, date, sender, subject, outcome) as JSONL",
)
def process(
    config: str | None,
    model: str | None,
    batch_size: int | None,
    concurrent_limit: int | None,
    dry_run: bool | None,
    email: str | None,
    max_emails: int | None,
    keywords: tuple,
    whitelist_domain: tuple,
    rules_path: str | None,
    combiner_path: str | None,
    history_path: str | None,
    use_ai: bool | None,
    only_uids_path: str | None,
    report_path: str | None,
):
    """Process emails through the classification pipeline with LangGraph.

    This command runs the email classification pipeline using LangGraph for
    orchestration. Each run scans the whole inbox; there is no resume state.

    \b
    Example:
        inbox-reaper process --config config.yaml
        inbox-reaper process --email user@example.com --keywords "important"
        inbox-reaper process --email user@example.com --max-emails 10
        inbox-reaper process --email user@example.com --rules rules.json
    """
    click.echo("Inbox Reaper - Email Classification System (MLX)")
    click.echo("=" * 60)

    cli_args = {
        "email": email,
        "model_name": model,
        "batch_size": batch_size,
        "concurrent_ai_limit": concurrent_limit,
        "dry_run": dry_run,
        "max_emails": max_emails,
        "keywords": list(keywords) if keywords else None,
        "whitelist_domains": list(whitelist_domain) if whitelist_domain else None,
        "rules_path": rules_path,
        "combiner_path": combiner_path,
        "history_path": history_path,
        "use_ai": use_ai,
        "only_uids_path": only_uids_path,
        "report_path": report_path,
    }
    config_obj = build_config(config, cli_args)

    click.echo("\nConfiguration:")
    click.echo(f"  MLX Model: {config_obj.model_name}")
    click.echo(f"  Batch size: {config_obj.batch_size}")
    click.echo(f"  Concurrent limit: {config_obj.concurrent_ai_limit}")
    click.echo(f"  Dry run: {config_obj.dry_run}")
    click.echo(f"  Keywords: {config_obj.keywords or 'None'}")
    click.echo(f"  Whitelisted domains: {config_obj.whitelist_domains or 'None'}")
    click.echo(f"  Rules: {config_obj.rules_path or 'None'}")
    click.echo(f"  Combiner: {config_obj.combiner_path or 'None'}")
    click.echo(f"  History: {config_obj.history_path or 'None'}")
    click.echo(f"  AI: {'on' if config_obj.use_ai else 'off'}")
    if config_obj.only_uids_path:
        click.echo(f"  Only UIDs from: {config_obj.only_uids_path}")

    # Run the LangGraph pipeline
    try:
        click.echo("\nStarting LangGraph pipeline execution (Streaming Mode)...")
        click.echo("-" * 60)

        # Deferred: importing LangGraph takes ~2s, which would slow every command
        from .langgraph_streaming import run_streaming_pipeline  # noqa: PLC0415

        asyncio.run(run_streaming_pipeline(config_obj))

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
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file (YAML or JSON)",
)
@click.option("--email", help="Email address to learn from")
@click.option(
    "--output",
    "-o",
    default="rules.json",
    type=click.Path(dir_okay=False),
    help="Where to write the learned rules",
    show_default=True,
)
@click.option("--model", default=None, help="MLX model name for AI classification")
@click.option("--max-emails", type=int, help="Maximum number of emails to scan")
@click.option(
    "--min-support",
    default=3,
    type=click.IntRange(min=1),
    help="Emails required from a sender/domain before a rule is learned",
    show_default=True,
)
@click.option(
    "--min-agreement",
    default=0.9,
    type=click.FloatRange(0.5, 1.0),
    help="Share of emails that must agree on keep/delete",
    show_default=True,
)
@click.option(
    "--ai/--no-ai",
    default=True,
    help="Use AI for emails the deterministic filters do not decide",
    show_default=True,
)
@click.option("--keywords", multiple=True, help="Critical keywords (KEEP)")
@click.option("--whitelist-domain", multiple=True, help="Whitelisted domains (KEEP)")
def learn(
    config: str | None,
    email: str | None,
    output: str,
    model: str | None,
    max_emails: int | None,
    min_support: int,
    min_agreement: float,
    ai: bool,
    keywords: tuple,
    whitelist_domain: tuple,
):
    """Learn deterministic sender/domain rules without changing the mailbox.

    Classifies emails read-only, then writes rules for senders and domains that
    were classified consistently. Review or edit the file, then apply it with
    'inbox-reaper process --rules'.

    \b
    Example:
        inbox-reaper learn --email user@example.com -o rules.json
        inbox-reaper process --email user@example.com --rules rules.json
    """
    # Deferred: importing LangGraph takes ~2s, which would slow every command
    from .langgraph_streaming import run_learning_pipeline  # noqa: PLC0415

    config_obj = build_config(
        config,
        {
            "email": email,
            "model_name": model,
            "max_emails": max_emails,
            "keywords": list(keywords) if keywords else None,
            "whitelist_domains": list(whitelist_domain) if whitelist_domain else None,
        },
    )

    click.echo(f"Learning rules from {config_obj.email} (read-only)...")
    rule_set = asyncio.run(
        run_learning_pipeline(
            config_obj,
            min_support=min_support,
            min_agreement=min_agreement,
            use_ai=ai,
        )
    )
    rule_set.save(output)

    deletes = sum(r.action == "delete" for r in rule_set.rules)
    click.echo(
        f"\nLearned {len(rule_set.rules)} rules ({deletes} delete, "
        f"{len(rule_set.rules) - deletes} keep) from "
        f"{rule_set.emails_observed} emails -> {output}"
    )
    for rule in rule_set.rules[:15]:
        click.echo(
            f"  {rule.action.value:<6} {rule.match.value:<6} {rule.value}  "
            f"({rule.support} emails, {rule.agreement:.0%} agree)"
        )
    if len(rule_set.rules) > 15:
        click.echo(f"  ... and {len(rule_set.rules) - 15} more")
    click.echo(f"\nReview {output}, then run: inbox-reaper process --rules {output}")


@cli.command()
def version():
    """Display version information."""
    click.echo("Inbox Reaper v0.1.0")
    click.echo("Email classification and cleaning system")


@cli.command()
@click.option(
    "--model",
    default="mlx-community/Llama-3.2-3B-Instruct-4bit",
    help="MLX model name to download (Hugging Face model ID)",
    show_default=True,
)
def download_model(model: str):
    """Download and cache MLX model for offline use.

    This command pre-fetches the specified MLX model to the local cache.
    Useful for CI/CD environments or preparing for offline use.

    The model will be downloaded to ~/.cache/huggingface/hub/ and can be
    used by subsequent runs of 'inbox-reaper process'.

    Example:
        inbox-reaper download-model
        inbox-reaper download-model --model mlx-community/Llama-3.2-1B-Instruct-4bit
    """
    click.echo("Inbox Reaper - Model Download")
    click.echo("=" * 60)
    click.echo(f"\nModel: {model}")
    click.echo("\nThis will download the model to your local cache.")
    click.echo("The first download may take several minutes.\n")

    success = download_mlx_model(model)

    if success:
        click.echo("\n" + "=" * 60)
        click.echo("✓ Model ready for use!")
        click.echo("=" * 60)
    else:
        click.echo("\n" + "=" * 60)
        click.echo("✗ Model download failed", err=True)
        click.echo("=" * 60)
        raise click.ClickException("Model download failed")


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
            "expires_at": expiry_after(tokens.get("expires_in", 3600)),
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
def accounts():
    """List all stored email accounts.

    Example:
        inbox-reaper accounts

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
        if refresh_if_expiring(email, creds):
            click.echo("✓ Token refreshed successfully")
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
            if refresh_if_expiring(email, creds):
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
                await asyncio.to_thread(
                    Path(output_path).write_text, eml_content, encoding="utf-8"
                )

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
                expires_at = parse_timestamp(expires_at_str)
                now = utcnow()
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
                exp_dt = parse_timestamp(exp_timestamp)
                now = utcnow()
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
                click.echo(f"  Would expire at: {expiry_after(expires_in)}")
            else:
                click.echo(
                    "\n✗ WARNING: expires_in NOT present in refresh response",
                    err=True,
                )

        except Exception as e:
            click.echo(f"\nError during token refresh: {e}", err=True)
            click.echo(traceback.format_exc(), err=True)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--email", help="Email address to sample from")
@click.option("-n", "count", default=200, show_default=True, help="Emails to sample")
@click.option("--seed", default=0, show_default=True, help="Random seed")
@click.option(
    "--exclude",
    "exclude_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Existing samples file whose emails should not be sampled again",
)
@click.option("-o", "--output", default="samples.jsonl", show_default=True)
def sample(config, email, count, seed, exclude_path, output):
    """Save a random sample of emails for labelling (read-only).

    The samples file holds email contents; keep it private.

    \b
    Example:
        inbox-reaper sample --email user@example.com -n 300 -o samples.jsonl
    """
    from .labeling import Sample, append_jsonl, load_samples  # noqa: PLC0415
    from .langgraph_streaming import sample_inbox  # noqa: PLC0415

    config_obj = build_config(config, {"email": email})
    exclude = {s.uid for s in load_samples(exclude_path)} if exclude_path else set()
    emails = asyncio.run(sample_inbox(config_obj, count, seed=seed, exclude=exclude))
    sampled_at = utcnow()
    for e in emails:
        append_jsonl(output, Sample(uid=e.uid, sampled_at=sampled_at, email=e))
    click.echo(f"Sampled {len(emails)} emails -> {output}")
    click.echo(f"Next: inbox-reaper label {output}")


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--email", help="Email address to read")
@click.option("-o", "--output", default="history.json", show_default=True)
@click.option("--full", is_flag=True, help="Rescan everything instead of updating")
def history(config, email, output, full):
    """Save your mailbox history for training and evaluation (read-only).

    Records the sender, date and read/flagged state of every INBOX message
    and who you have written to (from the Sent folder). Nothing is marked read
    or changed. If OUTPUT exists it is updated: only new messages are fetched
    and read state is refreshed. The file holds addresses from your mailbox;
    keep it private.

    \b
    Example:
        inbox-reaper history --email user@example.com -o history.json
    """
    from .langgraph_streaming import read_history  # noqa: PLC0415

    config_obj = build_config(config, {"email": email, "history_path": output})
    asyncio.run(read_history(config_obj, full=full))
    click.echo(f"Saved -> {output}")


@cli.command()
@click.argument("samples_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default="labels.jsonl", show_default=True)
@click.option("--labeler", default="human", show_default=True)
@click.option(
    "--prioritize",
    "reference_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Labels files (e.g. model labels) whose disagreements you label first",
)
@click.option("--limit", type=int, help="Stop after this many emails")
def label(samples_path, output, labeler, reference_paths, limit):
    """Label sampled emails along several dimensions, interactively.

    For each email: keep or delete, how long it stays useful, a category, and
    how bad it would be to lose it. Progress is saved after every email; run
    again to resume. Other labellers' answers are never shown.

    \b
    Example:
        inbox-reaper label samples.jsonl --prioritize model-labels.jsonl
    """
    from .labeling import label_interactively, load_labels, load_samples, review_order  # noqa: PLC0415

    samples = load_samples(samples_path)
    references = [load_labels(p) for p in reference_paths]
    ordered = review_order(samples, references) if references else samples
    count = label_interactively(ordered, output, labeler, limit)
    click.echo(f"Labelled {count} emails -> {output}")


@cli.command("compare-labels")
@click.argument(
    "label_paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--samples",
    "samples_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Samples file, to show subjects of disagreements",
)
@click.option("--show", default=20, show_default=True, help="Disagreements to list")
def compare_labels(label_paths, samples_path, show):
    """Compare labellers: agreement per dimension and keep/delete disagreements."""
    from .labeling import compare, disagreements, load_labels, load_samples  # noqa: PLC0415

    sets = {Path(p).stem: load_labels(p) for p in label_paths}
    for line in compare(sets):
        click.echo(line)
    samples = {s.uid: s for s in load_samples(samples_path)} if samples_path else {}
    rows = list(disagreements(sets, samples))
    click.echo(f"\n{len(rows)} keep/delete disagreements")
    for row in rows[:show]:
        click.echo(row)


@cli.command("train")
@click.argument("samples_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--labels",
    "label_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Labels files; when several label the same email, the first wins",
)
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--rules", "rules_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--history",
    "history_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Mailbox history from 'inbox-reaper history' (adds history features)",
)
@click.option("-o", "--output", default="combiner.json", show_default=True)
def train_command(samples_path, label_paths, config, rules_path, history_path, output):
    """Train the combiner from labelled samples.

    Keep-labelled emails are weighted by how bad losing them would be. The
    reported metrics are cross-validated. Use the result with
    'inbox-reaper process --combiner'.
    """
    from .history import MailboxHistory  # noqa: PLC0415
    from .labeling import load_labels, load_samples, train_combiner  # noqa: PLC0415
    from .rules import RuleSet  # noqa: PLC0415

    # Only filter settings (keywords, whitelist) matter here, not the account
    config_obj = build_config(config, {}, require_email=False)
    rules = RuleSet.load(rules_path) if rules_path else None
    combiner = train_combiner(
        load_samples(samples_path),
        [load_labels(p) for p in label_paths],
        config_obj,
        rules,
        MailboxHistory.load(history_path) if history_path else None,
    )
    combiner.save(output)
    m = combiner.metrics
    click.echo(
        f"Trained on {combiner.trained_on} emails ({m['keep_examples']:.0f} keep, "
        f"{m['delete_examples']:.0f} delete) -> {output}"
    )
    click.echo(
        f"Cross-validated: {m['false_deletes']:.0f} keep-labelled emails would be "
        f"deleted, {m['deletes_caught']:.0f}/{m['delete_examples']:.0f} deletable "
        f"caught, {m['contested_share']:.0%} contested (sent to the AI)"
    )
    click.echo(
        f"Thresholds: delete if P >= {combiner.delete_threshold:.2f}, "
        f"keep if P <= {combiner.keep_threshold:.2f}"
    )
    click.echo("Strongest features:")
    for name, weight in combiner.top_weights(8):
        click.echo(f"  {weight:+.2f}  {name}")


@cli.command("evaluate")
@click.argument("samples_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--labels",
    "label_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Labels files; when several label the same email, the first wins",
)
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--rules", "rules_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--combiner",
    "combiner_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Also evaluate the pipeline with this combiner",
)
@click.option(
    "--history",
    "history_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Mailbox history from 'inbox-reaper history' (adds history features)",
)
@click.option("--show", default=15, show_default=True, help="Wrong deletes to list")
def evaluate_command(
    samples_path, label_paths, config, rules_path, combiner_path, history_path, show
):
    """Delete precision and recall of each deletion path on labelled mail.

    Replays the deterministic stages (and the combiner) without running the
    AI. Compare 'today' against '--combiner' before trusting a new model: a
    wrong delete is the error that matters.
    """
    from .combiner import Combiner  # noqa: PLC0415
    from .history import MailboxHistory  # noqa: PLC0415
    from .labeling import evaluate, format_evaluation, load_labels, load_samples  # noqa: PLC0415
    from .rules import RuleSet  # noqa: PLC0415

    config_obj = build_config(config, {}, require_email=False)
    rules = RuleSet.load(rules_path) if rules_path else None
    samples = load_samples(samples_path)
    labels = [load_labels(p) for p in label_paths]
    history = MailboxHistory.load(history_path) if history_path else None
    runs = [evaluate(samples, labels, config_obj, rules)]
    if combiner_path:
        combiner = Combiner.load(combiner_path)
        if combiner.needs_history and history is None:
            raise click.UsageError("this combiner needs --history")
        runs.append(evaluate(samples, labels, config_obj, rules, combiner, history))
    for ev in runs:
        for line in format_evaluation(ev):
            click.echo(line)
        for row in ev.wrong[:show]:
            click.echo(f"    wrong delete - {row}")
        click.echo()
