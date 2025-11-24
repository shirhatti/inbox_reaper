"""Streaming implementation of LangGraph pipeline for email processing.

This module implements a streaming architecture where emails are yielded one by one
from an IMAP producer and processed through a LangGraph workflow.
"""

import logging
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .batch_coordinator import BatchCoordinator
from .credential_helper import get_credentials
from .imap_client import AsyncIMAPClient
from .oauth_flow import refresh_access_token
from .state import Config

logger = logging.getLogger(__name__)


async def create_imap_client(email: str) -> AsyncIMAPClient:
    """Create an AsyncIMAPClient with credentials from keyring.

    Args:
        email: User's email address

    Returns:
        Connected AsyncIMAPClient instance

    Raises:
        ValueError: If credentials not found or expired
    """
    # Load credentials from keyring
    creds = get_credentials(email)
    if not creds:
        raise ValueError(
            f"No credentials found for {email}. Please run 'inbox-reaper login' first."
        )

    # Check if token is expired and refresh if needed
    try:
        expires_at = datetime.fromisoformat(creds.get("expires_at", ""))
        if expires_at < datetime.now():
            logger.info("Access token expired, refreshing...")
            tokens = refresh_access_token(creds["refresh_token"], creds["provider"])
            creds["access_token"] = tokens["access_token"]
            creds["expires_at"] = (
                datetime.now() + timedelta(seconds=tokens.get("expires_in", 3600))
            ).isoformat()
            # Note: We don't update stored credentials here to avoid import cycles
            # The token will be valid for this session
    except Exception as e:
        logger.warning(f"Could not refresh token: {e}, using existing token")

    # Create and return async client
    client = AsyncIMAPClient(
        email_address=email,
        access_token=creds["access_token"],
        provider=creds["provider"],
    )

    return client


class StreamingState(TypedDict):
    """State for a single email processing flow."""

    email_uid: str
    email: dict  # Header dict
    operation: str  # "process", "delete", "keep", "error"
    decision: str | None
    reason: str | None
    config: dict  # Serialized Config


async def imap_producer(config: Config) -> AsyncIterator[dict]:
    """Async generator that yields emails from IMAP.

    Args:
        config: Pipeline configuration
    Yields:
        Dictionary representing the initial state for an email
    """
    # Create async IMAP client with credentials
    client = await create_imap_client(config.email)

    try:
        # Connect and select mailbox
        async with client:
            await client.select_mailbox("INBOX")

            # 1. Search for all UIDs
            logger.info("Searching for all UIDs...")
            all_uids_str = await client.search_uids(criteria="ALL")

            if not all_uids_str:
                logger.info("No emails found.")
                return

            # Convert to int and find max
            all_uids = [int(uid) for uid in all_uids_str]
            max_uid = max(all_uids)
            logger.info(f"Found {len(all_uids)} emails, max UID is {max_uid}")

            # Sort descending for newest-first processing
            all_uids.sort(reverse=True)

            # Apply max_emails limit if set
            if config.max_emails:
                all_uids = all_uids[: config.max_emails]
                logger.info(f"Limiting to {len(all_uids)} newest emails")

            # 2. Fetch headers in batches
            fetch_batch_size = config.fetch_size
            total_yielded = 0

            for i in range(0, len(all_uids), fetch_batch_size):
                batch = all_uids[i : i + fetch_batch_size]
                batch_str = [str(uid) for uid in batch]

                logger.info(
                    f"Fetching headers for {len(batch)} emails "
                    f"(batch {i // fetch_batch_size + 1})"
                )

                # Native async header fetch - no thread wrapper needed!
                headers = await client.fetch_headers(uids=batch_str)

                # Yield each email
                for uid_int in batch:
                    uid_str = str(uid_int)
                    if uid_str in headers:
                        yield {
                            "email_uid": uid_str,
                            "email": headers[uid_str],
                            "operation": "process",
                            "decision": None,
                            "reason": None,
                            "config": config.model_dump(),
                        }
                        total_yielded += 1

    except Exception as e:
        logger.error(f"Error in IMAP producer: {e}")
        raise


async def process_email_node(state: StreamingState) -> StreamingState:
    """Process a single email.

    Runs deterministic filters and optionally AI classification.
    Fetches email body and applies full classification pipeline.
    """
    import asyncio

    from .agents import (
        check_attachments,
        check_keywords,
        check_whitelist,
        classify_with_ai,
    )
    from .state import Config, Email

    uid = state["email_uid"]
    headers = state["email"]

    logger.info(f"Processing email {uid}: {headers.get('subject')}")

    # Reconstruct Config from serialized dict
    config = Config.model_validate(state["config"])

    # Create async IMAP client for fetching body
    client = await create_imap_client(config.email)

    try:
        # Connect and fetch the full email body - native async!
        async with client:
            await client.select_mailbox("INBOX")
            bodies = await client.fetch_bodies([uid])

            # Check if we got the body
            if uid not in bodies:
                logger.warning(
                    f"Failed to fetch body for email {uid}, keeping by default"
                )
                return {
                    **state,
                    "operation": "keep",
                    "decision": "keep",
                    "reason": "fetch_failed",
                }

            body_data = bodies[uid]

            # Create Email object
            email = Email(
                uid=uid,
                subject=body_data["subject"],
                sender=body_data["sender"],
                body=body_data["body"],
                date=body_data["date"],
                attachments=body_data["attachments"],
            )

        # Apply deterministic filters in order
        decision_obj = None

        # 1. Check attachments
        decision_obj = check_attachments(email, config)
        if decision_obj:
            logger.info(
                f"[{decision_obj.decision.value.upper()}] Email {uid}: "
                f"{email.subject[:50]}... -> {decision_obj.reason.value} "
                f"(confidence: {decision_obj.confidence})"
            )
            return {
                **state,
                "operation": decision_obj.decision.value,
                "decision": decision_obj.decision.value,
                "reason": decision_obj.reason.value,
            }

        # 2. Check keywords
        decision_obj = check_keywords(email, config)
        if decision_obj:
            logger.info(
                f"[{decision_obj.decision.value.upper()}] Email {uid}: "
                f"{email.subject[:50]}... -> {decision_obj.reason.value} "
                f"(confidence: {decision_obj.confidence})"
            )
            return {
                **state,
                "operation": decision_obj.decision.value,
                "decision": decision_obj.decision.value,
                "reason": decision_obj.reason.value,
            }

        # 3. Check whitelist
        decision_obj = check_whitelist(email, config)
        if decision_obj:
            logger.info(
                f"[{decision_obj.decision.value.upper()}] Email {uid}: "
                f"{email.subject[:50]}... -> {decision_obj.reason.value} "
                f"(confidence: {decision_obj.confidence})"
            )
            return {
                **state,
                "operation": decision_obj.decision.value,
                "decision": decision_obj.decision.value,
                "reason": decision_obj.reason.value,
            }

        # 4. If no deterministic filter matched, use AI classifier
        # Note: classify_with_ai is still blocking (uses Ollama sync client)
        # We wrap it in a thread to avoid blocking the event loop
        decision_obj = await asyncio.to_thread(classify_with_ai, email, config)

        logger.info(
            f"[{decision_obj.decision.value.upper()}] Email {uid}: "
            f"{email.subject[:50]}... -> {decision_obj.reason.value} "
            f"(confidence: {decision_obj.confidence})"
        )

        return {
            **state,
            "operation": decision_obj.decision.value,
            "decision": decision_obj.decision.value,
            "reason": decision_obj.reason.value,
        }

    except Exception as e:
        logger.error(f"Error processing email {uid}: {e}")
        # Safe default: keep the email on error
        return {
            **state,
            "operation": "keep",
            "decision": "keep",
            "reason": f"error: {str(e)}",
        }


def build_streaming_graph() -> CompiledStateGraph:
    """Build the streaming LangGraph."""
    graph = StateGraph(StreamingState)

    graph.add_node("process_email", process_email_node)

    graph.add_edge(START, "process_email")
    graph.add_edge("process_email", END)

    return graph.compile()


async def run_streaming_pipeline(config: Config):
    """Run the streaming pipeline."""
    graph = build_streaming_graph()

    # Coordinator for batch deletes
    async def execute_deletes(uids: list[str]):
        logger.info(f"Executing batch delete for {len(uids)} emails")
        if config.dry_run:
            logger.info(f"DRY RUN: Would delete {uids}")
            return

        # Create async IMAP client for deletion - native async!
        client = await create_imap_client(config.email)
        try:
            async with client:
                await client.select_mailbox("INBOX")
                await client.delete_emails(uids)
        except Exception as e:
            logger.error(f"Error deleting emails: {e}")

    batch_coordinator = BatchCoordinator(
        processor=execute_deletes,
        batch_size=50,  # Delete in batches of 50
    )

    logger.info("Starting streaming pipeline...")

    stats = {"processed": 0, "deleted": 0, "kept": 0, "errors": 0}

    # Stream input from producer
    # Note: We iterate over the producer and invoke the graph for each item
    # This is how we bridge the generator to the graph in this architecture
    async for initial_state in imap_producer(config):
        # We use ainvoke for each item.
        # To get parallelism, we could use asyncio.gather on a window of items,
        # but for simplicity and true streaming, we'll process sequentially here
        # or rely on LangGraph's internal handling if we were using a different pattern.
        # The user's request implies "LangGraph pulls from generator", which might
        # require a specific LangGraph feature or just this loop.

        # If we want parallelism, we can spawn tasks.
        # For now, let's do simple await to ensure correctness first.
        try:
            result = await graph.ainvoke(initial_state)

            op = result.get("operation")
            if op == "delete":
                await batch_coordinator.add(result["email_uid"])
                stats["deleted"] += 1
            elif op == "keep":
                stats["kept"] += 1
            else:
                stats["errors"] += 1

            stats["processed"] += 1

            if stats["processed"] % 10 == 0:
                print(f"Processed {stats['processed']} emails...")

        except Exception as e:
            logger.error(f"Error processing email: {e}")
            stats["errors"] += 1

    # Flush remaining deletes
    await batch_coordinator.flush()

    logger.info("Pipeline complete.")
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Total Processed: {stats['processed']}")
    print(f"Total Deleted:   {stats['deleted']}")
    print(f"Total Kept:      {stats['kept']}")
    print(f"Errors:          {stats['errors']}")
    print("=" * 40)
