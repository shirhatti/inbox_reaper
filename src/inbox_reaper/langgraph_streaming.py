"""Streaming implementation of LangGraph pipeline for email processing.

This module implements a streaming architecture where emails are yielded one by one
from an IMAP producer and processed through a LangGraph workflow.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .batch_coordinator import BatchCoordinator
from .imap_client import IMAPClient
from .state import Config

logger = logging.getLogger(__name__)


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
    # Create a dedicated client for the producer
    client = IMAPClient(config.email)

    # We need to run blocking IMAP calls in a thread
    def _connect():
        client.connect()

    await asyncio.to_thread(_connect)

    try:
        # 1. Get all UIDs to find the real maximum (in case latest_uid is a sentinel)
        logger.info("Searching for all UIDs...")
        all_uids_bytes = await asyncio.to_thread(client.search_uids, criteria="ALL")

        if not all_uids_bytes:
            logger.info("No emails found.")
            return

        # Convert to int and find max
        all_uids = [int(uid) for uid in all_uids_bytes]
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

            headers = await asyncio.to_thread(client.fetch_headers, uids=batch_str)

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
    finally:
        await asyncio.to_thread(client.disconnect)


async def process_email_node(state: StreamingState) -> StreamingState:
    """Process a single email.

    Runs deterministic filters and optionally AI classification.
    Fetches email body and applies full classification pipeline.
    """
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

    # Fetch email body using a dedicated connection
    # We can't share the producer's connection, so we create a new one
    client = IMAPClient(config.email)

    try:
        # Connect and fetch the full email body
        await asyncio.to_thread(client.connect)
        bodies = await asyncio.to_thread(client.batch_fetch_bodies, [uid])

        # Check if we got the body
        if uid not in bodies:
            logger.warning(f"Failed to fetch body for email {uid}, keeping by default")
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
        # Note: We run this in a thread since classify_with_ai is blocking
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
    finally:
        await asyncio.to_thread(client.disconnect)


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

        # Create a transient client for deletion
        # In a real app, use a connection pool
        client = IMAPClient(config.email)
        try:
            await asyncio.to_thread(client.connect)
            await asyncio.to_thread(client.batch_delete, uids, dry_run=False)
        finally:
            await asyncio.to_thread(client.disconnect)

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
