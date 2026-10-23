"""Streaming implementation of LangGraph pipeline for email processing.

This module implements a streaming architecture where emails are fetched from IMAP
in batches over a single shared connection and processed one by one through a
LangGraph workflow. The next batch is prefetched while the current one is being
classified, so network latency overlaps with (much slower) AI inference.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any, TypedDict

import aioimaplib
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import (
    check_attachments,
    check_keywords,
    check_marketing_indicators,
    check_transactional_patterns,
    check_whitelist,
    classify_with_ai,
)
from .batch_coordinator import BatchCoordinator
from .credential_helper import get_credentials, store_credentials
from .imap_client import AsyncIMAPClient, IMAPConnectionError
from .oauth_flow import refresh_access_token
from .rules import RuleLearner, RuleSet
from .state import Config, Email, EmailDecision

logger = logging.getLogger(__name__)

# Refresh tokens that expire within this window so a fresh connection does not
# get dropped mid-batch.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)

# Upper bound on full messages per UID FETCH, to cap memory when emails carry
# large attachments.
MAX_MESSAGES_PER_FETCH = 25

# Explicit user intent; always applied first and never overridden by rules.
HARD_KEEP_FILTERS = (
    check_attachments,
    check_keywords,
    check_whitelist,
)

# Heuristics applied after learned rules, before AI.
HEURISTIC_FILTERS = (
    check_transactional_patterns,  # receipts, orders, shipping
    check_marketing_indicators,  # unsubscribe, view in browser
)

# Failures that indicate a dropped/expired connection worth one reconnect.
_RECONNECTABLE_ERRORS = (
    IMAPConnectionError,
    aioimaplib.AioImapException,
    ConnectionError,
    OSError,
    TimeoutError,
)


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
        expires_at_str = creds.get("expires_at")
        if expires_at_str and isinstance(expires_at_str, str):
            expires_at = datetime.fromisoformat(expires_at_str)
            if expires_at < datetime.now() + TOKEN_REFRESH_MARGIN:
                logger.info("Access token expired, refreshing...")
                tokens = refresh_access_token(creds["refresh_token"], creds["provider"])
                creds["access_token"] = tokens["access_token"]
                creds["expires_at"] = (
                    datetime.now() + timedelta(seconds=tokens.get("expires_in", 3600))
                ).isoformat()
                # Persist the refreshed credentials
                store_credentials(email, creds)
                logger.info("Refreshed credentials persisted to keyring")
        else:
            # Token expiration info missing, proactively refresh to be safe
            logger.warning(
                "Token expiration time missing or invalid, attempting refresh..."
            )
            if "refresh_token" in creds:
                tokens = refresh_access_token(creds["refresh_token"], creds["provider"])
                creds["access_token"] = tokens["access_token"]
                creds["expires_at"] = (
                    datetime.now() + timedelta(seconds=tokens.get("expires_in", 3600))
                ).isoformat()
                # Persist the refreshed credentials
                store_credentials(email, creds)
                logger.info("Token refreshed and persisted successfully")
            else:
                logger.warning(
                    "No refresh token available, using existing access token"
                )
    except Exception as e:
        logger.warning(f"Could not refresh token: {e}, using existing token")

    # Create and return async client
    return AsyncIMAPClient(
        email_address=email,
        access_token=creds["access_token"],
        provider=creds["provider"],
    )


class IMAPSession:
    """A single authenticated IMAP connection shared by the whole pipeline.

    Operations are serialized with a lock (aioimaplib does not safely interleave
    commands on one connection). If an operation fails because the connection
    dropped -- e.g. the server ended the session when the OAuth token expired --
    the session reconnects with fresh credentials and retries once.
    """

    def __init__(self, email: str):
        self.email = email
        self._client: AsyncIMAPClient | None = None
        self._lock = asyncio.Lock()

    async def _connect(self) -> AsyncIMAPClient:
        client = await create_imap_client(self.email)
        await client.connect()
        await client.select_mailbox("INBOX")
        self._client = client
        return client

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.disconnect()

    async def run[T](self, operation: Callable[[AsyncIMAPClient], Awaitable[T]]) -> T:
        """Run ``operation(client)``, reconnecting once on connection failure."""
        async with self._lock:
            client = self._client or await self._connect()
            try:
                return await operation(client)
            except _RECONNECTABLE_ERRORS as e:
                logger.warning(f"IMAP operation failed ({e}), reconnecting...")
                await self._disconnect()
            client = await self._connect()
            return await operation(client)

    async def close(self) -> None:
        async with self._lock:
            await self._disconnect()

    async def __aenter__(self) -> "IMAPSession":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class StreamingState(TypedDict):
    """State for a single email processing flow."""

    email_uid: str
    email: Email
    operation: str  # "process", "delete", "keep", "error"
    decision: str | None
    reason: str | None
    config: Config
    rules: RuleSet | None


async def imap_producer(
    config: Config, session: IMAPSession
) -> AsyncGenerator[StreamingState]:
    """Async generator that yields emails from IMAP.

    Args:
        config: Pipeline configuration
        session: Shared IMAP session
    Yields:
        Dictionary representing the initial state for an email
    """
    # 1. Search for all UIDs
    logger.info("Searching for UIDs...")
    all_uids = await session.run(lambda client: client.search_uids(criteria="ALL"))

    if not all_uids:
        logger.info("No emails found.")
        return

    logger.info(f"Found {len(all_uids)} emails")

    # Process in chronological order (UIDs are already ascending)
    # Apply max_emails limit if set (take oldest N emails)
    if config.max_emails:
        all_uids = all_uids[: config.max_emails]
        logger.info(f"Limiting to {len(all_uids)} oldest emails")

    # 2. Fetch full messages in batches -- one round trip per batch
    fetch_batch_size = max(1, min(config.fetch_size, MAX_MESSAGES_PER_FETCH))

    for i in range(0, len(all_uids), fetch_batch_size):
        batch = all_uids[i : i + fetch_batch_size]

        logger.info(f"Fetching {len(batch)} emails (batch {i // fetch_batch_size + 1})")

        async def fetch(
            client: AsyncIMAPClient, uids: list[str] = batch
        ) -> dict[str, Email]:
            return await client.fetch_messages(uids)

        messages = await session.run(fetch)

        for uid in batch:
            if uid in messages:
                yield {
                    "email_uid": uid,
                    "email": messages[uid],
                    "operation": "process",
                    "decision": None,
                    "reason": None,
                    "config": config,
                    "rules": None,
                }
            else:
                logger.warning(f"Could not fetch email {uid}, skipping")


async def prefetch[T](source: AsyncGenerator[T], depth: int) -> AsyncIterator[T]:
    """Drive ``source`` in a background task, staying up to ``depth`` items ahead.

    ``source`` is closed when iteration stops, including when the consumer exits
    early or fails.
    """
    done = object()
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max(1, depth))
    failure: list[BaseException] = []

    async def pump() -> None:
        try:
            async with contextlib.aclosing(source):
                async for item in source:
                    await queue.put(item)
        except Exception as e:
            failure.append(e)
        await queue.put(done)

    task = asyncio.create_task(pump())
    try:
        while (item := await queue.get()) is not done:
            yield item
        if failure:
            raise failure[0]
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def classify_deterministic(
    email: Email, config: Config, rules: RuleSet | None = None
) -> EmailDecision | None:
    """Hard keeps, then learned rules, then heuristics; None if nothing matched."""
    for check in HARD_KEEP_FILTERS:
        if decision := check(email, config):
            return decision

    if rules and (decision := rules.decide(email)):
        return decision

    for check in HEURISTIC_FILTERS:
        if decision := check(email, config):
            return decision
    return None


async def classify_email(
    email: Email, config: Config, rules: RuleSet | None = None
) -> EmailDecision:
    """Apply deterministic filters and rules, falling back to AI classification."""
    if decision := classify_deterministic(email, config, rules):
        return decision

    # classify_with_ai is blocking (MLX inference); run it off the event loop
    # so IMAP prefetching and deletes keep making progress meanwhile.
    return await asyncio.to_thread(classify_with_ai, email, config)


async def process_email_node(state: StreamingState) -> StreamingState:
    """Process a single email.

    Runs deterministic filters and, if none match, AI classification.
    """
    uid = state["email_uid"]
    email = state["email"]

    logger.info(f"Processing email {uid}: {email.subject}")

    try:
        decision_obj = await classify_email(email, state["config"], state["rules"])
    except Exception as e:
        logger.exception(f"Error processing email {uid}: {e}")
        # Safe default: keep the email on error
        return {
            **state,
            "operation": "keep",
            "decision": "keep",
            "reason": f"error: {e!s}",
        }

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

    # Load rules up front so a bad file fails before touching the mailbox
    rules = RuleSet.load(config.rules_path) if config.rules_path else None
    if rules:
        logger.info(f"Loaded {len(rules.rules)} rules from {config.rules_path}")

    logger.info("Starting streaming pipeline...")

    stats = {"processed": 0, "deleted": 0, "kept": 0, "errors": 0}
    started = time.monotonic()

    async with IMAPSession(config.email) as session:
        # Coordinator for batch deletes (shares the producer's connection)
        async def execute_deletes(uids: list[str]):
            logger.info(f"Executing batch delete for {len(uids)} emails")
            if config.dry_run:
                logger.info(f"DRY RUN: Would delete {uids}")
                return

            try:
                await session.run(lambda client: client.delete_emails(uids))
            except Exception as e:
                logger.exception(f"Error deleting emails: {e}")

        batch_coordinator = BatchCoordinator(
            processor=execute_deletes,
            batch_size=config.batch_size,
        )

        # Keep about one fetch batch ready while the current one is classified
        prefetch_depth = max(1, min(config.fetch_size, MAX_MESSAGES_PER_FETCH))

        async for initial_state in prefetch(
            imap_producer(config, session), depth=prefetch_depth
        ):
            try:
                result = await graph.ainvoke({**initial_state, "rules": rules})

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
                    rate = stats["processed"] / (time.monotonic() - started)
                    print(f"Processed {stats['processed']} emails ({rate:.1f}/s)...")

            except Exception as e:
                logger.exception(f"Error processing email: {e}")
                stats["errors"] += 1

        # Flush remaining deletes
        await batch_coordinator.flush()

    elapsed = time.monotonic() - started
    logger.info("Pipeline complete.")
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Total Processed: {stats['processed']}")
    print(f"Total Deleted:   {stats['deleted']}")
    print(f"Total Kept:      {stats['kept']}")
    print(f"Errors:          {stats['errors']}")
    print(f"Elapsed:         {elapsed:.1f}s")
    print("=" * 40)


async def run_learning_pipeline(
    config: Config,
    min_support: int = 3,
    min_agreement: float = 0.9,
    use_ai: bool = True,
) -> RuleSet:
    """Classify the mailbox read-only and learn sender/domain rules.

    Never mutates the mailbox: there is no delete path here, regardless of
    ``config.dry_run``. With ``use_ai=False`` only emails matched by the
    deterministic filters contribute evidence (fast, no model needed).
    """
    learner = RuleLearner()
    started = time.monotonic()
    skipped = 0

    async with IMAPSession(config.email) as session:
        prefetch_depth = max(1, min(config.fetch_size, MAX_MESSAGES_PER_FETCH))
        async for state in prefetch(
            imap_producer(config, session), depth=prefetch_depth
        ):
            email = state["email"]
            decision: EmailDecision | None
            try:
                if use_ai:
                    decision = await classify_email(email, config)
                else:
                    decision = classify_deterministic(email, config)
            except Exception as e:
                logger.exception(f"Error classifying email {email.uid}: {e}")
                decision = None

            if decision is None:
                skipped += 1
            else:
                learner.observe(decision)

            seen = learner.observed + skipped
            if seen % 50 == 0:
                rate = seen / (time.monotonic() - started)
                print(f"Classified {seen} emails ({rate:.1f}/s)...")

    if skipped:
        logger.info(f"{skipped} emails had no decision and were not used")
    return learner.build(
        min_support=min_support, min_agreement=min_agreement, account=config.email
    )
