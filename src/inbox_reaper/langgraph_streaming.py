"""Streaming implementation of LangGraph pipeline for email processing.

This module implements a streaming architecture where emails are fetched from IMAP
in batches over a single shared connection and processed one by one through a
LangGraph workflow. The next batch is prefetched while the current one is being
classified, so network latency overlaps with (much slower) AI inference.
"""

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any, Self, TypedDict

import aioimaplib
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import (
    check_attachments,
    check_keywords,
    check_transactional_patterns,
    check_whitelist,
    classify_batch_with_ai,
)
from .batch_coordinator import BatchCoordinator
from .combiner import Combiner
from .credential_helper import get_credentials
from .features import FeatureContext, check_expiry, describe_features, extract_features
from .imap_client import AsyncIMAPClient, IMAPConnectionError
from .oauth_flow import refresh_if_expiring
from .rules import RuleLearner, RuleSet
from .state import Config, Decision, Email, EmailDecision, FilterReason

logger = logging.getLogger(__name__)

# Upper bound on full messages per UID FETCH, to cap memory when emails carry
# large attachments.
MAX_MESSAGES_PER_FETCH = 25

# Explicit user intent; always applied first and never overridden by rules.
HARD_KEEP_FILTERS = (
    check_attachments,
    check_keywords,
    check_whitelist,
)

# Heuristics applied after learned rules, before AI. They only ever keep:
# marketing indicators (unsubscribe footers, "view in browser") also appear on
# bank alerts and statements, so they are a combiner feature, never a delete.
HEURISTIC_FILTERS = (
    check_transactional_patterns,  # receipts, orders, shipping
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

    try:
        refresh_if_expiring(email, creds)
    except Exception as e:
        logger.warning("Could not refresh token: %s, using existing token", e)

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
                logger.warning("IMAP operation failed (%s), reconnecting...", e)
                await self._disconnect()
            client = await self._connect()
            return await operation(client)

    async def close(self) -> None:
        async with self._lock:
            await self._disconnect()

    async def __aenter__(self) -> Self:
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
    ai: "AIClassifier | None"
    combiner: Combiner | None


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

    logger.info("Found %s emails", len(all_uids))

    # Process in chronological order (UIDs are already ascending)
    # Apply max_emails limit if set (take oldest N emails)
    if config.max_emails:
        all_uids = all_uids[: config.max_emails]
        logger.info("Limiting to %s oldest emails", len(all_uids))

    # 2. Fetch full messages in batches -- one round trip per batch
    fetch_batch_size = max(1, min(config.fetch_size, MAX_MESSAGES_PER_FETCH))

    for i in range(0, len(all_uids), fetch_batch_size):
        batch = all_uids[i : i + fetch_batch_size]

        logger.info(
            "Fetching %s emails (batch %s)", len(batch), i // fetch_batch_size + 1
        )

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
                    "ai": None,
                    "combiner": None,
                }
            else:
                logger.warning("Could not fetch email %s, skipping", uid)


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


# (email, signals summary) -> decision
type AIClassifier = Callable[[Email, str], Awaitable[EmailDecision]]


class AIBatcher:
    """Groups AI classification requests into batched MLX calls.

    A single worker runs one batch at a time in a thread, so the model is never
    called concurrently. Requests that arrive while a batch is running form the
    next batch (up to ``max_batch``), so batches fill up as long as callers
    keep more than one email in flight.
    """

    def __init__(self, config: Config, max_batch: int):
        self._config = config
        self._max_batch = max(1, max_batch)
        self._queue: asyncio.Queue[tuple[Email, str, asyncio.Future[EmailDecision]]] = (
            asyncio.Queue()
        )
        self._worker: asyncio.Task[None] | None = None
        self.batches = 0

    async def __aenter__(self) -> Self:
        self._worker = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        # Nothing should be waiting by now, but never leave a caller hanging
        while not self._queue.empty():
            *_, future = self._queue.get_nowait()
            future.cancel()

    async def classify(self, email: Email, signals: str = "") -> EmailDecision:
        """Queue ``email`` for the next batch and wait for its decision."""
        future: asyncio.Future[EmailDecision] = (
            asyncio.get_running_loop().create_future()
        )
        await self._queue.put((email, signals, future))
        return await future

    async def _run(self) -> None:
        while True:
            batch = [await self._queue.get()]
            while len(batch) < self._max_batch and not self._queue.empty():
                batch.append(self._queue.get_nowait())

            emails = [email for email, _, _ in batch]
            signals = [sig for _, sig, _ in batch]
            self.batches += 1
            logger.info("AI classifying a batch of %s emails", len(emails))
            try:
                # Blocking MLX inference; keep the event loop free for IMAP
                decisions = await asyncio.to_thread(
                    classify_batch_with_ai, emails, self._config, signals
                )
            except Exception as e:
                for *_, future in batch:
                    if not future.done():
                        future.set_exception(e)
                continue

            for (*_, future), decision in zip(batch, decisions, strict=True):
                if not future.done():
                    future.set_result(decision)


def in_flight_limit(config: Config) -> int:
    """Emails processed at once: two AI batches' worth, so one can fill while
    the other runs and deterministic decisions never wait behind the model."""
    return 2 * max(1, config.concurrent_ai_limit)


async def for_each_concurrently[T](
    source: AsyncIterator[T],
    handler: Callable[[T], Awaitable[None]],
    limit: int,
) -> None:
    """Run ``handler`` on each item with at most ``limit`` running at once.

    ``handler`` should handle its own errors: one that escapes cancels the
    remaining handlers and is re-raised (in an ExceptionGroup).
    """
    slots = asyncio.Semaphore(max(1, limit))

    async def run(item: T) -> None:
        try:
            await handler(item)
        finally:
            slots.release()

    async with asyncio.TaskGroup() as tg:
        async for item in source:
            await slots.acquire()
            tg.create_task(run(item))


def classify_deterministic(
    email: Email,
    config: Config,
    rules: RuleSet | None = None,
    *,
    heuristics: bool = True,
    now: datetime | None = None,
) -> EmailDecision | None:
    """Hard keeps, expiry, learned rules, then heuristics; None if nothing matched.

    With ``heuristics=False`` the heuristic filters are skipped (the combiner
    sees them as features instead).
    """
    for check in HARD_KEEP_FILTERS:
        if decision := check(email, config):
            return decision

    # Time-sensitive mail: expired codes go even if a sender rule would keep them
    if decision := check_expiry(email, config, now):
        return decision

    if rules and (decision := rules.decide(email)):
        return decision

    if not heuristics:
        return None
    for check in HEURISTIC_FILTERS:
        if decision := check(email, config):
            return decision
    return None


async def classify_email(
    email: Email,
    config: Config,
    rules: RuleSet | None,
    ai: AIClassifier,
    combiner: Combiner | None = None,
) -> EmailDecision:
    """Classify one email.

    Hard keeps, expiry and learned rules always come first. Then, with a
    combiner, the feature vector decides when it is confident and contested
    mail goes to ``ai`` along with a summary of the signals; without one, the
    heuristic filters run and ``ai`` handles whatever they leave.
    """
    if decision := classify_deterministic(
        email, config, rules, heuristics=combiner is None
    ):
        return decision
    if combiner is None:
        return await ai(email, "")

    features = extract_features(email, FeatureContext(config=config, rules=rules))
    verdict, p_delete = combiner.decide(features)
    if verdict is not None:
        return EmailDecision(
            email=email,
            decision=verdict,
            reason=FilterReason.COMBINER,
            confidence=p_delete if verdict == Decision.DELETE else 1 - p_delete,
        )
    return await ai(email, describe_features(features))


async def process_email_node(state: StreamingState) -> StreamingState:
    """Process a single email.

    Runs deterministic filters and, if none match, AI classification.
    """
    uid = state["email_uid"]
    email = state["email"]
    ai = state["ai"]
    if ai is None:
        raise ValueError("StreamingState needs an AI classifier")

    logger.info("Processing email %s: %s", uid, email.subject)

    try:
        decision_obj = await classify_email(
            email, state["config"], state["rules"], ai, state.get("combiner")
        )
    except Exception as e:
        logger.exception("Error processing email %s", uid)
        # Safe default: keep the email on error
        return {
            **state,
            "operation": "keep",
            "decision": "keep",
            "reason": f"error: {e!s}",
        }

    logger.info(
        "[%s] Email %s: %s... -> %s (confidence: %s)",
        decision_obj.decision.value.upper(),
        uid,
        email.subject[:50],
        decision_obj.reason.value,
        decision_obj.confidence,
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
    combiner = Combiner.load(config.combiner_path) if config.combiner_path else None
    if combiner:
        logger.info(
            "Loaded combiner trained on %s emails from %s",
            combiner.trained_on,
            config.combiner_path,
        )
    if rules:
        logger.info("Loaded %s rules from %s", len(rules.rules), config.rules_path)

    logger.info("Starting streaming pipeline...")

    stats = {"processed": 0, "deleted": 0, "kept": 0, "errors": 0}
    started = time.monotonic()

    async with IMAPSession(config.email) as session:
        # Coordinator for batch deletes (shares the producer's connection)
        async def execute_deletes(uids: list[str]):
            logger.info("Executing batch delete for %s emails", len(uids))
            if config.dry_run:
                logger.info("DRY RUN: Would delete %s", uids)
                return

            try:
                await session.run(lambda client: client.delete_emails(uids))
            except Exception:
                logger.exception("Error deleting emails")

        batch_coordinator = BatchCoordinator(
            processor=execute_deletes,
            batch_size=config.batch_size,
        )

        # Keep about one fetch batch ready while the current one is classified
        prefetch_depth = max(1, min(config.fetch_size, MAX_MESSAGES_PER_FETCH))

        async def handle(initial_state: StreamingState) -> None:
            try:
                result = await graph.ainvoke(
                    {
                        **initial_state,
                        "rules": rules,
                        "ai": batcher.classify,
                        "combiner": combiner,
                    }
                )

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

            except Exception:
                logger.exception("Error processing email")
                stats["errors"] += 1

        async with AIBatcher(config, config.concurrent_ai_limit) as batcher:
            await for_each_concurrently(
                prefetch(imap_producer(config, session), depth=prefetch_depth),
                handle,
                in_flight_limit(config),
            )

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

    async with (
        IMAPSession(config.email) as session,
        AIBatcher(config, config.concurrent_ai_limit) as batcher,
    ):

        async def handle(state: StreamingState) -> None:
            nonlocal skipped
            email = state["email"]
            decision: EmailDecision | None
            try:
                if use_ai:
                    decision = await classify_email(
                        email, config, None, batcher.classify
                    )
                else:
                    decision = classify_deterministic(email, config)
            except Exception:
                logger.exception("Error classifying email %s", email.uid)
                decision = None

            if decision is None:
                skipped += 1
            else:
                learner.observe(decision)

            seen = learner.observed + skipped
            if seen % 50 == 0:
                rate = seen / (time.monotonic() - started)
                print(f"Classified {seen} emails ({rate:.1f}/s)...")

        prefetch_depth = max(1, min(config.fetch_size, MAX_MESSAGES_PER_FETCH))
        await for_each_concurrently(
            prefetch(imap_producer(config, session), depth=prefetch_depth),
            handle,
            in_flight_limit(config),
        )

    if skipped:
        logger.info("%s emails had no decision and were not used", skipped)
    return learner.build(
        min_support=min_support, min_agreement=min_agreement, account=config.email
    )


async def sample_inbox(
    config: Config, count: int, *, seed: int = 0, exclude: set[str] | None = None
) -> list[Email]:
    """Fetch a random sample of INBOX emails, read-only (messages stay unread)."""
    rng = random.Random(seed)
    async with IMAPSession(config.email) as session:
        uids = await session.run(lambda client: client.search_uids(criteria="ALL"))
        pool = [u for u in uids if u not in (exclude or set())]
        picked = sorted(rng.sample(pool, min(count, len(pool))), key=int)
        emails: list[Email] = []
        for i in range(0, len(picked), MAX_MESSAGES_PER_FETCH):
            batch = picked[i : i + MAX_MESSAGES_PER_FETCH]

            async def fetch(
                client: AsyncIMAPClient, uids: list[str] = batch
            ) -> dict[str, Email]:
                return await client.fetch_messages(uids)

            messages = await session.run(fetch)
            emails.extend(messages[u] for u in batch if u in messages)
    return emails
