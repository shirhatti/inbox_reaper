import asyncio
import threading
import time
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from inbox_reaper.imap_client import IMAPConnectionError
from inbox_reaper.langgraph_streaming import (
    AIBatcher,
    IMAPSession,
    classify_email,
    for_each_concurrently,
    prefetch,
    run_streaming_pipeline,
)
from inbox_reaper.state import Config, Decision, Email, EmailDecision, FilterReason


def make_email(uid: str, subject: str, sender: str, body: str = "") -> Email:
    return Email(
        uid=uid,
        subject=subject,
        sender=sender,
        body=body,
        date=datetime(2025, 1, 1),
    )


def ai_keep(email: Email) -> EmailDecision:
    return EmailDecision(
        email=email,
        decision=Decision.KEEP,
        reason=FilterReason.AI_CLASSIFIED,
        confidence=0.0,
    )


def make_config(**overrides) -> Config:
    defaults = {
        "email": "test@example.com",
        "model_name": "test",
        "batch_size": 10,
        "concurrent_ai_limit": 5,
        "dry_run": True,
        "keywords": ["important"],
        "whitelist_domains": [],
    }
    return Config(**{**defaults, **overrides})


def make_mock_client(messages: dict[str, Email]) -> AsyncMock:
    client = AsyncMock()
    client.search_uids.return_value = list(messages)
    client.fetch_messages.side_effect = lambda uids: {
        uid: messages[uid] for uid in uids if uid in messages
    }
    return client


MESSAGES = {
    # Marketing: unsubscribe footer + marketing subdomain -> delete
    "1": make_email(
        "1",
        "Big sale",
        "Deals <deals@e.shop.example>",
        "Save 50% today. Unsubscribe here.",
    ),
    # Keyword match -> keep
    "2": make_email("2", "Important update", "Boss <boss@corp.example>"),
}


class TestStreamingPipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Never load a real model in tests; AI says "keep" unless a test overrides
        patcher = patch(
            "inbox_reaper.langgraph_streaming.classify_batch_with_ai",
            side_effect=lambda emails, config, signals=None: [
                ai_keep(email) for email in emails
            ],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_streaming_pipeline_uses_one_connection(self, mock_create_client):
        mock_client = make_mock_client(MESSAGES)
        mock_create_client.return_value = mock_client

        await run_streaming_pipeline(make_config())

        # A single connection serves search, fetch and the whole run
        mock_create_client.assert_called_once_with("test@example.com")
        mock_client.connect.assert_awaited_once()
        mock_client.select_mailbox.assert_called_once_with("INBOX")
        mock_client.search_uids.assert_called_with(criteria="ALL")
        # Messages are fetched in one batch, not one round trip per email
        mock_client.fetch_messages.assert_called_once_with(["1", "2"])
        mock_client.disconnect.assert_awaited_once()
        # Dry run never deletes
        mock_client.delete_emails.assert_not_called()

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_streaming_pipeline_deletes(self, mock_create_client):
        mock_client = make_mock_client(MESSAGES)
        mock_create_client.return_value = mock_client

        await run_streaming_pipeline(make_config(dry_run=False))

        mock_client.delete_emails.assert_awaited_once_with(["1"])
        mock_create_client.assert_called_once()

    @patch("inbox_reaper.langgraph_streaming.classify_batch_with_ai")
    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_unmatched_emails_go_to_ai(self, mock_create_client, mock_ai):
        email = make_email("7", "Hello", "Robot <bot@service.example>", "Hi there")
        mock_create_client.return_value = make_mock_client({"7": email})
        mock_ai.return_value = [
            EmailDecision(
                email=email,
                decision=Decision.DELETE,
                reason=FilterReason.AI_CLASSIFIED,
                confidence=0.9,
            )
        ]

        await run_streaming_pipeline(make_config(dry_run=False))

        mock_ai.assert_called_once()
        mock_create_client.return_value.delete_emails.assert_awaited_once_with(["7"])

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_streaming_pipeline_max_emails(self, mock_create_client):
        all_uids = [str(i) for i in range(2, 101, 2)]  # 2, 4, 6, ..., 100
        messages = {
            uid: make_email(uid, f"Email {uid}", "test@example.com") for uid in all_uids
        }
        mock_client = make_mock_client(messages)
        mock_create_client.return_value = mock_client

        # Config with max_emails = 3, fetch_size = 10
        await run_streaming_pipeline(make_config(fetch_size=10, max_emails=3))

        # Oldest 3 UIDs, in chronological order
        mock_client.fetch_messages.assert_called_once_with(["2", "4", "6"])

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_fetches_in_capped_batches(self, mock_create_client):
        uids = [str(i) for i in range(1, 61)]
        messages = {uid: make_email(uid, "x", "a@b.example") for uid in uids}
        mock_client = make_mock_client(messages)
        mock_create_client.return_value = mock_client

        await run_streaming_pipeline(make_config(fetch_size=100))

        batch_sizes = [
            len(c.args[0]) for c in mock_client.fetch_messages.call_args_list
        ]
        self.assertEqual(batch_sizes, [25, 25, 10])


class TestIMAPSession(unittest.IsolatedAsyncioTestCase):
    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_reuses_connection(self, mock_create_client):
        mock_create_client.return_value = AsyncMock()

        async with IMAPSession("u@example.com") as session:
            await session.run(lambda c: c.search_uids())
            await session.run(lambda c: c.search_uids())

        mock_create_client.assert_called_once()

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_reconnects_once_on_dropped_connection(self, mock_create_client):
        dead, fresh = AsyncMock(), AsyncMock()
        dead.search_uids.side_effect = IMAPConnectionError("session expired")
        fresh.search_uids.return_value = ["1"]
        mock_create_client.side_effect = [dead, fresh]

        async with IMAPSession("u@example.com") as session:
            result = await session.run(lambda c: c.search_uids())

        self.assertEqual(result, ["1"])
        self.assertEqual(mock_create_client.call_count, 2)
        dead.disconnect.assert_awaited_once()

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_does_not_retry_unrelated_errors(self, mock_create_client):
        client = AsyncMock()
        client.search_uids.side_effect = ValueError("bug")
        mock_create_client.return_value = client

        session = IMAPSession("u@example.com")
        with self.assertRaises(ValueError):
            await session.run(lambda c: c.search_uids())
        mock_create_client.assert_called_once()


class TestPrefetch(unittest.IsolatedAsyncioTestCase):
    async def test_yields_all_items_in_order(self):
        async def source():
            for i in range(5):
                yield i

        self.assertEqual(
            [i async for i in prefetch(source(), depth=2)], [0, 1, 2, 3, 4]
        )

    async def test_propagates_source_errors(self):
        async def source():
            yield 1
            raise RuntimeError("boom")

        seen = []
        with self.assertRaises(RuntimeError):
            async for item in prefetch(source(), depth=1):
                seen.append(item)
        self.assertEqual(seen, [1])

    async def test_cancels_producer_when_consumer_stops(self):
        cleanup = MagicMock()

        async def source():
            try:
                for i in range(100):
                    yield i
            finally:
                cleanup()

        gen = prefetch(source(), depth=1)
        async for item in gen:
            if item == 2:
                break
        await gen.aclose()
        # Producer task was cancelled rather than left running
        cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class TestAIBatcher(unittest.IsolatedAsyncioTestCase):
    async def test_requests_waiting_during_a_batch_form_the_next_batch(self):
        started = threading.Event()
        release = threading.Event()
        seen: list[list[str]] = []

        def fake_batch(emails, config, signals=None):
            seen.append([e.uid for e in emails])
            if len(seen) == 1:
                started.set()
                release.wait(timeout=5)  # hold the first batch while others queue
            return [ai_keep(e) for e in emails]

        emails = [make_email(str(i), "s", "a@b.example") for i in range(7)]
        with patch(
            "inbox_reaper.langgraph_streaming.classify_batch_with_ai",
            side_effect=fake_batch,
        ):
            async with AIBatcher(make_config(), max_batch=4) as batcher:
                first = asyncio.create_task(batcher.classify(emails[0]))
                await asyncio.to_thread(started.wait, 5)
                rest = [asyncio.create_task(batcher.classify(e)) for e in emails[1:]]
                await asyncio.sleep(0.05)
                release.set()
                results = await asyncio.gather(first, *rest)

        assert seen == [["0"], ["1", "2", "3", "4"], ["5", "6"]]
        assert [r.email.uid for r in results] == [e.uid for e in emails]

    async def test_batch_failure_reaches_every_caller(self):
        emails = [make_email(str(i), "s", "a@b.example") for i in range(2)]
        with patch(
            "inbox_reaper.langgraph_streaming.classify_batch_with_ai",
            side_effect=RuntimeError("boom"),
        ):
            async with AIBatcher(make_config(), max_batch=4) as batcher:
                results = await asyncio.gather(
                    *(batcher.classify(e) for e in emails), return_exceptions=True
                )
        assert all(isinstance(r, RuntimeError) for r in results)


class TestForEachConcurrently(unittest.IsolatedAsyncioTestCase):
    async def test_respects_limit_and_handles_every_item(self):
        running = peak = 0
        done: list[int] = []

        async def source():
            for i in range(10):
                yield i

        async def handler(i):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            done.append(i)

        await for_each_concurrently(source(), handler, limit=3)
        assert peak == 3
        assert sorted(done) == list(range(10))


class TestConcurrentPipeline(unittest.IsolatedAsyncioTestCase):
    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_ai_emails_are_batched_and_deterministic_ones_dont_wait(
        self, mock_create_client
    ):
        # 6 emails for the AI interleaved with 6 keyword keeps
        messages = {}
        for i in range(12):
            uid = str(i + 1)
            if i % 2:
                messages[uid] = make_email(uid, "Important", "boss@corp.example")
            else:
                messages[uid] = make_email(uid, "Hello", "bot@svc.example", "Hi")
        mock_create_client.return_value = make_mock_client(messages)
        batches: list[int] = []

        def fake_batch(emails, config, signals=None):
            batches.append(len(emails))
            time.sleep(0.05)  # model is slow; more AI requests queue meanwhile
            return [
                EmailDecision(
                    email=e,
                    decision=Decision.DELETE,
                    reason=FilterReason.AI_CLASSIFIED,
                    confidence=0.9,
                )
                for e in emails
            ]

        with patch(
            "inbox_reaper.langgraph_streaming.classify_batch_with_ai",
            side_effect=fake_batch,
        ):
            await run_streaming_pipeline(
                make_config(dry_run=False, concurrent_ai_limit=4, batch_size=100)
            )

        assert sum(batches) == 6
        assert max(batches) > 1  # requests were grouped
        assert max(batches) <= 4  # never above the limit
        deleted = mock_create_client.return_value.delete_emails.await_args.args[0]
        assert sorted(deleted, key=int) == ["1", "3", "5", "7", "9", "11"]


class TestCombinerRouting(unittest.IsolatedAsyncioTestCase):
    def combiner(self, p_delete: float):
        combiner = MagicMock()
        combiner.decide.return_value = (
            Decision.DELETE
            if p_delete >= 0.9
            else Decision.KEEP
            if p_delete <= 0.1
            else None,
            p_delete,
        )
        return combiner

    async def test_confident_combiner_decides_without_ai(self):
        ai = AsyncMock()
        email = make_email("5", "Weekly deals", "news@e.shop.example", "Hi")
        decision = await classify_email(
            email, make_config(), None, ai, self.combiner(0.97)
        )
        assert decision.decision == Decision.DELETE
        assert decision.reason == FilterReason.COMBINER
        assert decision.confidence == 0.97
        ai.assert_not_awaited()

    async def test_contested_goes_to_ai_with_signals(self):
        email = make_email("5", "Hello", "news@e.shop.example", "Hi")
        email = email.model_copy(update={"headers": {"precedence": "bulk"}})
        ai = AsyncMock(return_value=ai_keep(email))
        decision = await classify_email(
            email, make_config(), None, ai, self.combiner(0.5)
        )
        assert decision.reason == FilterReason.AI_CLASSIFIED
        (_, signals), _ = ai.await_args
        assert "bulk mail" in signals

    async def test_hard_keeps_and_expiry_run_before_the_combiner(self):
        combiner = self.combiner(0.99)
        ai = AsyncMock()
        keyword = make_email("6", "Important update", "boss@corp.example")
        assert (
            await classify_email(keyword, make_config(), None, ai, combiner)
        ).decision == (Decision.KEEP)
        code = make_email(
            "7", "Your verification code is 123456", "no-reply@acct.example"
        )
        expired = await classify_email(code, make_config(), None, ai, combiner)
        assert expired.reason == FilterReason.EXPIRED
        combiner.decide.assert_not_called()

    async def test_without_combiner_heuristics_still_decide(self):
        ai = AsyncMock()
        decision = await classify_email(MESSAGES["1"], make_config(), None, ai)
        assert decision.decision == Decision.DELETE
        ai.assert_not_awaited()
