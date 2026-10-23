import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from inbox_reaper.imap_client import IMAPConnectionError
from inbox_reaper.langgraph_streaming import (
    IMAPSession,
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
            "inbox_reaper.langgraph_streaming.classify_with_ai",
            side_effect=lambda email, config: EmailDecision(
                email=email,
                decision=Decision.KEEP,
                reason=FilterReason.AI_CLASSIFIED,
                confidence=0.0,
            ),
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

    @patch("inbox_reaper.langgraph_streaming.classify_with_ai")
    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_unmatched_emails_go_to_ai(self, mock_create_client, mock_ai):
        email = make_email("7", "Hello", "Robot <bot@service.example>", "Hi there")
        mock_create_client.return_value = make_mock_client({"7": email})
        mock_ai.return_value = EmailDecision(
            email=email,
            decision=Decision.DELETE,
            reason=FilterReason.AI_CLASSIFIED,
            confidence=0.9,
        )

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
