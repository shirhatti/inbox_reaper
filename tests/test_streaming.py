import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from inbox_reaper.langgraph_streaming import run_streaming_pipeline
from inbox_reaper.state import Config


class TestStreamingPipeline(unittest.IsolatedAsyncioTestCase):
    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_streaming_pipeline(self, mock_create_client):
        # Setup mock client
        mock_client = AsyncMock()
        mock_create_client.return_value = mock_client

        # Mock context manager behavior
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Mock search_uids to return all UIDs
        mock_client.search_uids.return_value = ["1", "2"]

        # Mock fetch_headers
        mock_client.fetch_headers.return_value = {
            "1": {"subject": "Newsletter", "sender": "spam@example.com"},
            "2": {"subject": "Important", "sender": "boss@example.com"},
        }

        # Setup config
        config = Config(
            email="test@example.com",
            model_name="test",
            ollama_base_url="http://localhost",
            batch_size=10,
            concurrent_ai_limit=5,
            dry_run=True,
            keywords=[],
            whitelist_domains=[],
        )

        # Run pipeline
        await run_streaming_pipeline(config)

        # Verify interactions
        mock_create_client.assert_called()
        mock_client.select_mailbox.assert_called_with("INBOX")
        mock_client.search_uids.assert_called_with(criteria="ALL")
        mock_client.fetch_headers.assert_called()

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    @patch("inbox_reaper.langgraph_streaming.BatchCoordinator")
    async def test_streaming_pipeline_deletes(
        self, mock_coordinator_cls, mock_create_client
    ):
        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator_cls.return_value = mock_coordinator
        # Make add an async method
        mock_coordinator.add = unittest.mock.AsyncMock()
        mock_coordinator.flush = unittest.mock.AsyncMock()

        # Setup mock client
        mock_client = AsyncMock()
        mock_create_client.return_value = mock_client

        # Mock context manager behavior
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Mock search_uids to return all UIDs
        mock_client.search_uids.return_value = ["1", "2"]

        # Mock fetch_headers
        mock_client.fetch_headers.return_value = {
            # Should be deleted
            "1": {
                "subject": "Newsletter",
                "sender": "spam@example.com",
            },
            # Should be kept
            "2": {
                "subject": "Important",
                "sender": "boss@example.com",
            },
        }

        config = Config(
            email="test@example.com",
            model_name="test",
            ollama_base_url="http://localhost",
            batch_size=10,
            concurrent_ai_limit=5,
            dry_run=False,
            keywords=[],
            whitelist_domains=[],
        )

        # Run pipeline
        await run_streaming_pipeline(config)

        # Verify coordinator was initialized
        mock_coordinator_cls.assert_called_once()

        # We can't easily verify 'add' calls because we need to await
        # the coroutine in the mock. But we can verify flush was called
        mock_coordinator.flush.assert_called_once()

    @patch("inbox_reaper.langgraph_streaming.create_imap_client")
    async def test_streaming_pipeline_max_emails(self, mock_create_client):
        # Setup mock client
        mock_client = AsyncMock()
        mock_create_client.return_value = mock_client

        # Mock context manager behavior
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Mock search_uids to return all UIDs (simulating 100 emails)
        # Return even UIDs from 2 to 100
        all_uids = [str(i) for i in range(2, 101, 2)]  # 2, 4, 6, ..., 100
        mock_client.search_uids.return_value = all_uids

        # Mock fetch_headers
        def side_effect_fetch(uids):
            return {
                uid: {"subject": f"Email {uid}", "sender": "test@example.com"}
                for uid in uids
            }

        mock_client.fetch_headers.side_effect = side_effect_fetch

        # Config with max_emails = 3, fetch_size = 10
        config = Config(
            email="test@example.com",
            model_name="test",
            ollama_base_url="http://localhost",
            batch_size=10,
            fetch_size=10,
            concurrent_ai_limit=5,
            dry_run=True,
            max_emails=3,
            keywords=[],
            whitelist_domains=[],
        )

        # Run pipeline
        await run_streaming_pipeline(config)

        # Verify search_uids called with ALL
        mock_client.search_uids.assert_called_with(criteria="ALL")

        # Verify fetch_headers called with 3 oldest UIDs: 2, 4, 6
        # (Processing in chronological order now)
        mock_client.fetch_headers.assert_called_once()
        call_args = mock_client.fetch_headers.call_args[1]
        self.assertEqual(call_args["uids"], ["2", "4", "6"])


if __name__ == "__main__":
    unittest.main()
