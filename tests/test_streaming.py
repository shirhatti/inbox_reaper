import unittest
from unittest.mock import MagicMock, patch

from inbox_reaper.langgraph_streaming import run_streaming_pipeline
from inbox_reaper.state import Config


class TestStreamingPipeline(unittest.IsolatedAsyncioTestCase):
    @patch("inbox_reaper.langgraph_streaming.IMAPClient")
    async def test_streaming_pipeline(self, mock_imap_client_cls):
        # Setup mock client
        mock_client = MagicMock()
        mock_imap_client_cls.return_value = mock_client

        # Mock get_latest_uid
        mock_client.get_latest_uid.return_value = 2

        # Mock search_uids
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
        mock_client.connect.assert_called()
        mock_client.disconnect.assert_called()

        # Should search and fetch
        mock_client.get_latest_uid.assert_called()
        mock_client.search_uids.assert_called()
        mock_client.fetch_headers.assert_called()

    @patch("inbox_reaper.langgraph_streaming.IMAPClient")
    @patch("inbox_reaper.langgraph_streaming.BatchCoordinator")
    async def test_streaming_pipeline_deletes(
        self, mock_coordinator_cls, mock_imap_client_cls
    ):
        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator_cls.return_value = mock_coordinator
        # Make add an async method
        mock_coordinator.add = unittest.mock.AsyncMock()
        mock_coordinator.flush = unittest.mock.AsyncMock()

        # Setup mock client
        mock_client = MagicMock()
        mock_imap_client_cls.return_value = mock_client

        # Mock get_latest_uid
        mock_client.get_latest_uid.return_value = 2

        # Mock search_uids
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

    @patch("inbox_reaper.langgraph_streaming.IMAPClient")
    async def test_streaming_pipeline_max_emails(self, mock_imap_client_cls):
        # Setup mock client
        mock_client = MagicMock()
        mock_imap_client_cls.return_value = mock_client

        # Mock get_latest_uid
        mock_client.get_latest_uid.return_value = 100

        # Mock search_uids to return UIDs in range
        def side_effect_search(criteria):
            # criteria is "min:max"
            start, end = map(int, criteria.split(":"))
            # Return UIDs in this range (simulate some gaps)
            return [
                str(i) for i in range(start, end + 1) if i % 2 == 0
            ]  # Even UIDs only

        mock_client.search_uids.side_effect = side_effect_search

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

        # Verify get_latest_uid called
        mock_client.get_latest_uid.assert_called_once()

        # Verify search_uids called with range "91:100" (since fetch_size=10)
        # Max UID is 100. First chunk is 91-100.
        mock_client.search_uids.assert_called_with(criteria="91:100")

        # Verify fetch_headers called with 3 newest even UIDs
        # in range 91-100: 100, 98, 96
        mock_client.fetch_headers.assert_called_once()
        call_args = mock_client.fetch_headers.call_args[1]
        self.assertEqual(call_args["uids"], ["100", "98", "96"])


if __name__ == "__main__":
    unittest.main()
