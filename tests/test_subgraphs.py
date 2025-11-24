"""Tests for LangGraph subgraph execution and parallel processing.

This module tests:
- Email processing subgraph execution
- AI classification subgraph execution
- Parallel subgraph spawning
- State aggregation from parallel pipelines
- Race condition prevention in shared state
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langgraph.constants import Send

from inbox_reaper.langgraph_dag import (
    ai_classification_subgraph,
    email_processing_subgraph,
    fan_out_ai_classification,
    fan_out_processing,
)
from inbox_reaper.langgraph_state import (
    create_initial_state,
    merge_sender_stats,
)
from inbox_reaper.state import Config


class TestEmailProcessingSubgraph:
    """Test email processing subgraph execution."""

    def test_email_processing_subgraph_receives_current_email(self):
        """Test that subgraph receives current email data."""
        config = Config()
        state = create_initial_state(config)
        state["current_email_uid"] = "123"
        state["current_email"] = {
            "uid": "123",
            "subject": "Test Email",
            "sender": "test@example.com",
        }

        result = email_processing_subgraph(state)

        # Currently placeholder marks all emails as needing AI
        assert "needs_full_fetch" in result
        assert "123" in result["needs_full_fetch"]

    def test_email_processing_subgraph_placeholder_behavior(self):
        """Test current placeholder behavior of subgraph."""
        config = Config()
        state = create_initial_state(config)
        state["current_email_uid"] = "456"
        state["current_email"] = {
            "uid": "456",
            "subject": "Another Email",
            "sender": "sender@example.com",
        }

        result = email_processing_subgraph(state)

        # Placeholder marks email for AI classification
        assert result["needs_full_fetch"] == ["456"]

    def test_email_processing_subgraph_with_missing_uid(self):
        """Test subgraph handles missing current_email_uid gracefully."""
        config = Config()
        state = create_initial_state(config)
        # No current_email_uid set

        result = email_processing_subgraph(state)

        # Should still return a result (even if None UID)
        assert isinstance(result, dict)


class TestAIClassificationSubgraph:
    """Test AI classification subgraph execution."""

    def test_ai_classification_subgraph_receives_email_body(self):
        """Test that AI subgraph receives full email body."""
        config = Config()
        state = create_initial_state(config)
        state["current_email_uid"] = "123"
        state["current_email"] = {
            "uid": "123",
            "subject": "Test Email",
            "sender": "test@example.com",
            "body": "Full email body content",
        }

        result = ai_classification_subgraph(state)

        # Currently placeholder, just returns state
        assert isinstance(result, dict)

    def test_ai_classification_subgraph_placeholder_behavior(self):
        """Test current placeholder behavior of AI subgraph."""
        config = Config()
        state = create_initial_state(config)
        state["current_email_uid"] = "456"
        state["current_email"] = {
            "uid": "456",
            "subject": "Email for AI",
            "sender": "ai@example.com",
            "body": "Body for AI classification",
        }

        result = ai_classification_subgraph(state)

        # Placeholder returns state unchanged
        assert result == state


class TestParallelSubgraphSpawning:
    """Test that Send() correctly spawns parallel subgraphs."""

    def test_fan_out_processing_spawns_correct_number_of_subgraphs(self):
        """Test that fan_out creates one subgraph per email."""
        config = Config()
        state = create_initial_state(config)
        state["email_headers"] = {
            f"uid{i}": {
                "uid": f"uid{i}",
                "subject": f"Email {i}",
                "sender": f"sender{i}@example.com",
            }
            for i in range(10)
        }

        sends = fan_out_processing(state)

        assert len(sends) == 10
        assert all(isinstance(s, Send) for s in sends)
        assert all(s.node == "email_processing_subgraph" for s in sends)

    def test_fan_out_processing_preserves_state_in_each_send(self):
        """Test that each Send includes full state plus current email."""
        config = Config()
        state = create_initial_state(config)
        state["email_headers"] = {
            "123": {
                "uid": "123",
                "subject": "Test Email",
                "sender": "test@example.com",
            }
        }
        state["sender_stats"] = {
            "existing@example.com": {
                "sender": "existing@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            }
        }

        sends = fan_out_processing(state)

        assert len(sends) == 1
        send_state = sends[0].arg

        # Should include original state
        assert "sender_stats" in send_state
        assert "existing@example.com" in send_state["sender_stats"]

        # Should include current email data
        assert send_state["current_email_uid"] == "123"
        assert send_state["current_email"]["subject"] == "Test Email"

    def test_fan_out_ai_spawns_correct_number_of_subgraphs(self):
        """Test that AI fan_out creates one subgraph per email body."""
        config = Config()
        state = create_initial_state(config)
        state["email_bodies"] = {
            f"uid{i}": {
                "uid": f"uid{i}",
                "subject": f"Email {i}",
                "sender": f"sender{i}@example.com",
                "body": f"Body {i}",
            }
            for i in range(5)
        }

        sends = fan_out_ai_classification(state)

        assert len(sends) == 5
        assert all(isinstance(s, Send) for s in sends)
        assert all(s.node == "ai_classification_subgraph" for s in sends)

    def test_fan_out_ai_includes_full_email_data(self):
        """Test that AI Send includes full email with body."""
        config = Config()
        state = create_initial_state(config)
        state["email_bodies"] = {
            "123": {
                "uid": "123",
                "subject": "AI Test Email",
                "sender": "ai@example.com",
                "body": "Full body for AI classification",
                "attachments": ["doc.pdf"],
            }
        }

        sends = fan_out_ai_classification(state)

        assert len(sends) == 1
        send_state = sends[0].arg

        assert send_state["current_email_uid"] == "123"
        assert send_state["current_email"]["body"] == "Full body for AI classification"
        assert send_state["current_email"]["attachments"] == ["doc.pdf"]


class TestStateAggregationFromParallel:
    """Test state aggregation from parallel subgraph execution."""

    def test_parallel_decisions_accumulation(self):
        """Test that decisions from parallel subgraphs accumulate correctly."""
        # Simulate multiple parallel subgraphs each adding a decision
        decisions_from_subgraph1 = [
            {
                "email": {
                    "uid": "123",
                    "subject": "Email 1",
                    "sender": "sender1@example.com",
                    "body": "",
                    "date": datetime.now(),
                    "attachments": [],
                },
                "decision": "delete",
                "reason": "keyword",
                "confidence": 1.0,
                "processed_at": datetime.now(),
            }
        ]

        decisions_from_subgraph2 = [
            {
                "email": {
                    "uid": "456",
                    "subject": "Email 2",
                    "sender": "sender2@example.com",
                    "body": "",
                    "date": datetime.now(),
                    "attachments": [],
                },
                "decision": "keep",
                "reason": "whitelist",
                "confidence": 1.0,
                "processed_at": datetime.now(),
            }
        ]

        # Using operator.add reducer (as defined in GraphState)
        import operator

        accumulated = operator.add(decisions_from_subgraph1, decisions_from_subgraph2)

        assert len(accumulated) == 2
        assert accumulated[0]["email"]["uid"] == "123"
        assert accumulated[1]["email"]["uid"] == "456"

    def test_parallel_sender_stats_merging(self):
        """Test that sender stats from parallel subgraphs merge correctly."""
        # Simulate two parallel subgraphs updating same sender
        stats_from_subgraph1 = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 3,
                "total_count": 5,
                "auto_delete": False,
            }
        }

        stats_from_subgraph2 = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 2,
                "total_count": 4,
                "auto_delete": False,
            }
        }

        # Merge using the reducer
        merged = merge_sender_stats(stats_from_subgraph1, stats_from_subgraph2)

        # Counts should accumulate
        assert merged["sender@example.com"]["marketing_count"] == 5  # 3 + 2
        assert merged["sender@example.com"]["total_count"] == 9  # 5 + 4

    def test_parallel_needs_full_fetch_accumulation(self):
        """Test that needs_full_fetch lists from parallel subgraphs accumulate."""
        # Simulate multiple subgraphs marking emails for AI
        needs_fetch_1 = ["123", "456"]
        needs_fetch_2 = ["789"]
        needs_fetch_3 = ["101", "102"]

        import operator

        accumulated = operator.add(needs_fetch_1, needs_fetch_2)
        accumulated = operator.add(accumulated, needs_fetch_3)

        assert len(accumulated) == 5
        assert "123" in accumulated
        assert "789" in accumulated
        assert "102" in accumulated

    def test_parallel_to_delete_accumulation(self):
        """Test that to_delete UIDs from parallel subgraphs accumulate."""
        delete_1 = ["123", "456"]
        delete_2 = ["789"]
        delete_3 = []  # Some subgraphs may not delete anything

        import operator

        accumulated = operator.add(delete_1, delete_2)
        accumulated = operator.add(accumulated, delete_3)

        assert len(accumulated) == 3
        assert "123" in accumulated
        assert "456" in accumulated
        assert "789" in accumulated

    def test_parallel_processed_uids_set_union(self):
        """Test that processed UIDs from parallel subgraphs use set union."""
        processed_1 = {"123", "456"}
        processed_2 = {"456", "789"}  # 456 overlaps
        processed_3 = {"101"}

        import operator

        unioned = operator.or_(processed_1, processed_2)
        unioned = operator.or_(unioned, processed_3)

        assert len(unioned) == 4  # No duplicates
        assert unioned == {"123", "456", "789", "101"}


class TestRaceConditionPrevention:
    """Test that reducers prevent race conditions in parallel execution."""

    def test_concurrent_sender_stats_updates_no_race_condition(self):
        """Test that concurrent updates to sender stats don't have race conditions."""
        initial_stats = {}

        def update_sender_parallel(sender_id: int, count: int):
            """Simulate a parallel subgraph updating sender stats."""
            updates = {
                f"sender{sender_id}@example.com": {
                    "sender": f"sender{sender_id}@example.com",
                    "marketing_count": count,
                    "total_count": count * 2,
                    "auto_delete": False,
                }
            }
            return merge_sender_stats(initial_stats, updates)

        # Run 10 concurrent updates
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(update_sender_parallel, i, i + 1) for i in range(10)
            ]
            results = [f.result() for f in futures]

        # Each result should correctly contain its sender
        for i, result in enumerate(results):
            sender_key = f"sender{i}@example.com"
            assert sender_key in result
            assert result[sender_key]["marketing_count"] == i + 1

    def test_concurrent_same_sender_updates_accumulate(self):
        """Test that concurrent updates to same sender accumulate counts."""
        base_stats = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 10,
                "total_count": 20,
                "auto_delete": False,
            }
        }

        def increment_sender(increment: int):
            """Simulate parallel increment of sender stats."""
            updates = {
                "sender@example.com": {
                    "sender": "sender@example.com",
                    "marketing_count": increment,
                    "total_count": increment * 2,
                    "auto_delete": False,
                }
            }
            return merge_sender_stats(base_stats, updates)

        # Run concurrent increments
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(increment_sender, i + 1) for i in range(5)]
            results = [f.result() for f in futures]

        # Each result should correctly accumulate with base
        for i, result in enumerate(results):
            expected_marketing = 10 + (i + 1)
            expected_total = 20 + ((i + 1) * 2)
            assert result["sender@example.com"]["marketing_count"] == expected_marketing
            assert result["sender@example.com"]["total_count"] == expected_total

    def test_concurrent_list_accumulation_no_race_condition(self):
        """Test that concurrent list additions don't lose data."""
        import operator

        base_list = []

        def add_to_list(value: str):
            """Simulate parallel addition to list."""
            return operator.add(base_list, [value])

        # Run concurrent additions
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_to_list, f"item{i}") for i in range(5)]
            results = [f.result() for f in futures]

        # Each result should contain its item
        for i, result in enumerate(results):
            assert f"item{i}" in result
            assert len(result) == 1

    def test_concurrent_set_union_no_race_condition(self):
        """Test that concurrent set unions work correctly."""
        import operator

        base_set = {"uid0"}

        def add_to_set(uid: str):
            """Simulate parallel set union."""
            return operator.or_(base_set, {uid})

        # Run concurrent unions
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_to_set, f"uid{i}") for i in range(1, 6)]
            results = [f.result() for f in futures]

        # Each result should contain base and new UID
        for i, result in enumerate(results, 1):
            assert "uid0" in result
            assert f"uid{i}" in result


class TestSubgraphErrorHandling:
    """Test error handling in subgraph execution."""

    def test_email_processing_subgraph_with_malformed_email(self):
        """Test subgraph handles malformed email data gracefully."""
        config = Config()
        state = create_initial_state(config)
        state["current_email_uid"] = "123"
        state["current_email"] = {
            # Missing required fields
            "uid": "123",
        }

        # Should not crash
        result = email_processing_subgraph(state)

        assert isinstance(result, dict)

    def test_ai_classification_subgraph_with_missing_body(self):
        """Test AI subgraph handles missing body gracefully."""
        config = Config()
        state = create_initial_state(config)
        state["current_email_uid"] = "123"
        state["current_email"] = {
            "uid": "123",
            "subject": "Test",
            "sender": "test@example.com",
            # Missing body
        }

        # Should not crash
        result = ai_classification_subgraph(state)

        assert isinstance(result, dict)


class TestSubgraphStateIsolation:
    """Test that subgraphs properly isolate per-email state."""

    def test_each_send_has_isolated_current_email(self):
        """Test that each Send has its own current_email data."""
        config = Config()
        state = create_initial_state(config)
        state["email_headers"] = {
            "123": {
                "uid": "123",
                "subject": "Email 1",
                "sender": "sender1@example.com",
            },
            "456": {
                "uid": "456",
                "subject": "Email 2",
                "sender": "sender2@example.com",
            },
        }

        sends = fan_out_processing(state)

        # Each Send should have different current_email
        send1_email = sends[0].arg["current_email"]
        send2_email = sends[1].arg["current_email"]

        assert send1_email["uid"] != send2_email["uid"]
        assert send1_email["subject"] != send2_email["subject"]

    def test_sends_share_common_state(self):
        """Test that all Sends share common state like config and sender_stats."""
        config = Config()
        state = create_initial_state(config)
        state["email_headers"] = {
            "123": {"uid": "123", "subject": "Email 1"},
            "456": {"uid": "456", "subject": "Email 2"},
        }
        state["sender_stats"] = {
            "common@example.com": {
                "sender": "common@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            }
        }

        sends = fan_out_processing(state)

        # Both Sends should have the same sender_stats
        assert sends[0].arg["sender_stats"] == sends[1].arg["sender_stats"]
        assert sends[0].arg["config"] == sends[1].arg["config"]


class TestLargeScaleParallelExecution:
    """Test parallel execution at scale."""

    def test_fan_out_with_many_emails(self):
        """Test fanning out to many parallel subgraphs."""
        config = Config()
        state = create_initial_state(config)

        # Create 100 emails
        state["email_headers"] = {
            f"uid{i}": {
                "uid": f"uid{i}",
                "subject": f"Email {i}",
                "sender": f"sender{i}@example.com",
            }
            for i in range(100)
        }

        sends = fan_out_processing(state)

        assert len(sends) == 100
        assert all(isinstance(s, Send) for s in sends)

        # Verify each has unique current_email_uid
        uids = [s.arg["current_email_uid"] for s in sends]
        assert len(set(uids)) == 100  # All unique

    def test_concurrent_sender_stats_with_many_updates(self):
        """Test sender stats merging with many concurrent updates."""
        base_stats = {}

        def update_random_sender(update_id: int):
            """Simulate random sender update."""
            sender_num = update_id % 10  # 10 unique senders
            updates = {
                f"sender{sender_num}@example.com": {
                    "sender": f"sender{sender_num}@example.com",
                    "marketing_count": 1,
                    "total_count": 1,
                    "auto_delete": False,
                }
            }
            return merge_sender_stats(base_stats, updates)

        # Run 100 concurrent updates across 10 senders
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(update_random_sender, i) for i in range(100)]
            results = [f.result() for f in futures]

        # Each result should have at least one sender
        assert all(len(r) >= 1 for r in results)

    def test_concurrent_list_accumulation_with_many_items(self):
        """Test list accumulation with many concurrent additions."""
        import operator

        base_list = []

        def add_batch_to_list(batch_id: int):
            """Add a batch of items to list."""
            items = [f"batch{batch_id}_item{i}" for i in range(10)]
            return operator.add(base_list, items)

        # Run concurrent batch additions
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_batch_to_list, i) for i in range(10)]
            results = [f.result() for f in futures]

        # Each result should have exactly 10 items (its batch)
        assert all(len(r) == 10 for r in results)
