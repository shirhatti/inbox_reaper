"""Tests for LangGraph state schema and reducers.

This module tests:
- State schema validation
- Reducer functions (thread-safety)
- State serialization/deserialization
- Concurrent updates to shared state
"""

import operator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from inbox_reaper.langgraph_state import (
    create_initial_state,
    decision_dict_to_model,
    email_dict_to_model,
    merge_email_headers,
    merge_sender_stats,
    sender_stats_dict_to_model,
)
from inbox_reaper.state import Config, Decision, Email, FilterReason


class TestMergeSenderStats:
    """Test the merge_sender_stats reducer for thread-safe parallel updates."""

    def test_merge_empty_existing(self):
        """Test merging into empty existing stats."""
        existing = {}
        updates = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            }
        }

        result = merge_sender_stats(existing, updates)

        assert len(result) == 1
        assert result["sender@example.com"]["marketing_count"] == 5
        assert result["sender@example.com"]["total_count"] == 10
        assert result["sender@example.com"]["auto_delete"] is False

    def test_merge_new_sender(self):
        """Test adding a new sender to existing stats."""
        existing = {
            "existing@example.com": {
                "sender": "existing@example.com",
                "marketing_count": 3,
                "total_count": 5,
                "auto_delete": False,
            }
        }
        updates = {
            "new@example.com": {
                "sender": "new@example.com",
                "marketing_count": 2,
                "total_count": 4,
                "auto_delete": True,
            }
        }

        result = merge_sender_stats(existing, updates)

        assert len(result) == 2
        assert "existing@example.com" in result
        assert "new@example.com" in result
        assert result["new@example.com"]["auto_delete"] is True

    def test_merge_existing_sender_accumulates_counts(self):
        """Test that merging existing sender accumulates counts."""
        existing = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            }
        }
        updates = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 3,
                "total_count": 5,
                "auto_delete": False,
            }
        }

        result = merge_sender_stats(existing, updates)

        assert len(result) == 1
        assert result["sender@example.com"]["marketing_count"] == 8  # 5 + 3
        assert result["sender@example.com"]["total_count"] == 15  # 10 + 5
        assert result["sender@example.com"]["auto_delete"] is False

    def test_merge_auto_delete_flag_is_or_operation(self):
        """Test that auto_delete flag uses OR logic (True if either is True)."""
        # Case 1: existing False, update True -> True
        existing = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 3,
                "total_count": 5,
                "auto_delete": False,
            }
        }
        updates = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 2,
                "total_count": 3,
                "auto_delete": True,
            }
        }

        result = merge_sender_stats(existing, updates)
        assert result["sender@example.com"]["auto_delete"] is True

        # Case 2: existing True, update False -> True
        existing = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 3,
                "total_count": 5,
                "auto_delete": True,
            }
        }
        updates = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 2,
                "total_count": 3,
                "auto_delete": False,
            }
        }

        result = merge_sender_stats(existing, updates)
        assert result["sender@example.com"]["auto_delete"] is True

    def test_merge_multiple_senders(self):
        """Test merging multiple senders in one operation."""
        existing = {
            "sender1@example.com": {
                "sender": "sender1@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            }
        }
        updates = {
            "sender1@example.com": {
                "sender": "sender1@example.com",
                "marketing_count": 2,
                "total_count": 3,
                "auto_delete": True,
            },
            "sender2@example.com": {
                "sender": "sender2@example.com",
                "marketing_count": 1,
                "total_count": 2,
                "auto_delete": False,
            },
        }

        result = merge_sender_stats(existing, updates)

        assert len(result) == 2
        assert result["sender1@example.com"]["marketing_count"] == 7
        assert result["sender1@example.com"]["auto_delete"] is True
        assert result["sender2@example.com"]["marketing_count"] == 1

    def test_merge_does_not_mutate_input(self):
        """Test that merge operations don't mutate input dictionaries."""
        existing = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            }
        }
        updates = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 2,
                "total_count": 3,
                "auto_delete": False,
            }
        }

        # Make deep copies to verify no mutation
        existing_copy = {k: v.copy() for k, v in existing.items()}
        updates_copy = {k: v.copy() for k, v in updates.items()}

        merge_sender_stats(existing, updates)

        # Verify original dicts unchanged
        assert existing == existing_copy
        assert updates == updates_copy

    def test_merge_concurrent_updates(self):
        """Test thread-safety of merge_sender_stats with concurrent updates.

        This simulates parallel subgraph execution updating sender stats.
        """
        # Starting state
        existing = {}

        def update_sender(sender_id: int, count: int):
            """Simulate a parallel subgraph updating sender stats."""
            updates = {
                f"sender{sender_id}@example.com": {
                    "sender": f"sender{sender_id}@example.com",
                    "marketing_count": count,
                    "total_count": count * 2,
                    "auto_delete": False,
                }
            }
            return merge_sender_stats(existing, updates)

        # Simulate 10 parallel updates
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_sender, i, i + 1) for i in range(10)]
            results = [f.result() for f in futures]

        # Each result should have the sender it was updating
        for i, result in enumerate(results):
            sender_key = f"sender{i}@example.com"
            assert sender_key in result
            assert result[sender_key]["marketing_count"] == i + 1


class TestMergeEmailHeaders:
    """Test the merge_email_headers reducer."""

    def test_merge_empty_existing(self):
        """Test merging into empty headers."""
        existing = {}
        updates = {
            "123": {
                "uid": "123",
                "subject": "Test Email",
                "sender": "test@example.com",
            }
        }

        result = merge_email_headers(existing, updates)

        assert len(result) == 1
        assert result["123"]["subject"] == "Test Email"

    def test_merge_new_headers(self):
        """Test adding new headers to existing ones."""
        existing = {
            "123": {
                "uid": "123",
                "subject": "Email 1",
            }
        }
        updates = {
            "456": {
                "uid": "456",
                "subject": "Email 2",
            }
        }

        result = merge_email_headers(existing, updates)

        assert len(result) == 2
        assert "123" in result
        assert "456" in result

    def test_merge_overlapping_keys_updates_values(self):
        """Test that overlapping UIDs get updated with new values."""
        existing = {
            "123": {
                "uid": "123",
                "subject": "Old Subject",
            }
        }
        updates = {
            "123": {
                "uid": "123",
                "subject": "New Subject",
            }
        }

        result = merge_email_headers(existing, updates)

        assert len(result) == 1
        assert result["123"]["subject"] == "New Subject"

    def test_merge_does_not_mutate_input(self):
        """Test that merge doesn't mutate input dicts."""
        existing = {"123": {"uid": "123", "subject": "Test"}}
        updates = {"456": {"uid": "456", "subject": "Test 2"}}

        existing_copy = existing.copy()
        updates_copy = updates.copy()

        merge_email_headers(existing, updates)

        assert existing == existing_copy
        assert updates == updates_copy


class TestGraphState:
    """Test GraphState TypedDict schema."""

    def test_create_initial_state(self):
        """Test creating initial state from config."""
        config = Config(
            dry_run=True,
            fetch_size=50,
            auto_delete_threshold=3,
        )

        state = create_initial_state(config)

        assert isinstance(state, dict)
        assert "config" in state
        assert state["email_headers"] == {}
        assert state["email_bodies"] == {}
        assert state["sender_stats"] == {}
        assert state["decisions"] == []
        assert state["needs_full_fetch"] == []
        assert state["to_delete"] == []
        assert state["processed_uids"] == set()
        assert state["errors"] == []
        assert state["total_processed"] == 0
        assert state["total_deleted"] == 0
        assert state["total_kept"] == 0
        assert state["min_uid"] is None
        assert state["max_uid"] is None
        assert state["consecutive_empty_batches"] == 0
        assert state["imap_server"] is None
        assert state["imap_username"] is None

    def test_config_serialization(self):
        """Test that config is properly serialized in state."""
        config = Config(
            dry_run=False,
            fetch_size=100,
            model_name="test-model",
        )

        state = create_initial_state(config)

        assert state["config"]["dry_run"] is False
        assert state["config"]["fetch_size"] == 100
        assert state["config"]["model_name"] == "test-model"


class TestAnnotatedReducers:
    """Test that Annotated reducers work correctly in state updates."""

    def test_operator_add_reducer_for_lists(self):
        """Test operator.add reducer accumulates list items."""
        existing = ["item1", "item2"]
        updates = ["item3", "item4"]

        result = operator.add(existing, updates)

        assert result == ["item1", "item2", "item3", "item4"]
        assert len(result) == 4

    def test_operator_or_reducer_for_sets(self):
        """Test operator.or_ reducer merges sets."""
        existing = {"uid1", "uid2"}
        updates = {"uid2", "uid3"}

        result = operator.or_(existing, updates)

        assert result == {"uid1", "uid2", "uid3"}
        assert len(result) == 3


class TestStateSerialization:
    """Test serialization and deserialization of state objects."""

    def test_email_dict_to_model(self):
        """Test converting email dict to Email model."""
        email_dict = {
            "uid": "123",
            "subject": "Test Subject",
            "sender": "test@example.com",
            "body": "Test body",
            "date": datetime(2025, 11, 24, 10, 30),
            "attachments": ["file.pdf"],
        }

        email = email_dict_to_model(email_dict)

        assert isinstance(email, Email)
        assert email.uid == "123"
        assert email.subject == "Test Subject"
        assert email.sender == "test@example.com"
        assert email.body == "Test body"
        assert email.attachments == ["file.pdf"]

    def test_decision_dict_to_model(self):
        """Test converting decision dict to EmailDecision model."""
        decision_dict = {
            "email": {
                "uid": "123",
                "subject": "Test",
                "sender": "test@example.com",
                "body": "Body",
                "date": datetime(2025, 11, 24, 10, 30),
                "attachments": [],
            },
            "decision": "delete",
            "reason": "keyword",
            "confidence": 0.95,
            "processed_at": datetime(2025, 11, 24, 10, 35),
        }

        decision = decision_dict_to_model(decision_dict)

        assert decision.email.uid == "123"
        assert decision.decision == Decision.DELETE
        assert decision.reason == FilterReason.KEYWORD
        assert decision.confidence == 0.95

    def test_sender_stats_dict_to_model(self):
        """Test converting sender stats dict to SenderStats model."""
        stats_dict = {
            "sender": "marketing@example.com",
            "marketing_count": 10,
            "total_count": 15,
            "auto_delete": True,
        }

        stats = sender_stats_dict_to_model(stats_dict)

        assert stats.sender == "marketing@example.com"
        assert stats.marketing_count == 10
        assert stats.total_count == 15
        assert stats.auto_delete is True


class TestConcurrentStateUpdates:
    """Test concurrent updates to state using reducers."""

    def test_concurrent_sender_stats_updates(self):
        """Test that concurrent sender stats updates are handled correctly."""
        # Simulate multiple parallel subgraphs updating the same sender
        base_stats = {
            "sender@example.com": {
                "sender": "sender@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            }
        }

        def parallel_update(increment: int):
            """Simulate a parallel subgraph update."""
            updates = {
                "sender@example.com": {
                    "sender": "sender@example.com",
                    "marketing_count": increment,
                    "total_count": increment * 2,
                    "auto_delete": False,
                }
            }
            return merge_sender_stats(base_stats, updates)

        # Run 5 parallel updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(parallel_update, i + 1) for i in range(5)]
            results = [f.result() for f in futures]

        # Each result should correctly merge with base_stats
        for i, result in enumerate(results):
            expected_marketing = 5 + (i + 1)
            expected_total = 10 + ((i + 1) * 2)
            assert result["sender@example.com"]["marketing_count"] == expected_marketing
            assert result["sender@example.com"]["total_count"] == expected_total

    def test_concurrent_decisions_accumulation(self):
        """Test that decisions from parallel subgraphs accumulate correctly."""
        # Simulate parallel subgraphs each adding decisions
        base_decisions = []

        def create_decision(uid: str):
            """Create a decision dict."""
            return {
                "email": {
                    "uid": uid,
                    "subject": f"Email {uid}",
                    "sender": "test@example.com",
                    "body": "Body",
                    "date": datetime.now(),
                    "attachments": [],
                },
                "decision": "delete",
                "reason": "keyword",
                "confidence": 1.0,
                "processed_at": datetime.now(),
            }

        # Using operator.add to accumulate
        decisions = [create_decision(f"uid{i}") for i in range(5)]

        accumulated = base_decisions
        for decision in decisions:
            accumulated = operator.add(accumulated, [decision])

        assert len(accumulated) == 5
        assert all("email" in d for d in accumulated)

    def test_concurrent_uids_set_merge(self):
        """Test that processed UIDs from parallel subgraphs merge correctly."""
        base_uids = {"uid1", "uid2"}

        # Simulate parallel subgraphs processing different UIDs
        def process_uid(uid: str):
            """Simulate processing a UID."""
            return operator.or_(base_uids, {uid})

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_uid, f"uid{i}") for i in range(3, 6)]
            results = [f.result() for f in futures]

        # Each result should have base UIDs plus the new one
        for i, result in enumerate(results):
            expected_uid = f"uid{i + 3}"
            assert expected_uid in result
            assert "uid1" in result
            assert "uid2" in result
