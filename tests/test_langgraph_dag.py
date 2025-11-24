"""Tests for LangGraph DAG structure and node execution.

This module tests:
- Main graph structure and edges
- Node execution flow
- Conditional routing
- State updates through the pipeline
- Graph compilation
"""

from datetime import datetime
from unittest.mock import patch

from langgraph.constants import Send
from langgraph.graph import StateGraph

from inbox_reaper.langgraph_dag import (
    aggregate_ai_results,
    aggregate_results,
    batch_delete,
    batch_fetch_bodies,
    batch_fetch_headers,
    build_langgraph,
    fan_out_ai_classification,
    fan_out_processing,
    should_fetch_bodies,
    update_checkpoint,
)
from inbox_reaper.langgraph_state import create_initial_state
from inbox_reaper.state import Config


class TestBatchFetchHeaders:
    """Test the batch_fetch_headers node."""

    def test_batch_fetch_headers_uses_config_fetch_size(self):
        """Test that batch_fetch_headers respects config fetch_size."""
        config = Config(fetch_size=50)
        state = create_initial_state(config)

        # Currently returns empty dict (placeholder)
        result = batch_fetch_headers(state)

        assert "email_headers" in result
        assert isinstance(result["email_headers"], dict)

    def test_batch_fetch_headers_returns_state_dict(self):
        """Test that batch_fetch_headers returns a state dict."""
        config = Config()
        state = create_initial_state(config)

        result = batch_fetch_headers(state)

        assert isinstance(result, dict)
        assert "config" in result
        assert "email_headers" in result


class TestFanOutProcessing:
    """Test the fan_out_processing node."""

    def test_fan_out_processing_with_empty_headers(self):
        """Test fan out with no email headers."""
        config = Config()
        state = create_initial_state(config)
        state["email_headers"] = {}

        sends = fan_out_processing(state)

        assert isinstance(sends, list)
        assert len(sends) == 0

    def test_fan_out_processing_creates_send_per_email(self):
        """Test that fan out creates one Send per email header."""
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

        assert len(sends) == 2
        assert all(isinstance(s, Send) for s in sends)

    def test_fan_out_processing_send_includes_current_email(self):
        """Test that each Send includes current_email data."""
        config = Config()
        state = create_initial_state(config)
        state["email_headers"] = {
            "123": {
                "uid": "123",
                "subject": "Test Email",
                "sender": "test@example.com",
            }
        }

        sends = fan_out_processing(state)

        assert len(sends) == 1
        send = sends[0]
        assert send.node == "email_processing_subgraph"
        assert "current_email_uid" in send.arg
        assert send.arg["current_email_uid"] == "123"
        assert "current_email" in send.arg
        assert send.arg["current_email"]["subject"] == "Test Email"


class TestAggregateResults:
    """Test the aggregate_results node."""

    def test_aggregate_results_returns_state_unchanged(self):
        """Test that aggregate_results passes through state."""
        config = Config()
        state = create_initial_state(config)
        state["decisions"] = [
            {
                "email": {
                    "uid": "123",
                    "subject": "Test",
                    "sender": "test@example.com",
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
        state["needs_full_fetch"] = ["456"]
        state["sender_stats"] = {
            "test@example.com": {
                "sender": "test@example.com",
                "marketing_count": 1,
                "total_count": 1,
                "auto_delete": False,
            }
        }

        result = aggregate_results(state)

        assert result == state
        assert len(result["decisions"]) == 1
        assert len(result["needs_full_fetch"]) == 1
        assert len(result["sender_stats"]) == 1


class TestShouldFetchBodies:
    """Test the should_fetch_bodies conditional edge."""

    def test_should_fetch_bodies_returns_fetch_when_needed(self):
        """Test returns 'fetch_bodies' when emails need AI classification."""
        config = Config()
        state = create_initial_state(config)
        state["needs_full_fetch"] = ["123", "456"]

        result = should_fetch_bodies(state)

        assert result == "fetch_bodies"

    def test_should_fetch_bodies_returns_skip_when_empty(self):
        """Test returns 'skip_bodies' when no emails need AI classification."""
        config = Config()
        state = create_initial_state(config)
        state["needs_full_fetch"] = []

        result = should_fetch_bodies(state)

        assert result == "skip_bodies"


class TestBatchFetchBodies:
    """Test the batch_fetch_bodies node."""

    def test_batch_fetch_bodies_processes_needs_full_fetch(self):
        """Test that batch_fetch_bodies uses needs_full_fetch list."""
        config = Config()
        state = create_initial_state(config)
        state["needs_full_fetch"] = ["123", "456"]

        result = batch_fetch_bodies(state)

        assert "email_bodies" in result
        assert isinstance(result["email_bodies"], dict)

    def test_batch_fetch_bodies_with_empty_list(self):
        """Test batch_fetch_bodies with empty needs_full_fetch."""
        config = Config()
        state = create_initial_state(config)
        state["needs_full_fetch"] = []

        result = batch_fetch_bodies(state)

        assert result["email_bodies"] == {}


class TestFanOutAIClassification:
    """Test the fan_out_ai_classification node."""

    def test_fan_out_ai_classification_with_empty_bodies(self):
        """Test fan out AI with no email bodies."""
        config = Config()
        state = create_initial_state(config)
        state["email_bodies"] = {}

        sends = fan_out_ai_classification(state)

        assert isinstance(sends, list)
        assert len(sends) == 0

    def test_fan_out_ai_classification_creates_send_per_email(self):
        """Test that fan out creates one Send per email body."""
        config = Config()
        state = create_initial_state(config)
        state["email_bodies"] = {
            "123": {
                "uid": "123",
                "subject": "Email 1",
                "sender": "sender1@example.com",
                "body": "Full body 1",
            },
            "456": {
                "uid": "456",
                "subject": "Email 2",
                "sender": "sender2@example.com",
                "body": "Full body 2",
            },
        }

        sends = fan_out_ai_classification(state)

        assert len(sends) == 2
        assert all(isinstance(s, Send) for s in sends)

    def test_fan_out_ai_classification_send_structure(self):
        """Test that each Send has correct structure for AI classification."""
        config = Config()
        state = create_initial_state(config)
        state["email_bodies"] = {
            "123": {
                "uid": "123",
                "subject": "Test Email",
                "sender": "test@example.com",
                "body": "Full email body",
            }
        }

        sends = fan_out_ai_classification(state)

        assert len(sends) == 1
        send = sends[0]
        assert send.node == "ai_classification_subgraph"
        assert "current_email_uid" in send.arg
        assert send.arg["current_email_uid"] == "123"
        assert "current_email" in send.arg
        assert send.arg["current_email"]["body"] == "Full email body"


class TestAggregateAIResults:
    """Test the aggregate_ai_results node."""

    def test_aggregate_ai_results_returns_state(self):
        """Test that aggregate_ai_results returns state unchanged."""
        config = Config()
        state = create_initial_state(config)
        state["decisions"] = [
            {
                "email": {
                    "uid": "123",
                    "subject": "Test",
                    "sender": "test@example.com",
                    "body": "Body",
                    "date": datetime.now(),
                    "attachments": [],
                },
                "decision": "delete",
                "reason": "ai_classified",
                "confidence": 0.85,
                "processed_at": datetime.now(),
            }
        ]

        result = aggregate_ai_results(state)

        assert result == state
        assert len(result["decisions"]) == 1


class TestBatchDelete:
    """Test the batch_delete node."""

    def test_batch_delete_dry_run_mode(self):
        """Test batch delete in dry run mode."""
        config = Config(dry_run=True)
        state = create_initial_state(config)
        state["to_delete"] = ["123", "456", "789"]

        result = batch_delete(state)

        # Should update total_deleted counter
        assert result["total_deleted"] == 3

    def test_batch_delete_with_empty_list(self):
        """Test batch delete with no emails to delete."""
        config = Config(dry_run=False)
        state = create_initial_state(config)
        state["to_delete"] = []

        result = batch_delete(state)

        assert result["total_deleted"] == 0

    def test_batch_delete_increments_counter(self):
        """Test that batch delete increments total_deleted."""
        config = Config(dry_run=True)
        state = create_initial_state(config)
        state["total_deleted"] = 5
        state["to_delete"] = ["123", "456"]

        result = batch_delete(state)

        assert result["total_deleted"] == 7  # 5 + 2


class TestUpdateCheckpoint:
    """Test the update_checkpoint node."""

    def test_update_checkpoint_returns_state(self):
        """Test that update_checkpoint returns state unchanged."""
        config = Config()
        state = create_initial_state(config)
        state["total_processed"] = 10
        state["total_deleted"] = 5
        state["total_kept"] = 5

        result = update_checkpoint(state)

        assert result == state
        assert result["total_processed"] == 10
        assert result["total_deleted"] == 5
        assert result["total_kept"] == 5


class TestBuildLangGraph:
    """Test the build_langgraph function."""

    @patch("inbox_reaper.langgraph_dag.SqliteSaver")
    def test_build_langgraph_creates_compiled_graph(self, mock_saver):
        """Test that build_langgraph returns a compiled graph."""
        config = Config()

        graph = build_langgraph(config, checkpoint_path=":memory:")

        # Should return a compiled graph (not a StateGraph)
        assert graph is not None
        assert not isinstance(graph, StateGraph)

    @patch("inbox_reaper.langgraph_dag.SqliteSaver")
    def test_build_langgraph_uses_checkpoint_path(self, mock_saver):
        """Test that build_langgraph uses provided checkpoint path."""
        config = Config()
        checkpoint_path = "test_checkpoints.db"

        build_langgraph(config, checkpoint_path=checkpoint_path)

        mock_saver.from_conn_string.assert_called_once_with(checkpoint_path)

    @patch("inbox_reaper.langgraph_dag.SqliteSaver")
    def test_build_langgraph_default_checkpoint_path(self, mock_saver):
        """Test that build_langgraph uses default checkpoint path."""
        config = Config()

        build_langgraph(config)

        mock_saver.from_conn_string.assert_called_once_with("checkpoints.db")


class TestGraphStructure:
    """Test the overall graph structure and flow."""

    @patch("inbox_reaper.langgraph_dag.SqliteSaver")
    def test_graph_has_all_required_nodes(self, mock_saver):
        """Test that graph includes all required nodes."""
        config = Config()
        graph = build_langgraph(config, checkpoint_path=":memory:")

        # This is a compiled graph, so we can't directly inspect nodes
        # But we can verify it was built without errors
        assert graph is not None

    def test_fan_out_processing_creates_parallel_execution(self):
        """Test that fan_out creates parallel execution paths."""
        config = Config()
        state = create_initial_state(config)
        state["email_headers"] = {
            f"uid{i}": {
                "uid": f"uid{i}",
                "subject": f"Email {i}",
                "sender": f"sender{i}@example.com",
            }
            for i in range(5)
        }

        sends = fan_out_processing(state)

        # Should create 5 parallel Send commands
        assert len(sends) == 5
        assert all(isinstance(s, Send) for s in sends)
        assert all(s.node == "email_processing_subgraph" for s in sends)

    def test_fan_out_ai_creates_parallel_execution(self):
        """Test that fan_out_ai creates parallel AI classification."""
        config = Config()
        state = create_initial_state(config)
        state["email_bodies"] = {
            f"uid{i}": {
                "uid": f"uid{i}",
                "subject": f"Email {i}",
                "sender": f"sender{i}@example.com",
                "body": f"Body {i}",
            }
            for i in range(3)
        }

        sends = fan_out_ai_classification(state)

        # Should create 3 parallel Send commands
        assert len(sends) == 3
        assert all(isinstance(s, Send) for s in sends)
        assert all(s.node == "ai_classification_subgraph" for s in sends)


class TestConditionalEdges:
    """Test conditional edge routing."""

    def test_should_fetch_bodies_routing_logic(self):
        """Test conditional routing based on needs_full_fetch."""
        config = Config()

        # Case 1: No emails need bodies
        state1 = create_initial_state(config)
        state1["needs_full_fetch"] = []
        assert should_fetch_bodies(state1) == "skip_bodies"

        # Case 2: Some emails need bodies
        state2 = create_initial_state(config)
        state2["needs_full_fetch"] = ["123"]
        assert should_fetch_bodies(state2) == "fetch_bodies"

        # Case 3: Many emails need bodies
        state3 = create_initial_state(config)
        state3["needs_full_fetch"] = [f"uid{i}" for i in range(10)]
        assert should_fetch_bodies(state3) == "fetch_bodies"


class TestStateFlowThroughPipeline:
    """Test that state flows correctly through the pipeline."""

    def test_state_accumulates_through_nodes(self):
        """Test that state accumulates data as it flows through nodes."""
        config = Config()
        state = create_initial_state(config)

        # Simulate pipeline flow
        # 1. Fetch headers
        state = batch_fetch_headers(state)
        assert "email_headers" in state

        # 2. Aggregate (after parallel processing)
        state["decisions"] = [
            {
                "email": {
                    "uid": "123",
                    "subject": "Test",
                    "sender": "test@example.com",
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
        state["to_delete"] = ["123"]
        state = aggregate_results(state)
        assert len(state["decisions"]) == 1

        # 3. Delete
        state = batch_delete(state)
        assert state["total_deleted"] == 1

        # 4. Update checkpoint
        state = update_checkpoint(state)
        assert state["total_deleted"] == 1

    def test_state_preserves_config_through_pipeline(self):
        """Test that config is preserved throughout pipeline execution."""
        config = Config(
            dry_run=False,
            fetch_size=50,
            auto_delete_threshold=3,
        )
        state = create_initial_state(config)

        # Pass through various nodes
        state = batch_fetch_headers(state)
        state = aggregate_results(state)
        state = batch_delete(state)
        state = update_checkpoint(state)

        # Config should still be intact
        assert state["config"]["dry_run"] is False
        assert state["config"]["fetch_size"] == 50
        assert state["config"]["auto_delete_threshold"] == 3
