"""Tests for checkpoint manager and progress tracking.

This module tests:
- Watermark persistence and retrieval
- Progress statistics tracking
- Resume capability
- Checkpoint integrity validation
- Progress display
"""

import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from inbox_reaper.checkpoint_manager import CheckpointManager
from inbox_reaper.langgraph_state import create_initial_state
from inbox_reaper.state import Config


@pytest.fixture
def temp_checkpoint_db():
    """Create a temporary checkpoint database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def checkpoint_manager(temp_checkpoint_db):
    """Create a CheckpointManager instance with temporary database."""
    return CheckpointManager(temp_checkpoint_db)


class TestCheckpointInitialization:
    """Test checkpoint manager initialization."""

    def test_init_creates_database_file(self, temp_checkpoint_db):
        """Test that initialization creates the database file."""
        CheckpointManager(temp_checkpoint_db)

        assert Path(temp_checkpoint_db).exists()

    def test_init_creates_watermark_table(self, temp_checkpoint_db):
        """Test that initialization creates watermark table."""
        CheckpointManager(temp_checkpoint_db)

        conn = sqlite3.connect(temp_checkpoint_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='watermarks'
            """
        )
        result = cursor.fetchone()
        assert result is not None

        conn.close()

    def test_init_creates_default_watermark_row(self, temp_checkpoint_db):
        """Test that initialization creates default watermark row."""
        CheckpointManager(temp_checkpoint_db)

        conn = sqlite3.connect(temp_checkpoint_db)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM watermarks WHERE id = 1")
        result = cursor.fetchone()
        assert result is not None

        conn.close()

    @patch("inbox_reaper.checkpoint_manager.SqliteSaver")
    def test_init_creates_checkpointer(self, mock_saver, temp_checkpoint_db):
        """Test that initialization creates SqliteSaver checkpointer."""
        CheckpointManager(temp_checkpoint_db)

        mock_saver.from_conn_string.assert_called_once()


class TestWatermarkOperations:
    """Test watermark persistence and retrieval."""

    def test_get_last_processed_uid_initially_none(self, checkpoint_manager):
        """Test that initial watermark is None."""
        last_uid = checkpoint_manager.get_last_processed_uid()

        assert last_uid is None

    def test_update_watermark(self, checkpoint_manager):
        """Test updating watermark with UID."""
        checkpoint_manager.update_watermark("12345")

        last_uid = checkpoint_manager.get_last_processed_uid()
        assert last_uid == "12345"

    def test_update_watermark_multiple_times(self, checkpoint_manager):
        """Test that watermark can be updated multiple times."""
        checkpoint_manager.update_watermark("12345")
        checkpoint_manager.update_watermark("67890")
        checkpoint_manager.update_watermark("11111")

        last_uid = checkpoint_manager.get_last_processed_uid()
        assert last_uid == "11111"

    def test_update_watermark_sets_timestamp(
        self, checkpoint_manager, temp_checkpoint_db
    ):
        """Test that updating watermark sets last_update_time."""
        checkpoint_manager.update_watermark("12345")

        conn = sqlite3.connect(temp_checkpoint_db)
        cursor = conn.cursor()

        cursor.execute("SELECT last_update_time FROM watermarks WHERE id = 1")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] is not None

        conn.close()


class TestProgressStats:
    """Test progress statistics tracking."""

    def test_get_progress_stats_initial_state(self, checkpoint_manager):
        """Test that initial progress stats are zero."""
        stats = checkpoint_manager.get_progress_stats()

        assert stats["last_processed_uid"] is None
        assert stats["last_update_time"] is None
        assert stats["total_processed"] == 0
        assert stats["total_deleted"] == 0
        assert stats["total_kept"] == 0
        assert stats["total_errors"] == 0

    def test_update_stats(self, checkpoint_manager):
        """Test updating progress statistics."""
        checkpoint_manager.update_stats(
            total_processed=100,
            total_deleted=60,
            total_kept=35,
            total_errors=5,
        )

        stats = checkpoint_manager.get_progress_stats()
        assert stats["total_processed"] == 100
        assert stats["total_deleted"] == 60
        assert stats["total_kept"] == 35
        assert stats["total_errors"] == 5

    def test_update_stats_multiple_times(self, checkpoint_manager):
        """Test that stats can be updated multiple times."""
        checkpoint_manager.update_stats(50, 30, 20, 0)
        checkpoint_manager.update_stats(100, 60, 40, 0)
        checkpoint_manager.update_stats(150, 90, 55, 5)

        stats = checkpoint_manager.get_progress_stats()
        assert stats["total_processed"] == 150
        assert stats["total_deleted"] == 90
        assert stats["total_kept"] == 55
        assert stats["total_errors"] == 5


class TestClearCheckpoint:
    """Test checkpoint clearing functionality."""

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_clear_checkpoint_resets_watermark(self, mock_echo, checkpoint_manager):
        """Test that clearing checkpoint resets watermark."""
        checkpoint_manager.update_watermark("12345")
        checkpoint_manager.update_stats(100, 60, 40, 0)

        checkpoint_manager.clear_checkpoint()

        stats = checkpoint_manager.get_progress_stats()
        assert stats["last_processed_uid"] is None
        assert stats["total_processed"] == 0
        assert stats["total_deleted"] == 0
        assert stats["total_kept"] == 0

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_clear_checkpoint_displays_message(self, mock_echo, checkpoint_manager):
        """Test that clearing checkpoint displays success message."""
        checkpoint_manager.clear_checkpoint()

        mock_echo.assert_called_once()
        assert "Checkpoint cleared successfully" in str(mock_echo.call_args)


class TestCheckpointIntegrity:
    """Test checkpoint integrity validation."""

    def test_verify_checkpoint_integrity_valid(self, checkpoint_manager):
        """Test verifying integrity of valid checkpoint."""
        is_valid, message = checkpoint_manager.verify_checkpoint_integrity()

        assert is_valid is True
        assert "verified" in message.lower()

    def test_verify_checkpoint_integrity_nonexistent_file(self):
        """Test verifying integrity when database doesn't exist."""
        CheckpointManager("nonexistent.db")

        is_valid, message = checkpoint_manager.verify_checkpoint_integrity()

        assert is_valid is False
        assert "does not exist" in message.lower()

    def test_verify_checkpoint_integrity_with_data(self, checkpoint_manager):
        """Test verifying integrity with populated checkpoint."""
        checkpoint_manager.update_watermark("12345")
        checkpoint_manager.update_stats(100, 60, 40, 0)

        is_valid, message = checkpoint_manager.verify_checkpoint_integrity()

        assert is_valid is True


class TestResumeCapability:
    """Test resume capability and information."""

    def test_get_resume_info_no_checkpoint(self, checkpoint_manager):
        """Test getting resume info when no checkpoint exists."""
        info = checkpoint_manager.get_resume_info()

        assert info["can_resume"] is False
        assert info["last_processed_uid"] is None
        assert "No previous processing session" in info["message"]

    def test_get_resume_info_with_checkpoint(self, checkpoint_manager):
        """Test getting resume info when checkpoint exists."""
        checkpoint_manager.update_watermark("12345")
        checkpoint_manager.update_stats(100, 60, 40, 0)

        info = checkpoint_manager.get_resume_info()

        assert info["can_resume"] is True
        assert info["last_processed_uid"] == "12345"
        assert "100 emails processed" in info["message"]

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    @patch("inbox_reaper.checkpoint_manager.click.confirm")
    def test_display_resume_prompt_with_checkpoint(
        self, mock_confirm, mock_echo, checkpoint_manager
    ):
        """Test displaying resume prompt when checkpoint exists."""
        checkpoint_manager.update_watermark("12345")
        checkpoint_manager.update_stats(100, 60, 40, 0)

        mock_confirm.return_value = True

        result = checkpoint_manager.display_resume_prompt()

        assert result is True
        assert mock_echo.call_count > 0
        mock_confirm.assert_called_once()

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_resume_prompt_without_checkpoint(
        self, mock_echo, checkpoint_manager
    ):
        """Test displaying resume prompt when no checkpoint exists."""
        result = checkpoint_manager.display_resume_prompt()

        assert result is False
        mock_echo.assert_called_once()
        assert "Starting fresh" in str(mock_echo.call_args)


class TestProgressDisplay:
    """Test progress display functionality."""

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_progress_first_call(self, mock_echo, checkpoint_manager):
        """Test displaying progress on first call."""
        checkpoint_manager.display_progress(
            current=10,
            total=100,
            decisions={"delete": 6, "keep": 4},
            parallel_count=5,
        )

        mock_echo.assert_called_once()
        output = str(mock_echo.call_args[0][0])
        assert "10/100" in output
        assert "delete: 6" in output
        assert "keep: 4" in output

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_progress_calculates_percentage(
        self, mock_echo, checkpoint_manager
    ):
        """Test that progress display calculates percentage."""
        checkpoint_manager.display_progress(
            current=50,
            total=100,
            decisions={},
        )

        output = str(mock_echo.call_args[0][0])
        assert "50.0%" in output or "50%" in output

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_progress_shows_eta(self, mock_echo, checkpoint_manager):
        """Test that progress display shows ETA."""
        # First call to initialize start time
        checkpoint_manager.display_progress(10, 100, {})

        # Wait a bit
        time.sleep(0.1)

        # Second call should show ETA
        checkpoint_manager.display_progress(20, 100, {})

        output = str(mock_echo.call_args[0][0])
        assert "ETA" in output

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_progress_completion(self, mock_echo, checkpoint_manager):
        """Test progress display at completion."""
        checkpoint_manager.display_progress(
            current=100,
            total=100,
            decisions={"delete": 60, "keep": 40},
        )

        # Should call echo twice: once for progress, once for newline
        assert mock_echo.call_count == 2

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_progress_with_parallel_count(self, mock_echo, checkpoint_manager):
        """Test progress display includes parallel execution count."""
        checkpoint_manager.display_progress(
            current=50,
            total=100,
            decisions={},
            parallel_count=10,
        )

        output = str(mock_echo.call_args[0][0])
        assert "Parallel: 10" in output

    def test_format_time_seconds(self, checkpoint_manager):
        """Test time formatting for seconds."""
        formatted = checkpoint_manager._format_time(45)

        assert formatted == "45s"

    def test_format_time_minutes(self, checkpoint_manager):
        """Test time formatting for minutes."""
        formatted = checkpoint_manager._format_time(125)  # 2m 5s

        assert "2m" in formatted
        assert "5s" in formatted

    def test_format_time_hours(self, checkpoint_manager):
        """Test time formatting for hours."""
        formatted = checkpoint_manager._format_time(3725)  # 1h 2m

        assert "1h" in formatted
        assert "2m" in formatted


class TestSummaryDisplay:
    """Test final summary display."""

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_summary(self, mock_echo, checkpoint_manager):
        """Test displaying final processing summary."""
        config = Config()
        state = create_initial_state(config)
        state["total_processed"] = 100
        state["total_deleted"] = 60
        state["total_kept"] = 40
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
        state["sender_stats"] = {
            "test@example.com": {
                "sender": "test@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": True,
            }
        }
        state["errors"] = []

        checkpoint_manager.display_summary(state)

        # Should have multiple echo calls for summary sections
        assert mock_echo.call_count > 5

        # Check that key information is displayed
        calls = [str(call) for call in mock_echo.call_args_list]
        output = " ".join(calls)

        assert "100" in output  # total_processed
        assert "60" in output  # total_deleted
        assert "40" in output  # total_kept

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_summary_with_errors(self, mock_echo, checkpoint_manager):
        """Test displaying summary with errors."""
        config = Config()
        state = create_initial_state(config)
        state["total_processed"] = 10
        state["total_deleted"] = 5
        state["total_kept"] = 5
        state["decisions"] = []
        state["errors"] = [
            {"message": "Error 1"},
            {"message": "Error 2"},
            {"message": "Error 3"},
        ]

        checkpoint_manager.display_summary(state)

        calls = [str(call) for call in mock_echo.call_args_list]
        output = " ".join(calls)

        assert "Error 1" in output
        assert "Error 2" in output
        assert "Error 3" in output

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_summary_with_decisions_breakdown(
        self, mock_echo, checkpoint_manager
    ):
        """Test summary shows decisions breakdown."""
        config = Config()
        state = create_initial_state(config)
        state["total_processed"] = 10
        state["total_deleted"] = 6
        state["total_kept"] = 4
        state["decisions"] = [
            {
                "email": {
                    "uid": f"uid{i}",
                    "subject": "Test",
                    "sender": "test@example.com",
                    "body": "",
                    "date": datetime.now(),
                    "attachments": [],
                },
                "decision": "delete" if i < 6 else "keep",
                "reason": "keyword",
                "confidence": 1.0,
                "processed_at": datetime.now(),
            }
            for i in range(10)
        ]
        state["errors"] = []

        checkpoint_manager.display_summary(state)

        calls = [str(call) for call in mock_echo.call_args_list]
        output = " ".join(calls)

        assert "Delete" in output
        assert "Keep" in output

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_summary_shows_processing_time(self, mock_echo, checkpoint_manager):
        """Test summary shows total processing time."""
        # Initialize start time
        checkpoint_manager._start_time = time.time() - 60  # 1 minute ago

        config = Config()
        state = create_initial_state(config)
        state["total_processed"] = 100
        state["total_deleted"] = 50
        state["total_kept"] = 50
        state["decisions"] = []
        state["errors"] = []

        checkpoint_manager.display_summary(state)

        calls = [str(call) for call in mock_echo.call_args_list]
        output = " ".join(calls)

        assert "Total Time" in output
        assert "Average Rate" in output


class TestCheckpointPersistence:
    """Test checkpoint persistence across sessions."""

    def test_checkpoint_survives_manager_recreation(self, temp_checkpoint_db):
        """Test that checkpoint data persists across manager instances."""
        # First session
        manager1 = CheckpointManager(temp_checkpoint_db)
        manager1.update_watermark("12345")
        manager1.update_stats(100, 60, 40, 0)

        # Second session (new manager instance)
        manager2 = CheckpointManager(temp_checkpoint_db)

        # Data should still be there
        assert manager2.get_last_processed_uid() == "12345"
        stats = manager2.get_progress_stats()
        assert stats["total_processed"] == 100
        assert stats["total_deleted"] == 60

    def test_multiple_managers_same_database(self, temp_checkpoint_db):
        """Test multiple managers can access same database safely."""
        manager1 = CheckpointManager(temp_checkpoint_db)
        manager2 = CheckpointManager(temp_checkpoint_db)

        manager1.update_watermark("12345")
        assert manager2.get_last_processed_uid() == "12345"

        manager2.update_stats(100, 60, 40, 0)
        stats = manager1.get_progress_stats()
        assert stats["total_processed"] == 100


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_progress_with_zero_total(self, checkpoint_manager):
        """Test progress display with zero total."""
        with patch("inbox_reaper.checkpoint_manager.click.echo"):
            # Should not crash
            checkpoint_manager.display_progress(0, 0, {})

    def test_watermark_with_very_long_uid(self, checkpoint_manager):
        """Test watermark with very long UID."""
        long_uid = "x" * 1000
        checkpoint_manager.update_watermark(long_uid)

        assert checkpoint_manager.get_last_processed_uid() == long_uid

    def test_stats_with_negative_values(self, checkpoint_manager):
        """Test handling of negative stat values."""
        # This shouldn't happen in practice, but test robustness
        checkpoint_manager.update_stats(-10, -5, -5, -1)

        stats = checkpoint_manager.get_progress_stats()
        # Values should be stored as-is (no validation in current implementation)
        assert stats["total_processed"] == -10

    @patch("inbox_reaper.checkpoint_manager.click.echo")
    def test_display_summary_with_empty_state(self, mock_echo, checkpoint_manager):
        """Test displaying summary with minimal/empty state."""
        config = Config()
        state = create_initial_state(config)

        # Should not crash
        checkpoint_manager.display_summary(state)

        assert mock_echo.call_count > 0
