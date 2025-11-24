"""Checkpoint management for progress tracking and resume capability.

This module provides the CheckpointManager class that integrates with LangGraph's
SqliteSaver to enable resumable email processing with progress tracking and
watermarking.
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from langgraph.checkpoint.sqlite import SqliteSaver

from .langgraph_state import GraphState


class CheckpointManager:
    """Manages checkpointing and progress tracking for email processing.

    This class provides:
    - UID watermarking for resumable processing
    - Progress tracking with statistics
    - Integration with LangGraph's SqliteSaver
    - Progress display with ETA calculations
    """

    def __init__(self, checkpoint_path: str = "checkpoints.db"):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_path: Path to the SQLite checkpoint database
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpointer = SqliteSaver.from_conn_string(str(self.checkpoint_path))

        # Progress tracking
        self._start_time: float | None = None
        self._last_update_time: float | None = None
        self._processing_rate: float = 0.0  # emails per second

        # Create watermark table if it doesn't exist
        self._init_watermark_table()

    def _init_watermark_table(self) -> None:
        """Initialize the watermark table for UID tracking."""
        conn = sqlite3.connect(self.checkpoint_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS watermarks (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_processed_uid TEXT,
                    last_update_time TEXT,
                    total_processed INTEGER DEFAULT 0,
                    total_deleted INTEGER DEFAULT 0,
                    total_kept INTEGER DEFAULT 0,
                    total_errors INTEGER DEFAULT 0
                )
                """
            )
            # Initialize with default row if empty
            cursor.execute(
                """
                INSERT OR IGNORE INTO watermarks (id, last_processed_uid, last_update_time)
                VALUES (1, NULL, NULL)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_last_processed_uid(self) -> str | None:
        """Get the last successfully processed UID watermark.

        Returns:
            The last processed UID, or None if no checkpoint exists
        """
        conn = sqlite3.connect(self.checkpoint_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_processed_uid FROM watermarks WHERE id = 1"
            )
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    def update_watermark(self, uid: str) -> None:
        """Update the progress watermark with the last processed UID.

        Args:
            uid: The UID of the last successfully processed email
        """
        conn = sqlite3.connect(self.checkpoint_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE watermarks
                SET last_processed_uid = ?,
                    last_update_time = ?
                WHERE id = 1
                """,
                (uid, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def update_stats(
        self,
        total_processed: int,
        total_deleted: int,
        total_kept: int,
        total_errors: int,
    ) -> None:
        """Update progress statistics in the watermark table.

        Args:
            total_processed: Total number of emails processed
            total_deleted: Total number of emails deleted
            total_kept: Total number of emails kept
            total_errors: Total number of errors encountered
        """
        conn = sqlite3.connect(self.checkpoint_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE watermarks
                SET total_processed = ?,
                    total_deleted = ?,
                    total_kept = ?,
                    total_errors = ?
                WHERE id = 1
                """,
                (total_processed, total_deleted, total_kept, total_errors),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_checkpoint(self) -> None:
        """Clear checkpoint data on successful completion.

        This removes the watermark and resets statistics, but preserves
        the LangGraph checkpoint history for debugging.
        """
        conn = sqlite3.connect(self.checkpoint_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE watermarks
                SET last_processed_uid = NULL,
                    last_update_time = NULL,
                    total_processed = 0,
                    total_deleted = 0,
                    total_kept = 0,
                    total_errors = 0
                WHERE id = 1
                """
            )
            conn.commit()
            click.echo("✓ Checkpoint cleared successfully")
        finally:
            conn.close()

    def get_progress_stats(self) -> dict[str, Any]:
        """Get current progress statistics from the checkpoint.

        Returns:
            Dictionary containing:
            - last_processed_uid: Last processed UID or None
            - last_update_time: ISO timestamp of last update or None
            - total_processed: Total emails processed
            - total_deleted: Total emails deleted
            - total_kept: Total emails kept
            - total_errors: Total errors encountered
        """
        conn = sqlite3.connect(self.checkpoint_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT last_processed_uid, last_update_time,
                       total_processed, total_deleted, total_kept, total_errors
                FROM watermarks WHERE id = 1
                """
            )
            result = cursor.fetchone()

            if not result:
                return {
                    "last_processed_uid": None,
                    "last_update_time": None,
                    "total_processed": 0,
                    "total_deleted": 0,
                    "total_kept": 0,
                    "total_errors": 0,
                }

            return {
                "last_processed_uid": result[0],
                "last_update_time": result[1],
                "total_processed": result[2] or 0,
                "total_deleted": result[3] or 0,
                "total_kept": result[4] or 0,
                "total_errors": result[5] or 0,
            }
        finally:
            conn.close()

    def display_progress(
        self,
        current: int,
        total: int,
        decisions: dict[str, int],
        parallel_count: int = 0,
    ) -> None:
        """Display progress bar and statistics during processing.

        Args:
            current: Current number of emails processed
            total: Total number of emails to process
            decisions: Dictionary with counts for each decision type
                      (e.g., {"delete": 10, "keep": 5, "uncertain": 2})
            parallel_count: Number of parallel subgraph executions active
        """
        # Initialize start time on first call
        if self._start_time is None:
            self._start_time = time.time()
            self._last_update_time = self._start_time

        # Calculate progress
        percent = (current / total * 100) if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        # Calculate processing rate and ETA
        current_time = time.time()
        elapsed = current_time - self._start_time

        if elapsed > 0 and current > 0:
            self._processing_rate = current / elapsed
            remaining = total - current
            eta_seconds = remaining / self._processing_rate if self._processing_rate > 0 else 0
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        # Format elapsed time
        elapsed_str = self._format_time(elapsed)

        # Build progress line
        progress_line = (
            f"\r{bar} {current}/{total} ({percent:.1f}%) | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str}"
        )

        # Add processing rate
        rate_str = f" | Rate: {self._processing_rate:.1f} emails/s"
        progress_line += rate_str

        # Display decisions breakdown
        decisions_str = " | ".join(
            f"{decision}: {count}" for decision, count in decisions.items()
        )
        if decisions_str:
            progress_line += f" | {decisions_str}"

        # Add parallel execution info
        if parallel_count > 0:
            progress_line += f" | Parallel: {parallel_count}"

        # Output progress (overwrite line)
        click.echo(progress_line, nl=False)

        # Update last update time
        self._last_update_time = current_time

        # Add newline when complete
        if current >= total:
            click.echo()  # New line after completion

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (e.g., "1h 23m 45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def display_summary(self, state: GraphState) -> None:
        """Display final processing summary.

        Args:
            state: Final GraphState with all processing results
        """
        click.echo("\n" + "=" * 60)
        click.echo("PROCESSING SUMMARY")
        click.echo("=" * 60)

        # Overall statistics
        click.echo(f"Total Processed: {state['total_processed']}")
        click.echo(f"Total Deleted:   {state['total_deleted']}")
        click.echo(f"Total Kept:      {state['total_kept']}")
        click.echo(f"Errors:          {len(state['errors'])}")

        # Decisions breakdown
        decisions = state["decisions"]
        if decisions:
            click.echo("\nDecisions Breakdown:")
            delete_count = sum(1 for d in decisions if d["decision"] == "delete")
            keep_count = sum(1 for d in decisions if d["decision"] == "keep")
            uncertain_count = sum(
                1 for d in decisions if d["decision"] == "uncertain"
            )

            click.echo(f"  Delete:    {delete_count}")
            click.echo(f"  Keep:      {keep_count}")
            click.echo(f"  Uncertain: {uncertain_count}")

        # Processing time
        if self._start_time:
            elapsed = time.time() - self._start_time
            click.echo(f"\nTotal Time: {self._format_time(elapsed)}")
            if state["total_processed"] > 0:
                avg_rate = state["total_processed"] / elapsed
                click.echo(f"Average Rate: {avg_rate:.2f} emails/s")

        # Sender statistics
        sender_stats = state["sender_stats"]
        if sender_stats:
            click.echo(f"\nTracked Senders: {len(sender_stats)}")
            auto_delete_senders = [
                s for s in sender_stats.values() if s.get("auto_delete", False)
            ]
            if auto_delete_senders:
                click.echo(f"Auto-delete Senders: {len(auto_delete_senders)}")

        # Errors
        errors = state["errors"]
        if errors:
            click.echo("\nErrors Encountered:")
            for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
                error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                click.echo(f"  {i}. {error_msg}")
            if len(errors) > 5:
                click.echo(f"  ... and {len(errors) - 5} more")

        click.echo("=" * 60)

    def verify_checkpoint_integrity(self) -> tuple[bool, str]:
        """Verify the integrity of the checkpoint database.

        Returns:
            Tuple of (is_valid, message)
            - is_valid: True if checkpoint is valid, False otherwise
            - message: Description of validation result
        """
        if not self.checkpoint_path.exists():
            return False, "Checkpoint database does not exist"

        try:
            conn = sqlite3.connect(self.checkpoint_path)
            cursor = conn.cursor()

            # Check if watermarks table exists
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='watermarks'
                """
            )
            if not cursor.fetchone():
                conn.close()
                return False, "Watermarks table missing"

            # Check if LangGraph checkpoint tables exist
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='checkpoints'
                """
            )
            has_checkpoints = cursor.fetchone() is not None

            # Get watermark stats
            cursor.execute("SELECT * FROM watermarks WHERE id = 1")
            watermark = cursor.fetchone()

            conn.close()

            if not watermark:
                return False, "Watermark record missing"

            if has_checkpoints:
                return True, "Checkpoint integrity verified (with LangGraph checkpoints)"
            else:
                return True, "Checkpoint integrity verified (watermark only)"

        except sqlite3.Error as e:
            return False, f"Database error: {e}"

    def get_resume_info(self) -> dict[str, Any]:
        """Get information needed to resume processing.

        Returns:
            Dictionary containing:
            - can_resume: Whether resuming is possible
            - last_processed_uid: Last UID processed
            - stats: Current progress statistics
            - message: Human-readable status message
        """
        is_valid, integrity_msg = self.verify_checkpoint_integrity()

        if not is_valid:
            return {
                "can_resume": False,
                "last_processed_uid": None,
                "stats": {},
                "message": f"Cannot resume: {integrity_msg}",
            }

        stats = self.get_progress_stats()
        last_uid = stats["last_processed_uid"]

        if last_uid is None:
            return {
                "can_resume": False,
                "last_processed_uid": None,
                "stats": stats,
                "message": "No previous processing session found",
            }

        return {
            "can_resume": True,
            "last_processed_uid": last_uid,
            "stats": stats,
            "message": f"Can resume from UID {last_uid} "
            f"({stats['total_processed']} emails processed)",
        }

    def display_resume_prompt(self) -> bool:
        """Display resume information and prompt user to continue.

        Returns:
            True if user wants to resume, False to start fresh
        """
        resume_info = self.get_resume_info()

        if not resume_info["can_resume"]:
            click.echo(f"Starting fresh: {resume_info['message']}")
            return False

        stats = resume_info["stats"]
        click.echo("\n" + "=" * 60)
        click.echo("RESUMABLE SESSION DETECTED")
        click.echo("=" * 60)
        click.echo(f"Last processed UID: {resume_info['last_processed_uid']}")
        click.echo(f"Last update: {stats['last_update_time']}")
        click.echo(f"Progress:")
        click.echo(f"  - Processed: {stats['total_processed']} emails")
        click.echo(f"  - Deleted:   {stats['total_deleted']} emails")
        click.echo(f"  - Kept:      {stats['total_kept']} emails")
        click.echo(f"  - Errors:    {stats['total_errors']}")
        click.echo("=" * 60)

        return click.confirm("\nDo you want to resume from the last checkpoint?", default=True)
