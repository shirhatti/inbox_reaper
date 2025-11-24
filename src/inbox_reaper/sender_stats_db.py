"""SQLite persistence layer for sender statistics.

Provides thread-safe storage and retrieval of sender statistics
with efficient bulk operations and querying capabilities.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from inbox_reaper.state import SenderStats


class SenderStatsDB:
    """Thread-safe SQLite database for sender statistics."""

    def __init__(self, db_path: str | Path = "sender_stats.db"):
        """Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create sender_stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sender_stats (
                    sender TEXT PRIMARY KEY,
                    marketing_count INTEGER NOT NULL DEFAULT 0,
                    total_count INTEGER NOT NULL DEFAULT 0,
                    auto_delete INTEGER NOT NULL DEFAULT 0,
                    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_auto_delete
                ON sender_stats(auto_delete)
                WHERE auto_delete = 1
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_updated
                ON sender_stats(last_updated)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections.

        Ensures proper cleanup and thread safety.
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                yield conn
        finally:
            conn.close()

    def load_all_stats(self) -> dict[str, dict[str, Any]]:
        """Load all sender statistics from the database.

        Returns:
            Dictionary mapping sender email to stats dict with keys:
            - sender: str
            - marketing_count: int
            - total_count: int
            - auto_delete: bool
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT sender, marketing_count, total_count, auto_delete
                FROM sender_stats
            """)

            stats = {}
            for row in cursor.fetchall():
                stats[row["sender"]] = {
                    "sender": row["sender"],
                    "marketing_count": row["marketing_count"],
                    "total_count": row["total_count"],
                    "auto_delete": bool(row["auto_delete"]),
                }

            return stats

    def save_stats(self, sender_stats: dict[str, dict[str, Any]]) -> None:
        """Persist all sender stats to the database.

        This is a convenience method that wraps bulk_update.

        Args:
            sender_stats: Dictionary of sender email to stats dict
        """
        self.bulk_update(sender_stats)

    def get_auto_delete_senders(self, threshold: int | None = None) -> list[str]:
        """Query senders marked for auto-delete.

        Args:
            threshold: Optional marketing count threshold. If provided,
                      returns senders with marketing_count >= threshold.
                      If None, returns senders with auto_delete = True.

        Returns:
            List of sender email addresses
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if threshold is not None:
                cursor.execute(
                    """
                    SELECT sender
                    FROM sender_stats
                    WHERE marketing_count >= ?
                    ORDER BY marketing_count DESC
                """,
                    (threshold,),
                )
            else:
                cursor.execute("""
                    SELECT sender
                    FROM sender_stats
                    WHERE auto_delete = 1
                    ORDER BY marketing_count DESC
                """)

            return [row["sender"] for row in cursor.fetchall()]

    def update_sender(
        self,
        sender: str,
        marketing_count: int,
        total_count: int,
        auto_delete: bool,
    ) -> None:
        """Update statistics for a single sender.

        Args:
            sender: Sender email address
            marketing_count: Number of marketing emails from this sender
            total_count: Total number of emails from this sender
            auto_delete: Whether to auto-delete emails from this sender
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sender_stats
                    (sender, marketing_count, total_count, auto_delete, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(sender) DO UPDATE SET
                    marketing_count = excluded.marketing_count,
                    total_count = excluded.total_count,
                    auto_delete = excluded.auto_delete,
                    last_updated = excluded.last_updated
            """,
                (
                    sender,
                    marketing_count,
                    total_count,
                    1 if auto_delete else 0,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def bulk_update(self, stats: dict[str, dict[str, Any]]) -> None:
        """Batch update multiple senders efficiently.

        Uses a single transaction for all updates to maximize performance.

        Args:
            stats: Dictionary mapping sender email to stats dict.
                  Each stats dict should have keys: sender, marketing_count,
                  total_count, auto_delete
        """
        if not stats:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Prepare batch data
            now = datetime.now().isoformat()
            batch_data = []
            for sender, stat_dict in stats.items():
                batch_data.append(
                    (
                        sender,
                        stat_dict.get("marketing_count", 0),
                        stat_dict.get("total_count", 0),
                        1 if stat_dict.get("auto_delete", False) else 0,
                        now,
                    )
                )

            # Execute batch upsert in a single transaction
            cursor.executemany(
                """
                INSERT INTO sender_stats
                    (sender, marketing_count, total_count, auto_delete, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(sender) DO UPDATE SET
                    marketing_count = excluded.marketing_count,
                    total_count = excluded.total_count,
                    auto_delete = excluded.auto_delete,
                    last_updated = excluded.last_updated
            """,
                batch_data,
            )

            conn.commit()

    def get_sender_stats(self, sender: str) -> dict[str, Any] | None:
        """Get statistics for a specific sender.

        Args:
            sender: Sender email address

        Returns:
            Stats dict or None if sender not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sender, marketing_count, total_count, auto_delete
                FROM sender_stats
                WHERE sender = ?
            """,
                (sender,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "sender": row["sender"],
                    "marketing_count": row["marketing_count"],
                    "total_count": row["total_count"],
                    "auto_delete": bool(row["auto_delete"]),
                }
            return None

    def delete_sender(self, sender: str) -> bool:
        """Delete a sender's statistics.

        Args:
            sender: Sender email address

        Returns:
            True if sender was deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sender_stats WHERE sender = ?", (sender,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Clear all sender statistics.

        Returns:
            Number of rows deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sender_stats")
            conn.commit()
            return cursor.rowcount

    def get_stats_count(self) -> int:
        """Get total number of senders in the database.

        Returns:
            Total count of sender records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM sender_stats")
            row = cursor.fetchone()
            return row["count"] if row else 0

    def get_top_senders(
        self, limit: int = 10, by: str = "marketing"
    ) -> list[dict[str, Any]]:
        """Get top senders by marketing or total email count.

        Args:
            limit: Maximum number of senders to return
            by: Sort by "marketing" (marketing_count) or "total" (total_count)

        Returns:
            List of sender stats dicts
        """
        order_column = "marketing_count" if by == "marketing" else "total_count"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT sender, marketing_count, total_count, auto_delete
                FROM sender_stats
                ORDER BY {order_column} DESC
                LIMIT ?
            """,
                (limit,),
            )

            return [
                {
                    "sender": row["sender"],
                    "marketing_count": row["marketing_count"],
                    "total_count": row["total_count"],
                    "auto_delete": bool(row["auto_delete"]),
                }
                for row in cursor.fetchall()
            ]

    @staticmethod
    def stats_to_dict(sender_stats: SenderStats) -> dict[str, Any]:
        """Convert SenderStats model to dict for database storage.

        Args:
            sender_stats: SenderStats Pydantic model

        Returns:
            Dictionary representation
        """
        return {
            "sender": sender_stats.sender,
            "marketing_count": sender_stats.marketing_count,
            "total_count": sender_stats.total_count,
            "auto_delete": sender_stats.auto_delete,
        }

    @staticmethod
    def dict_to_stats(stats_dict: dict[str, Any]) -> SenderStats:
        """Convert dict to SenderStats model.

        Args:
            stats_dict: Dictionary with sender stats

        Returns:
            SenderStats Pydantic model
        """
        return SenderStats(
            sender=stats_dict["sender"],
            marketing_count=stats_dict.get("marketing_count", 0),
            total_count=stats_dict.get("total_count", 0),
            auto_delete=stats_dict.get("auto_delete", False),
        )

    def load_all_as_models(self) -> dict[str, SenderStats]:
        """Load all sender statistics as SenderStats models.

        Returns:
            Dictionary mapping sender email to SenderStats model
        """
        stats_dicts = self.load_all_stats()
        return {
            sender: self.dict_to_stats(stats_dict)
            for sender, stats_dict in stats_dicts.items()
        }

    def save_models(self, sender_stats: dict[str, SenderStats]) -> None:
        """Save SenderStats models to the database.

        Args:
            sender_stats: Dictionary mapping sender to SenderStats model
        """
        stats_dicts = {
            sender: self.stats_to_dict(stats) for sender, stats in sender_stats.items()
        }
        self.bulk_update(stats_dicts)
