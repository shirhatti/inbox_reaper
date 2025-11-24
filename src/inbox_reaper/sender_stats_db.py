"""SQLite persistence layer for sender statistics.

Provides thread-safe storage and retrieval of sender statistics
with efficient bulk operations and querying capabilities.
"""

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, cast

from peewee import SqliteDatabase

from inbox_reaper.models import SenderStatsModel
from inbox_reaper.state import SenderStats


class SenderStatsDB:
    """Thread-safe SQLite database for sender statistics using Peewee ORM."""

    def __init__(self, db_path: str | Path = "sender_stats.db"):
        """Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = Lock()

        # Initialize database
        self.db = SqliteDatabase(
            str(self.db_path),
            pragmas={
                "foreign_keys": 1,
                "journal_mode": "wal",
                "synchronous": "normal",
            },
        )

        # Bind model to database
        SenderStatsModel._meta.database = self.db

        self._init_db()

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._lock:
            self.db.connect(reuse_if_open=True)
            self.db.create_tables([SenderStatsModel], safe=True)

            # Create partial index for auto_delete if not exists
            self.db.execute_sql(
                """
                CREATE INDEX IF NOT EXISTS idx_auto_delete
                ON sender_stats(auto_delete)
                WHERE auto_delete = 1
                """
            )

            # Create index for last_updated if not exists
            self.db.execute_sql(
                """
                CREATE INDEX IF NOT EXISTS idx_last_updated
                ON sender_stats(last_updated)
                """
            )

    def load_all_stats(self) -> dict[str, dict[str, Any]]:
        """Load all sender statistics from the database.

        Returns:
            Dictionary mapping sender email to stats dict with keys:
            - sender: str
            - marketing_count: int
            - total_count: int
            - auto_delete: bool
        """
        with self._lock:
            stats = {}
            for sender_model in SenderStatsModel.select():
                stats[sender_model.sender] = {
                    "sender": sender_model.sender,
                    "marketing_count": sender_model.marketing_count,
                    "total_count": sender_model.total_count,
                    "auto_delete": bool(sender_model.auto_delete),
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
        with self._lock:
            if threshold is not None:
                query = (
                    SenderStatsModel.select()
                    .where(SenderStatsModel.marketing_count >= threshold)
                    .order_by(SenderStatsModel.marketing_count.desc())
                )
            else:
                query = (
                    SenderStatsModel.select()
                    .where(SenderStatsModel.auto_delete == 1)
                    .order_by(SenderStatsModel.marketing_count.desc())
                )

            return [sender.sender for sender in query]

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
        with self._lock:
            SenderStatsModel.insert(
                sender=sender,
                marketing_count=marketing_count,
                total_count=total_count,
                auto_delete=1 if auto_delete else 0,
                last_updated=datetime.now(),
            ).on_conflict(
                conflict_target=[SenderStatsModel.sender],
                update={
                    SenderStatsModel.marketing_count: marketing_count,
                    SenderStatsModel.total_count: total_count,
                    SenderStatsModel.auto_delete: 1 if auto_delete else 0,
                    SenderStatsModel.last_updated: datetime.now(),
                },
            ).execute()

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

        with self._lock:
            with self.db.atomic():
                # Prepare batch data
                now = datetime.now()
                batch_data = []
                for sender, stat_dict in stats.items():
                    batch_data.append({
                        "sender": sender,
                        "marketing_count": stat_dict.get("marketing_count", 0),
                        "total_count": stat_dict.get("total_count", 0),
                        "auto_delete": 1 if stat_dict.get("auto_delete", False) else 0,
                        "last_updated": now,
                    })

                # Insert or update in batches
                for batch_item in batch_data:
                    SenderStatsModel.insert(**batch_item).on_conflict(
                        conflict_target=[SenderStatsModel.sender],
                        update={
                            SenderStatsModel.marketing_count: batch_item[
                                "marketing_count"
                            ],
                            SenderStatsModel.total_count: batch_item["total_count"],
                            SenderStatsModel.auto_delete: batch_item["auto_delete"],
                            SenderStatsModel.last_updated: batch_item["last_updated"],
                        },
                    ).execute()

    def get_sender_stats(self, sender: str) -> dict[str, Any] | None:
        """Get statistics for a specific sender.

        Args:
            sender: Sender email address

        Returns:
            Stats dict or None if sender not found
        """
        with self._lock:
            try:
                sender_model = SenderStatsModel.get(SenderStatsModel.sender == sender)
                return {
                    "sender": sender_model.sender,
                    "marketing_count": sender_model.marketing_count,
                    "total_count": sender_model.total_count,
                    "auto_delete": bool(sender_model.auto_delete),
                }
            except SenderStatsModel.DoesNotExist:
                return None

    def delete_sender(self, sender: str) -> bool:
        """Delete a sender's statistics.

        Args:
            sender: Sender email address

        Returns:
            True if sender was deleted, False if not found
        """
        with self._lock:
            deleted_count = (
                SenderStatsModel.delete()
                .where(SenderStatsModel.sender == sender)
                .execute()
            )
            return cast(bool, deleted_count > 0)

    def clear_all(self) -> int:
        """Clear all sender statistics.

        Returns:
            Number of rows deleted
        """
        with self._lock:
            return cast(int, SenderStatsModel.delete().execute())

    def get_stats_count(self) -> int:
        """Get total number of senders in the database.

        Returns:
            Total count of sender records
        """
        with self._lock:
            return cast(int, SenderStatsModel.select().count())

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
        with self._lock:
            order_column = (
                SenderStatsModel.marketing_count
                if by == "marketing"
                else SenderStatsModel.total_count
            )

            query = (
                SenderStatsModel.select()
                .order_by(order_column.desc())
                .limit(limit)
            )

            return [
                {
                    "sender": sender.sender,
                    "marketing_count": sender.marketing_count,
                    "total_count": sender.total_count,
                    "auto_delete": bool(sender.auto_delete),
                }
                for sender in query
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
