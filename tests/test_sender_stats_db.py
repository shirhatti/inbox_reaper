"""Tests for sender statistics database operations.

This module tests:
- Database initialization
- CRUD operations
- Bulk updates
- Thread safety
- Query operations
- Model conversion
"""

import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from inbox_reaper.sender_stats_db import SenderStatsDB
from inbox_reaper.state import SenderStats


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def db(temp_db):
    """Create a SenderStatsDB instance with temporary database."""
    return SenderStatsDB(temp_db)


class TestDatabaseInitialization:
    """Test database initialization and schema creation."""

    def test_init_creates_database_file(self, temp_db):
        """Test that initialization creates the database file."""
        SenderStatsDB(temp_db)

        assert Path(temp_db).exists()

    def test_init_creates_tables(self, temp_db):
        """Test that initialization creates required tables."""
        SenderStatsDB(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check sender_stats table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='sender_stats'
            """
        )
        result = cursor.fetchone()
        assert result is not None

        conn.close()

    def test_init_creates_indexes(self, temp_db):
        """Test that initialization creates performance indexes."""
        SenderStatsDB(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check indexes exist
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='index' AND (name='idx_auto_delete' OR name='idx_last_updated')
            """
        )
        results = cursor.fetchall()
        assert len(results) == 2

        conn.close()

    def test_init_can_reopen_existing_database(self, temp_db):
        """Test that opening existing database doesn't corrupt data."""
        # Create and populate database
        db1 = SenderStatsDB(temp_db)
        db1.update_sender("test@example.com", 5, 10, False)

        # Reopen database
        db2 = SenderStatsDB(temp_db)
        stats = db2.get_sender_stats("test@example.com")

        assert stats is not None
        assert stats["marketing_count"] == 5


class TestUpdateSender:
    """Test single sender update operations."""

    def test_update_new_sender(self, db):
        """Test updating a new sender creates record."""
        db.update_sender("new@example.com", 5, 10, False)

        stats = db.get_sender_stats("new@example.com")
        assert stats is not None
        assert stats["sender"] == "new@example.com"
        assert stats["marketing_count"] == 5
        assert stats["total_count"] == 10
        assert stats["auto_delete"] is False

    def test_update_existing_sender(self, db):
        """Test updating existing sender replaces values."""
        db.update_sender("test@example.com", 5, 10, False)
        db.update_sender("test@example.com", 8, 15, True)

        stats = db.get_sender_stats("test@example.com")
        assert stats["marketing_count"] == 8  # Updated
        assert stats["total_count"] == 15  # Updated
        assert stats["auto_delete"] is True  # Updated

    def test_update_sender_auto_delete_flag(self, db):
        """Test auto_delete flag is stored correctly."""
        db.update_sender("delete@example.com", 10, 20, True)

        stats = db.get_sender_stats("delete@example.com")
        assert stats["auto_delete"] is True


class TestGetSenderStats:
    """Test retrieving sender statistics."""

    def test_get_nonexistent_sender_returns_none(self, db):
        """Test getting stats for non-existent sender returns None."""
        stats = db.get_sender_stats("nonexistent@example.com")

        assert stats is None

    def test_get_existing_sender_returns_stats(self, db):
        """Test getting stats for existing sender."""
        db.update_sender("test@example.com", 5, 10, False)

        stats = db.get_sender_stats("test@example.com")

        assert stats is not None
        assert stats["sender"] == "test@example.com"
        assert stats["marketing_count"] == 5
        assert stats["total_count"] == 10
        assert stats["auto_delete"] is False


class TestLoadAllStats:
    """Test loading all sender statistics."""

    def test_load_all_stats_empty_database(self, db):
        """Test loading all stats from empty database."""
        stats = db.load_all_stats()

        assert stats == {}

    def test_load_all_stats_with_data(self, db):
        """Test loading all stats from populated database."""
        db.update_sender("sender1@example.com", 5, 10, False)
        db.update_sender("sender2@example.com", 3, 6, True)
        db.update_sender("sender3@example.com", 8, 12, False)

        stats = db.load_all_stats()

        assert len(stats) == 3
        assert "sender1@example.com" in stats
        assert "sender2@example.com" in stats
        assert "sender3@example.com" in stats
        assert stats["sender2@example.com"]["auto_delete"] is True


class TestBulkUpdate:
    """Test bulk update operations."""

    def test_bulk_update_empty_dict(self, db):
        """Test bulk update with empty dictionary."""
        db.bulk_update({})

        stats = db.load_all_stats()
        assert stats == {}

    def test_bulk_update_multiple_senders(self, db):
        """Test bulk updating multiple senders at once."""
        updates = {
            "sender1@example.com": {
                "sender": "sender1@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            },
            "sender2@example.com": {
                "sender": "sender2@example.com",
                "marketing_count": 3,
                "total_count": 6,
                "auto_delete": True,
            },
        }

        db.bulk_update(updates)

        stats = db.load_all_stats()
        assert len(stats) == 2
        assert stats["sender1@example.com"]["marketing_count"] == 5
        assert stats["sender2@example.com"]["auto_delete"] is True

    def test_bulk_update_overwrites_existing(self, db):
        """Test that bulk update overwrites existing records."""
        # Initial data
        db.update_sender("test@example.com", 5, 10, False)

        # Bulk update with new value
        updates = {
            "test@example.com": {
                "sender": "test@example.com",
                "marketing_count": 10,
                "total_count": 20,
                "auto_delete": True,
            }
        }
        db.bulk_update(updates)

        stats = db.get_sender_stats("test@example.com")
        assert stats["marketing_count"] == 10
        assert stats["auto_delete"] is True

    def test_save_stats_uses_bulk_update(self, db):
        """Test that save_stats is an alias for bulk_update."""
        stats = {
            "test@example.com": {
                "sender": "test@example.com",
                "marketing_count": 5,
                "total_count": 10,
                "auto_delete": False,
            }
        }

        db.save_stats(stats)

        loaded = db.get_sender_stats("test@example.com")
        assert loaded is not None
        assert loaded["marketing_count"] == 5


class TestQueryOperations:
    """Test query and filtering operations."""

    def test_get_auto_delete_senders_by_flag(self, db):
        """Test getting senders with auto_delete=True."""
        db.update_sender("sender1@example.com", 10, 20, True)
        db.update_sender("sender2@example.com", 3, 6, False)
        db.update_sender("sender3@example.com", 8, 12, True)

        auto_delete = db.get_auto_delete_senders()

        assert len(auto_delete) == 2
        assert "sender1@example.com" in auto_delete
        assert "sender3@example.com" in auto_delete
        assert "sender2@example.com" not in auto_delete

    def test_get_auto_delete_senders_by_threshold(self, db):
        """Test getting senders above marketing count threshold."""
        db.update_sender("sender1@example.com", 10, 20, False)
        db.update_sender("sender2@example.com", 3, 6, False)
        db.update_sender("sender3@example.com", 8, 12, False)

        auto_delete = db.get_auto_delete_senders(threshold=5)

        assert len(auto_delete) == 2
        assert "sender1@example.com" in auto_delete
        assert "sender3@example.com" in auto_delete

    def test_get_auto_delete_senders_ordered_by_count(self, db):
        """Test that auto_delete senders are ordered by marketing count."""
        db.update_sender("sender1@example.com", 5, 10, True)
        db.update_sender("sender2@example.com", 10, 20, True)
        db.update_sender("sender3@example.com", 3, 6, True)

        auto_delete = db.get_auto_delete_senders()

        # Should be ordered descending by marketing_count
        assert auto_delete[0] == "sender2@example.com"  # 10
        assert auto_delete[1] == "sender1@example.com"  # 5
        assert auto_delete[2] == "sender3@example.com"  # 3

    def test_get_top_senders_by_marketing(self, db):
        """Test getting top senders by marketing count."""
        db.update_sender("sender1@example.com", 10, 20, False)
        db.update_sender("sender2@example.com", 5, 10, False)
        db.update_sender("sender3@example.com", 8, 15, False)

        top = db.get_top_senders(limit=2, by="marketing")

        assert len(top) == 2
        assert top[0]["sender"] == "sender1@example.com"
        assert top[1]["sender"] == "sender3@example.com"

    def test_get_top_senders_by_total(self, db):
        """Test getting top senders by total count."""
        db.update_sender("sender1@example.com", 5, 20, False)
        db.update_sender("sender2@example.com", 10, 15, False)
        db.update_sender("sender3@example.com", 3, 25, False)

        top = db.get_top_senders(limit=2, by="total")

        assert len(top) == 2
        assert top[0]["sender"] == "sender3@example.com"  # 25
        assert top[1]["sender"] == "sender1@example.com"  # 20

    def test_get_stats_count(self, db):
        """Test getting total count of senders."""
        assert db.get_stats_count() == 0

        db.update_sender("sender1@example.com", 5, 10, False)
        db.update_sender("sender2@example.com", 3, 6, False)

        assert db.get_stats_count() == 2


class TestDeleteOperations:
    """Test delete operations."""

    def test_delete_existing_sender(self, db):
        """Test deleting an existing sender."""
        db.update_sender("test@example.com", 5, 10, False)

        result = db.delete_sender("test@example.com")

        assert result is True
        assert db.get_sender_stats("test@example.com") is None

    def test_delete_nonexistent_sender(self, db):
        """Test deleting non-existent sender returns False."""
        result = db.delete_sender("nonexistent@example.com")

        assert result is False

    def test_clear_all(self, db):
        """Test clearing all sender statistics."""
        db.update_sender("sender1@example.com", 5, 10, False)
        db.update_sender("sender2@example.com", 3, 6, False)

        count = db.clear_all()

        assert count == 2
        assert db.get_stats_count() == 0
        assert db.load_all_stats() == {}


class TestModelConversion:
    """Test conversion between dicts and Pydantic models."""

    def test_stats_to_dict(self):
        """Test converting SenderStats model to dict."""
        stats = SenderStats(
            sender="test@example.com",
            marketing_count=5,
            total_count=10,
            auto_delete=True,
        )

        stats_dict = SenderStatsDB.stats_to_dict(stats)

        assert stats_dict["sender"] == "test@example.com"
        assert stats_dict["marketing_count"] == 5
        assert stats_dict["total_count"] == 10
        assert stats_dict["auto_delete"] is True

    def test_dict_to_stats(self):
        """Test converting dict to SenderStats model."""
        stats_dict = {
            "sender": "test@example.com",
            "marketing_count": 5,
            "total_count": 10,
            "auto_delete": True,
        }

        stats = SenderStatsDB.dict_to_stats(stats_dict)

        assert isinstance(stats, SenderStats)
        assert stats.sender == "test@example.com"
        assert stats.marketing_count == 5
        assert stats.total_count == 10
        assert stats.auto_delete is True

    def test_load_all_as_models(self, db):
        """Test loading all stats as SenderStats models."""
        db.update_sender("sender1@example.com", 5, 10, False)
        db.update_sender("sender2@example.com", 3, 6, True)

        models = db.load_all_as_models()

        assert len(models) == 2
        assert all(isinstance(s, SenderStats) for s in models.values())
        assert models["sender1@example.com"].marketing_count == 5
        assert models["sender2@example.com"].auto_delete is True

    def test_save_models(self, db):
        """Test saving SenderStats models to database."""
        models = {
            "sender1@example.com": SenderStats(
                sender="sender1@example.com",
                marketing_count=5,
                total_count=10,
                auto_delete=False,
            ),
            "sender2@example.com": SenderStats(
                sender="sender2@example.com",
                marketing_count=3,
                total_count=6,
                auto_delete=True,
            ),
        }

        db.save_models(models)

        stats = db.load_all_stats()
        assert len(stats) == 2
        assert stats["sender1@example.com"]["marketing_count"] == 5
        assert stats["sender2@example.com"]["auto_delete"] is True


class TestThreadSafety:
    """Test thread-safe concurrent database operations."""

    def test_concurrent_updates_different_senders(self, db):
        """Test concurrent updates to different senders."""

        def update_sender(sender_id: int):
            sender = f"sender{sender_id}@example.com"
            db.update_sender(sender, sender_id, sender_id * 2, False)

        # Run 10 concurrent updates
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_sender, i) for i in range(10)]
            [f.result() for f in futures]

        # Verify all updates succeeded
        stats = db.load_all_stats()
        assert len(stats) == 10

        for i in range(10):
            sender = f"sender{i}@example.com"
            assert stats[sender]["marketing_count"] == i

    def test_concurrent_updates_same_sender(self, db):
        """Test concurrent updates to the same sender (last write wins)."""
        sender = "test@example.com"

        def update_with_count(count: int):
            db.update_sender(sender, count, count * 2, False)

        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_with_count, i) for i in range(1, 6)]
            [f.result() for f in futures]

        # One update should win (which one is non-deterministic)
        stats = db.get_sender_stats(sender)
        assert stats is not None
        assert stats["marketing_count"] in range(1, 6)

    def test_concurrent_bulk_updates(self, db):
        """Test concurrent bulk update operations."""

        def bulk_update_batch(batch_id: int):
            updates = {}
            for i in range(10):
                sender = f"batch{batch_id}_sender{i}@example.com"
                updates[sender] = {
                    "sender": sender,
                    "marketing_count": batch_id,
                    "total_count": batch_id * 2,
                    "auto_delete": False,
                }
            db.bulk_update(updates)

        # Run 5 concurrent bulk updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(bulk_update_batch, i) for i in range(5)]
            [f.result() for f in futures]

        # Verify all bulk updates succeeded
        stats = db.load_all_stats()
        assert len(stats) == 50  # 5 batches × 10 senders

    def test_concurrent_reads_and_writes(self, db):
        """Test concurrent read and write operations."""
        # Prepopulate database
        for i in range(10):
            db.update_sender(f"sender{i}@example.com", i, i * 2, False)

        def read_random_sender(sender_id: int):
            return db.get_sender_stats(f"sender{sender_id}@example.com")

        def write_random_sender(sender_id: int):
            db.update_sender(
                f"sender{sender_id}@example.com",
                sender_id + 10,
                (sender_id + 10) * 2,
                True,
            )

        # Mix reads and writes
        with ThreadPoolExecutor(max_workers=10) as executor:
            read_futures = [
                executor.submit(read_random_sender, i % 10) for i in range(20)
            ]
            write_futures = [
                executor.submit(write_random_sender, i % 10) for i in range(20)
            ]

            # Wait for all operations
            [f.result() for f in read_futures + write_futures]

        # Database should still be consistent
        assert db.get_stats_count() == 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_sender_with_special_characters(self, db):
        """Test sender emails with special characters."""
        sender = "user+tag@example.com"
        db.update_sender(sender, 5, 10, False)

        stats = db.get_sender_stats(sender)
        assert stats is not None
        assert stats["sender"] == sender

    def test_sender_with_unicode(self, db):
        """Test sender emails with unicode characters."""
        sender = "用户@example.com"
        db.update_sender(sender, 5, 10, False)

        stats = db.get_sender_stats(sender)
        assert stats is not None
        assert stats["sender"] == sender

    def test_zero_counts(self, db):
        """Test handling zero counts."""
        db.update_sender("zero@example.com", 0, 0, False)

        stats = db.get_sender_stats("zero@example.com")
        assert stats["marketing_count"] == 0
        assert stats["total_count"] == 0

    def test_very_large_counts(self, db):
        """Test handling very large counts."""
        large_count = 1_000_000
        db.update_sender("large@example.com", large_count, large_count * 2, True)

        stats = db.get_sender_stats("large@example.com")
        assert stats["marketing_count"] == large_count
        assert stats["total_count"] == large_count * 2
