"""Peewee ORM models for inbox_reaper database tables.

Provides declarative models for:
- SenderStats: Sender statistics and auto-delete flags
- Watermark: Progress tracking for resumable processing
- Quarantine: Failed email storage and retry management
"""

from datetime import datetime

from peewee import (
    SQL,
    BooleanField,
    CharField,
    DateTimeField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

# Database proxy - will be initialized with actual path by each manager
database_proxy = SqliteDatabase(None)


class BaseModel(Model):
    """Base model with common configuration."""

    class Meta:
        database = database_proxy


class SenderStatsModel(BaseModel):
    """Model for sender statistics table."""

    sender = CharField(primary_key=True)
    marketing_count = IntegerField(default=0)
    total_count = IntegerField(default=0)
    auto_delete = IntegerField(default=0)  # SQLite uses INTEGER for boolean
    last_updated = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "sender_stats"
        indexes = (
            # Partial index for auto_delete
            (("auto_delete",), False),
            (("last_updated",), False),
        )


class WatermarkModel(BaseModel):
    """Model for watermark/progress tracking table."""

    id = IntegerField(primary_key=True, constraints=[SQL("CHECK (id = 1)")])
    last_processed_uid = CharField(null=True)
    last_update_time = CharField(null=True)
    total_processed = IntegerField(default=0)
    total_deleted = IntegerField(default=0)
    total_kept = IntegerField(default=0)
    total_errors = IntegerField(default=0)

    class Meta:
        table_name = "watermarks"


class QuarantineModel(BaseModel):
    """Model for quarantine table."""

    uid = CharField(primary_key=True)
    email_data = TextField()  # JSON-serialized email data
    error_type = CharField()
    error_message = TextField()
    attempts = IntegerField(default=1)
    quarantined_at = CharField()
    last_retry_at = CharField(null=True)
    resolved = BooleanField(default=False)

    class Meta:
        table_name = "quarantine"
        indexes = (
            (("resolved",), False),
            (("error_type",), False),
        )
