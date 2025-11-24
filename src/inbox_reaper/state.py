"""State models for the email classification pipeline.

All state is encapsulated in immutable Pydantic models to enable
pure functional programming patterns.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """Classification decision for an email."""

    KEEP = "keep"
    DELETE = "delete"
    UNCERTAIN = "uncertain"


class FilterReason(str, Enum):
    """Reason for filtering decision (before AI classification)."""

    ATTACHMENT = "attachment"
    KEYWORD = "keyword"
    WHITELIST = "whitelist"
    SENDER_PATTERN = "sender_pattern"
    AI_CLASSIFIED = "ai_classified"
    NONE = "none"


class Email(BaseModel):
    """Immutable email data."""

    uid: str
    subject: str
    sender: str
    body: str
    date: datetime
    attachments: list[str] = Field(default_factory=list)

    class Config:
        frozen = True


class EmailDecision(BaseModel):
    """Decision for a single email with metadata."""

    email: Email
    decision: Decision
    reason: FilterReason
    confidence: float = 1.0
    processed_at: datetime = Field(default_factory=datetime.now)

    class Config:
        frozen = True


class SenderStats(BaseModel):
    """Statistics for a sender across multiple emails."""

    sender: str
    marketing_count: int = 0
    total_count: int = 0
    auto_delete: bool = False

    class Config:
        frozen = True

    def should_auto_delete(self, threshold: int = 5) -> bool:
        """Check if sender has enough marketing emails to auto-delete."""
        return self.marketing_count >= threshold


class Config(BaseModel):
    """Configuration for the email classification pipeline."""

    # AI settings
    concurrent_ai_limit: int = 25
    model_name: str = "gemma2:2b"
    ollama_base_url: str = "http://localhost:11434"

    # Email settings
    email: str = ""

    # IMAP settings
    fetch_size: int = 100
    batch_size: int = 50
    max_emails: int | None = None

    # Decision thresholds
    auto_delete_threshold: int = 5

    # Keywords & whitelist
    keywords: list[str] = Field(default_factory=list)
    whitelist_domains: list[str] = Field(default_factory=list)
    important_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".doc", ".docx"]
    )

    # Performance
    enable_pipelining: bool = True
    enable_sender_tracking: bool = True

    # Safety
    dry_run: bool = True

    class Config:
        frozen = True


class ProcessingState(BaseModel):
    """Complete state for the email processing pipeline.

    This is the single source of truth that flows through the DAG.
    All transformations are pure functions: State -> State.
    """

    # Configuration
    config: Config

    # Current batch being processed
    emails: list[Email] = Field(default_factory=list)

    # Decisions made for current batch
    decisions: list[EmailDecision] = Field(default_factory=list)

    # Historical data (for resume and sender tracking)
    processed_uids: set[str] = Field(default_factory=set)
    sender_stats: dict[str, SenderStats] = Field(default_factory=dict)

    # Pagination/windowing
    min_uid: str | None = None
    max_uid: str | None = None

    # Progress tracking
    total_processed: int = 0
    total_deleted: int = 0
    total_kept: int = 0
    consecutive_empty_batches: int = 0

    # Error handling
    errors: list[str] = Field(default_factory=list)

    class Config:
        frozen = True

    def add_decision(self, decision: EmailDecision) -> "ProcessingState":
        """Pure function to add a decision and update counters."""
        new_decisions = self.decisions + [decision]
        new_processed_uids = self.processed_uids | {decision.email.uid}

        # Update sender stats
        sender = decision.email.sender
        current_stats = self.sender_stats.get(sender, SenderStats(sender=sender))

        new_sender_stats = self.sender_stats.copy()
        new_sender_stats[sender] = SenderStats(
            sender=sender,
            total_count=current_stats.total_count + 1,
            marketing_count=current_stats.marketing_count
            + (1 if decision.decision == Decision.DELETE else 0),
            auto_delete=(
                current_stats.marketing_count + 1 >= self.config.auto_delete_threshold
                if decision.decision == Decision.DELETE
                else current_stats.auto_delete
            ),
        )

        return self.model_copy(
            update={
                "decisions": new_decisions,
                "processed_uids": new_processed_uids,
                "sender_stats": new_sender_stats,
                "total_processed": self.total_processed + 1,
                "total_deleted": self.total_deleted
                + (1 if decision.decision == Decision.DELETE else 0),
                "total_kept": self.total_kept
                + (1 if decision.decision == Decision.KEEP else 0),
            }
        )

    def add_error(self, error: str) -> "ProcessingState":
        """Pure function to add an error."""
        return self.model_copy(update={"errors": self.errors + [error]})

    def update_batch(self, emails: list[Email]) -> "ProcessingState":
        """Pure function to update current batch and pagination."""
        if not emails:
            return self.model_copy(
                update={"consecutive_empty_batches": self.consecutive_empty_batches + 1}
            )

        uids = [email.uid for email in emails]
        new_min_uid = min(uids)
        new_max_uid = max(uids)

        return self.model_copy(
            update={
                "emails": emails,
                "decisions": [],  # Reset decisions for new batch
                "min_uid": new_min_uid,
                "max_uid": new_max_uid,
                "consecutive_empty_batches": 0,
            }
        )
