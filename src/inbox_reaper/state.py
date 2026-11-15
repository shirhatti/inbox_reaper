"""State models for the email classification pipeline.

All state is encapsulated in immutable Pydantic models to enable
pure functional programming patterns.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .timeutil import utcnow


class Decision(StrEnum):
    """Classification decision for an email."""

    KEEP = "keep"
    DELETE = "delete"
    UNCERTAIN = "uncertain"


class FilterReason(StrEnum):
    """Reason for filtering decision (before AI classification)."""

    ATTACHMENT = "attachment"
    KEYWORD = "keyword"
    WHITELIST = "whitelist"
    SENDER_PATTERN = "sender_pattern"
    RULE = "rule"  # learned/manual sender or domain rule
    EXPIRED = (
        "expired"  # time-sensitive mail (codes, sign-in links) past its useful life
    )
    TIME_SENSITIVE = (
        "time_sensitive"  # time-sensitive mail still within its useful life
    )
    COMBINER = "combiner"  # trained combiner over the feature vector
    AI_CLASSIFIED = "ai_classified"
    NONE = "none"


class Email(BaseModel):
    """Immutable email data."""

    model_config = ConfigDict(frozen=True)

    uid: str
    subject: str
    sender: str
    body: str
    date: datetime
    attachments: list[str] = Field(default_factory=list)
    # True when the message carries a List-Unsubscribe header (bulk mail)
    has_list_unsubscribe: bool = False
    # Selected headers (lowercased name -> value; repeats joined by newlines)
    headers: dict[str, str] = Field(default_factory=dict)
    # Link destinations from the HTML and text bodies (Safe Links unwrapped)
    links: list[str] = Field(default_factory=list)
    image_count: int = 0
    tracking_pixels: int = 0


class EmailDecision(BaseModel):
    """Decision for a single email with metadata."""

    model_config = ConfigDict(frozen=True)

    email: Email
    decision: Decision
    reason: FilterReason
    confidence: float = 1.0
    processed_at: datetime = Field(default_factory=utcnow)


class Config(BaseModel):
    """Configuration for the email classification pipeline."""

    model_config = ConfigDict(frozen=True)

    # AI settings (MLX)
    # Emails per batched MLX call; past ~16 a batch gets slower per email
    concurrent_ai_limit: int = 8
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    # Email settings
    email: str = ""

    # IMAP settings
    fetch_size: int = 100
    batch_size: int = 50
    max_emails: int | None = None

    # AI deletes only at or above this stated confidence
    ai_confidence_threshold: float = 0.7

    # Keywords & whitelist
    keywords: list[str] = Field(default_factory=list)
    whitelist_domains: list[str] = Field(default_factory=list)
    important_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".doc", ".docx"]
    )

    # Learned rules file (from `inbox-reaper learn`), applied before AI
    rules_path: str | None = None
    # Trained combiner (from `inbox-reaper train`): decides from the feature vector
    combiner_path: str | None = None

    # Safety
    dry_run: bool = True
