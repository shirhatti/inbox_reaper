"""Dynamic batching optimization for email processing.

This module provides intelligent batching strategies that adapt to:
- System performance (processing rate, memory usage)
- Email characteristics (sender patterns, deletion likelihood)
- Cache locality (grouping by sender for better cache hits)

Features:
- Adaptive batch sizing based on real-time metrics
- Email prioritization for faster cleanup
- Sender-based grouping for cache optimization
- Performance tracking and adjustment
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from .state import Config, Email, SenderStats

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Performance metrics for batch processing."""

    processing_rate: float = 0.0  # emails/sec
    memory_usage_mb: float = 0.0
    memory_trend: float = 0.0  # MB change per batch
    error_rate: float = 0.0  # ratio of errors to total
    batch_duration: float = 0.0  # seconds
    emails_processed: int = 0
    errors_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        """Check if metrics indicate healthy processing."""
        return self.error_rate < 0.1 and self.memory_usage_mb < 1000


@dataclass
class EmailScore:
    """Score for email prioritization."""

    email: Email
    score: float  # Higher score = higher priority for deletion
    reason: str  # Reason for the score


class DynamicBatcher:
    """Optimizes email batching with adaptive sizing and prioritization.

    This class implements dynamic batching strategies that improve
    processing performance through:

    1. Adaptive batch sizing - adjusts batch size based on performance
    2. Email prioritization - processes likely deletions first
    3. Sender grouping - groups emails by sender for cache locality
    4. Performance tracking - monitors and adjusts strategy

    Example:
        batcher = DynamicBatcher(config, sender_stats)

        # Get optimized batches
        for batch in batcher.create_batches(emails):
            process_batch(batch)

        # Update metrics after processing
        batcher.update_metrics(
            processing_rate=10.5,
            memory_usage_mb=250.0,
            error_rate=0.02
        )
    """

    def __init__(
        self,
        config: Config,
        sender_stats: dict[str, SenderStats] | None = None,
        min_batch_size: int = 10,
        max_batch_size: int = 200,
    ):
        """Initialize dynamic batcher.

        Args:
            config: Configuration with initial batch_size
            sender_stats: Historical sender statistics for prioritization
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
        """
        self.config = config
        self.sender_stats = sender_stats or {}
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Current optimal batch size (starts with config value)
        self.current_batch_size = config.batch_size

        # Metrics tracking
        self._metrics_history: list[BatchMetrics] = []
        self._last_adjustment_time = time.time()
        self._adjustment_interval = 60.0  # Adjust every 60 seconds

        # Performance tracking
        self._total_batches = 0
        self._total_emails = 0
        self._total_duration = 0.0

        logger.info(
            f"Initialized DynamicBatcher: batch_size={self.current_batch_size}, "
            f"range=[{min_batch_size}, {max_batch_size}]"
        )

    def calculate_optimal_batch_size(self, metrics: dict[str, Any]) -> int:
        """Calculate optimal batch size based on performance metrics.

        Strategy:
        - Start with config.batch_size
        - Increase if: fast processing + low memory usage + low errors
        - Decrease if: slow processing OR high memory usage OR high errors
        - Stay within min/max bounds

        Args:
            metrics: Dictionary with keys:
                - processing_rate: emails/sec (optional)
                - memory_usage_mb: current memory in MB (optional)
                - error_rate: ratio of errors (optional)
                - memory_trend: MB change per batch (optional)

        Returns:
            Optimal batch size (between min_batch_size and max_batch_size)
        """
        current_size = self.current_batch_size

        # Extract metrics with defaults
        processing_rate = metrics.get("processing_rate", 0.0)
        memory_usage_mb = metrics.get("memory_usage_mb", 0.0)
        error_rate = metrics.get("error_rate", 0.0)
        memory_trend = metrics.get("memory_trend", 0.0)

        # Get system memory info
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024

        # Decision factors
        is_fast = processing_rate > 5.0  # More than 5 emails/sec
        has_memory = available_memory_mb > 500 and memory_usage_mb < 500
        low_errors = error_rate < 0.05
        memory_stable = abs(memory_trend) < 10  # Less than 10MB change

        # Increase batch size if conditions are good
        if is_fast and has_memory and low_errors and memory_stable:
            new_size = min(int(current_size * 1.2), self.max_batch_size)
            if new_size != current_size:
                logger.info(
                    f"Increasing batch size: {current_size} -> {new_size} "
                    f"(rate={processing_rate:.2f}/s, mem={memory_usage_mb:.0f}MB, "
                    f"errors={error_rate:.2%})"
                )
            return new_size

        # Decrease batch size if conditions are poor
        if error_rate > 0.1 or memory_usage_mb > 800 or memory_trend > 50:
            new_size = max(int(current_size * 0.7), self.min_batch_size)
            if new_size != current_size:
                logger.warning(
                    f"Decreasing batch size: {current_size} -> {new_size} "
                    f"(rate={processing_rate:.2f}/s, mem={memory_usage_mb:.0f}MB, "
                    f"errors={error_rate:.2%}, trend={memory_trend:+.1f}MB)"
                )
            return new_size

        # Moderate decrease if processing is slow
        if processing_rate > 0 and processing_rate < 2.0:
            new_size = max(int(current_size * 0.85), self.min_batch_size)
            if new_size != current_size:
                logger.info(
                    f"Moderately decreasing batch size due to slow rate: "
                    f"{current_size} -> {new_size} ({processing_rate:.2f}/s)"
                )
            return new_size

        # Keep current size if conditions are moderate
        return current_size

    def prioritize_emails(self, emails: list[Email]) -> list[Email]:
        """Prioritize emails based on likelihood of deletion.

        Scoring strategy:
        - High priority (score 3.0): Known marketing senders (auto_delete=True)
        - Medium priority (score 2.0): Senders with delete patterns (>50% delete rate)
        - Low priority (score 1.0): Unknown senders

        This ensures likely deletions are processed first for faster cleanup.

        Args:
            emails: List of emails to prioritize

        Returns:
            List of emails sorted by priority (highest first)
        """
        if not emails:
            return []

        scored_emails: list[EmailScore] = []

        for email in emails:
            sender = email.sender
            stats = self.sender_stats.get(sender)

            if stats is None:
                # Unknown sender - low priority
                score = 1.0
                reason = "unknown_sender"
            elif stats.auto_delete:
                # Known marketing sender - high priority
                score = 3.0
                reason = "auto_delete_sender"
            elif stats.total_count > 0:
                # Known sender with history - calculate delete rate
                delete_rate = stats.marketing_count / stats.total_count
                if delete_rate > 0.5:
                    score = 2.0
                    reason = f"delete_pattern_{delete_rate:.0%}"
                else:
                    score = 1.0
                    reason = f"keep_pattern_{1 - delete_rate:.0%}"
            else:
                score = 1.0
                reason = "no_history"

            scored_emails.append(EmailScore(email=email, score=score, reason=reason))

        # Sort by score (highest first)
        scored_emails.sort(key=lambda x: x.score, reverse=True)

        # Log prioritization summary
        high_priority = sum(1 for se in scored_emails if se.score >= 3.0)
        medium_priority = sum(1 for se in scored_emails if 2.0 <= se.score < 3.0)
        low_priority = sum(1 for se in scored_emails if se.score < 2.0)

        logger.info(
            f"Prioritized {len(emails)} emails: "
            f"high={high_priority}, medium={medium_priority}, low={low_priority}"
        )

        return [se.email for se in scored_emails]

    def group_by_sender(self, emails: list[Email]) -> list[list[Email]]:
        """Group consecutive emails from the same sender.

        This improves cache locality for sender_stats lookups and
        enhances pattern detection within sender groups.

        Strategy:
        - Preserve overall priority order
        - Group consecutive emails from same sender
        - Keep groups reasonably sized (max 20 per group)

        Args:
            emails: List of emails (should be pre-prioritized)

        Returns:
            List of email groups, each group contains emails from same sender
        """
        if not emails:
            return []

        groups: list[list[Email]] = []
        current_group: list[Email] = []
        current_sender: str | None = None
        max_group_size = 20

        for email in emails:
            if current_sender is None:
                # Start first group
                current_sender = email.sender
                current_group = [email]
            elif email.sender == current_sender and len(current_group) < max_group_size:
                # Continue current group
                current_group.append(email)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_sender = email.sender
                current_group = [email]

        # Add final group
        if current_group:
            groups.append(current_group)

        # Log grouping summary
        group_sizes = [len(g) for g in groups]
        avg_group_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0

        logger.info(
            f"Grouped {len(emails)} emails into {len(groups)} groups: "
            f"avg_size={avg_group_size:.1f}, "
            f"max_size={max(group_sizes) if group_sizes else 0}"
        )

        return groups

    def create_batches(
        self,
        emails: list[Email],
        prioritize: bool = True,
        group_by_sender: bool = True,
    ) -> list[list[Email]]:
        """Create optimized batches from emails.

        This is the main method that combines all optimization strategies:
        1. Prioritize emails (if enabled)
        2. Group by sender (if enabled)
        3. Create batches of optimal size

        Args:
            emails: List of emails to batch
            prioritize: Whether to prioritize emails by deletion likelihood
            group_by_sender: Whether to group by sender for cache locality

        Returns:
            List of batches, each batch is a list of emails
        """
        if not emails:
            return []

        logger.info(f"Creating batches for {len(emails)} emails")

        # Step 1: Prioritize emails
        if prioritize:
            emails = self.prioritize_emails(emails)

        # Step 2: Group by sender
        if group_by_sender:
            sender_groups = self.group_by_sender(emails)
            # Flatten groups back to single list while preserving order
            emails = [email for group in sender_groups for email in group]

        # Step 3: Create batches of optimal size
        batches: list[list[Email]] = []
        batch_size = self.current_batch_size

        for i in range(0, len(emails), batch_size):
            batch = emails[i : i + batch_size]
            batches.append(batch)

        logger.info(
            f"Created {len(batches)} batches with size ~{batch_size} "
            f"(range: {min(len(b) for b in batches)}-{max(len(b) for b in batches)})"
        )

        return batches

    def update_metrics(
        self,
        processing_rate: float = 0.0,
        memory_usage_mb: float = 0.0,
        error_rate: float = 0.0,
        batch_duration: float = 0.0,
        emails_processed: int = 0,
        errors_count: int = 0,
    ) -> None:
        """Update performance metrics and adjust batch size if needed.

        This should be called after processing each batch to track
        performance and trigger adaptive adjustments.

        Args:
            processing_rate: Emails processed per second
            memory_usage_mb: Current memory usage in MB
            error_rate: Ratio of errors to total emails
            batch_duration: Time taken to process batch in seconds
            emails_processed: Number of emails processed in batch
            errors_count: Number of errors encountered
        """
        # Calculate memory trend
        memory_trend = 0.0
        if self._metrics_history:
            last_memory = self._metrics_history[-1].memory_usage_mb
            memory_trend = memory_usage_mb - last_memory

        # Create metrics record
        metrics = BatchMetrics(
            processing_rate=processing_rate,
            memory_usage_mb=memory_usage_mb,
            memory_trend=memory_trend,
            error_rate=error_rate,
            batch_duration=batch_duration,
            emails_processed=emails_processed,
            errors_count=errors_count,
        )

        self._metrics_history.append(metrics)
        self._total_batches += 1
        self._total_emails += emails_processed
        self._total_duration += batch_duration

        # Keep only recent metrics (last 100 batches)
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]

        # Log metrics
        logger.debug(
            f"Batch metrics: rate={processing_rate:.2f}/s, "
            f"mem={memory_usage_mb:.0f}MB ({memory_trend:+.1f}MB), "
            f"errors={error_rate:.2%}, duration={batch_duration:.2f}s"
        )

        # Adjust batch size periodically
        time_since_adjustment = time.time() - self._last_adjustment_time
        if time_since_adjustment >= self._adjustment_interval:
            self._adjust_batch_size()
            self._last_adjustment_time = time.time()

    def _adjust_batch_size(self) -> None:
        """Adjust batch size based on recent metrics.

        Uses average of recent metrics to make informed decisions.
        """
        if not self._metrics_history:
            return

        # Calculate average metrics from recent history
        recent = self._metrics_history[-10:]  # Last 10 batches

        avg_rate = sum(m.processing_rate for m in recent) / len(recent)
        avg_memory = sum(m.memory_usage_mb for m in recent) / len(recent)
        avg_error_rate = sum(m.error_rate for m in recent) / len(recent)
        avg_memory_trend = sum(m.memory_trend for m in recent) / len(recent)

        metrics_dict = {
            "processing_rate": avg_rate,
            "memory_usage_mb": avg_memory,
            "error_rate": avg_error_rate,
            "memory_trend": avg_memory_trend,
        }

        # Calculate new optimal size
        new_size = self.calculate_optimal_batch_size(metrics_dict)

        if new_size != self.current_batch_size:
            logger.info(
                f"Adjusted batch size: {self.current_batch_size} -> {new_size} "
                f"based on recent performance"
            )
            self.current_batch_size = new_size

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of performance metrics and current settings.

        Returns:
            Dictionary with performance statistics
        """
        avg_rate = (
            self._total_emails / self._total_duration
            if self._total_duration > 0
            else 0.0
        )

        recent_metrics = self._metrics_history[-10:] if self._metrics_history else []
        recent_avg_rate = (
            sum(m.processing_rate for m in recent_metrics) / len(recent_metrics)
            if recent_metrics
            else 0.0
        )

        return {
            "current_batch_size": self.current_batch_size,
            "batch_size_range": [self.min_batch_size, self.max_batch_size],
            "total_batches": self._total_batches,
            "total_emails": self._total_emails,
            "total_duration": self._total_duration,
            "avg_processing_rate": avg_rate,
            "recent_avg_rate": recent_avg_rate,
            "metrics_count": len(self._metrics_history),
            "healthy": (
                self._metrics_history[-1].is_healthy()
                if self._metrics_history
                else True
            ),
        }

    def reset_metrics(self) -> None:
        """Reset all performance metrics.

        Useful when starting a new processing session.
        """
        self._metrics_history.clear()
        self._total_batches = 0
        self._total_emails = 0
        self._total_duration = 0.0
        self._last_adjustment_time = time.time()
        self.current_batch_size = self.config.batch_size

        logger.info("Reset all metrics and batch size to initial value")
