"""Observability and metrics module for email processing pipeline.

This module provides:
- Structured JSON logging for all pipeline events
- Comprehensive metrics collection for performance tracking
- Prometheus-compatible metrics export
- Dashboard data generation for visualization
- Real-time observability with correlation IDs
- Alert generation for threshold-based notifications
"""

import json
import logging
import os
import psutil
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Event Types and Data Structures
# ============================================================================


class EventType(str, Enum):
    """Types of events that can be logged."""

    SUBGRAPH_START = "subgraph_start"
    SUBGRAPH_END = "subgraph_end"
    DECISION_MADE = "decision_made"
    ERROR_OCCURRED = "error_occurred"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    IMAP_OPERATION = "imap_operation"
    AI_CLASSIFICATION = "ai_classification"
    CHECKPOINT_UPDATED = "checkpoint_updated"
    ALERT_TRIGGERED = "alert_triggered"


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StructuredEvent:
    """Structured event for logging."""

    event_type: EventType
    timestamp: datetime
    correlation_id: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "context": self.context,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Alert:
    """Alert for threshold-based notifications."""

    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "context": self.context,
        }


# ============================================================================
# Structured Logger
# ============================================================================


class StructuredLogger:
    """JSON-formatted structured logging for pipeline events.

    Logs events in JSON format for easy ingestion by log aggregation tools
    like ELK stack, Splunk, or CloudWatch.

    Features:
    - JSON-formatted logs for machine parsing
    - Event types for categorization
    - Context enrichment (uid, sender, timestamps, etc.)
    - Correlation IDs for tracing through parallel subgraphs

    Example:
        logger = StructuredLogger(log_file="events.jsonl")
        logger.log_subgraph_start(
            correlation_id="123",
            subgraph_name="email_processing",
            uid="12345",
            sender="user@example.com"
        )
    """

    def __init__(
        self,
        log_file: str | None = None,
        console_output: bool = True,
        log_level: int = logging.INFO,
    ):
        """Initialize structured logger.

        Args:
            log_file: Optional path to JSON lines log file
            console_output: Whether to also log to console
            log_level: Minimum log level for console output
        """
        self.log_file = Path(log_file) if log_file else None
        self.console_output = console_output
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"{__name__}.StructuredLogger")
        self._logger.setLevel(log_level)

        # Ensure log directory exists
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write_event(self, event: StructuredEvent):
        """Write event to log destinations.

        Args:
            event: Structured event to log
        """
        json_line = event.to_json()

        with self._lock:
            # Write to file
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(json_line + "\n")

            # Write to console
            if self.console_output:
                self._logger.info(json_line)

    def log_subgraph_start(
        self, correlation_id: str, subgraph_name: str, **context
    ):
        """Log start of subgraph execution.

        Args:
            correlation_id: Unique ID for tracing this execution
            subgraph_name: Name of the subgraph
            **context: Additional context (uid, sender, etc.)
        """
        event = StructuredEvent(
            event_type=EventType.SUBGRAPH_START,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            context={
                "subgraph_name": subgraph_name,
                **context,
            },
        )
        self._write_event(event)

    def log_subgraph_end(
        self,
        correlation_id: str,
        subgraph_name: str,
        duration: float,
        **context,
    ):
        """Log end of subgraph execution.

        Args:
            correlation_id: Unique ID for tracing this execution
            subgraph_name: Name of the subgraph
            duration: Execution duration in seconds
            **context: Additional context (decision, reason, etc.)
        """
        event = StructuredEvent(
            event_type=EventType.SUBGRAPH_END,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            context={
                "subgraph_name": subgraph_name,
                "duration": duration,
                **context,
            },
        )
        self._write_event(event)

    def log_decision_made(
        self,
        correlation_id: str,
        uid: str,
        sender: str,
        decision: str,
        reason: str,
        confidence: float,
        **context,
    ):
        """Log email classification decision.

        Args:
            correlation_id: Unique ID for tracing
            uid: Email UID
            sender: Email sender
            decision: Decision made (delete/keep/uncertain)
            reason: Reason for decision
            confidence: Confidence score
            **context: Additional context
        """
        event = StructuredEvent(
            event_type=EventType.DECISION_MADE,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            context={
                "uid": uid,
                "sender": sender,
                "decision": decision,
                "reason": reason,
                "confidence": confidence,
                **context,
            },
        )
        self._write_event(event)

    def log_error_occurred(
        self, correlation_id: str, error_type: str, error_message: str, **context
    ):
        """Log error occurrence.

        Args:
            correlation_id: Unique ID for tracing
            error_type: Type of error
            error_message: Error message
            **context: Additional context (uid, sender, stack trace, etc.)
        """
        event = StructuredEvent(
            event_type=EventType.ERROR_OCCURRED,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            context={
                "error_type": error_type,
                "error_message": error_message,
                **context,
            },
        )
        self._write_event(event)

    def log_batch_start(self, correlation_id: str, batch_size: int, **context):
        """Log start of batch processing.

        Args:
            correlation_id: Unique ID for tracing
            batch_size: Number of emails in batch
            **context: Additional context
        """
        event = StructuredEvent(
            event_type=EventType.BATCH_START,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            context={
                "batch_size": batch_size,
                **context,
            },
        )
        self._write_event(event)

    def log_batch_end(
        self, correlation_id: str, batch_size: int, duration: float, **context
    ):
        """Log end of batch processing.

        Args:
            correlation_id: Unique ID for tracing
            batch_size: Number of emails processed
            duration: Processing duration in seconds
            **context: Additional context
        """
        event = StructuredEvent(
            event_type=EventType.BATCH_END,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            context={
                "batch_size": batch_size,
                "duration": duration,
                "throughput": batch_size / duration if duration > 0 else 0,
                **context,
            },
        )
        self._write_event(event)

    def log_imap_operation(
        self, correlation_id: str, operation: str, duration: float, **context
    ):
        """Log IMAP operation.

        Args:
            correlation_id: Unique ID for tracing
            operation: Operation name (fetch_headers, fetch_bodies, delete, etc.)
            duration: Operation duration in seconds
            **context: Additional context
        """
        event = StructuredEvent(
            event_type=EventType.IMAP_OPERATION,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            context={
                "operation": operation,
                "duration": duration,
                **context,
            },
        )
        self._write_event(event)

    def log_ai_classification(
        self, correlation_id: str, uid: str, duration: float, **context
    ):
        """Log AI classification operation.

        Args:
            correlation_id: Unique ID for tracing
            uid: Email UID
            duration: Classification duration in seconds
            **context: Additional context (decision, confidence, etc.)
        """
        event = StructuredEvent(
            event_type=EventType.AI_CLASSIFICATION,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            context={
                "uid": uid,
                "duration": duration,
                **context,
            },
        )
        self._write_event(event)


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """Collect and track metrics for pipeline performance.

    Tracks:
    - Parallel execution timing per subgraph
    - Decision distribution (delete/keep/uncertain)
    - Sender statistics trends
    - Processing throughput
    - Memory usage per batch
    - Error rates by type
    - IMAP operation latencies
    - AI classification latencies

    Example:
        collector = MetricsCollector()
        with collector.track_operation("imap_fetch"):
            headers = client.batch_fetch_headers(100)

        collector.record_decision("delete")
        metrics = collector.export_prometheus()
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._lock = threading.Lock()

        # Counters
        self._emails_processed = 0
        self._emails_deleted = 0
        self._emails_kept = 0
        self._emails_uncertain = 0
        self._errors_total = 0

        # Error tracking by type
        self._errors_by_type: dict[str, int] = defaultdict(int)

        # Operation timing (name -> list of durations)
        self._operation_durations: dict[str, list[float]] = defaultdict(list)

        # Subgraph timing
        self._subgraph_durations: dict[str, list[float]] = defaultdict(list)

        # IMAP operation timing
        self._imap_operation_durations: dict[str, list[float]] = defaultdict(list)

        # AI classification timing
        self._ai_classification_durations: list[float] = []

        # Sender statistics
        self._sender_email_counts: dict[str, int] = defaultdict(int)

        # Throughput tracking (timestamp -> count)
        self._processing_timeline: list[tuple[float, int]] = []

        # Memory tracking (timestamp -> memory_mb)
        self._memory_timeline: list[tuple[float, float]] = []

        # Active parallel subgraphs
        self._active_subgraphs = 0

        # Start time for rate calculations
        self._start_time = time.time()

    def record_emails_processed(self, count: int = 1):
        """Record processed emails.

        Args:
            count: Number of emails processed
        """
        with self._lock:
            self._emails_processed += count
            self._processing_timeline.append((time.time(), count))

    def record_decision(self, decision: str):
        """Record classification decision.

        Args:
            decision: Decision type (delete, keep, uncertain)
        """
        with self._lock:
            if decision.lower() == "delete":
                self._emails_deleted += 1
            elif decision.lower() == "keep":
                self._emails_kept += 1
            elif decision.lower() == "uncertain":
                self._emails_uncertain += 1

    def record_error(self, error_type: str = "unknown"):
        """Record error occurrence.

        Args:
            error_type: Type of error
        """
        with self._lock:
            self._errors_total += 1
            self._errors_by_type[error_type] += 1

    def record_sender(self, sender: str):
        """Record sender for statistics.

        Args:
            sender: Email sender address
        """
        with self._lock:
            self._sender_email_counts[sender] += 1

    def record_operation_duration(self, operation: str, duration: float):
        """Record operation duration.

        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        with self._lock:
            self._operation_durations[operation].append(duration)

    def record_subgraph_duration(self, subgraph_name: str, duration: float):
        """Record subgraph execution duration.

        Args:
            subgraph_name: Name of the subgraph
            duration: Duration in seconds
        """
        with self._lock:
            self._subgraph_durations[subgraph_name].append(duration)

    def record_imap_operation(self, operation: str, duration: float):
        """Record IMAP operation latency.

        Args:
            operation: IMAP operation name
            duration: Duration in seconds
        """
        with self._lock:
            self._imap_operation_durations[operation].append(duration)

    def record_ai_classification(self, duration: float):
        """Record AI classification latency.

        Args:
            duration: Duration in seconds
        """
        with self._lock:
            self._ai_classification_durations.append(duration)

    def record_memory_usage(self, memory_mb: float | None = None):
        """Record current memory usage.

        Args:
            memory_mb: Memory usage in MB (auto-detected if None)
        """
        if memory_mb is None:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

        with self._lock:
            self._memory_timeline.append((time.time(), memory_mb))

    def increment_active_subgraphs(self):
        """Increment active parallel subgraph count."""
        with self._lock:
            self._active_subgraphs += 1

    def decrement_active_subgraphs(self):
        """Decrement active parallel subgraph count."""
        with self._lock:
            self._active_subgraphs = max(0, self._active_subgraphs - 1)

    @contextmanager
    def track_operation(self, operation: str):
        """Context manager to track operation timing.

        Args:
            operation: Operation name

        Example:
            with collector.track_operation("fetch_emails"):
                emails = client.batch_fetch_headers(100)
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.record_operation_duration(operation, duration)

    def get_summary(self) -> dict[str, Any]:
        """Get current metrics summary.

        Returns:
            Dictionary with current metrics
        """
        with self._lock:
            elapsed = time.time() - self._start_time
            processing_rate = self._emails_processed / elapsed if elapsed > 0 else 0

            return {
                "emails_processed": self._emails_processed,
                "emails_deleted": self._emails_deleted,
                "emails_kept": self._emails_kept,
                "emails_uncertain": self._emails_uncertain,
                "errors_total": self._errors_total,
                "errors_by_type": dict(self._errors_by_type),
                "processing_rate": processing_rate,
                "active_subgraphs": self._active_subgraphs,
                "elapsed_seconds": elapsed,
            }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string

        Example output:
            # HELP emails_processed_total Total emails processed
            # TYPE emails_processed_total counter
            emails_processed_total 150

            # HELP processing_duration_seconds Processing duration histogram
            # TYPE processing_duration_seconds histogram
            processing_duration_seconds_bucket{le="0.1"} 50
            processing_duration_seconds_bucket{le="0.5"} 120
            processing_duration_seconds_bucket{le="+Inf"} 150
            processing_duration_seconds_sum 45.5
            processing_duration_seconds_count 150
        """
        lines = []

        with self._lock:
            # Counters
            lines.extend(
                [
                    "# HELP emails_processed_total Total emails processed",
                    "# TYPE emails_processed_total counter",
                    f"emails_processed_total {self._emails_processed}",
                    "",
                    "# HELP emails_deleted_total Total emails deleted",
                    "# TYPE emails_deleted_total counter",
                    f"emails_deleted_total {self._emails_deleted}",
                    "",
                    "# HELP errors_total Total errors encountered",
                    "# TYPE errors_total counter",
                    f"errors_total {self._errors_total}",
                    "",
                ]
            )

            # Error breakdown by type
            for error_type, count in self._errors_by_type.items():
                lines.append(f'errors_total{{type="{error_type}"}} {count}')
            lines.append("")

            # Gauge: Active parallel subgraphs
            lines.extend(
                [
                    "# HELP parallel_subgraphs_active Currently active parallel subgraphs",
                    "# TYPE parallel_subgraphs_active gauge",
                    f"parallel_subgraphs_active {self._active_subgraphs}",
                    "",
                ]
            )

            # Gauge: Memory usage
            if self._memory_timeline:
                current_memory = self._memory_timeline[-1][1]
                lines.extend(
                    [
                        "# HELP memory_usage_bytes Current memory usage in bytes",
                        "# TYPE memory_usage_bytes gauge",
                        f"memory_usage_bytes {int(current_memory * 1024 * 1024)}",
                        "",
                    ]
                )

            # Histogram: Processing duration
            lines.extend(
                [
                    "# HELP processing_duration_seconds Processing duration histogram",
                    "# TYPE processing_duration_seconds histogram",
                ]
            )
            all_durations = []
            for durations in self._operation_durations.values():
                all_durations.extend(durations)

            if all_durations:
                histogram = self._create_histogram(all_durations, "processing_duration_seconds")
                lines.extend(histogram)
            lines.append("")

            # Histogram: IMAP operation duration
            lines.extend(
                [
                    "# HELP imap_operation_duration_seconds IMAP operation latency histogram",
                    "# TYPE imap_operation_duration_seconds histogram",
                ]
            )
            for operation, durations in self._imap_operation_durations.items():
                if durations:
                    histogram = self._create_histogram(
                        durations,
                        "imap_operation_duration_seconds",
                        labels={"operation": operation},
                    )
                    lines.extend(histogram)
            lines.append("")

            # Histogram: AI classification duration
            if self._ai_classification_durations:
                lines.extend(
                    [
                        "# HELP ai_classification_duration_seconds AI classification latency histogram",
                        "# TYPE ai_classification_duration_seconds histogram",
                    ]
                )
                histogram = self._create_histogram(
                    self._ai_classification_durations,
                    "ai_classification_duration_seconds",
                )
                lines.extend(histogram)
                lines.append("")

        return "\n".join(lines)

    def _create_histogram(
        self,
        values: list[float],
        metric_name: str,
        buckets: list[float] | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[str]:
        """Create Prometheus histogram format.

        Args:
            values: List of observed values
            metric_name: Metric name
            buckets: Bucket boundaries (default: [0.1, 0.5, 1.0, 5.0, 10.0])
            labels: Optional labels dict

        Returns:
            List of histogram lines
        """
        if buckets is None:
            buckets = [0.1, 0.5, 1.0, 5.0, 10.0]

        label_str = ""
        if labels:
            label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
            label_str = "{" + ",".join(label_pairs) + "}"

        lines = []
        total = len(values)
        sum_val = sum(values)

        # Bucket counts
        for bucket in buckets:
            count = sum(1 for v in values if v <= bucket)
            lines.append(f'{metric_name}_bucket{label_str}{{le="{bucket}"}} {count}')

        lines.append(f'{metric_name}_bucket{label_str}{{le="+Inf"}} {total}')
        lines.append(f"{metric_name}_sum{label_str} {sum_val:.6f}")
        lines.append(f"{metric_name}_count{label_str} {total}")

        return lines

    def generate_dashboard_data(self) -> dict[str, Any]:
        """Generate data for dashboard visualization.

        Returns:
            Dictionary with dashboard-ready data including:
            - Processing rate over time (time series)
            - Decision breakdown (pie chart data)
            - Top senders by volume (bar chart data)
            - Error trends (time series)
            - Performance metrics (processing time, throughput)
        """
        with self._lock:
            elapsed = time.time() - self._start_time

            # Processing rate over time (time series)
            processing_timeline = []
            cumulative = 0
            for timestamp, count in self._processing_timeline:
                cumulative += count
                processing_timeline.append(
                    {
                        "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                        "emails_processed": cumulative,
                        "rate": count / ((timestamp - self._start_time) or 1),
                    }
                )

            # Decision breakdown (pie chart)
            decision_breakdown = {
                "delete": self._emails_deleted,
                "keep": self._emails_kept,
                "uncertain": self._emails_uncertain,
            }

            # Top senders by volume (bar chart)
            top_senders = sorted(
                self._sender_email_counts.items(), key=lambda x: x[1], reverse=True
            )[:20]
            sender_chart_data = [
                {"sender": sender, "count": count} for sender, count in top_senders
            ]

            # Error trends (time series) - simplified from error timeline
            error_breakdown = [
                {"error_type": error_type, "count": count}
                for error_type, count in self._errors_by_type.items()
            ]

            # Performance metrics
            avg_processing_time = 0.0
            if self._operation_durations:
                all_durations = []
                for durations in self._operation_durations.values():
                    all_durations.extend(durations)
                if all_durations:
                    avg_processing_time = sum(all_durations) / len(all_durations)

            throughput = self._emails_processed / elapsed if elapsed > 0 else 0

            # Memory usage over time
            memory_timeline = [
                {
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                    "memory_mb": memory_mb,
                }
                for timestamp, memory_mb in self._memory_timeline
            ]

            # IMAP latency stats
            imap_latency_stats = {}
            for operation, durations in self._imap_operation_durations.items():
                if durations:
                    imap_latency_stats[operation] = {
                        "avg": sum(durations) / len(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "count": len(durations),
                    }

            # AI classification stats
            ai_classification_stats = {}
            if self._ai_classification_durations:
                ai_classification_stats = {
                    "avg": sum(self._ai_classification_durations)
                    / len(self._ai_classification_durations),
                    "min": min(self._ai_classification_durations),
                    "max": max(self._ai_classification_durations),
                    "count": len(self._ai_classification_durations),
                }

            return {
                "summary": {
                    "total_processed": self._emails_processed,
                    "total_deleted": self._emails_deleted,
                    "total_kept": self._emails_kept,
                    "total_uncertain": self._emails_uncertain,
                    "total_errors": self._errors_total,
                    "elapsed_seconds": elapsed,
                    "throughput": throughput,
                    "avg_processing_time": avg_processing_time,
                },
                "processing_timeline": processing_timeline,
                "decision_breakdown": decision_breakdown,
                "top_senders": sender_chart_data,
                "error_breakdown": error_breakdown,
                "memory_timeline": memory_timeline,
                "imap_latency_stats": imap_latency_stats,
                "ai_classification_stats": ai_classification_stats,
            }

    def export_dashboard_json(self, output_file: str):
        """Export dashboard data to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        dashboard_data = self.generate_dashboard_data()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(dashboard_data, f, indent=2)

        logger.info(f"Dashboard data exported to {output_file}")


# ============================================================================
# Observability Context Manager
# ============================================================================


class ObservabilityContext:
    """Context manager for operation tracking with automatic timing and logging.

    Integrates StructuredLogger and MetricsCollector for comprehensive
    observability with minimal code changes.

    Features:
    - Automatic timing of operations
    - Correlation IDs for tracing
    - Structured logging
    - Metrics collection
    - Error tracking

    Example:
        obs = ObservabilityContext(logger, collector)

        with obs.track("email_processing", uid="12345", sender="user@example.com") as ctx:
            # Process email
            decision = classify_email(email)
            ctx.add_context(decision=decision)
    """

    def __init__(
        self,
        structured_logger: StructuredLogger,
        metrics_collector: MetricsCollector,
        checkpoint_manager: Any = None,
    ):
        """Initialize observability context.

        Args:
            structured_logger: Structured logger instance
            metrics_collector: Metrics collector instance
            checkpoint_manager: Optional checkpoint manager for persistence
        """
        self.logger = structured_logger
        self.collector = metrics_collector
        self.checkpoint_manager = checkpoint_manager

    @contextmanager
    def track(self, operation: str, **initial_context):
        """Track an operation with automatic timing and logging.

        Args:
            operation: Operation name
            **initial_context: Initial context data

        Yields:
            Context object for adding additional context
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        context = {"correlation_id": correlation_id, **initial_context}

        # Log operation start
        self.logger.log_subgraph_start(
            correlation_id=correlation_id,
            subgraph_name=operation,
            **initial_context,
        )

        # Track active operation
        if "subgraph" in operation.lower():
            self.collector.increment_active_subgraphs()

        try:
            # Yield context object for adding data during execution
            yield OperationContext(context)

        except Exception as e:
            # Log error
            self.logger.log_error_occurred(
                correlation_id=correlation_id,
                error_type=type(e).__name__,
                error_message=str(e),
                **context,
            )
            self.collector.record_error(error_type=type(e).__name__)
            raise

        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Log operation end
            self.logger.log_subgraph_end(
                correlation_id=correlation_id,
                subgraph_name=operation,
                duration=duration,
                **context,
            )

            # Record metrics
            self.collector.record_operation_duration(operation, duration)

            if "subgraph" in operation.lower():
                self.collector.record_subgraph_duration(operation, duration)
                self.collector.decrement_active_subgraphs()


class OperationContext:
    """Context object for adding data during operation execution."""

    def __init__(self, context: dict[str, Any]):
        """Initialize operation context.

        Args:
            context: Context dictionary
        """
        self.context = context

    def add_context(self, **kwargs):
        """Add additional context data.

        Args:
            **kwargs: Context data to add
        """
        self.context.update(kwargs)


# ============================================================================
# Alert Manager
# ============================================================================


class AlertManager:
    """Threshold-based alert generation.

    Monitors metrics and generates alerts when thresholds are exceeded.

    Alert conditions:
    - Error rate exceeds threshold
    - Processing rate drops significantly
    - Memory usage exceeds limit
    - Too many uncertain decisions

    Alert outputs:
    - Log to file
    - Console output
    - Callback function (for external integrations)

    Example:
        alert_manager = AlertManager(
            error_rate_threshold=0.1,
            memory_limit_mb=1024,
            callback=send_slack_notification
        )

        alert_manager.check_error_rate(
            errors=10,
            total=100,
            context={"batch_id": "123"}
        )
    """

    def __init__(
        self,
        error_rate_threshold: float = 0.1,
        min_processing_rate: float = 1.0,
        memory_limit_mb: float = 2048,
        uncertain_rate_threshold: float = 0.3,
        alert_log_file: str | None = None,
        callback: Callable[[Alert], None] | None = None,
    ):
        """Initialize alert manager.

        Args:
            error_rate_threshold: Maximum acceptable error rate (0-1)
            min_processing_rate: Minimum processing rate (emails/sec)
            memory_limit_mb: Maximum memory usage in MB
            uncertain_rate_threshold: Maximum uncertain decision rate (0-1)
            alert_log_file: Optional file to log alerts
            callback: Optional callback function for alerts
        """
        self.error_rate_threshold = error_rate_threshold
        self.min_processing_rate = min_processing_rate
        self.memory_limit_mb = memory_limit_mb
        self.uncertain_rate_threshold = uncertain_rate_threshold
        self.alert_log_file = Path(alert_log_file) if alert_log_file else None
        self.callback = callback

        self._alerts: list[Alert] = []
        self._lock = threading.Lock()

        # Ensure alert log directory exists
        if self.alert_log_file:
            self.alert_log_file.parent.mkdir(parents=True, exist_ok=True)

    def _trigger_alert(self, alert: Alert):
        """Trigger an alert through all configured outputs.

        Args:
            alert: Alert to trigger
        """
        with self._lock:
            self._alerts.append(alert)

        # Log to file
        if self.alert_log_file:
            with open(self.alert_log_file, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")

        # Log to console
        logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")

        # Call callback
        if self.callback:
            try:
                self.callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def check_error_rate(
        self, errors: int, total: int, context: dict[str, Any] | None = None
    ):
        """Check if error rate exceeds threshold.

        Args:
            errors: Number of errors
            total: Total operations
            context: Optional context data
        """
        if total == 0:
            return

        error_rate = errors / total

        if error_rate > self.error_rate_threshold:
            alert = Alert(
                level=AlertLevel.ERROR,
                message=f"Error rate ({error_rate:.1%}) exceeds threshold ({self.error_rate_threshold:.1%})",
                timestamp=datetime.now(),
                metric_name="error_rate",
                current_value=error_rate,
                threshold=self.error_rate_threshold,
                context=context or {},
            )
            self._trigger_alert(alert)

    def check_processing_rate(
        self, current_rate: float, context: dict[str, Any] | None = None
    ):
        """Check if processing rate is too low.

        Args:
            current_rate: Current processing rate (emails/sec)
            context: Optional context data
        """
        if current_rate < self.min_processing_rate:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=f"Processing rate ({current_rate:.2f} emails/sec) is below minimum ({self.min_processing_rate:.2f} emails/sec)",
                timestamp=datetime.now(),
                metric_name="processing_rate",
                current_value=current_rate,
                threshold=self.min_processing_rate,
                context=context or {},
            )
            self._trigger_alert(alert)

    def check_memory_usage(
        self, memory_mb: float | None = None, context: dict[str, Any] | None = None
    ):
        """Check if memory usage exceeds limit.

        Args:
            memory_mb: Current memory usage in MB (auto-detected if None)
            context: Optional context data
        """
        if memory_mb is None:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

        if memory_mb > self.memory_limit_mb:
            alert = Alert(
                level=AlertLevel.CRITICAL,
                message=f"Memory usage ({memory_mb:.0f} MB) exceeds limit ({self.memory_limit_mb:.0f} MB)",
                timestamp=datetime.now(),
                metric_name="memory_usage_mb",
                current_value=memory_mb,
                threshold=self.memory_limit_mb,
                context=context or {},
            )
            self._trigger_alert(alert)

    def check_uncertain_rate(
        self, uncertain: int, total: int, context: dict[str, Any] | None = None
    ):
        """Check if uncertain decision rate is too high.

        Args:
            uncertain: Number of uncertain decisions
            total: Total decisions
            context: Optional context data
        """
        if total == 0:
            return

        uncertain_rate = uncertain / total

        if uncertain_rate > self.uncertain_rate_threshold:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=f"Uncertain decision rate ({uncertain_rate:.1%}) exceeds threshold ({self.uncertain_rate_threshold:.1%})",
                timestamp=datetime.now(),
                metric_name="uncertain_rate",
                current_value=uncertain_rate,
                threshold=self.uncertain_rate_threshold,
                context=context or {},
            )
            self._trigger_alert(alert)

    def get_alerts(
        self, level: AlertLevel | None = None, since: datetime | None = None
    ) -> list[Alert]:
        """Get triggered alerts with optional filtering.

        Args:
            level: Filter by alert level
            since: Filter by timestamp

        Returns:
            List of alerts
        """
        with self._lock:
            alerts = self._alerts.copy()

        if level:
            alerts = [a for a in alerts if a.level == level]

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        return alerts

    def clear_alerts(self):
        """Clear all stored alerts."""
        with self._lock:
            self._alerts.clear()


# ============================================================================
# HTTP Endpoint for Prometheus (Optional)
# ============================================================================


def create_prometheus_endpoint(
    metrics_collector: MetricsCollector, host: str = "0.0.0.0", port: int = 9090
):
    """Create HTTP endpoint for Prometheus scraping.

    This is optional and requires the 'http.server' module.

    Args:
        metrics_collector: MetricsCollector instance
        host: Host to bind to
        port: Port to bind to

    Example:
        collector = MetricsCollector()
        # Run in background thread
        thread = threading.Thread(
            target=create_prometheus_endpoint,
            args=(collector,),
            daemon=True
        )
        thread.start()
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                metrics = metrics_collector.export_prometheus()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(metrics.encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress default logging
            pass

    server = HTTPServer((host, port), MetricsHandler)
    logger.info(f"Prometheus metrics endpoint running at http://{host}:{port}/metrics")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Prometheus endpoint")
        server.shutdown()
