"""Error handling and retry logic for robust email processing.

This module provides:
- Exponential backoff for transient failures
- Partial batch processing with fallback
- Email quarantine system for problematic messages
- Error categorization and recovery strategies
- Circuit breaker pattern for repeated failures
"""

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorCategory(str, Enum):
    """Category of error for determining recovery strategy."""

    TRANSIENT = "transient"  # Network issues, timeouts → Retry
    PERMANENT = "permanent"  # Invalid format, missing fields → Quarantine
    RATE_LIMIT = "rate_limit"  # Rate limiting → Backoff and retry
    AUTHENTICATION = "authentication"  # Auth failures → Token refresh
    PARSE = "parse"  # Email parsing failures → Quarantine
    IMAP = "imap"  # IMAP connection errors → Reconnect and retry
    OLLAMA = "ollama"  # Ollama/AI errors → Use fallback decision
    CHECKPOINT = "checkpoint"  # Checkpoint errors → Continue without checkpointing
    DATABASE = "database"  # Database errors → Retry with backoff
    UNKNOWN = "unknown"  # Unknown errors → Log and quarantine


class RecoveryStrategy(str, Enum):
    """Recovery strategy for different error types."""

    RETRY = "retry"  # Retry with backoff
    RETRY_WITH_RECONNECT = "retry_with_reconnect"  # Reconnect and retry
    RETRY_WITH_REFRESH = "retry_with_refresh"  # Refresh token and retry
    QUARANTINE = "quarantine"  # Move to quarantine
    FALLBACK = "fallback"  # Use fallback value/decision
    SKIP = "skip"  # Skip and continue
    FAIL = "fail"  # Fail immediately


class CircuitState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Too many failures, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class ErrorTaxonomy:
    """Error taxonomy mapping error types to recovery strategies."""

    # Map error patterns to categories
    ERROR_PATTERNS = {
        ErrorCategory.TRANSIENT: [
            "timeout",
            "connection reset",
            "connection refused",
            "temporary failure",
            "network",
            "socket",
        ],
        ErrorCategory.RATE_LIMIT: [
            "rate limit",
            "too many requests",
            "quota exceeded",
            "throttled",
        ],
        ErrorCategory.AUTHENTICATION: [
            "authentication failed",
            "authenticationfailed",
            "invalid credentials",
            "token expired",
            "unauthorized",
        ],
        ErrorCategory.PARSE: [
            "parse error",
            "invalid format",
            "decode error",
            "encoding",
            "malformed",
        ],
        ErrorCategory.IMAP: [
            "imap",
            "abort",
            "bye",
            "protocol error",
        ],
        ErrorCategory.OLLAMA: [
            "ollama",
            "model",
            "inference",
            "llm",
        ],
        ErrorCategory.CHECKPOINT: [
            "checkpoint",
            "sqlite",
            "database locked",
        ],
        ErrorCategory.DATABASE: [
            "database",
            "sql",
            "integrity error",
        ],
    }

    # Map categories to recovery strategies
    RECOVERY_STRATEGIES = {
        ErrorCategory.TRANSIENT: RecoveryStrategy.RETRY,
        ErrorCategory.PERMANENT: RecoveryStrategy.QUARANTINE,
        ErrorCategory.RATE_LIMIT: RecoveryStrategy.RETRY,
        ErrorCategory.AUTHENTICATION: RecoveryStrategy.RETRY_WITH_REFRESH,
        ErrorCategory.PARSE: RecoveryStrategy.QUARANTINE,
        ErrorCategory.IMAP: RecoveryStrategy.RETRY_WITH_RECONNECT,
        ErrorCategory.OLLAMA: RecoveryStrategy.FALLBACK,
        ErrorCategory.CHECKPOINT: RecoveryStrategy.SKIP,
        ErrorCategory.DATABASE: RecoveryStrategy.RETRY,
        ErrorCategory.UNKNOWN: RecoveryStrategy.QUARANTINE,
    }

    @classmethod
    def categorize_error(cls, error: Exception) -> ErrorCategory:
        """Categorize an error based on its message and type.

        Args:
            error: Exception to categorize

        Returns:
            ErrorCategory for the error
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        for category, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_str or pattern in error_type:
                    return category

        return ErrorCategory.UNKNOWN

    @classmethod
    def get_recovery_strategy(cls, error: Exception) -> RecoveryStrategy:
        """Get recovery strategy for an error.

        Args:
            error: Exception to get recovery strategy for

        Returns:
            RecoveryStrategy for the error
        """
        category = cls.categorize_error(error)
        return cls.RECOVERY_STRATEGIES[category]


class CircuitBreaker:
    """Circuit breaker pattern for repeated failures.

    Prevents cascading failures by stopping requests when error rate is too high.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class QuarantineManager:
    """Manages quarantined emails with SQLite storage using Peewee ORM."""

    def __init__(self, quarantine_path: str = "quarantine.db"):
        """Initialize quarantine manager.

        Args:
            quarantine_path: Path to SQLite quarantine database
        """
        from peewee import SqliteDatabase

        from inbox_reaper.models import QuarantineModel

        self.quarantine_path = Path(quarantine_path)

        # Initialize database
        self.db = SqliteDatabase(str(self.quarantine_path))
        QuarantineModel._meta.database = self.db

        self._init_database()

    def _init_database(self) -> None:
        """Initialize quarantine database schema."""
        from inbox_reaper.models import QuarantineModel

        self.db.connect(reuse_if_open=True)
        self.db.create_tables([QuarantineModel], safe=True)
        logger.info(f"Quarantine database initialized at {self.quarantine_path}")

    def add_to_quarantine(
        self,
        uid: str,
        email_data: dict,
        error: Exception,
        error_category: ErrorCategory | None = None,
    ) -> None:
        """Add an email to quarantine.

        Args:
            uid: Email UID
            email_data: Complete email data dictionary
            error: Exception that caused quarantine
            error_category: Optional error category
        """
        import json

        from inbox_reaper.models import QuarantineModel

        if error_category is None:
            error_category = ErrorTaxonomy.categorize_error(error)

        email_json = json.dumps(email_data)

        try:
            # Check if already quarantined
            existing = QuarantineModel.get(QuarantineModel.uid == uid)
            # Update existing quarantine record
            attempts = existing.attempts + 1
            QuarantineModel.update(
                attempts=attempts,
                error_type=error_category.value,
                error_message=str(error),
                last_retry_at=datetime.now().isoformat(),
            ).where(QuarantineModel.uid == uid).execute()
            logger.warning(
                f"Updated quarantine for UID {uid} (attempt {attempts}): {error}"
            )
        except QuarantineModel.DoesNotExist:
            # Insert new quarantine record
            QuarantineModel.create(
                uid=uid,
                email_data=email_json,
                error_type=error_category.value,
                error_message=str(error),
                quarantined_at=datetime.now().isoformat(),
            )
            logger.warning(f"Quarantined email UID {uid}: {error}")

    def get_quarantined_emails(
        self, error_type: ErrorCategory | None = None, resolved: bool = False
    ) -> list[dict]:
        """Get quarantined emails.

        Args:
            error_type: Optional filter by error type
            resolved: If True, get resolved emails; if False, get unresolved

        Returns:
            List of quarantined email records
        """
        import json

        from inbox_reaper.models import QuarantineModel

        query = QuarantineModel.select().where(QuarantineModel.resolved == resolved)

        if error_type:
            query = query.where(QuarantineModel.error_type == error_type.value)

        query = query.order_by(QuarantineModel.quarantined_at.desc())

        results = []
        for record in query:
            results.append({
                "uid": record.uid,
                "email_data": json.loads(record.email_data),
                "error_type": record.error_type,
                "error_message": record.error_message,
                "attempts": record.attempts,
                "quarantined_at": record.quarantined_at,
                "last_retry_at": record.last_retry_at,
                "resolved": record.resolved,
            })

        return results

    def mark_resolved(self, uid: str) -> None:
        """Mark a quarantined email as resolved.

        Args:
            uid: Email UID to mark as resolved
        """
        from inbox_reaper.models import QuarantineModel

        QuarantineModel.update(resolved=True).where(
            QuarantineModel.uid == uid
        ).execute()
        logger.info(f"Marked quarantined email {uid} as resolved")

    def export_quarantine(self, output_path: str) -> None:
        """Export quarantine to JSON file for manual review.

        Args:
            output_path: Path to output JSON file
        """
        import json

        emails = self.get_quarantined_emails()

        with open(output_path, "w") as f:
            json.dump(emails, f, indent=2, default=str)

        logger.info(f"Exported {len(emails)} quarantined emails to {output_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get quarantine statistics.

        Returns:
            Dictionary with quarantine stats
        """
        from peewee import fn

        from inbox_reaper.models import QuarantineModel

        # Total counts
        total_quarantined = (
            QuarantineModel.select()
            .where(~QuarantineModel.resolved)
            .count()
        )

        total_resolved = (
            QuarantineModel.select()
            .where(QuarantineModel.resolved)
            .count()
        )

        # By error type
        by_error_type_query = (
            QuarantineModel.select(
                QuarantineModel.error_type,
                fn.COUNT(QuarantineModel.uid).alias("count"),
            )
            .where(~QuarantineModel.resolved)
            .group_by(QuarantineModel.error_type)
        )
        by_error_type = {row.error_type: row.count for row in by_error_type_query}

        # High attempt emails
        high_attempts = (
            QuarantineModel.select()
            .where((QuarantineModel.attempts >= 3) & (~QuarantineModel.resolved))
            .count()
        )

        return {
            "total_quarantined": total_quarantined,
            "total_resolved": total_resolved,
            "by_error_type": by_error_type,
            "high_attempts": high_attempts,
        }


class ErrorHandler:
    """Comprehensive error handling with retry logic and quarantine."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        quarantine_path: str = "quarantine.db",
    ):
        """Initialize error handler.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            quarantine_path: Path to quarantine database
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        self.quarantine_manager = QuarantineManager(quarantine_path)
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

    def retry_with_backoff(
        self,
        func: Callable[..., T],
        *args: Any,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        **kwargs: Any,
    ) -> T:
        """Retry a function with exponential backoff.

        Args:
            func: Function to retry
            *args: Positional arguments for function
            max_retries: Override default max retries
            base_delay: Override default base delay
            max_delay: Override default max delay
            **kwargs: Keyword arguments for function

        Returns:
            Result of successful function call

        Raises:
            Exception: If all retries fail
        """
        max_retries = max_retries or self.max_retries
        base_delay = base_delay or self.base_delay
        max_delay = max_delay or self.max_delay

        last_exception = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                category = ErrorTaxonomy.categorize_error(e)
                strategy = ErrorTaxonomy.get_recovery_strategy(e)

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e} "
                    f"(category: {category.value}, strategy: {strategy.value})"
                )

                # Don't retry if strategy is not RETRY
                if strategy not in [
                    RecoveryStrategy.RETRY,
                    RecoveryStrategy.RETRY_WITH_RECONNECT,
                ]:
                    raise e

                # Calculate backoff delay
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    # Add jitter to prevent thundering herd
                    import random

                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

        raise last_exception or Exception("All retries failed")

    @contextmanager
    def with_retry(
        self,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
    ):
        """Context manager for automatic retry.

        Args:
            max_retries: Override default max retries
            base_delay: Override default base delay
            max_delay: Override default max delay

        Yields:
            None

        Example:
            with error_handler.with_retry():
                risky_operation()
        """
        max_retries = max_retries or self.max_retries
        base_delay = base_delay or self.base_delay
        max_delay = max_delay or self.max_delay

        last_exception = None

        for attempt in range(max_retries):
            try:
                yield
                return
            except Exception as e:
                last_exception = e
                category = ErrorTaxonomy.categorize_error(e)
                strategy = ErrorTaxonomy.get_recovery_strategy(e)

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e} "
                    f"(category: {category.value}, strategy: {strategy.value})"
                )

                # Don't retry if strategy is not RETRY
                if strategy not in [
                    RecoveryStrategy.RETRY,
                    RecoveryStrategy.RETRY_WITH_RECONNECT,
                ]:
                    raise e

                # Calculate backoff delay
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    import random

                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

        raise last_exception or Exception("All retries failed")

    def process_batch_with_fallback(
        self,
        emails: list[dict],
        process_fn: Callable[[dict], Any],
        quarantine_on_failure: bool = True,
    ) -> tuple[list[Any], list[dict]]:
        """Process batch with fallback to individual processing.

        On batch failure, retry individual emails to maximize successful processing.

        Args:
            emails: List of email dictionaries to process
            process_fn: Function to process each email
            quarantine_on_failure: Whether to quarantine failed emails

        Returns:
            Tuple of (successful_results, failed_emails)
        """
        if not emails:
            return [], []

        successful_results = []
        failed_emails = []

        # Try batch processing first
        try:
            logger.info(f"Processing batch of {len(emails)} emails")
            results = process_fn(emails)  # type: ignore[arg-type]
            # Ensure results is a list for return type
            results_list = results if isinstance(results, list) else [results]
            return results_list, []
        except Exception as batch_error:
            logger.warning(
                f"Batch processing failed: {batch_error}. "
                f"Falling back to individual processing..."
            )

            # Fallback to individual processing
            for email in emails:
                uid = email.get("uid", "unknown")
                try:
                    result = process_fn(email)
                    successful_results.append(result)
                except Exception as email_error:
                    category = ErrorTaxonomy.categorize_error(email_error)
                    strategy = ErrorTaxonomy.get_recovery_strategy(email_error)

                    logger.error(
                        f"Failed to process email {uid}: {email_error} "
                        f"(category: {category.value}, strategy: {strategy.value})"
                    )

                    failed_emails.append(
                        {
                            "email": email,
                            "error": str(email_error),
                            "error_type": category.value,
                        }
                    )

                    # Quarantine if requested and strategy indicates
                    if (
                        quarantine_on_failure
                        and strategy == RecoveryStrategy.QUARANTINE
                    ):
                        self.quarantine_manager.add_to_quarantine(
                            uid, email, email_error, category
                        )

        logger.info(
            f"Batch processing complete: {len(successful_results)} successful, "
            f"{len(failed_emails)} failed"
        )

        return successful_results, failed_emails

    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a named operation.

        Args:
            name: Name of the operation
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again

        Returns:
            CircuitBreaker for the operation
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return self.circuit_breakers[name]

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        email_data: dict | None = None,
    ) -> dict[str, Any]:
        """Handle an error with appropriate recovery strategy.

        Args:
            error: Exception to handle
            context: Optional context information
            email_data: Optional email data for quarantine

        Returns:
            Dictionary with error handling result:
            - category: ErrorCategory
            - strategy: RecoveryStrategy
            - message: Human-readable message
            - quarantined: Whether email was quarantined
        """
        category = ErrorTaxonomy.categorize_error(error)
        strategy = ErrorTaxonomy.get_recovery_strategy(error)

        context = context or {}
        quarantined = False

        # Log the error
        logger.error(
            f"Handling error: {error} "
            f"(category: {category.value}, strategy: {strategy.value})"
        )

        # Execute recovery strategy
        if strategy == RecoveryStrategy.QUARANTINE and email_data:
            uid = email_data.get("uid", "unknown")
            self.quarantine_manager.add_to_quarantine(uid, email_data, error, category)
            quarantined = True

        return {
            "category": category.value,
            "strategy": strategy.value,
            "message": str(error),
            "quarantined": quarantined,
            "context": context,
        }

    def retry_quarantined_emails(
        self,
        process_fn: Callable[[dict], Any],
        error_type: ErrorCategory | None = None,
        max_attempts: int = 3,
    ) -> tuple[list[Any], list[dict]]:
        """Retry processing quarantined emails.

        Args:
            process_fn: Function to process emails
            error_type: Optional filter by error type
            max_attempts: Only retry emails with fewer attempts than this

        Returns:
            Tuple of (successful_results, still_failed)
        """
        quarantined = self.quarantine_manager.get_quarantined_emails(
            error_type=error_type, resolved=False
        )

        # Filter by max attempts
        to_retry = [q for q in quarantined if q["attempts"] < max_attempts]

        if not to_retry:
            logger.info("No quarantined emails to retry")
            return [], []

        logger.info(f"Retrying {len(to_retry)} quarantined emails")

        successful_results = []
        still_failed = []

        for quarantined_item in to_retry:
            uid = quarantined_item["uid"]
            email_data = quarantined_item["email_data"]

            try:
                result = self.retry_with_backoff(process_fn, email_data)
                successful_results.append(result)
                self.quarantine_manager.mark_resolved(uid)
                logger.info(f"Successfully processed quarantined email {uid}")
            except Exception as e:
                logger.error(f"Failed to retry quarantined email {uid}: {e}")
                # Re-add to quarantine (increments attempt counter)
                self.quarantine_manager.add_to_quarantine(uid, email_data, e)
                still_failed.append(quarantined_item)

        logger.info(
            f"Quarantine retry complete: {len(successful_results)} successful, "
            f"{len(still_failed)} still failed"
        )

        return successful_results, still_failed


def with_error_handling(
    error_handler: ErrorHandler | None = None,
    quarantine_on_failure: bool = False,
):
    """Decorator for automatic error handling.

    Args:
        error_handler: ErrorHandler instance to use
        quarantine_on_failure: Whether to quarantine on failure

    Returns:
        Decorated function

    Example:
        @with_error_handling(error_handler)
        def process_email(email: dict) -> dict:
            # Process email
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()

            try:
                return func(*args, **kwargs)
            except Exception as e:
                email_data = None
                if args and isinstance(args[0], dict) and "uid" in args[0]:
                    email_data = args[0]

                result = handler.handle_error(
                    e,
                    context={"function": func.__name__},
                    email_data=email_data if quarantine_on_failure else None,
                )

                # Re-raise unless strategy is SKIP or FALLBACK
                strategy = result["strategy"]
                if strategy not in [
                    RecoveryStrategy.SKIP.value,
                    RecoveryStrategy.FALLBACK.value,
                ]:
                    raise e

                return None

        return wrapper

    return decorator
