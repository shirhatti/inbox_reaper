"""Performance optimization and tuning module.

This module provides:
- Auto-tuning of configuration parameters based on system resources
- Benchmarking utilities for AI and IMAP operations
- Performance monitoring and profiling
- IMAP connection pooling for concurrent operations
- Memory usage tracking
"""

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psutil

from .imap_client import IMAPClient
from .state import Config

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark operation."""

    operation: str
    duration: float
    throughput: float | None = None
    memory_delta: float | None = None
    success_rate: float = 1.0
    samples: int = 1
    metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        """Human-readable benchmark result."""
        parts = [f"{self.operation}: {self.duration:.3f}s"]
        if self.throughput is not None:
            parts.append(f"throughput={self.throughput:.2f} ops/sec")
        if self.memory_delta is not None:
            parts.append(f"memory_delta={self.memory_delta:.2f} MB")
        parts.append(f"success_rate={self.success_rate:.1%}")
        return ", ".join(parts)


@dataclass
class PerformanceReport:
    """Complete performance report."""

    timestamp: datetime
    config: Config
    benchmarks: list[BenchmarkResult]
    system_info: dict[str, Any]
    recommendations: list[str]

    def __str__(self) -> str:
        """Human-readable performance report."""
        lines = [
            f"Performance Report - {self.timestamp.isoformat()}",
            "=" * 60,
            "",
            "System Information:",
        ]
        for key, value in self.system_info.items():
            lines.append(f"  {key}: {value}")

        lines.extend(["", "Benchmarks:"])
        for bench in self.benchmarks:
            lines.append(f"  {bench}")

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class IMAPConnectionPool:
    """Thread-safe connection pool for IMAP operations.

    Features:
    - Reusable connections to reduce overhead
    - Thread-safe checkout/checkin
    - Automatic connection recycling
    - Health checking
    - Configurable pool size

    Example:
        pool = IMAPConnectionPool(email="user@gmail.com", pool_size=5)
        with pool.checkout() as client:
            headers = client.batch_fetch_headers(100)
    """

    def __init__(
        self,
        email: str,
        pool_size: int = 5,
        max_connection_age: float = 300.0,
        provider: str | None = None,
    ):
        """Initialize connection pool.

        Args:
            email: User's email address
            pool_size: Maximum number of connections in pool
            max_connection_age: Maximum age of connection in seconds before recycling
            provider: Email provider ('gmail' or 'outlook')
        """
        self.email = email
        self.pool_size = pool_size
        self.max_connection_age = max_connection_age
        self.provider = provider

        self._pool: queue.Queue[tuple[IMAPClient, float]] = queue.Queue(
            maxsize=pool_size
        )
        self._lock = threading.Lock()
        self._created_count = 0
        self._checkout_count = 0
        self._checkin_count = 0

        logger.info(f"Initialized IMAP connection pool with size={pool_size}")

    def _create_connection(self) -> IMAPClient:
        """Create a new IMAP connection."""
        client = IMAPClient(email=self.email, provider=self.provider)
        client.connect()
        return client

    def _is_connection_valid(self, client: IMAPClient, created_at: float) -> bool:
        """Check if connection is still valid.

        Args:
            client: IMAP client to check
            created_at: Timestamp when connection was created

        Returns:
            True if connection is valid and not too old
        """
        age = time.time() - created_at
        if age > self.max_connection_age:
            logger.debug(f"Connection too old ({age:.1f}s), will recycle")
            return False

        # Check if connection is still alive
        if not client._connected:
            logger.debug("Connection not connected, will recycle")
            return False

        return True

    @contextmanager
    def checkout(self):
        """Checkout a connection from the pool.

        Yields:
            IMAPClient: Active IMAP connection

        Example:
            with pool.checkout() as client:
                headers = client.batch_fetch_headers(100)
        """
        client = None
        created_at = None
        reused = False

        try:
            # Try to get existing connection from pool
            try:
                client, created_at = self._pool.get_nowait()
                reused = True

                # Validate connection
                if not self._is_connection_valid(client, created_at):
                    try:
                        client.disconnect()
                    except Exception:
                        pass
                    client = None
                    reused = False
            except queue.Empty:
                pass

            # Create new connection if needed
            if client is None:
                with self._lock:
                    if self._created_count < self.pool_size:
                        client = self._create_connection()
                        created_at = time.time()
                        self._created_count += 1
                        logger.debug(
                            f"Created new connection "
                            f"({self._created_count}/{self.pool_size})"
                        )
                    else:
                        # Pool is full, wait for a connection to be returned
                        logger.debug("Pool exhausted, waiting for available connection")
                        client, created_at = self._pool.get()
                        reused = True

            self._checkout_count += 1
            logger.debug(f"Checked out connection (reused={reused})")

            yield client

        finally:
            # Return connection to pool
            if client is not None and created_at is not None:
                try:
                    # Validate before returning
                    if self._is_connection_valid(client, created_at):
                        self._pool.put((client, created_at))
                        self._checkin_count += 1
                        logger.debug("Returned connection to pool")
                    else:
                        # Don't return invalid connection
                        try:
                            client.disconnect()
                        except Exception:
                            pass
                        with self._lock:
                            self._created_count -= 1
                        logger.debug("Discarded invalid connection")
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")

    def close_all(self):
        """Close all connections in the pool."""
        logger.info("Closing all connections in pool")
        closed = 0

        while not self._pool.empty():
            try:
                client, _ = self._pool.get_nowait()
                try:
                    client.disconnect()
                    closed += 1
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            except queue.Empty:
                break

        logger.info(f"Closed {closed} connections")
        self._created_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        return {
            "pool_size": self.pool_size,
            "created_count": self._created_count,
            "available": self._pool.qsize(),
            "checkout_count": self._checkout_count,
            "checkin_count": self._checkin_count,
        }


class PerformanceOptimizer:
    """Auto-tuning and performance optimization for email processing.

    Features:
    - Automatic parameter tuning based on system resources
    - Benchmarking of AI and IMAP operations
    - Memory usage tracking
    - Performance monitoring
    - Configuration recommendations

    Example:
        optimizer = PerformanceOptimizer()
        tuned_config = optimizer.auto_tune_config(config)
        report = optimizer.generate_report(config)
    """

    def __init__(self):
        """Initialize performance optimizer."""
        self._benchmarks: list[BenchmarkResult] = []
        self._start_memory = self._get_memory_usage()
        logger.info("Initialized PerformanceOptimizer")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Current memory usage in megabytes
        """
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / 1024 / 1024)

    def _get_cpu_count(self) -> int:
        """Get number of CPU cores.

        Returns:
            Number of CPU cores
        """
        return psutil.cpu_count(logical=True) or 4

    def _get_available_memory(self) -> float:
        """Get available system memory in MB.

        Returns:
            Available memory in megabytes
        """
        return float(psutil.virtual_memory().available / 1024 / 1024)

    def measure_memory_usage(self) -> float:
        """Track memory usage delta since initialization.

        Returns:
            Memory delta in megabytes
        """
        current = self._get_memory_usage()
        delta = current - self._start_memory
        logger.debug(f"Memory delta: {delta:.2f} MB")
        return float(delta)

    @contextmanager
    def measure_operation(self, operation: str):
        """Context manager to measure operation performance.

        Args:
            operation: Name of the operation being measured

        Yields:
            Dictionary to store metadata

        Example:
            with optimizer.measure_operation("fetch_emails") as meta:
                emails = client.batch_fetch_headers(100)
                meta["count"] = len(emails)
        """
        metadata: dict[str, Any] = {}
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield metadata
        finally:
            duration = time.time() - start_time
            memory_delta = self._get_memory_usage() - start_memory

            # Calculate throughput if count is available
            throughput = None
            if "count" in metadata and metadata["count"] > 0:
                throughput = metadata["count"] / duration if duration > 0 else 0

            success_rate = metadata.get("success_rate", 1.0)

            result = BenchmarkResult(
                operation=operation,
                duration=duration,
                throughput=throughput,
                memory_delta=memory_delta,
                success_rate=success_rate,
                samples=metadata.get("samples", 1),
                metadata=metadata,
            )

            self._benchmarks.append(result)
            logger.info(f"Benchmark: {result}")

    def benchmark_ollama_throughput(
        self,
        model_name: str = "gemma2:2b",
        ollama_base_url: str = "http://localhost:11434",
        sample_count: int = 10,
    ) -> BenchmarkResult:
        """Measure AI classification throughput.

        Args:
            model_name: Ollama model name
            ollama_base_url: Ollama API base URL
            sample_count: Number of samples to test

        Returns:
            BenchmarkResult with throughput measurements
        """
        logger.info(f"Benchmarking Ollama throughput with {sample_count} samples")

        try:
            import requests

            # Sample email text for classification
            sample_text = """
            Subject: Limited Time Offer - 50% Off!

            Don't miss out on our exclusive sale. Click here to shop now!
            Unsubscribe link at bottom.
            """

            payload = {
                "model": model_name,
                "prompt": (
                    f"Classify this email as marketing or personal:\n\n"
                    f"{sample_text}\n\nAnswer with one word:"
                ),
                "stream": False,
            }

            successes = 0
            total_duration = 0.0
            self._get_memory_usage()

            with self.measure_operation("ollama_throughput") as meta:
                for i in range(sample_count):
                    try:
                        start = time.time()
                        response = requests.post(
                            f"{ollama_base_url}/api/generate",
                            json=payload,
                            timeout=30,
                        )
                        if response.status_code == 200:
                            successes += 1
                        total_duration += time.time() - start
                    except Exception as e:
                        logger.warning(f"Ollama request {i + 1} failed: {e}")

                meta["count"] = successes
                meta["samples"] = sample_count
                meta["success_rate"] = (
                    successes / sample_count if sample_count > 0 else 0
                )
                meta["avg_latency"] = (
                    total_duration / sample_count if sample_count > 0 else 0
                )

            result = self._benchmarks[-1]
            logger.info(
                f"Ollama throughput: {result.throughput:.2f} req/sec"
                if result.throughput
                else "N/A"
            )
            return result

        except ImportError:
            logger.error("requests library not available for benchmarking")
            return BenchmarkResult(
                operation="ollama_throughput",
                duration=0,
                success_rate=0,
                metadata={"error": "requests not installed"},
            )
        except Exception as e:
            logger.error(f"Error benchmarking Ollama: {e}")
            return BenchmarkResult(
                operation="ollama_throughput",
                duration=0,
                success_rate=0,
                metadata={"error": str(e)},
            )

    def benchmark_imap_latency(
        self,
        email: str,
        provider: str | None = None,
        operation: str = "headers",
        sample_count: int = 5,
    ) -> BenchmarkResult:
        """Measure IMAP operation latency.

        Args:
            email: User's email address
            provider: Email provider
            operation: Operation to benchmark ('headers' or 'bodies')
            sample_count: Number of samples to test

        Returns:
            BenchmarkResult with latency measurements
        """
        logger.info(
            f"Benchmarking IMAP {operation} latency with {sample_count} samples"
        )

        try:
            successes = 0
            total_duration = 0.0

            with self.measure_operation(f"imap_{operation}_latency") as meta:
                with IMAPClient(email=email, provider=provider) as client:
                    for i in range(sample_count):
                        try:
                            start = time.time()

                            if operation == "headers":
                                result = client.batch_fetch_headers(limit=10)
                            elif operation == "bodies":
                                # Fetch headers first to get UIDs
                                headers = client.batch_fetch_headers(limit=5)
                                if headers:
                                    uids = list(headers.keys())[:3]
                                    result = client.batch_fetch_bodies(uids)
                                else:
                                    result = {}
                            else:
                                raise ValueError(f"Unknown operation: {operation}")

                            if result:
                                successes += 1
                            total_duration += time.time() - start

                        except Exception as e:
                            logger.warning(f"IMAP request {i + 1} failed: {e}")

                meta["samples"] = sample_count
                meta["success_rate"] = (
                    successes / sample_count if sample_count > 0 else 0
                )
                meta["avg_latency"] = (
                    total_duration / sample_count if sample_count > 0 else 0
                )

            benchmark_result = self._benchmarks[-1]
            logger.info(
                f"IMAP {operation} latency: "
                f"{benchmark_result.metadata.get('avg_latency', 0) if benchmark_result.metadata else 0:.3f}s avg"
            )
            return benchmark_result

        except Exception as e:
            logger.error(f"Error benchmarking IMAP: {e}")
            return BenchmarkResult(
                operation=f"imap_{operation}_latency",
                duration=0,
                success_rate=0,
                metadata={"error": str(e)},
            )

    def profile_batch_processing(
        self,
        process_func: Callable,
        batch_sizes: list[int] | None = None,
    ) -> dict[int, BenchmarkResult]:
        """Profile batch processing performance with different batch sizes.

        Args:
            process_func: Function that takes batch_size as argument
            batch_sizes: List of batch sizes to test (default: [10, 25, 50, 100])

        Returns:
            Dictionary mapping batch_size to BenchmarkResult
        """
        if batch_sizes is None:
            batch_sizes = [10, 25, 50, 100]

        logger.info(f"Profiling batch processing with sizes: {batch_sizes}")
        results = {}

        for batch_size in batch_sizes:
            logger.info(f"Testing batch_size={batch_size}")

            with self.measure_operation(f"batch_processing_size_{batch_size}") as meta:
                try:
                    process_func(batch_size)
                    meta["batch_size"] = batch_size
                    meta["success_rate"] = 1.0
                except Exception as e:
                    logger.error(f"Batch processing failed for size {batch_size}: {e}")
                    meta["batch_size"] = batch_size
                    meta["success_rate"] = 0.0
                    meta["error"] = str(e)

            results[batch_size] = self._benchmarks[-1]

        return results

    def auto_tune_config(self, config: Config) -> Config:
        """Automatically tune configuration based on system resources.

        Analyzes:
        - CPU cores for concurrent_ai_limit
        - Available memory for batch_size and fetch_size
        - System resources for optimal performance

        Args:
            config: Base configuration to tune

        Returns:
            Tuned configuration
        """
        logger.info("Auto-tuning configuration based on system resources")

        cpu_count = self._get_cpu_count()
        available_memory = self._get_available_memory()

        # Tune concurrent_ai_limit based on CPU cores
        # Use 2-3x CPU cores for I/O-bound operations
        optimal_concurrent_limit = min(cpu_count * 3, 50)

        # Tune batch_size based on available memory
        # Assume ~1MB per email on average
        # Reserve 50% of available memory for other operations
        memory_for_batch = available_memory * 0.5
        optimal_batch_size = min(int(memory_for_batch / 1.0), 100)
        optimal_batch_size = max(optimal_batch_size, 10)  # Minimum 10

        # Tune fetch_size (should be larger than batch_size)
        optimal_fetch_size = optimal_batch_size * 2
        optimal_fetch_size = min(optimal_fetch_size, 200)  # Cap at 200

        logger.info(
            f"System: {cpu_count} CPUs, {available_memory:.0f} MB available memory"
        )
        logger.info(
            f"Tuned concurrent_ai_limit: {config.concurrent_ai_limit} -> "
            f"{optimal_concurrent_limit}"
        )
        logger.info(f"Tuned batch_size: {config.batch_size} -> {optimal_batch_size}")
        logger.info(f"Tuned fetch_size: {config.fetch_size} -> {optimal_fetch_size}")

        # Create tuned config
        tuned_config = config.model_copy(
            update={
                "concurrent_ai_limit": optimal_concurrent_limit,
                "batch_size": optimal_batch_size,
                "fetch_size": optimal_fetch_size,
            }
        )

        return tuned_config

    def get_system_info(self) -> dict[str, Any]:
        """Get system information for performance analysis.

        Returns:
            Dictionary with system information
        """
        cpu_count = self._get_cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        return {
            "cpu_count": cpu_count,
            "cpu_percent": f"{cpu_percent:.1f}%",
            "total_memory_mb": f"{memory.total / 1024 / 1024:.0f}",
            "available_memory_mb": f"{memory.available / 1024 / 1024:.0f}",
            "memory_percent": f"{memory.percent:.1f}%",
            "process_memory_mb": f"{self._get_memory_usage():.2f}",
        }

    def generate_recommendations(self, config: Config) -> list[str]:
        """Generate performance recommendations based on benchmarks and system state.

        Args:
            config: Current configuration

        Returns:
            List of recommendation strings
        """
        recommendations = []
        cpu_count = self._get_cpu_count()
        available_memory = self._get_available_memory()

        # Check concurrent_ai_limit
        if config.concurrent_ai_limit < cpu_count:
            recommendations.append(
                f"Increase concurrent_ai_limit to {cpu_count * 2} to better "
                f"utilize {cpu_count} CPU cores"
            )
        elif config.concurrent_ai_limit > cpu_count * 5:
            recommendations.append(
                f"Reduce concurrent_ai_limit to {cpu_count * 3} to prevent "
                "resource exhaustion"
            )

        # Check batch_size vs available memory
        estimated_memory_per_batch = config.batch_size * 1.0  # MB
        if estimated_memory_per_batch > available_memory * 0.5:
            recommended_batch_size = int(available_memory * 0.5)
            recommendations.append(
                f"Reduce batch_size to {recommended_batch_size} to prevent "
                f"memory issues (current: {config.batch_size}, available memory: "
                f"{available_memory:.0f} MB)"
            )

        # Check if fetch_size is optimal
        if config.fetch_size < config.batch_size:
            recommendations.append(
                f"Increase fetch_size to at least {config.batch_size * 2} "
                f"(should be larger than batch_size)"
            )

        # Memory pressure check
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            recommendations.append(
                f"High memory usage ({memory.percent:.1f}%) - consider reducing "
                "batch_size and fetch_size"
            )

        # Analyze benchmarks if available
        for benchmark in self._benchmarks:
            if benchmark.operation == "ollama_throughput":
                if benchmark.success_rate < 0.8:
                    recommendations.append(
                        f"Ollama success rate is low ({benchmark.success_rate:.1%}) - "
                        "check if Ollama service is running properly"
                    )
                elif benchmark.throughput and benchmark.throughput < 1.0:
                    recommendations.append(
                        f"Ollama throughput is slow "
                        f"({benchmark.throughput:.2f} req/sec) - "
                        "consider using a smaller model or reducing concurrent_ai_limit"
                    )

        if not recommendations:
            recommendations.append(
                "Configuration looks optimal for current system resources"
            )

        return recommendations

    def generate_report(self, config: Config) -> PerformanceReport:
        """Generate comprehensive performance report.

        Args:
            config: Current configuration

        Returns:
            PerformanceReport with all metrics and recommendations
        """
        logger.info("Generating performance report")

        return PerformanceReport(
            timestamp=datetime.now(),
            config=config,
            benchmarks=self._benchmarks.copy(),
            system_info=self.get_system_info(),
            recommendations=self.generate_recommendations(config),
        )

    def clear_benchmarks(self):
        """Clear stored benchmark results."""
        self._benchmarks.clear()
        logger.debug("Cleared benchmark results")


class PerformanceMonitor:
    """Real-time performance monitoring during email processing.

    Features:
    - Track processing rate (emails/sec)
    - Monitor memory usage
    - Log slow operations
    - Generate real-time statistics

    Example:
        monitor = PerformanceMonitor(log_interval=10.0)
        monitor.start()

        for email in emails:
            with monitor.track_operation("process_email"):
                process(email)
            monitor.record_processed(1)

        stats = monitor.get_stats()
        monitor.stop()
    """

    def __init__(
        self, log_interval: float = 10.0, slow_operation_threshold: float = 5.0
    ):
        """Initialize performance monitor.

        Args:
            log_interval: Interval in seconds to log statistics
            slow_operation_threshold: Threshold in seconds to log slow operations
        """
        self.log_interval = log_interval
        self.slow_operation_threshold = slow_operation_threshold

        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._emails_processed = 0
        self._operations: dict[str, list[float]] = {}
        self._running = False
        self._lock = threading.Lock()

        logger.info("Initialized PerformanceMonitor")

    def start(self):
        """Start monitoring."""
        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._running = True
        logger.info("Started performance monitoring")

    def stop(self):
        """Stop monitoring and log final statistics."""
        self._running = False
        self._log_stats(final=True)
        logger.info("Stopped performance monitoring")

    def record_processed(self, count: int = 1):
        """Record processed emails.

        Args:
            count: Number of emails processed
        """
        with self._lock:
            self._emails_processed += count

        # Log periodically
        if time.time() - self._last_log_time >= self.log_interval:
            self._log_stats()

    @contextmanager
    def track_operation(self, operation: str):
        """Track operation duration.

        Args:
            operation: Name of the operation

        Yields:
            None

        Example:
            with monitor.track_operation("fetch_emails"):
                emails = client.batch_fetch_headers(100)
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start

            with self._lock:
                if operation not in self._operations:
                    self._operations[operation] = []
                self._operations[operation].append(duration)

            # Log slow operations
            if duration > self.slow_operation_threshold:
                logger.warning(
                    f"Slow operation detected: {operation} took {duration:.2f}s"
                )

    def _log_stats(self, final: bool = False):
        """Log current statistics.

        Args:
            final: Whether this is the final log
        """
        with self._lock:
            elapsed = time.time() - self._start_time
            rate = self._emails_processed / elapsed if elapsed > 0 else 0

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            prefix = "FINAL " if final else ""
            logger.info(
                f"{prefix}Performance: {self._emails_processed} emails processed, "
                f"{rate:.2f} emails/sec, {memory_mb:.2f} MB memory"
            )

            # Log operation statistics
            for operation, durations in self._operations.items():
                avg_duration = sum(durations) / len(durations) if durations else 0
                logger.info(
                    f"  {operation}: {len(durations)} calls, "
                    f"avg {avg_duration:.3f}s, "
                    f"total {sum(durations):.2f}s"
                )

            self._last_log_time = time.time()

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics.

        Returns:
            Dictionary with performance statistics
        """
        with self._lock:
            elapsed = time.time() - self._start_time
            rate = self._emails_processed / elapsed if elapsed > 0 else 0

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            operation_stats = {}
            for operation, durations in self._operations.items():
                if durations:
                    operation_stats[operation] = {
                        "calls": len(durations),
                        "total_duration": sum(durations),
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                    }

            return {
                "elapsed_seconds": elapsed,
                "emails_processed": self._emails_processed,
                "processing_rate": rate,
                "memory_mb": memory_mb,
                "operations": operation_stats,
            }
