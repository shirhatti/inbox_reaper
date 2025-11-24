"""Batch coordinator for accumulating and processing items.

This module provides a BatchCoordinator class that accumulates items and processes
them in batches when a size limit is reached or explicitly flushed.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BatchCoordinator:
    """Coordinator for batch processing of items.

    Accumulates items and calls a processor function when the batch size
    limit is reached.
    """

    def __init__(
        self,
        processor: Callable[[list[T]], Awaitable[Any]],
        batch_size: int = 50,
    ):
        """Initialize BatchCoordinator.

        Args:
            processor: Async function to process a batch of items
            batch_size: Maximum size of a batch before processing
        """
        self.processor = processor
        self.batch_size = batch_size
        self.buffer: list[T] = []
        self._lock = asyncio.Lock()

    async def add(self, item: T) -> None:
        """Add an item to the buffer and process if full.

        Args:
            item: Item to add
        """
        async with self._lock:
            self.buffer.append(item)  # type: ignore[arg-type]

            if len(self.buffer) >= self.batch_size:
                await self._process_batch()

    async def flush(self) -> None:
        """Process any remaining items in the buffer."""
        async with self._lock:
            if self.buffer:
                await self._process_batch()

    async def _process_batch(self) -> None:
        """Process the current buffer."""
        if not self.buffer:
            return

        items_to_process = list(self.buffer)
        self.buffer.clear()

        try:
            await self.processor(items_to_process)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # In a real system, we might want to retry or handle partial failures
            # For now, we just log the error
