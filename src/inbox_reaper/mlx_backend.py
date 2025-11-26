"""MLX backend for LLM inference on Apple Silicon.

This module provides a clean interface for MLX-based LLM inference with:
- Automatic model downloading from Hugging Face on first use
- Model caching for subsequent runs
- Progress indication for downloads
- Simple sync and async interfaces
"""

import asyncio
import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .ui.events import EventEmitter

# Global model cache to avoid reloading models
_model_cache: dict[str, tuple[Any, Any]] = {}


def is_mlx_available() -> bool:
    """Check if MLX is available on this system.

    Returns:
        True if MLX can be imported, False otherwise
    """
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def get_mlx_model(
    model_name: str, event_emitter: Optional["EventEmitter"] = None
) -> tuple[Any, Any]:
    """Load MLX model with caching.

    Downloads model from Hugging Face if not cached locally.
    Subsequent calls return the cached model.

    Args:
        model_name: Hugging Face model ID
            (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")
        event_emitter: Optional event emitter for progress updates

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ImportError: If mlx-lm is not installed
        Exception: If model loading fails
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from mlx_lm import load
    except ImportError as e:
        raise ImportError(
            "mlx-lm is not installed. Install it with: pip install mlx-lm"
        ) from e

    # Emit loading status or fallback to print
    if event_emitter:
        from .ui.events import EventType

        event_emitter.emit(
            EventType.PROCESSING_START,
            f"Loading MLX model: {model_name}",
            model=model_name,
            note="First run will download model weights, cached for future use",
        )
    else:
        print(f"Loading MLX model: {model_name}", file=sys.stderr)
        print(
            "(First run will download model weights, cached for future use)",
            file=sys.stderr,
        )

    try:
        # Request config to get predictable 3-value return, discard it
        model, tokenizer, _ = load(model_name, return_config=True)  # type: ignore[misc]
        _model_cache[model_name] = (model, tokenizer)

        # Emit success or fallback to print
        if event_emitter:
            from .ui.events import EventType

            event_emitter.emit(
                EventType.PROCESSING_COMPLETE,
                f"✓ Model loaded successfully: {model_name}",
                model=model_name,
            )
        else:
            print(f"✓ Model loaded successfully: {model_name}", file=sys.stderr)

        return model, tokenizer
    except Exception as e:
        # Emit error or fallback to print
        if event_emitter:
            from .ui.events import EventType

            event_emitter.emit(
                EventType.PROCESSING_ERROR,
                f"✗ Failed to load model {model_name}: {e}",
                model=model_name,
                error=str(e),
            )
        else:
            print(f"✗ Failed to load model {model_name}: {e}", file=sys.stderr)
        raise


def generate_text(
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int = 100,
    event_emitter: Optional["EventEmitter"] = None,
) -> str:
    """Generate text using MLX model with chat template (synchronous).

    Args:
        model_name: Hugging Face model ID
        messages: Chat messages (list of dicts with 'role' and 'content')
        max_tokens: Maximum tokens to generate
        event_emitter: Optional event emitter for progress updates

    Returns:
        Generated text string

    Raises:
        ImportError: If mlx-lm is not installed
    """
    try:
        from mlx_lm import generate
    except ImportError as e:
        raise ImportError(
            "mlx-lm is not installed. Install it with: pip install mlx-lm"
        ) from e

    model, tokenizer = get_mlx_model(model_name, event_emitter)

    # Use chat template for instruct models (much better JSON compliance)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate text with explicit parameters
    # Note: MLX generate() returns the completion (without the prompt)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,  # Suppress token-by-token output
    )

    # Explicit cast since mlx_lm doesn't have type stubs
    response_str = str(response)

    # Debug logging if response is suspiciously short
    if len(response_str) < 10:
        import logging

        logging.getLogger(__name__).warning(
            f"MLX returned very short response ({len(response_str)} chars): "
            f"'{response_str}'"
        )

    return response_str


async def generate_text_async(
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int = 100,
    event_emitter: Optional["EventEmitter"] = None,
) -> str:
    """Generate text using MLX model with chat template (asynchronous).

    Runs generation in thread pool to avoid blocking event loop.

    Args:
        model_name: Hugging Face model ID
        messages: Chat messages (list of dicts with 'role' and 'content')
        max_tokens: Maximum tokens to generate
        event_emitter: Optional event emitter for progress updates

    Returns:
        Generated text string

    Raises:
        ImportError: If mlx-lm is not installed
    """
    # Run synchronous generate in thread pool to avoid blocking
    from functools import partial

    loop = asyncio.get_event_loop()
    generate_fn = partial(
        generate_text,
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        event_emitter=event_emitter,
    )
    return await loop.run_in_executor(None, generate_fn)


def download_model(model_name: str) -> bool:
    """Download MLX model to local cache.

    Useful for pre-fetching models in CI/CD or before running offline.

    Args:
        model_name: Hugging Face model ID

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading model: {model_name}")
        print("This may take a few minutes depending on model size...")

        # Loading the model will trigger download if not cached
        model, tokenizer = get_mlx_model(model_name)

        # Run a tiny test to verify it works
        print("\nVerifying model...")
        test_messages = [{"role": "user", "content": "Hello"}]
        test_output = generate_text(model_name, test_messages, max_tokens=5)
        print(f"Test generation: 'Hello' → '{test_output}'")

        print(f"\n✓ Model downloaded and verified: {model_name}")
        return True

    except Exception as e:
        print(f"\n✗ Failed to download model: {e}", file=sys.stderr)
        return False


def clear_model_cache():
    """Clear in-memory model cache.

    Useful for testing or to free memory. Note: This does not delete
    the on-disk Hugging Face cache.
    """
    global _model_cache
    _model_cache.clear()
    print("Model cache cleared")


def get_cached_models() -> list[str]:
    """Get list of currently cached models in memory.

    Returns:
        List of model names currently cached
    """
    return list(_model_cache.keys())
