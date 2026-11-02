"""MLX backend for LLM inference on Apple Silicon.

This module provides a clean interface for MLX-based LLM inference with:
- Automatic model downloading from Hugging Face on first use
- Model caching for subsequent runs
- Progress indication for downloads
- Simple sync and async interfaces
"""

import asyncio
import importlib.util
import logging
import sys
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading models
_model_cache: dict[str, tuple[Any, Any]] = {}


def _mlx_lm() -> ModuleType:
    """Import mlx_lm on first use: it is slow to import and Apple Silicon only."""
    try:
        import mlx_lm  # noqa: PLC0415 - deferred heavy/platform-specific import
    except ImportError as e:
        raise ImportError(
            "mlx-lm is not installed. Install it with: uv add mlx-lm"
        ) from e
    return mlx_lm


def is_mlx_available() -> bool:
    """Check if MLX is available on this system.

    Returns:
        True if MLX can be imported, False otherwise
    """
    return importlib.util.find_spec("mlx_lm") is not None


def get_mlx_model(model_name: str) -> tuple[Any, Any]:
    """Load MLX model with caching.

    Downloads model from Hugging Face if not cached locally.
    Subsequent calls return the cached model.

    Args:
        model_name: Hugging Face model ID
            (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ImportError: If mlx-lm is not installed
        Exception: If model loading fails
    """
    if cached := _model_cache.get(model_name):
        return cached

    load = _mlx_lm().load

    print(f"Loading MLX model: {model_name}", file=sys.stderr)
    print(
        "(First run will download model weights, cached for future use)",
        file=sys.stderr,
    )

    try:
        # Request config to get predictable 3-value return, discard it
        model, tokenizer, _ = load(model_name, return_config=True)
    except Exception as e:
        print(f"✗ Failed to load model {model_name}: {e}", file=sys.stderr)
        raise

    _model_cache[model_name] = (model, tokenizer)
    print(f"✓ Model loaded successfully: {model_name}", file=sys.stderr)
    return model, tokenizer


def generate_text(
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int = 100,
) -> str:
    """Generate text using MLX model with chat template (synchronous).

    Args:
        model_name: Hugging Face model ID
        messages: Chat messages (list of dicts with 'role' and 'content')
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text string

    Raises:
        ImportError: If mlx-lm is not installed
    """
    generate = _mlx_lm().generate
    model, tokenizer = get_mlx_model(model_name)

    # Use chat template for instruct models (much better JSON compliance)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Note: MLX generate() returns the completion (without the prompt)
    response = str(
        generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,  # Suppress token-by-token output
        )
    )

    if len(response) < 10:
        logger.warning(
            "MLX returned very short response (%s chars): %r", len(response), response
        )

    return response


async def generate_text_async(
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int = 100,
) -> str:
    """Generate text using MLX model with chat template (asynchronous).

    Runs generation in a worker thread to avoid blocking the event loop.
    """
    return await asyncio.to_thread(generate_text, model_name, messages, max_tokens)


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
        get_mlx_model(model_name)

        # Run a tiny test to verify it works
        print("\nVerifying model...")
        test_messages = [{"role": "user", "content": "Hello"}]
        test_output = generate_text(model_name, test_messages, max_tokens=5)
        print(f"Test generation: 'Hello' → '{test_output}'")
    except Exception as e:
        print(f"\n✗ Failed to download model: {e}", file=sys.stderr)
        return False

    print(f"\n✓ Model downloaded and verified: {model_name}")
    return True


def clear_model_cache() -> None:
    """Clear in-memory model cache.

    Useful for testing or to free memory. Note: This does not delete
    the on-disk Hugging Face cache.
    """
    _model_cache.clear()
    print("Model cache cleared")


def get_cached_models() -> list[str]:
    """Get list of currently cached models in memory.

    Returns:
        List of model names currently cached
    """
    return list(_model_cache)
