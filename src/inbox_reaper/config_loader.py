"""Configuration file loader for inbox-reaper.

Supports loading configuration from YAML and JSON files,
with CLI arguments taking precedence over file values.
"""

import json
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config_file(config_path: str) -> dict[str, Any]:
    """Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary of configuration values

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the file format is not supported or invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = path.suffix.lower()

    # Load based on file extension
    if suffix in [".yaml", ".yml"]:
        if not HAS_YAML:
            raise ValueError(
                "YAML support not available. Install PyYAML: pip install pyyaml"
            )
        return _load_yaml(path)
    elif suffix == ".json":
        return _load_json(path)
    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. "
            "Supported formats: .yaml, .yml, .json"
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: expected dict, got {type(config)}")

    return config


def _load_json(path: Path) -> dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(path, encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid JSON config: expected dict, got {type(config)}")

    return config


def merge_config_with_cli_args(
    file_config: dict[str, Any], cli_args: dict[str, Any]
) -> dict[str, Any]:
    """Merge configuration file values with CLI arguments.

    CLI arguments take precedence over file values.
    None values from CLI are ignored (file values are used).

    Args:
        file_config: Configuration loaded from file
        cli_args: Configuration from CLI arguments

    Returns:
        Merged configuration dictionary
    """
    # Start with file config
    merged = file_config.copy()

    # Overlay CLI args (only non-None values)
    for key, value in cli_args.items():
        if value is not None:
            # Special handling for lists (like keywords, whitelist_domains)
            # If CLI provides an empty tuple/list, don't override file config
            if isinstance(value, (tuple, list)) and len(value) == 0:
                continue
            merged[key] = value

    return merged
