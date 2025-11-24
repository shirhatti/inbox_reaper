"""Tests for configuration file loading."""

import json
import tempfile
from pathlib import Path

import pytest

from inbox_reaper.config_loader import (
    HAS_YAML,
    load_config_file,
    merge_config_with_cli_args,
)

# Import yaml for YAML-specific errors
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# Skip YAML tests if PyYAML is not installed
requires_yaml = pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")


class TestLoadConfigFile:
    """Tests for loading configuration files."""

    @requires_yaml
    def test_load_yaml_config(self):
        """Test loading a valid YAML configuration file."""
        # Create temporary YAML file
        config_content = """
email: test@example.com
model_name: test-model
batch_size: 100
dry_run: true
keywords:
  - important
  - urgent
whitelist_domains:
  - example.com
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            # Load config
            config = load_config_file(temp_path)

            # Verify values
            assert config["email"] == "test@example.com"
            assert config["model_name"] == "test-model"
            assert config["batch_size"] == 100
            assert config["dry_run"] is True
            assert config["keywords"] == ["important", "urgent"]
            assert config["whitelist_domains"] == ["example.com"]
        finally:
            Path(temp_path).unlink()

    @requires_yaml
    def test_load_yml_extension(self):
        """Test loading YAML file with .yml extension."""
        config_content = """
email: test@example.com
batch_size: 50
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            config = load_config_file(temp_path)
            assert config["email"] == "test@example.com"
            assert config["batch_size"] == 50
        finally:
            Path(temp_path).unlink()

    def test_load_json_config(self):
        """Test loading a valid JSON configuration file."""
        config_data = {
            "email": "json@example.com",
            "model_name": "json-model",
            "batch_size": 200,
            "dry_run": False,
            "keywords": ["test", "json"],
            "whitelist_domains": ["json.com"],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config_file(temp_path)
            assert config == config_data
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config_file("/nonexistent/config.yaml")

        assert "not found" in str(exc_info.value).lower()

    def test_load_unsupported_format(self):
        """Test loading a file with unsupported extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("email: test@example.com")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_config_file(temp_path)

            assert "Unsupported config file format" in str(exc_info.value)
            assert ".txt" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    @requires_yaml
    def test_load_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        # Invalid YAML with unclosed bracket
        invalid_yaml = """
email: test@example.com
keywords: [invalid
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            assert yaml is not None, "yaml module should be available"
            with pytest.raises(yaml.YAMLError):
                load_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_json(self):
        """Test loading an invalid JSON file."""
        invalid_json = '{"email": "test@example.com", "batch_size": }'

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(invalid_json)
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    @requires_yaml
    def test_load_yaml_with_null_values(self):
        """Test loading YAML with null values."""
        config_content = """
email: test@example.com
max_emails: null
batch_size: 50
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            config = load_config_file(temp_path)
            assert config["email"] == "test@example.com"
            assert config["max_emails"] is None
            assert config["batch_size"] == 50
        finally:
            Path(temp_path).unlink()

    def test_load_json_with_null_values(self):
        """Test loading JSON with null values."""
        config_data = {
            "email": "test@example.com",
            "max_emails": None,
            "batch_size": 50,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config_file(temp_path)
            assert config["email"] == "test@example.com"
            assert config["max_emails"] is None
            assert config["batch_size"] == 50
        finally:
            Path(temp_path).unlink()

    @requires_yaml
    def test_load_yaml_not_dict(self):
        """Test loading YAML that isn't a dictionary."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("- item1\n- item2\n")  # YAML list instead of dict
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_config_file(temp_path)

            assert "expected dict" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    def test_load_json_not_dict(self):
        """Test loading JSON that isn't a dictionary."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(["item1", "item2"], f)  # JSON array instead of object
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_config_file(temp_path)

            assert "expected dict" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    @requires_yaml
    def test_load_empty_yaml(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_config_file(temp_path)

            assert "expected dict" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    @pytest.mark.skipif(HAS_YAML, reason="Only test when PyYAML is not installed")
    def test_load_yaml_without_pyyaml_installed(self):
        """Test that proper error is shown when PyYAML is not installed."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("email: test@example.com")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_config_file(temp_path)

            assert "YAML support not available" in str(exc_info.value)
            assert "pip install pyyaml" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    @requires_yaml
    def test_load_yaml_with_complex_nested_structure(self):
        """Test loading YAML with nested structures."""
        config_content = """
email: test@example.com
keywords:
  - urgent
  - important
  - action required
important_extensions:
  - .pdf
  - .docx
  - .xlsx
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            config = load_config_file(temp_path)
            assert len(config["keywords"]) == 3
            assert "urgent" in config["keywords"]
            assert len(config["important_extensions"]) == 3
            assert ".pdf" in config["important_extensions"]
        finally:
            Path(temp_path).unlink()


class TestMergeConfigWithCliArgs:
    """Tests for merging configuration with CLI arguments."""

    def test_merge_cli_overrides_file(self):
        """Test that CLI arguments override file values."""
        file_config = {
            "email": "file@example.com",
            "batch_size": 50,
            "dry_run": True,
        }
        cli_args = {
            "email": "cli@example.com",
            "batch_size": 100,
            "dry_run": False,
        }

        merged = merge_config_with_cli_args(file_config, cli_args)

        # CLI values should override file values
        assert merged["email"] == "cli@example.com"
        assert merged["batch_size"] == 100
        assert merged["dry_run"] is False

    def test_merge_none_values_ignored(self):
        """Test that None values from CLI are ignored."""
        file_config = {
            "email": "file@example.com",
            "batch_size": 50,
            "model_name": "file-model",
        }
        cli_args = {
            "email": None,
            "batch_size": 100,
            "model_name": None,
        }

        merged = merge_config_with_cli_args(file_config, cli_args)

        # None values should not override file values
        assert merged["email"] == "file@example.com"
        assert merged["batch_size"] == 100  # Non-None CLI value overrides
        assert merged["model_name"] == "file-model"

    def test_merge_empty_lists_ignored(self):
        """Test that empty lists/tuples from CLI are ignored."""
        file_config = {
            "keywords": ["important", "urgent"],
            "whitelist_domains": ["example.com"],
        }
        cli_args = {
            "keywords": [],  # Empty list
            "whitelist_domains": (),  # Empty tuple
        }

        merged = merge_config_with_cli_args(file_config, cli_args)

        # Empty lists should not override file values
        assert merged["keywords"] == ["important", "urgent"]
        assert merged["whitelist_domains"] == ["example.com"]

    def test_merge_nonempty_lists_override(self):
        """Test that non-empty lists from CLI override file values."""
        file_config = {
            "keywords": ["old1", "old2"],
            "whitelist_domains": ["old.com"],
        }
        cli_args = {
            "keywords": ["new1", "new2", "new3"],
            "whitelist_domains": ["new.com"],
        }

        merged = merge_config_with_cli_args(file_config, cli_args)

        # Non-empty lists should override
        assert merged["keywords"] == ["new1", "new2", "new3"]
        assert merged["whitelist_domains"] == ["new.com"]

    def test_merge_cli_adds_new_keys(self):
        """Test that CLI can add new keys not in file config."""
        file_config = {
            "email": "test@example.com",
            "batch_size": 50,
        }
        cli_args = {
            "model_name": "cli-model",
            "dry_run": True,
        }

        merged = merge_config_with_cli_args(file_config, cli_args)

        # File values should be preserved
        assert merged["email"] == "test@example.com"
        assert merged["batch_size"] == 50
        # CLI values should be added
        assert merged["model_name"] == "cli-model"
        assert merged["dry_run"] is True

    def test_merge_empty_file_config(self):
        """Test merging with empty file config."""
        file_config = {}
        cli_args = {
            "email": "cli@example.com",
            "batch_size": 100,
            "dry_run": False,
        }

        merged = merge_config_with_cli_args(file_config, cli_args)

        # All CLI values should be present
        assert merged["email"] == "cli@example.com"
        assert merged["batch_size"] == 100
        assert merged["dry_run"] is False

    def test_merge_empty_cli_args(self):
        """Test merging with empty CLI args."""
        file_config = {
            "email": "file@example.com",
            "batch_size": 50,
            "dry_run": True,
        }
        cli_args = {}

        merged = merge_config_with_cli_args(file_config, cli_args)

        # All file values should be preserved
        assert merged == file_config

    def test_merge_preserves_file_config(self):
        """Test that merge doesn't modify original file config."""
        file_config = {
            "email": "file@example.com",
            "batch_size": 50,
        }
        original_file_config = file_config.copy()
        cli_args = {
            "email": "cli@example.com",
        }

        merge_config_with_cli_args(file_config, cli_args)

        # Original file_config should not be modified
        assert file_config == original_file_config

    def test_merge_with_all_config_fields(self):
        """Test merge with comprehensive config including all field types."""
        file_config = {
            "email": "file@example.com",
            "model_name": "file-model",
            "ollama_base_url": "http://file:11434",
            "batch_size": 50,
            "concurrent_ai_limit": 25,
            "dry_run": True,
            "max_emails": None,
            "keywords": ["file-keyword"],
            "whitelist_domains": ["file.com"],
            "auto_delete_threshold": 5,
            "fetch_size": 100,
            "important_extensions": [".pdf", ".doc"],
            "enable_pipelining": True,
            "enable_sender_tracking": True,
        }
        cli_args = {
            "email": None,  # Should keep file value
            "batch_size": 200,  # Should override
            "dry_run": False,  # Should override
            "max_emails": 1000,  # Should override
            "keywords": ["cli-keyword"],  # Should override
            "model_name": "cli-model",  # Should override
        }

        merged = merge_config_with_cli_args(file_config, cli_args)

        # Verify correct merging
        assert merged["email"] == "file@example.com"  # None ignored
        assert merged["model_name"] == "cli-model"  # Overridden
        assert merged["batch_size"] == 200  # Overridden
        assert merged["dry_run"] is False  # Overridden
        assert merged["max_emails"] == 1000  # Overridden
        assert merged["keywords"] == ["cli-keyword"]  # Overridden
        assert merged["whitelist_domains"] == ["file.com"]  # Not in CLI, kept
        assert merged["ollama_base_url"] == "http://file:11434"  # Not in CLI
        assert merged["auto_delete_threshold"] == 5  # Not in CLI
