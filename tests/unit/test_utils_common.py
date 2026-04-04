"""Tests for common utility functions."""

import os
import tempfile
import pytest
import yaml

from fall_detection.utils.common import load_config, save_config


class TestLoadConfig:
    """Test cases for the load_config function."""

    def test_load_valid_yaml(self, tmp_path):
        """Loading a valid YAML file should return the parsed dictionary."""
        config_path = tmp_path / "test_config.yaml"
        test_data = {"name": "test", "value": 42, "nested": {"key": "value"}}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(test_data, f)

        result = load_config(str(config_path))
        assert result == test_data

    def test_load_empty_yaml(self, tmp_path):
        """Loading an empty YAML file should return None."""
        config_path = tmp_path / "empty_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("")

        result = load_config(str(config_path))
        assert result is None

    def test_load_nested_structure(self, tmp_path):
        """Loading YAML with deeply nested structures."""
        config_path = tmp_path / "nested_config.yaml"
        test_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(test_data, f)

        result = load_config(str(config_path))
        assert result["level1"]["level2"]["level3"]["value"] == "deep"

    def test_load_list_values(self, tmp_path):
        """Loading YAML with list values."""
        config_path = tmp_path / "list_config.yaml"
        test_data = {"items": [1, 2, 3], "names": ["a", "b", "c"]}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(test_data, f)

        result = load_config(str(config_path))
        assert result["items"] == [1, 2, 3]
        assert result["names"] == ["a", "b", "c"]

    def test_load_unicode_content(self, tmp_path):
        """Loading YAML with unicode characters."""
        config_path = tmp_path / "unicode_config.yaml"
        test_data = {"name": "测试", "description": "日本語テキスト"}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(test_data, f, allow_unicode=True)

        result = load_config(str(config_path))
        assert result["name"] == "测试"
        assert result["description"] == "日本語テキスト"

    def test_load_nonexistent_file(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """Loading invalid YAML should raise an error."""
        config_path = tmp_path / "invalid_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            load_config(str(config_path))


class TestSaveConfig:
    """Test cases for the save_config function."""

    def test_save_simple_config(self, tmp_path):
        """Saving a simple config should create a valid YAML file."""
        config_path = tmp_path / "output_config.yaml"
        test_data = {"name": "test", "value": 42}

        save_config(test_data, str(config_path))

        assert config_path.exists()
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == test_data

    def test_save_creates_directories(self, tmp_path):
        """Saving config should create parent directories if they don't exist."""
        config_path = tmp_path / "subdir1" / "subdir2" / "config.yaml"
        test_data = {"key": "value"}

        save_config(test_data, str(config_path))

        assert config_path.exists()
        assert (tmp_path / "subdir1" / "subdir2").exists()

    def test_save_nested_config(self, tmp_path):
        """Saving nested config should preserve structure."""
        config_path = tmp_path / "nested_config.yaml"
        test_data = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }

        save_config(test_data, str(config_path))

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == test_data

    def test_save_list_values(self, tmp_path):
        """Saving config with list values should preserve lists."""
        config_path = tmp_path / "list_config.yaml"
        test_data = {"items": [1, 2, 3], "mixed": [1, "two", 3.0, True]}

        save_config(test_data, str(config_path))

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == test_data

    def test_save_unicode_content(self, tmp_path):
        """Saving config with unicode characters should preserve them."""
        config_path = tmp_path / "unicode_config.yaml"
        test_data = {"name": "测试", "emoji": "Hello"}

        save_config(test_data, str(config_path))

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == test_data

    def test_save_overwrites_existing(self, tmp_path):
        """Saving should overwrite existing file."""
        config_path = tmp_path / "config.yaml"

        # First save
        save_config({"version": 1}, str(config_path))

        # Second save with different content
        save_config({"version": 2}, str(config_path))

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["version"] == 2

    def test_save_none_values(self, tmp_path):
        """Saving config with None values should work."""
        config_path = tmp_path / "none_config.yaml"
        test_data = {"value": None, "other": "present"}

        save_config(test_data, str(config_path))

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["value"] is None
        assert loaded["other"] == "present"

    def test_save_boolean_values(self, tmp_path):
        """Saving config with boolean values should preserve them."""
        config_path = tmp_path / "bool_config.yaml"
        test_data = {"enabled": True, "disabled": False}

        save_config(test_data, str(config_path))

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["enabled"] is True
        assert loaded["disabled"] is False

    def test_save_numeric_types(self, tmp_path):
        """Saving config with different numeric types should preserve them."""
        config_path = tmp_path / "numeric_config.yaml"
        test_data = {"integer": 42, "float": 3.14, "negative": -10}

        save_config(test_data, str(config_path))

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["integer"] == 42
        assert loaded["float"] == 3.14
        assert loaded["negative"] == -10


class TestLoadSaveRoundtrip:
    """Test roundtrip: save then load should preserve data."""

    def test_roundtrip_simple(self, tmp_path):
        """Roundtrip with simple config."""
        config_path = tmp_path / "roundtrip.yaml"
        original = {"name": "test", "value": 42}

        save_config(original, str(config_path))
        loaded = load_config(str(config_path))

        assert loaded == original

    def test_roundtrip_complex(self, tmp_path):
        """Roundtrip with complex nested config."""
        config_path = tmp_path / "roundtrip.yaml"
        original = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "admin",
                    "password": "secret"
                }
            },
            "features": ["feature1", "feature2", "feature3"],
            "enabled": True,
            "threshold": 0.85
        }

        save_config(original, str(config_path))
        loaded = load_config(str(config_path))

        assert loaded == original
