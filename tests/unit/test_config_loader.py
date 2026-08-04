from __future__ import annotations

from pathlib import Path

import pytest

from config.config_loader import ConfigLoader
from utils.exceptions import ConfigError, ConfigValidationError


def test_load_real_settings_file() -> None:
    loader = ConfigLoader("config/settings.yaml")
    config = loader.load()

    assert config.app.name == "S.A.R.A."
    assert config.context_engine.enabled is True
    assert "system" in config.context_engine.enabled_collectors


def test_load_missing_file_raises(tmp_path: Path) -> None:
    loader = ConfigLoader(tmp_path / "does_not_exist.yaml")
    with pytest.raises(ConfigError):
        loader.load()


def test_load_invalid_yaml_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "settings.yaml"
    bad_file.write_text("app: [this is not: valid: yaml", encoding="utf-8")

    loader = ConfigLoader(bad_file)
    with pytest.raises(ConfigError):
        loader.load()


def test_unknown_top_level_key_fails_validation(tmp_path: Path) -> None:
    bad_file = tmp_path / "settings.yaml"
    bad_file.write_text("not_a_real_section:\n  foo: bar\n", encoding="utf-8")

    loader = ConfigLoader(bad_file)
    with pytest.raises(ConfigValidationError):
        loader.load()


def test_partial_config_fills_defaults(tmp_path: Path) -> None:
    partial_file = tmp_path / "settings.yaml"
    partial_file.write_text("app:\n  name: 'Custom Name'\n", encoding="utf-8")

    loader = ConfigLoader(partial_file)
    config = loader.load()

    assert config.app.name == "Custom Name"
    assert config.llm.model == "qwen2.5:7b"  # untouched section still gets its default


def test_empty_file_loads_all_defaults(tmp_path: Path) -> None:
    empty_file = tmp_path / "settings.yaml"
    empty_file.write_text("", encoding="utf-8")

    loader = ConfigLoader(empty_file)
    config = loader.load()

    assert config.app.name == "S.A.R.A."


def test_current_property_loads_lazily() -> None:
    loader = ConfigLoader("config/settings.yaml")
    config = loader.current  # not yet explicitly loaded
    assert config.app.name == "S.A.R.A."
