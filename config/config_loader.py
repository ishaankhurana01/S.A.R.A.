"""
Loads and validates S.A.R.A.'s configuration file.

``ConfigLoader`` is the only piece of code in the project that should read
``settings.yaml`` directly. Every other module receives a validated
``SaraConfig`` instance (typically resolved from ``core.service_registry``)
rather than parsing YAML itself — this is what lets ``config_schema.py``
be the single source of truth for what a valid config looks like.

Hot-reload
----------
``ConfigLoader.reload()`` re-reads and re-validates the file on demand.
Callers (e.g. a future settings UI) are expected to call this after the
user edits settings, and to re-register the fresh ``SaraConfig`` in the
service registry themselves — ``ConfigLoader`` does not reach into other
modules on your behalf, keeping its responsibility narrowly "produce a
valid SaraConfig from disk."
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from config.config_schema import SaraConfig
from utils.exceptions import ConfigError, ConfigValidationError
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Reads settings.yaml from disk and produces a validated SaraConfig.

    Example:
        loader = ConfigLoader("config/settings.yaml")
        config = loader.load()
        print(config.llm.model)
    """

    def __init__(self, config_path: str | Path = "config/settings.yaml") -> None:
        self._config_path = Path(config_path)
        self._config: SaraConfig | None = None

    @property
    def config_path(self) -> Path:
        return self._config_path

    def load(self) -> SaraConfig:
        """Load, parse, and validate settings.yaml.

        Returns:
            A fully validated ``SaraConfig`` instance.

        Raises:
            ConfigError: If the file is missing or contains invalid YAML.
            ConfigValidationError: If the YAML is well-formed but fails
                schema validation (unknown key, wrong type, etc.).
        """
        if not self._config_path.exists():
            raise ConfigError(
                f"Configuration file not found: {self._config_path}",
                context={"path": str(self._config_path)},
            )

        try:
            raw_text = self._config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"Failed to read configuration file: {exc}",
                context={"path": str(self._config_path)},
            ) from exc

        try:
            raw_data = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Configuration file contains invalid YAML: {exc}",
                context={"path": str(self._config_path)},
            ) from exc

        if not isinstance(raw_data, dict):
            raise ConfigError(
                "Configuration file must contain a YAML mapping at the top level.",
                context={"path": str(self._config_path), "parsed_type": type(raw_data).__name__},
            )

        try:
            self._config = SaraConfig(**raw_data)
        except ValidationError as exc:
            raise ConfigValidationError(
                f"Configuration failed schema validation: {exc}",
                context={"path": str(self._config_path), "errors": exc.errors()},
            ) from exc

        logger.info(
            "Configuration loaded and validated ({} v{}, environment={})",
            self._config.app.name,
            self._config.app.version,
            self._config.app.environment,
        )
        return self._config

    def reload(self) -> SaraConfig:
        """Re-read and re-validate the configuration file from disk.

        Returns:
            The freshly loaded ``SaraConfig``. Callers are responsible for
            propagating this (e.g. re-registering it in the service
            registry) to any component that needs the update.
        """
        logger.info("Reloading configuration from {}", self._config_path)
        return self.load()

    @property
    def current(self) -> SaraConfig:
        """Return the most recently loaded config, loading it if necessary."""
        if self._config is None:
            return self.load()
        return self._config
