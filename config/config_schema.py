"""
Pydantic schema for S.A.R.A.'s configuration file (settings.yaml).

Every tunable in the project — model names, hotkeys, feature toggles,
polling intervals, permission defaults — is declared here as a typed field.
``config_loader.ConfigLoader`` parses settings.yaml into an instance of
``SaraConfig`` and validates it against this schema, so a malformed or
incomplete config fails fast at startup with a clear error instead of
surfacing as a confusing ``KeyError`` deep inside some agent at runtime.

Sections are grouped by the module that owns them (logging, context,
memory, permissions, ...) so a new module's config lives in one obvious
place — add a new nested model + field, never touch existing ones.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Controls utils.logger.configure_logging."""

    log_dir: str = "logs"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    rotation: str = "10 MB"
    retention: str = "14 days"


class EventBusConfig(BaseModel):
    """Controls core.event_bus.EventBus."""

    # If True, an exception in one subscriber is logged and swallowed so
    # other subscribers still receive the event. If False, the first
    # failing subscriber's exception propagates to the publisher.
    isolate_subscriber_errors: bool = True


class ContextEngineConfig(BaseModel):
    """Controls context.context_engine.ContextEngine."""

    enabled: bool = True
    poll_interval_seconds: float = 2.0
    enabled_collectors: list[str] = Field(
        default_factory=lambda: [
            "system",
            "process",
            "window",
            "clipboard",
        ]
    )


class MemoryConfig(BaseModel):
    """Controls memory.* tiers (implemented in a later phase)."""

    working_memory_max_turns: int = 50
    episodic_db_path: str = "data/episodic.db"
    semantic_db_path: str = "data/semantic.db"
    semantic_vector_path: str = "data/chroma"
    procedural_db_path: str = "data/procedural.db"


class LLMConfig(BaseModel):
    """Controls llm.providers.* (llm.providers.ollama_provider.OllamaProvider, Phase 3)."""

    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    max_context_tokens: int = 8192
    request_timeout_seconds: float = 30.0


class VoiceConfig(BaseModel):
    """Controls voice/* (implemented in a later phase)."""

    wake_word_enabled: bool = False
    wake_word_phrase: str = "hey sara"
    stt_model_size: str = "base"
    tts_voice: str = "en_US-lessac-medium"


class AutomationConfig(BaseModel):
    """Controls automation.desktop_platform / DesktopAutomationAgent (Phase 4).

    ``enabled`` gates whether ``core.app.Application`` wires
    ``DesktopAutomationAgent`` in at all — set False to run S.A.R.A. with
    conversation only and no OS-level capabilities registered.
    """

    enabled: bool = True
    screenshot_directory: str = "data/screenshots"
    action_timeout_seconds: float = 10.0


class PermissionsConfig(BaseModel):
    """Controls permissions.permission_manager (implemented in a later phase).

    ``default_policy`` is intentionally conservative — "ask" rather than
    "allow" — so a fresh install never silently grants filesystem/mic/
    desktop-control access.
    """

    default_policy: str = "ask"  # one of: allow, ask, deny
    scopes: list[str] = Field(
        default_factory=lambda: [
            "filesystem.read",
            "filesystem.write",
            "browser.control",
            "microphone.listen",
            "desktop.input_control",
            "network.outbound",
        ]
    )


class UIConfig(BaseModel):
    """Controls ui/* (implemented in a later phase)."""

    always_on_top: bool = True
    theme: str = "dark"
    global_hotkey: str = "ctrl+alt+s"
    start_minimized_to_tray: bool = True


class AppConfig(BaseModel):
    """Top-level application metadata and lifecycle behavior."""

    name: str = "S.A.R.A."
    version: str = "0.3.0"
    launch_on_startup: bool = False
    environment: str = "development"  # development | production


class SaraConfig(BaseModel):
    """Root configuration object — the validated, in-memory form of settings.yaml.

    Nested models default to their own defaults, so a settings.yaml that
    only overrides a handful of values (or is entirely empty) still
    produces a fully valid, fully-populated ``SaraConfig``.
    """

    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    context_engine: ContextEngineConfig = Field(default_factory=ContextEngineConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    model_config = {
        "extra": "forbid",  # unknown keys in settings.yaml fail validation loudly
    }
