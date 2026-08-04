"""
Abstract interfaces for S.A.R.A.'s dependency-inversion boundaries.

Every interface defined here is what ``core.service_registry`` maps
concrete implementations against. High-level code (the Executive Agent,
the reasoning loop, ``llm.prompt_builder``, etc.) is written against these
ABCs, never against a concrete class like ``OllamaProvider`` or
``ChromaMemoryStore`` directly.

Phase 1 defines the full set now — including interfaces for modules not
implemented until later phases (``LLMProvider``, ``MemoryTier``,
``Plugin``) — so that when those phases arrive, the contract is already
fixed and agreed upon, and implementing them is additive rather than a
redesign. This mirrors the "declare the whole dependency list now" choice
made for ``requirements.txt``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ContextCollector(ABC):
    """A single signal source for the World Model (context/collectors/).

    Each collector is responsible for exactly one signal (active window,
    running processes, clipboard, ...). The Context Engine polls all
    enabled collectors and merges their output into one ``ContextSnapshot``.
    A collector that fails must raise so the engine can log it via
    ``events.event_types.CollectorFailed`` and retain the previous value —
    it must never crash the whole polling cycle.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique, config-referenceable name (e.g. 'system', 'window')."""

    @abstractmethod
    def collect(self) -> dict[str, Any]:
        """Gather this collector's current signal(s).

        Returns:
            A flat dict of field-name -> value to be merged into the
            ``ContextSnapshot``. Keys should be stable across calls so
            ``ContextEngine`` can diff snapshots to compute
            ``changed_fields``.

        Raises:
            utils.exceptions.CollectorError: On any failure to gather the
                signal (permission denied, platform not supported, the
                underlying library not installed, etc.).
        """


class Agent(ABC):
    """Base contract for every Executive/Worker agent (agents/).

    Reserved for Phase-2-and-later implementation. Defined now so the
    Executive Agent's ``capability_registry`` has a stable type to
    register against from the first agent onward, and so plugins
    (``plugins.plugin_interface.Plugin``) can be registered through the
    same registry as built-in agents.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name, used for delegation logging and the audit trail."""

    @property
    @abstractmethod
    def capabilities(self) -> tuple[str, ...]:
        """Capability identifiers this agent can handle (e.g. 'code_review')."""

    @abstractmethod
    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        """Execute a delegated task.

        Args:
            task_description: What the Executive Agent has asked this
                agent to do.
            context: The current World Model snapshot plus any relevant
                memory recall, as assembled by the reasoning loop's
                Context Gathering step.

        Returns:
            Agent-specific result, surfaced back to the Executive Agent's
            Reflection step.
        """


class MemoryTier(ABC):
    """Base contract for a memory tier (memory/working, episodic, semantic, procedural).

    Reserved for Phase 5 implementation. Each tier implements its own
    storage strategy behind this shared read/write contract so
    ``agents.memory_agent`` can address all four tiers uniformly.
    """

    @abstractmethod
    def write(self, record: dict[str, Any]) -> None:
        """Persist a record into this tier."""

    @abstractmethod
    def query(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve records relevant to ``query`` from this tier."""


class LLMProvider(ABC):
    """Base contract for a local LLM backend (llm/providers/).

    Reserved for Phase 3 implementation. Defined now so
    ``llm.prompt_builder`` and the Executive Agent can be written against
    this interface before ``OllamaProvider`` exists.
    """

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.7) -> str:
        """Generate a completion for ``prompt``."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the backend is reachable (e.g. Ollama server is up)."""


class Plugin(ABC):
    """Base contract for third-party plugins (plugins/installed/).

    Reserved for Phase 9 implementation. A plugin must declare a manifest
    (capabilities, commands, events, required permission scopes) and is
    registered with the same capability_registry as built-in Worker
    Agents once ``plugins.plugin_loader`` validates its manifest and
    required permissions.
    """

    @abstractmethod
    def manifest(self) -> dict[str, Any]:
        """Return this plugin's manifest: capabilities, commands, events, permissions."""
