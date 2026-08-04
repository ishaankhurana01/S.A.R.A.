"""
Service registry — S.A.R.A.'s dependency-injection container.

Architectural rule: high-level code depends on abstract interfaces
(defined in ``core.interfaces``), never on concrete implementations
directly. A module obtains its dependencies by resolving an interface
from this registry, e.g.:

    llm_provider = registry.resolve(LLMProvider)

rather than:

    from llm.providers.ollama_provider import OllamaProvider
    llm_provider = OllamaProvider(...)

This is what lets you swap ``OllamaProvider`` for a different backend, or
``ChromaMemoryStore`` for a different vector store, without touching any
of the code that consumes them — only the one ``register()`` call at
startup changes.

Design notes
------------
- **Instance registry, not a factory/auto-wiring container.** Phase 1
  keeps this deliberately simple: you construct a concrete instance
  yourself and register it against the interface it implements. Only add
  factory/auto-wiring behavior later if the manual wiring in
  ``core.app.Application.startup`` actually becomes unwieldy — this
  follows the same "don't build machinery you don't need yet" principle
  as the rest of Phase 1.
- **Explicit override.** Re-registering an already-registered interface
  raises unless ``allow_override=True`` is passed, so a plugin can't
  silently replace a core service by accident.
"""

from __future__ import annotations

import threading
from typing import Type, TypeVar

from utils.exceptions import ServiceAlreadyRegisteredError, ServiceNotRegisteredError
from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ServiceRegistry:
    """Maps abstract interface types to concrete singleton instances.

    Example:
        registry = ServiceRegistry()
        registry.register(LLMProvider, OllamaProvider(host="http://localhost:11434"))
        ...
        provider = registry.resolve(LLMProvider)
    """

    def __init__(self) -> None:
        self._services: dict[type, object] = {}
        self._lock = threading.RLock()

    def register(self, interface: Type[T], implementation: T, *, allow_override: bool = False) -> None:
        """Register a concrete implementation against an interface type.

        Args:
            interface: The abstract type other code will resolve by (e.g.
                a class from ``core.interfaces``, or any type usable as a
                lookup key).
            implementation: The concrete instance to return on resolve().
                Must be an instance, not a class — the registry stores
                singletons.
            allow_override: If False (default) and ``interface`` is
                already registered, raises ``ServiceAlreadyRegisteredError``.
                Pass True to intentionally replace an existing
                registration (e.g. swapping in a test double).

        Raises:
            ServiceAlreadyRegisteredError: If already registered and
                ``allow_override`` is False.
        """
        with self._lock:
            if interface in self._services and not allow_override:
                raise ServiceAlreadyRegisteredError(
                    f"{interface.__name__} is already registered. "
                    "Pass allow_override=True to replace it intentionally.",
                    context={"interface": interface.__name__},
                )
            self._services[interface] = implementation
        logger.debug(
            "Registered {} -> {}",
            getattr(interface, "__name__", str(interface)),
            type(implementation).__name__,
        )

    def resolve(self, interface: Type[T]) -> T:
        """Return the concrete instance registered for ``interface``.

        Args:
            interface: The interface type to resolve.

        Returns:
            The registered implementation instance.

        Raises:
            ServiceNotRegisteredError: If nothing is registered for this
                interface. This fails fast and loud rather than returning
                None, since a missing dependency should never be silently
                tolerated at runtime.
        """
        with self._lock:
            if interface not in self._services:
                raise ServiceNotRegisteredError(
                    f"No implementation registered for {interface.__name__}. "
                    "Was it registered during Application.startup()?",
                    context={"interface": interface.__name__},
                )
            return self._services[interface]  # type: ignore[return-value]

    def is_registered(self, interface: Type[T]) -> bool:
        """Return whether an implementation is currently registered for ``interface``."""
        with self._lock:
            return interface in self._services

    def unregister(self, interface: Type[T]) -> None:
        """Remove a registration. Safe to call even if not registered (no-op)."""
        with self._lock:
            self._services.pop(interface, None)
        logger.debug("Unregistered {}", getattr(interface, "__name__", str(interface)))

    def clear(self) -> None:
        """Remove all registrations. Intended for test teardown, not runtime use."""
        with self._lock:
            self._services.clear()
