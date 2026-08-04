"""
Custom exception hierarchy for S.A.R.A.

Every module-specific error should subclass ``SaraError`` (directly or via
one of the domain-level exceptions below) rather than raising bare
``Exception`` / ``ValueError`` / etc. This gives calling code — especially
``core.event_bus`` and ``agents.executive.executive_agent`` — a single
type hierarchy to catch against, so a failure in one worker agent can be
handled generically without the caller needing to know which subsystem
raised it.

Design notes
------------
- ``SaraError`` carries an optional ``context`` dict so error handlers and
  the audit log (``permissions.audit_log``) can log structured details
  without string-parsing a message.
- Domain exceptions (``ConfigError``, ``EventBusError``, ...) exist even
  before their modules are fully built out, so later phases can start
  raising the right exception type from day one instead of retrofitting.
"""

from __future__ import annotations

from typing import Any


class SaraError(Exception):
    """Base class for every exception raised within S.A.R.A.

    Attributes:
        message: Human-readable description of what went wrong.
        context: Optional structured details (e.g. {"key": "llm.model"})
            useful for logging/auditing without parsing the message string.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigError(SaraError):
    """Raised when settings.yaml is missing, malformed, or fails schema validation."""


class ConfigValidationError(ConfigError):
    """Raised specifically when Pydantic schema validation fails."""


# --------------------------------------------------------------------------- #
# Core (event bus / service registry)
# --------------------------------------------------------------------------- #
class EventBusError(SaraError):
    """Raised for event bus publish/subscribe failures."""


class HandlerExecutionError(EventBusError):
    """Raised when a subscriber callback raises during event dispatch.

    The event bus catches subscriber exceptions and wraps them in this type
    so one misbehaving handler cannot silently break event delivery to
    other subscribers, while the original traceback is still preserved via
    exception chaining (``raise ... from original``).
    """


class ServiceRegistryError(SaraError):
    """Base class for dependency-injection/service-registry failures."""


class ServiceNotRegisteredError(ServiceRegistryError):
    """Raised when resolving an interface that has no registered implementation."""


class ServiceAlreadyRegisteredError(ServiceRegistryError):
    """Raised when registering an interface that already has an implementation.

    Registration must be explicit about overwriting (via ``allow_override``)
    to avoid a plugin or late-loaded module silently replacing a core service.
    """


# --------------------------------------------------------------------------- #
# Context Engine
# --------------------------------------------------------------------------- #
class ContextError(SaraError):
    """Base class for World Model / Context Engine failures."""


class CollectorError(ContextError):
    """Raised when an individual context collector fails to gather its signal.

    Collector failures are non-fatal to the engine as a whole (see
    ``context.context_engine.ContextEngine``): a failing collector logs and
    keeps its previous value rather than crashing the whole snapshot.
    """


# --------------------------------------------------------------------------- #
# Agents (reserved for later phases, defined now for a stable contract)
# --------------------------------------------------------------------------- #
class AgentError(SaraError):
    """Base class for errors raised by Executive or Worker agents."""


class CapabilityNotFoundError(AgentError):
    """Raised when the Executive Agent cannot find any agent for a request."""


# --------------------------------------------------------------------------- #
# Permissions (reserved for later phases)
# --------------------------------------------------------------------------- #
class PermissionError_(AgentError):
    """Raised when a gated action is attempted without the required permission.

    Named ``PermissionError_`` (trailing underscore) to avoid shadowing the
    Python builtin ``PermissionError``.
    """


class PermissionDeniedError(PermissionError_):
    """Raised when the user has explicitly denied a permission scope."""
