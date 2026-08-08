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


class AgentAlreadyRegisteredError(AgentError):
    """Raised when registering an agent name that is already in the Capability Registry.

    Requires ``allow_override=True`` to replace intentionally, mirroring
    ``core.service_registry.ServiceRegistry``'s override protection — a
    second agent claiming the same name should never silently replace the
    first.
    """


class CapabilityAlreadyRegisteredError(AgentError):
    """Raised when two agents attempt to claim the same capability string.

    Each capability must resolve to exactly one agent so the Executive
    Agent's Planning step is unambiguous. If two agents legitimately need
    to share a capability name in a later phase, that requires an explicit
    routing/priority strategy — not a silent last-write-wins.
    """


class AgentTimeoutError(AgentError):
    """Raised (in contexts that propagate rather than return a failed TaskResult) on a task timeout."""


# --------------------------------------------------------------------------- #
# LLM (llm/providers/, Phase 3)
# --------------------------------------------------------------------------- #
class LLMError(SaraError):
    """Base class for every failure raised by an ``core.interfaces.LLMProvider``.

    ``agents.conversation_agent.ConversationAgent`` deliberately does not
    catch these — they propagate up to ``agents.base_agent.BaseAgent``,
    which already turns any exception from ``handle()`` into a
    ``TaskFailed`` event (reason=``"agent_exception"``). Catching them
    here too would just duplicate that handling; the exception hierarchy
    exists so the *message* is specific, not so the agent needs extra
    try/except blocks.
    """


class LLMProviderUnavailableError(LLMError):
    """Raised when the LLM backend cannot be reached at all (e.g. Ollama isn't running)."""


class LLMModelNotFoundError(LLMError):
    """Raised when the configured model is not available on the backend."""


class LLMTimeoutError(LLMError):
    """Raised when a request to the LLM backend exceeds ``LLMConfig.request_timeout_seconds``."""


class LLMInvalidResponseError(LLMError):
    """Raised when the backend responds successfully but the response body is malformed or empty."""


# --------------------------------------------------------------------------- #
# Desktop Automation (automation/, Phase 4)
# --------------------------------------------------------------------------- #
class DesktopAutomationError(SaraError):
    """Base class for every failure raised by a ``automation.desktop_platform.DesktopPlatform``.

    ``agents.desktop_automation_agent.DesktopAutomationAgent`` catches
    this specific hierarchy (not bare ``Exception``) so it can distinguish
    "the requested desktop action failed in an expected way" (encoded as
    ``{"success": False, ...}`` in the structured result, per requirement
    #9) from a genuine programming bug in the agent itself, which should
    still propagate to ``agents.base_agent.BaseAgent`` and become a
    ``TaskFailed`` event rather than being silently swallowed.
    """


class InvalidDesktopTargetError(DesktopAutomationError):
    """Raised when an application name/URL argument is empty, too long, or fails safety validation.

    This is the enforcement point for requirement #6 ("never allow
    arbitrary shell execution") at the input layer: every target string
    passes through validation before it ever reaches a subprocess argv
    list or ``webbrowser.open``.
    """


class ApplicationLaunchError(DesktopAutomationError):
    """Raised when an application (or URL) could not be launched/opened."""


class ApplicationCloseError(DesktopAutomationError):
    """Raised when no matching running process was found, or it could not be terminated."""


class ScreenshotCaptureError(DesktopAutomationError):
    """Raised when a screenshot could not be captured or saved to disk."""


class UnsupportedPlatformError(DesktopAutomationError):
    """Raised when the current OS has no ``DesktopPlatform`` implementation."""


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
