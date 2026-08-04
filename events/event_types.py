"""
Typed event definitions for the S.A.R.A. event bus.

Every event published on ``core.event_bus.EventBus`` is an instance of a
class defined here — never a raw dict or string. This module is the
contract every publisher and subscriber agrees on: if you're adding a new
kind of cross-module notification, it starts here as a new ``Event``
subclass, not as an ad-hoc payload shape invented at the call site.

Convention: event classes are named in the past tense (``ContextUpdated``,
not ``UpdateContext``) because they represent something that already
happened, which is what a pub/sub bus should be carrying — commands
("do this") belong in direct method calls or agent delegation, not on
the event bus.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """Base class for every event on the bus.

    Attributes:
        event_id: Unique identifier, useful for tracing a single event
            through multiple subscriber logs.
        timestamp: Unix timestamp (seconds) of when the event was created.
        source: Name of the module/agent that published the event, for
            debugging and for the future audit log.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()), kw_only=True)
    timestamp: float = field(default_factory=time.time, kw_only=True)
    source: str = field(default="unknown", kw_only=True)


# --------------------------------------------------------------------------- #
# Core lifecycle events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ApplicationStarted(Event):
    """Published once core.app.Application has finished startup."""


@dataclass(frozen=True)
class ApplicationShuttingDown(Event):
    """Published when a graceful shutdown has been requested."""


# --------------------------------------------------------------------------- #
# Context Engine events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContextUpdated(Event):
    """Published whenever the World Model's snapshot changes.

    Attributes:
        snapshot: The new ``context.context_model.ContextSnapshot``.
            Typed as ``Any`` here (rather than importing ContextSnapshot)
            to avoid a circular import between ``events`` and ``context``;
            ``events/`` is meant to be importable by every module,
            including ones that ``context/`` itself does not depend on.
        changed_fields: Names of the top-level fields that changed since
            the previous snapshot, so subscribers can cheaply ignore
            updates irrelevant to them (e.g. Notification Agent only
            cares about ``battery``).
    """

    snapshot: Any = None
    changed_fields: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CollectorFailed(Event):
    """Published when a single context collector raises during a poll cycle.

    Non-fatal by design: the Context Engine keeps running with the
    collector's previous value. This event exists so something (initially
    just logging; later possibly the Notification Agent) can surface
    persistent collector failures to the user.
    """

    collector_name: str = ""
    error_message: str = ""


# --------------------------------------------------------------------------- #
# Reserved for later phases — defined now so the contract is stable
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UserUtterance(Event):
    """A transcribed or typed user request, ready for the Executive Agent."""

    text: str = ""


@dataclass(frozen=True)
class AgentDelegated(Event):
    """Published by the Executive Agent when it delegates work to a Worker Agent."""

    agent_name: str = ""
    task_description: str = ""


@dataclass(frozen=True)
class AssistantResponse(Event):
    """A response ready to be spoken and/or displayed."""

    text: str = ""


@dataclass(frozen=True)
class PermissionRequested(Event):
    """Published when a gated action needs a permission decision."""

    scope: str = ""
    requesting_agent: str = ""
