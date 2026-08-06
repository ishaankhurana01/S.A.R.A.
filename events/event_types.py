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
class AssistantResponse(Event):
    """A response ready to be spoken and/or displayed."""

    text: str = ""


@dataclass(frozen=True)
class PermissionRequested(Event):
    """Published when a gated action needs a permission decision."""

    scope: str = ""
    requesting_agent: str = ""


# --------------------------------------------------------------------------- #
# Executive Agent Framework — task lifecycle events (Phase 2, Day 2)
#
# Every request the Executive Agent handles moves through this event
# sequence, always in this order for a given task_id:
#   TaskCreated -> AgentDelegated -> (TaskCompleted | TaskFailed | TaskTimeout)
#
# task_id correlates all events belonging to the same request, which is
# what lets agents.executive.reasoning_loop.ReasoningLoop match a
# TaskCompleted/TaskFailed event back to the task it is waiting on.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskCreated(Event):
    """Published the moment the Executive Agent accepts a request (reasoning loop step 1: Input)."""

    task_id: str = ""
    capability: str = ""
    description: str = ""


@dataclass(frozen=True)
class AgentDelegated(Event):
    """Published by the Executive Agent when it delegates a task to a specific Worker Agent.

    This is the *only* message a Worker Agent (``agents.base_agent.BaseAgent``)
    listens for. Every subscribed agent receives every ``AgentDelegated``
    event and is responsible for ignoring it if ``agent_name`` does not
    match its own name — this keeps the Executive Agent from ever needing
    a direct reference to a specific agent instance.
    """

    task_id: str = ""
    agent_name: str = ""
    capability: str = ""
    task_description: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskCompleted(Event):
    """Published by a Worker Agent when it finishes a delegated task successfully."""

    task_id: str = ""
    agent_name: str = ""
    result: Any = None


@dataclass(frozen=True)
class TaskFailed(Event):
    """Published when a task cannot be completed.

    Attributes:
        reason: A short machine-readable failure category, e.g.
            ``"unknown_capability"`` (Planning found no agent for the
            request) or ``"agent_exception"`` (the assigned agent raised
            while handling it). Distinct from ``error_message``, which is
            the human-readable detail — this split is what lets callers
            branch on failure type without string-matching messages.
    """

    task_id: str = ""
    agent_name: str = ""
    error_message: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TaskTimeout(Event):
    """Published when a delegated task does not complete within its allotted time.

    A Worker Agent that never publishes ``TaskCompleted``/``TaskFailed``
    (hung, crashed silently, or simply too slow) must not be able to block
    the Executive Agent forever — this event marks that boundary being hit.
    """

    task_id: str = ""
    capability: str = ""
    agent_name: str = ""
    timeout_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Capability Registry events (Phase 2, Day 2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentRegistered(Event):
    """Published when a Worker Agent successfully registers its capabilities."""

    agent_name: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AgentUnregistered(Event):
    """Published when a Worker Agent is removed from the Capability Registry."""

    agent_name: str = ""
