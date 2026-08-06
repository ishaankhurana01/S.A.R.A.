"""
Base class for every Worker Agent.

``BaseAgent`` implements everything a Worker Agent needs that is *not*
domain-specific: subscribing to the event bus, filtering
``AgentDelegated`` events down to the ones addressed to this agent,
timing the work, and publishing ``TaskCompleted``/``TaskFailed`` back onto
the bus. A concrete agent (``agents.desktop_agent.DesktopAgent``, etc.)
only implements ``capabilities`` and ``handle`` — everything else is
inherited.

This is the mechanism behind the architecture's "no direct coupling
between agents" rule: a ``BaseAgent`` never receives a reference to the
Executive Agent, the Capability Registry, or any other Worker Agent — its
only dependency is the event bus.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

from core.event_bus import EventBus
from core.interfaces import Agent
from events.event_types import AgentDelegated, TaskCompleted, TaskFailed
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(Agent):
    """Common event-bus plumbing for every Worker Agent.

    Subclasses must implement ``name``, ``capabilities``, and ``handle``.
    ``start()``/``stop()`` control whether this agent is actively
    listening on the bus — an agent that hasn't been started will never
    see delegated tasks even if it's registered in the Capability
    Registry, so ``agents.executive.executive_agent.ExecutiveAgent``
    starts every agent it registers.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._subscription: tuple[type, Any] | None = None

    def start(self) -> None:
        """Begin listening for tasks delegated to this agent. Idempotent."""
        if self._subscription is not None:
            return
        self._subscription = self._event_bus.subscribe(AgentDelegated, self._on_agent_delegated)
        logger.debug("{} started listening for delegated tasks", self.name)

    def stop(self) -> None:
        """Stop listening for delegated tasks. Idempotent."""
        if self._subscription is None:
            return
        self._event_bus.unsubscribe(self._subscription)
        self._subscription = None
        logger.debug("{} stopped listening for delegated tasks", self.name)

    def _on_agent_delegated(self, event: AgentDelegated) -> None:
        """Filter, execute, and report the outcome of a delegated task.

        Every ``BaseAgent`` instance subscribes to the *same*
        ``AgentDelegated`` event type — this handler is what turns a
        broadcast bus message into "only the addressed agent reacts,"
        rather than the Executive Agent needing per-agent event types.
        """
        if event.agent_name != self.name:
            return  # not addressed to us — every other agent also sees this and ignores it too

        logger.info(
            "{} received task {} (capability={}): {}",
            self.name,
            event.task_id,
            event.capability,
            event.task_description,
        )

        start = time.monotonic()
        try:
            result = self.handle(event.task_description, context=event.payload)
        except Exception as exc:  # noqa: BLE001 - any agent failure must become TaskFailed, never crash the bus
            duration = time.monotonic() - start
            logger.error("{} failed task {} after {:.3f}s: {}", self.name, event.task_id, duration, exc)
            self._event_bus.publish(
                TaskFailed(
                    source=self.name,
                    task_id=event.task_id,
                    agent_name=self.name,
                    error_message=str(exc),
                    reason="agent_exception",
                )
            )
            return

        duration = time.monotonic() - start
        logger.info("{} completed task {} in {:.3f}s", self.name, event.task_id, duration)
        self._event_bus.publish(
            TaskCompleted(
                source=self.name,
                task_id=event.task_id,
                agent_name=self.name,
                result=result,
            )
        )

    @abstractmethod
    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        """Perform this agent's domain-specific work for a delegated task.

        Args:
            task_description: What the Executive Agent asked this agent
                to do.
            context: The task payload assembled by the reasoning loop's
                Context Gathering step (includes a ``"_context"`` key with
                the current World Model snapshot, when available).

        Returns:
            Any JSON-serializable-ish result describing the outcome; it is
            carried verbatim on ``TaskCompleted.result``.

        Raises:
            Exception: Any exception raised here is caught by
                ``_on_agent_delegated`` and reported as ``TaskFailed`` —
                subclasses do not need their own try/except for this.
        """
