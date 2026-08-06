"""
Reasoning loop — the standard execution flow for every task the Executive
Agent handles.

Per the architecture doc: Input -> Context Gathering -> Planning ->
Execution -> Reflection -> Memory Update. This module implements the
*framework* for that pipeline — each stage is deliberately lightweight in
this phase (no LLM, no real memory) — so later phases can deepen
individual stages (e.g. Planning consulting an LLM for multi-step
breakdowns) without changing the six-stage shape itself or anything that
calls ``ReasoningLoop.run``.

Task/agent correlation
-----------------------
Execution is delegated asynchronously over the event bus (publish
``AgentDelegated``, wait for ``TaskCompleted``/``TaskFailed`` carrying the
same ``task_id``). ``ReasoningLoop`` subscribes to both outcome events
once, for the lifetime of the loop, and correlates them back to the
waiting caller via a per-task ``threading.Event`` — this supports
multiple tasks in flight concurrently (e.g. from different threads)
without cross-talk.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from context.context_engine import ContextEngine
from core.event_bus import EventBus
from events.event_types import AgentDelegated, TaskCompleted, TaskCreated, TaskFailed, TaskTimeout
from utils.exceptions import CapabilityNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class Task:
    """A single unit of work moving through the reasoning loop."""

    task_id: str
    capability: str
    description: str
    payload: dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class TaskResult:
    """The outcome of running a task through the reasoning loop.

    Attributes:
        success: Whether the task completed successfully.
        result: The Worker Agent's return value, when successful.
        error_message: Human-readable failure detail, when unsuccessful.
        reason: Machine-readable failure category (``"unknown_capability"``,
            ``"agent_exception"``, ``"timeout"``), when unsuccessful.
        duration_seconds: Wall-clock time from Input to this result.
    """

    task_id: str
    success: bool
    result: Any = None
    error_message: str | None = None
    reason: str | None = None
    duration_seconds: float = 0.0


class _CapabilityResolver:
    """Structural protocol so ReasoningLoop doesn't import CapabilityRegistry directly.

    Avoids a circular import (capability_registry -> events, reasoning_loop
    -> capability_registry would be fine actually, but keeping this
    boundary narrow means ReasoningLoop only depends on the one method it
    needs, making it trivially testable with a stub registry).
    """

    def resolve(self, capability: str) -> str:  # pragma: no cover - protocol only
        raise NotImplementedError


class ReasoningLoop:
    """Drives a task through Input -> Context -> Plan -> Execute -> Reflect -> Memory Update.

    Example:
        loop = ReasoningLoop(event_bus=bus, capability_registry=registry)
        result = loop.run("desktop.open_application", "Open VS Code")
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        capability_registry: _CapabilityResolver,
        context_engine: ContextEngine | None = None,
        default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._event_bus = event_bus
        self._capability_registry = capability_registry
        self._context_engine = context_engine
        self._default_timeout_seconds = default_timeout_seconds

        self._lock = threading.Lock()
        self._pending_events: dict[str, threading.Event] = {}
        self._pending_results: dict[str, TaskResult] = {}

        event_bus.subscribe(TaskCompleted, self._on_task_completed)
        event_bus.subscribe(TaskFailed, self._on_task_failed)

    def run(
        self,
        capability: str,
        description: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> TaskResult:
        """Run one full pass of the reasoning loop for a single request.

        Args:
            capability: The capability identifier this request needs
                (e.g. ``"desktop.open_application"``). In this phase the
                caller supplies it directly; a later phase's Planning
                stage may derive it from natural language instead.
            description: Human-readable description of the task, passed
                through to the Worker Agent and logged at every stage.
            payload: Extra structured data for the Worker Agent. The
                current World Model snapshot (if a Context Engine was
                provided) is merged in under the ``"_context"`` key.
            timeout_seconds: Overrides the loop's default execution
                timeout for this task only.

        Returns:
            A ``TaskResult`` describing success/failure — this method
            never raises for an ordinary task failure (unknown
            capability, agent exception, timeout); those are all
            represented as ``TaskResult(success=False, ...)`` so callers
            have one uniform way to handle outcomes.
        """
        start_time = time.monotonic()

        # --- 1. Input ---------------------------------------------------
        task = Task(
            task_id=str(uuid.uuid4()),
            capability=capability,
            description=description,
            payload=dict(payload or {}),
            created_at=time.time(),
        )
        self._event_bus.publish(
            TaskCreated(
                source="agents.executive.reasoning_loop.ReasoningLoop",
                task_id=task.task_id,
                capability=task.capability,
                description=task.description,
            )
        )
        logger.info("[Input] Task {} created (capability={})", task.task_id, task.capability)

        # --- 2. Context Gathering ---------------------------------------
        enriched_payload = self._gather_context(task)

        # --- 3. Planning --------------------------------------------------
        try:
            agent_name = self._capability_registry.resolve(task.capability)
        except CapabilityNotFoundError as exc:
            logger.warning("[Planning] No agent for capability '{}': {}", task.capability, exc)
            self._event_bus.publish(
                TaskFailed(
                    source="agents.executive.reasoning_loop.ReasoningLoop",
                    task_id=task.task_id,
                    agent_name="",
                    error_message=str(exc),
                    reason="unknown_capability",
                )
            )
            result = TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(exc),
                reason="unknown_capability",
                duration_seconds=time.monotonic() - start_time,
            )
            self._reflect(task, result)
            self._update_memory(task, result)
            return result

        logger.info("[Planning] Task {} routed to agent '{}'", task.task_id, agent_name)

        # --- 4. Execution -------------------------------------------------
        effective_timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds
        result = self._execute(task, agent_name, enriched_payload, effective_timeout, start_time)

        # --- 5. Reflection & 6. Memory Update -----------------------------
        self._reflect(task, result)
        self._update_memory(task, result)

        return result

    # ------------------------------------------------------------------ #
    # Stage implementations
    # ------------------------------------------------------------------ #
    def _gather_context(self, task: Task) -> dict[str, Any]:
        """Stage 2: attach the current World Model snapshot to the task payload.

        Lightweight by design for this phase: it does not query
        Semantic/Procedural memory (Phase 5) — it only pulls whatever the
        Context Engine currently has, and degrades to an empty context if
        no Context Engine was provided or it fails to respond.
        """
        payload = dict(task.payload)
        if self._context_engine is None:
            payload["_context"] = {}
            return payload

        try:
            payload["_context"] = self._context_engine.get_snapshot().as_dict()
        except Exception as exc:  # noqa: BLE001 - context unavailability must not block the task
            logger.warning("[Context Gathering] Failed to read World Model snapshot: {}", exc)
            payload["_context"] = {}
        return payload

    def _execute(
        self,
        task: Task,
        agent_name: str,
        payload: dict[str, Any],
        timeout_seconds: float,
        start_time: float,
    ) -> TaskResult:
        """Stage 4: delegate to the resolved agent over the event bus and await the outcome.

        The delegation publish happens on a background thread rather than
        inline. This matters because ``EventBus.publish`` dispatches
        subscribers *synchronously* in the calling thread: a Worker Agent's
        ``handle()`` runs to completion, and its ``TaskCompleted``/
        ``TaskFailed`` response is published, all before the publish call
        returns. If we published here inline, a slow-or-hung agent would
        block this method before ``wait_event.wait(timeout=...)`` ever
        started counting down — the timeout would never have a chance to
        fire. Running the publish on its own daemon thread lets this
        method's ``wait_event.wait(timeout=...)`` genuinely race against
        the agent's execution time, so a hung agent times out here instead
        of hanging the caller.
        """
        wait_event = threading.Event()
        with self._lock:
            self._pending_events[task.task_id] = wait_event

        delegation_event = AgentDelegated(
            source="agents.executive.reasoning_loop.ReasoningLoop",
            task_id=task.task_id,
            agent_name=agent_name,
            capability=task.capability,
            task_description=task.description,
            payload=payload,
        )
        delegation_thread = threading.Thread(
            target=self._event_bus.publish,
            args=(delegation_event,),
            name=f"delegate-{task.task_id[:8]}",
            daemon=True,
        )
        delegation_thread.start()
        logger.info("[Execution] Task {} delegated to '{}', awaiting outcome (timeout={}s)", task.task_id, agent_name, timeout_seconds)

        completed_in_time = wait_event.wait(timeout=timeout_seconds)

        with self._lock:
            self._pending_events.pop(task.task_id, None)
            stored_result = self._pending_results.pop(task.task_id, None)

        duration = time.monotonic() - start_time

        if not completed_in_time:
            logger.error("[Execution] Task {} timed out after {}s", task.task_id, timeout_seconds)
            self._event_bus.publish(
                TaskTimeout(
                    source="agents.executive.reasoning_loop.ReasoningLoop",
                    task_id=task.task_id,
                    capability=task.capability,
                    agent_name=agent_name,
                    timeout_seconds=timeout_seconds,
                )
            )
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=f"Task timed out after {timeout_seconds}s waiting on agent '{agent_name}'",
                reason="timeout",
                duration_seconds=duration,
            )

        if stored_result is None:
            # Defensive: the wait event was set but no result was recorded.
            # Should not happen in practice (_on_task_completed/_on_task_failed
            # always store a result before setting the event) — surfaced as a
            # distinct reason rather than silently fabricating success.
            logger.error("[Execution] Task {} signalled completion but no result was recorded", task.task_id)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message="Task signalled completion but no result was recorded",
                reason="internal_error",
                duration_seconds=duration,
            )

        return TaskResult(
            task_id=stored_result.task_id,
            success=stored_result.success,
            result=stored_result.result,
            error_message=stored_result.error_message,
            reason=stored_result.reason,
            duration_seconds=duration,
        )

    def _reflect(self, task: Task, result: TaskResult) -> None:
        """Stage 5: evaluate the outcome. Lightweight in this phase: structured logging only.

        A later phase can extend this to decide *what* about the outcome
        is worth remembering (e.g. "this capability keeps timing out")
        without changing the loop's shape.
        """
        if result.success:
            logger.info(
                "[Reflection] Task {} succeeded in {:.3f}s: {}",
                task.task_id,
                result.duration_seconds,
                result.result,
            )
        else:
            logger.warning(
                "[Reflection] Task {} failed in {:.3f}s (reason={}): {}",
                task.task_id,
                result.duration_seconds,
                result.reason,
                result.error_message,
            )

    def _update_memory(self, task: Task, result: TaskResult) -> None:
        """Stage 6: placeholder for tiered memory writes (Phase 5).

        No real memory module exists yet, so this only logs that the stage
        ran — it exists now so the six-stage pipeline is complete and
        callable end-to-end, and so Phase 5 has one obvious place to add
        real Episodic/Semantic/Procedural writes later.
        """
        logger.debug(
            "[Memory Update] (placeholder) task {} outcome noted, no memory tier implemented yet",
            task.task_id,
        )

    # ------------------------------------------------------------------ #
    # Event subscribers — correlate async outcomes back to waiting tasks
    # ------------------------------------------------------------------ #
    def _on_task_completed(self, event: TaskCompleted) -> None:
        result = TaskResult(task_id=event.task_id, success=True, result=event.result)
        self._store_result_and_wake(event.task_id, result)

    def _on_task_failed(self, event: TaskFailed) -> None:
        result = TaskResult(
            task_id=event.task_id,
            success=False,
            error_message=event.error_message,
            reason=event.reason,
        )
        self._store_result_and_wake(event.task_id, result)

    def _store_result_and_wake(self, task_id: str, result: TaskResult) -> None:
        with self._lock:
            wait_event = self._pending_events.get(task_id)
            if wait_event is None:
                # No one is waiting on this task_id (e.g. it already timed
                # out and was cleaned up, and the agent responded late).
                # Not an error — just nothing to wake.
                return
            self._pending_results[task_id] = result
        wait_event.set()
