"""
End-to-end tests for the Executive Agent Framework.

Unlike ``test_capability_registry.py`` (registry mechanics in isolation)
and ``test_base_agent.py`` (a single agent's event-bus plumbing in
isolation), these tests exercise the full path: ``ExecutiveAgent.submit_task``
-> ``ReasoningLoop`` (Input/Context/Planning/Execution/Reflection/Memory
Update) -> event bus -> a real ``BaseAgent`` subclass -> back through the
event bus to the waiting caller. This is what proves the framework holds
together end-to-end, not just its individual pieces.
"""

from __future__ import annotations

import time
from typing import Any

from agents.base_agent import BaseAgent
from agents.desktop_agent import DesktopAgent
from agents.memory_agent import MemoryAgent
from agents.notification_agent import NotificationAgent
from agents.executive.executive_agent import ExecutiveAgent
from core.event_bus import EventBus
from events.event_types import (
    AgentDelegated,
    AgentRegistered,
    AgentUnregistered,
    TaskCompleted,
    TaskCreated,
    TaskFailed,
    TaskTimeout,
)
from utils.exceptions import CapabilityNotFoundError


class _SlowAgent(BaseAgent):
    """Test double: sleeps longer than the caller's timeout, to exercise TaskTimeout."""

    def __init__(self, event_bus: EventBus, delay_seconds: float) -> None:
        super().__init__(event_bus)
        self._delay_seconds = delay_seconds

    @property
    def name(self) -> str:
        return "slow_agent"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return ("test.slow",)

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        time.sleep(self._delay_seconds)
        return {"status": "success", "note": "arrived too late to matter"}


# --------------------------------------------------------------------------- #
# Successful task routing
# --------------------------------------------------------------------------- #
def test_successful_task_routing_to_desktop_agent() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(DesktopAgent(event_bus=bus))

    result = executive.submit_task("desktop.open_application", "Open VS Code")

    assert result.success is True
    assert result.result["status"] == "success"
    assert result.result["agent"] == "desktop_agent"
    assert result.error_message is None


# --------------------------------------------------------------------------- #
# Multiple registered agents — routing must go to the *correct* one
# --------------------------------------------------------------------------- #
def test_multiple_registered_agents_route_correctly() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(DesktopAgent(event_bus=bus))
    executive.register_agent(MemoryAgent(event_bus=bus))
    executive.register_agent(NotificationAgent(event_bus=bus))

    assert set(executive.registered_agent_names) == {
        "desktop_agent",
        "memory_agent",
        "notification_agent",
    }

    desktop_result = executive.submit_task("desktop.close_application", "Close Chrome")
    memory_result = executive.submit_task("memory.recall", "What did I ask yesterday?")
    notify_result = executive.submit_task("notify.send", "Battery low")

    assert desktop_result.result["agent"] == "desktop_agent"
    assert memory_result.result["agent"] == "memory_agent"
    assert notify_result.result["agent"] == "notification_agent"


# --------------------------------------------------------------------------- #
# Unknown capability
# --------------------------------------------------------------------------- #
def test_unknown_capability_returns_failed_result_not_an_exception() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(DesktopAgent(event_bus=bus))

    result = executive.submit_task("spotify.play_track", "Play some music")

    assert result.success is False
    assert result.reason == "unknown_capability"
    assert "spotify.play_track" in result.error_message


def test_capability_registry_resolve_raises_directly_for_unknown_capability() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)

    try:
        executive.capability_registry.resolve("does.not.exist")
        assert False, "expected CapabilityNotFoundError"
    except CapabilityNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# Agent registration / unregistration
# --------------------------------------------------------------------------- #
def test_register_then_unregister_agent_via_executive() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(NotificationAgent(event_bus=bus))

    assert "notification_agent" in executive.registered_agent_names

    executive.unregister_agent("notification_agent")

    assert "notification_agent" not in executive.registered_agent_names
    result = executive.submit_task("notify.send", "should fail — agent gone")
    assert result.success is False
    assert result.reason == "unknown_capability"


def test_unregistered_agent_stops_receiving_delegations() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    agent = NotificationAgent(event_bus=bus)
    executive.register_agent(agent)
    executive.unregister_agent("notification_agent")

    # BaseAgent.stop() was called by unregister_agent — confirm the bus
    # subscription is actually gone, not just the capability mapping.
    assert bus.subscriber_count(AgentDelegated) == 0


# --------------------------------------------------------------------------- #
# Agent failure
# --------------------------------------------------------------------------- #
def test_agent_failure_surfaces_as_failed_task_result() -> None:
    bus = EventBus()

    class _ExplodingAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "exploding_agent"

        @property
        def capabilities(self) -> tuple[str, ...]:
            return ("test.explode",)

        def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
            raise RuntimeError("simulated failure")

    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(_ExplodingAgent(event_bus=bus))

    result = executive.submit_task("test.explode", "please explode")

    assert result.success is False
    assert result.reason == "agent_exception"
    assert "simulated failure" in result.error_message


# --------------------------------------------------------------------------- #
# Timeout handling
# --------------------------------------------------------------------------- #
def test_timeout_when_agent_takes_too_long() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(_SlowAgent(bus, delay_seconds=0.5))

    timeouts: list[TaskTimeout] = []
    bus.subscribe(TaskTimeout, timeouts.append)

    result = executive.submit_task("test.slow", "take your time", timeout_seconds=0.1)

    assert result.success is False
    assert result.reason == "timeout"
    assert len(timeouts) == 1
    assert timeouts[0].agent_name == "slow_agent"


def test_per_task_timeout_override_is_respected() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus, default_timeout_seconds=5.0)
    executive.register_agent(_SlowAgent(bus, delay_seconds=0.3))

    # Default timeout (5s) would succeed; an explicit short override must
    # still time out, proving the override is actually used.
    result = executive.submit_task("test.slow", "hurry up", timeout_seconds=0.05)

    assert result.success is False
    assert result.reason == "timeout"


# --------------------------------------------------------------------------- #
# Event propagation across the full pipeline
# --------------------------------------------------------------------------- #
def test_full_event_sequence_published_for_a_successful_task() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)

    created: list[TaskCreated] = []
    delegated: list[AgentDelegated] = []
    completed: list[TaskCompleted] = []
    bus.subscribe(TaskCreated, created.append)
    bus.subscribe(AgentDelegated, delegated.append)
    bus.subscribe(TaskCompleted, completed.append)

    executive.register_agent(DesktopAgent(event_bus=bus))
    result = executive.submit_task("desktop.open_application", "Open Terminal")

    assert len(created) == 1
    assert len(delegated) == 1
    assert len(completed) == 1
    # All three events correlate to the same task via task_id.
    assert created[0].task_id == delegated[0].task_id == completed[0].task_id == result.task_id
    assert delegated[0].agent_name == "desktop_agent"


def test_registration_events_are_published_through_executive_agent() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)

    registered: list[AgentRegistered] = []
    unregistered: list[AgentUnregistered] = []
    bus.subscribe(AgentRegistered, registered.append)
    bus.subscribe(AgentUnregistered, unregistered.append)

    executive.register_agent(MemoryAgent(event_bus=bus))
    executive.unregister_agent("memory_agent")

    assert len(registered) == 1
    assert registered[0].agent_name == "memory_agent"
    assert len(unregistered) == 1
    assert unregistered[0].agent_name == "memory_agent"


def test_failed_task_publishes_task_failed_not_task_completed() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)

    completed: list[TaskCompleted] = []
    failed: list[TaskFailed] = []
    bus.subscribe(TaskCompleted, completed.append)
    bus.subscribe(TaskFailed, failed.append)

    class _ExplodingAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "exploding_agent_2"

        @property
        def capabilities(self) -> tuple[str, ...]:
            return ("test.explode2",)

        def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
            raise ValueError("boom")

    executive.register_agent(_ExplodingAgent(event_bus=bus))
    executive.submit_task("test.explode2", "please explode")

    assert completed == []
    assert len(failed) == 1
    assert failed[0].reason == "agent_exception"
