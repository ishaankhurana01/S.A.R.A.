from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from core.event_bus import EventBus
from events.event_types import AgentDelegated, TaskCompleted, TaskFailed


class _EchoAgent(BaseAgent):
    """Test double: returns whatever task_description it received."""

    @property
    def name(self) -> str:
        return "echo_agent"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return ("test.echo",)

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        return {"echoed": task_description}


class _ExplodingAgent(BaseAgent):
    """Test double: always raises, to verify failure reporting."""

    @property
    def name(self) -> str:
        return "exploding_agent"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return ("test.explode",)

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        raise ValueError("simulated agent failure")


def test_agent_ignores_delegation_addressed_to_someone_else() -> None:
    bus = EventBus()
    agent = _EchoAgent(event_bus=bus)
    agent.start()

    completed: list[TaskCompleted] = []
    bus.subscribe(TaskCompleted, completed.append)

    bus.publish(
        AgentDelegated(
            task_id="t1",
            agent_name="someone_else",
            capability="test.echo",
            task_description="not for you",
        )
    )

    assert completed == []
    agent.stop()


def test_agent_handles_addressed_delegation_and_publishes_completion() -> None:
    bus = EventBus()
    agent = _EchoAgent(event_bus=bus)
    agent.start()

    completed: list[TaskCompleted] = []
    bus.subscribe(TaskCompleted, completed.append)

    bus.publish(
        AgentDelegated(
            task_id="t2",
            agent_name="echo_agent",
            capability="test.echo",
            task_description="say hi",
        )
    )

    assert len(completed) == 1
    assert completed[0].task_id == "t2"
    assert completed[0].agent_name == "echo_agent"
    assert completed[0].result == {"echoed": "say hi"}
    agent.stop()


def test_agent_exception_is_reported_as_task_failed() -> None:
    bus = EventBus()
    agent = _ExplodingAgent(event_bus=bus)
    agent.start()

    failed: list[TaskFailed] = []
    bus.subscribe(TaskFailed, failed.append)

    bus.publish(
        AgentDelegated(
            task_id="t3",
            agent_name="exploding_agent",
            capability="test.explode",
            task_description="boom please",
        )
    )

    assert len(failed) == 1
    assert failed[0].task_id == "t3"
    assert failed[0].agent_name == "exploding_agent"
    assert failed[0].reason == "agent_exception"
    assert "simulated agent failure" in failed[0].error_message
    agent.stop()


def test_stop_prevents_further_delegation_handling() -> None:
    bus = EventBus()
    agent = _EchoAgent(event_bus=bus)
    agent.start()
    agent.stop()

    completed: list[TaskCompleted] = []
    bus.subscribe(TaskCompleted, completed.append)

    bus.publish(
        AgentDelegated(
            task_id="t4",
            agent_name="echo_agent",
            capability="test.echo",
            task_description="should not be handled",
        )
    )

    assert completed == []


def test_start_and_stop_are_idempotent() -> None:
    bus = EventBus()
    agent = _EchoAgent(event_bus=bus)
    agent.start()
    agent.start()  # must not double-subscribe
    assert bus.subscriber_count(AgentDelegated) == 1
    agent.stop()
    agent.stop()  # must not raise
