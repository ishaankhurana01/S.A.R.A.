from __future__ import annotations

from typing import Any

import pytest

from agents.executive.capability_registry import CapabilityRegistry
from core.event_bus import EventBus
from core.interfaces import Agent
from events.event_types import AgentRegistered, AgentUnregistered
from utils.exceptions import (
    AgentAlreadyRegisteredError,
    CapabilityAlreadyRegisteredError,
    CapabilityNotFoundError,
)


class _StubAgent(Agent):
    """Minimal Agent double — no event bus, no BaseAgent machinery — just for registry tests."""

    def __init__(self, name: str, capabilities: tuple[str, ...]) -> None:
        self._name = name
        self._capabilities = capabilities

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self._capabilities

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        return {"status": "success"}


def test_register_and_resolve() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)
    agent = _StubAgent("desktop_agent", ("desktop.open_application",))

    registry.register(agent)

    assert registry.resolve("desktop.open_application") == "desktop_agent"
    assert registry.is_registered("desktop_agent") is True


def test_resolve_unknown_capability_raises() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)

    with pytest.raises(CapabilityNotFoundError):
        registry.resolve("does.not.exist")


def test_registering_agent_publishes_agent_registered_event() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)
    received: list[AgentRegistered] = []
    bus.subscribe(AgentRegistered, received.append)

    agent = _StubAgent("notification_agent", ("notify.send",))
    registry.register(agent)

    assert len(received) == 1
    assert received[0].agent_name == "notification_agent"
    assert received[0].capabilities == ("notify.send",)


def test_unregister_publishes_agent_unregistered_event_and_frees_capability() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)
    received: list[AgentUnregistered] = []
    bus.subscribe(AgentUnregistered, received.append)

    agent = _StubAgent("memory_agent", ("memory.recall",))
    registry.register(agent)
    registry.unregister("memory_agent")

    assert len(received) == 1
    assert received[0].agent_name == "memory_agent"
    assert registry.is_registered("memory_agent") is False
    with pytest.raises(CapabilityNotFoundError):
        registry.resolve("memory.recall")


def test_unregister_unknown_agent_is_noop() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)
    registry.unregister("never_registered")  # must not raise


def test_duplicate_agent_name_without_override_raises() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)
    registry.register(_StubAgent("desktop_agent", ("desktop.open_application",)))

    with pytest.raises(AgentAlreadyRegisteredError):
        registry.register(_StubAgent("desktop_agent", ("desktop.close_application",)))


def test_duplicate_capability_across_different_agents_raises() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)
    registry.register(_StubAgent("agent_one", ("shared.capability",)))

    with pytest.raises(CapabilityAlreadyRegisteredError):
        registry.register(_StubAgent("agent_two", ("shared.capability",)))


def test_multiple_agents_with_distinct_capabilities() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)
    registry.register(_StubAgent("desktop_agent", ("desktop.open_application",)))
    registry.register(_StubAgent("memory_agent", ("memory.recall", "memory.store")))
    registry.register(_StubAgent("notification_agent", ("notify.send",)))

    assert registry.resolve("desktop.open_application") == "desktop_agent"
    assert registry.resolve("memory.recall") == "memory_agent"
    assert registry.resolve("memory.store") == "memory_agent"
    assert registry.resolve("notify.send") == "notification_agent"
    assert set(registry.list_agents()) == {"desktop_agent", "memory_agent", "notification_agent"}
    assert set(registry.list_capabilities()) == {
        "desktop.open_application",
        "memory.recall",
        "memory.store",
        "notify.send",
    }


def test_allow_override_replaces_existing_registration() -> None:
    bus = EventBus()
    registry = CapabilityRegistry(event_bus=bus)
    registry.register(_StubAgent("desktop_agent", ("desktop.open_application",)))

    registry.register(
        _StubAgent("desktop_agent", ("desktop.open_application", "desktop.close_application")),
        allow_override=True,
    )

    assert registry.resolve("desktop.close_application") == "desktop_agent"
