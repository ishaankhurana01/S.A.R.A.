"""
Capability Registry.

This is what lets the Executive Agent be genuinely agent-agnostic: it
never contains a line like ``if capability == "desktop.open_application":
use DesktopAgent``. Instead, each ``agents.base_agent.BaseAgent``
declares its own ``capabilities`` tuple, registers itself here, and the
Executive Agent's Planning step asks the registry "who handles this
capability" at request time.

Adding a ninth Worker Agent later means: implement it, register it. Zero
changes to ``executive_agent.py`` or ``reasoning_loop.py`` — this is the
concrete mechanism behind the architecture doc's "adding a new capability
is additive, never a modification."
"""

from __future__ import annotations

import threading

from core.event_bus import EventBus
from core.interfaces import Agent
from events.event_types import AgentRegistered, AgentUnregistered
from utils.exceptions import (
    AgentAlreadyRegisteredError,
    CapabilityAlreadyRegisteredError,
    CapabilityNotFoundError,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class CapabilityRegistry:
    """Maps capability strings to the single agent that handles them.

    Example:
        registry = CapabilityRegistry(event_bus=bus)
        registry.register(DesktopAgent(event_bus=bus))
        agent_name = registry.resolve("desktop.open_application")  # "desktop_agent"
    """

    def __init__(self, *, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._agents: dict[str, Agent] = {}
        self._capability_to_agent: dict[str, str] = {}
        self._lock = threading.RLock()

    def register(self, agent: Agent, *, allow_override: bool = False) -> None:
        """Register a Worker Agent and every capability it declares.

        Args:
            agent: A concrete ``core.interfaces.Agent`` (in practice, a
                ``agents.base_agent.BaseAgent`` subclass instance).
            allow_override: If True, permits replacing an already-registered
                agent name or reassigning an already-claimed capability.
                Intended for tests/hot-reload scenarios, not routine use.

        Raises:
            AgentAlreadyRegisteredError: If ``agent.name`` is already
                registered and ``allow_override`` is False.
            CapabilityAlreadyRegisteredError: If any capability
                ``agent`` declares is already claimed by a different
                agent and ``allow_override`` is False.
        """
        with self._lock:
            if agent.name in self._agents and not allow_override:
                raise AgentAlreadyRegisteredError(
                    f"Agent '{agent.name}' is already registered.",
                    context={"agent_name": agent.name},
                )

            for capability in agent.capabilities:
                existing_owner = self._capability_to_agent.get(capability)
                if existing_owner is not None and existing_owner != agent.name and not allow_override:
                    raise CapabilityAlreadyRegisteredError(
                        f"Capability '{capability}' is already claimed by agent "
                        f"'{existing_owner}'; cannot also assign it to '{agent.name}'.",
                        context={"capability": capability, "existing_owner": existing_owner},
                    )

            self._agents[agent.name] = agent
            for capability in agent.capabilities:
                self._capability_to_agent[capability] = agent.name

        logger.info("Registered agent '{}' with capabilities {}", agent.name, agent.capabilities)
        self._event_bus.publish(
            AgentRegistered(
                source="agents.executive.capability_registry.CapabilityRegistry",
                agent_name=agent.name,
                capabilities=tuple(agent.capabilities),
            )
        )

    def unregister(self, agent_name: str) -> None:
        """Remove an agent and every capability it owned.

        Safe to call with an unknown ``agent_name`` — this is a no-op in
        that case rather than raising, matching
        ``core.service_registry.ServiceRegistry.unregister``'s behavior.
        """
        with self._lock:
            if agent_name not in self._agents:
                return
            del self._agents[agent_name]
            owned_capabilities = [
                capability
                for capability, owner in self._capability_to_agent.items()
                if owner == agent_name
            ]
            for capability in owned_capabilities:
                del self._capability_to_agent[capability]

        logger.info("Unregistered agent '{}'", agent_name)
        self._event_bus.publish(
            AgentUnregistered(
                source="agents.executive.capability_registry.CapabilityRegistry",
                agent_name=agent_name,
            )
        )

    def resolve(self, capability: str) -> str:
        """Return the name of the agent registered to handle ``capability``.

        Args:
            capability: The capability identifier requested by the
                Executive Agent's Planning step (e.g. ``"desktop.open_application"``).

        Returns:
            The registered agent's name.

        Raises:
            CapabilityNotFoundError: If no agent currently declares this
                capability. This is a normal, expected outcome (e.g. the
                user asked for something nothing implements yet) — callers
                should catch it and fail the task gracefully, not treat it
                as a bug.
        """
        with self._lock:
            agent_name = self._capability_to_agent.get(capability)
        if agent_name is None:
            raise CapabilityNotFoundError(
                f"No registered agent declares capability '{capability}'.",
                context={"capability": capability, "known_capabilities": self.list_capabilities()},
            )
        return agent_name

    def list_agents(self) -> tuple[str, ...]:
        """Return the names of all currently registered agents."""
        with self._lock:
            return tuple(self._agents.keys())

    def list_capabilities(self) -> tuple[str, ...]:
        """Return every capability currently claimed by some agent."""
        with self._lock:
            return tuple(self._capability_to_agent.keys())

    def is_registered(self, agent_name: str) -> bool:
        """Return whether ``agent_name`` is currently registered."""
        with self._lock:
            return agent_name in self._agents
