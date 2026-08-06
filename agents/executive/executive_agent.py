"""
Executive Agent.

The Executive Agent is a pure coordinator: it owns a
``agents.executive.capability_registry.CapabilityRegistry`` (who can do
what) and a ``agents.executive.reasoning_loop.ReasoningLoop`` (the
standard Input -> Context -> Plan -> Execute -> Reflect -> Memory Update
pipeline), and nothing else. It has no ``if capability == "..."``
branches, no domain logic, and no direct references to any specific
Worker Agent class — every interaction with a Worker Agent happens
through ``register_agent``/``unregister_agent`` (Capability Registry) and
``submit_task`` (reasoning loop, over the event bus). This is the
"never hardcodes agent names" requirement, structurally enforced rather
than just a coding convention.
"""

from __future__ import annotations

from context.context_engine import ContextEngine
from core.event_bus import EventBus
from core.interfaces import Agent
from agents.executive.capability_registry import CapabilityRegistry
from agents.executive.reasoning_loop import ReasoningLoop, TaskResult
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 5.0


class ExecutiveAgent:
    """Central coordinator: delegates tasks to registered Worker Agents.

    Example:
        executive = ExecutiveAgent(event_bus=bus, context_engine=engine)
        executive.register_agent(DesktopAgent(event_bus=bus))
        result = executive.submit_task(
            "desktop.open_application", "Open VS Code"
        )
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        context_engine: ContextEngine | None = None,
        default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._event_bus = event_bus
        self.capability_registry = CapabilityRegistry(event_bus=event_bus)
        self._reasoning_loop = ReasoningLoop(
            event_bus=event_bus,
            capability_registry=self.capability_registry,
            context_engine=context_engine,
            default_timeout_seconds=default_timeout_seconds,
        )
        self._agents: dict[str, Agent] = {}
        logger.info("Executive Agent initialized")

    def register_agent(self, agent: Agent, *, allow_override: bool = False) -> None:
        """Register a Worker Agent and start it listening on the event bus.

        Args:
            agent: A ``core.interfaces.Agent`` implementation (typically a
                ``agents.base_agent.BaseAgent`` subclass).
            allow_override: Forwarded to ``CapabilityRegistry.register``.
        """
        self.capability_registry.register(agent, allow_override=allow_override)
        self._agents[agent.name] = agent
        start = getattr(agent, "start", None)
        if callable(start):
            start()
        logger.info("Executive Agent: '{}' registered and started", agent.name)

    def unregister_agent(self, agent_name: str) -> None:
        """Stop and remove a Worker Agent.

        Safe to call with an unknown ``agent_name`` (no-op), matching
        ``CapabilityRegistry.unregister``.
        """
        agent = self._agents.pop(agent_name, None)
        if agent is not None:
            stop = getattr(agent, "stop", None)
            if callable(stop):
                stop()
        self.capability_registry.unregister(agent_name)
        logger.info("Executive Agent: '{}' unregistered", agent_name)

    def submit_task(
        self,
        capability: str,
        description: str,
        *,
        payload: dict | None = None,
        timeout_seconds: float | None = None,
    ) -> TaskResult:
        """Submit a task for delegation to whichever agent handles ``capability``.

        This is the Executive Agent's only "do work" method, and it does
        no work itself — it hands the request straight to the reasoning
        loop, which performs Planning (via the Capability Registry) and
        Execution (via the event bus).

        Args:
            capability: The capability identifier needed (e.g.
                ``"notify.send"``).
            description: Human-readable task description.
            payload: Extra structured data for the Worker Agent.
            timeout_seconds: Overrides the default execution timeout for
                this task only.

        Returns:
            A ``TaskResult`` — see ``agents.executive.reasoning_loop.TaskResult``.
        """
        return self._reasoning_loop.run(
            capability,
            description,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )

    @property
    def registered_agent_names(self) -> tuple[str, ...]:
        """Names of every currently registered Worker Agent."""
        return self.capability_registry.list_agents()

    @property
    def known_capabilities(self) -> tuple[str, ...]:
        """Every capability currently handled by some registered agent."""
        return self.capability_registry.list_capabilities()
