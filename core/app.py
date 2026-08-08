"""
Application lifecycle for S.A.R.A.

``Application`` is the only place that knows the correct startup order for
Phase 1's foundation pieces. Every module that follows (agents, memory,
voice, ...) hooks into this same sequence rather than each phase inventing
its own bootstrap — this is the concrete implementation of
"core/app.py: Application lifecycle (startup/shutdown sequencing)" from
the architecture doc.

Startup order (and why it's fixed):
    1. Load + validate config       — nothing else can configure itself without this
    2. Configure logging            — every subsequent step needs to be able to log
    3. Create the event bus         — context engine and (later) agents need it to exist
    4. Create the service registry  — and register the config + event bus into it,
                                       so any module can resolve them instead of being
                                       handed them via constructor threading
    5. Start the Context Engine     — first real subsystem; publishes ContextUpdated
                                       onto the now-existing event bus
    6. Build the Executive Agent    — registers the Worker Agents: Memory
                                       and Notification are still Phase 2
                                       placeholders; Conversation (Ollama)
                                       and DesktopAutomation (real OS
                                       actions, platform-selected) do real
                                       work, so the framework is live
                                       end-to-end for both talking and
                                       doing
"""

from __future__ import annotations

from config.config_loader import ConfigLoader
from config.config_schema import SaraConfig
from context.context_engine import ContextEngine
from core.event_bus import EventBus
from core.interfaces import LLMProvider
from core.service_registry import ServiceRegistry
from events.event_types import ApplicationShuttingDown, ApplicationStarted
from llm.providers.ollama_provider import OllamaProvider
from automation.desktop_platform import DesktopPlatform
from automation.platform_factory import get_platform
from utils.logger import configure_logging, get_logger
from agents.executive.executive_agent import ExecutiveAgent
from agents.conversation_agent import ConversationAgent
from agents.desktop_automation_agent import DesktopAutomationAgent
from agents.memory_agent import MemoryAgent
from agents.notification_agent import NotificationAgent

logger = get_logger(__name__)


class Application:
    """Owns the startup/shutdown sequence and the top-level object graph.

    Example:
        app = Application(config_path="config/settings.yaml")
        app.startup()
        ...
        app.shutdown()
    """

    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        self._config_path = config_path
        self._config_loader: ConfigLoader | None = None
        self.config: SaraConfig | None = None
        self.event_bus: EventBus | None = None
        self.registry: ServiceRegistry | None = None
        self.context_engine: ContextEngine | None = None
        self.llm_provider: OllamaProvider | None = None
        self.desktop_platform: DesktopPlatform | None = None
        self.executive_agent: ExecutiveAgent | None = None
        self._started = False

    def startup(self) -> None:
        """Boot every Phase 1 subsystem in the required order.

        Raises:
            config.ConfigError / config.ConfigValidationError: If
                settings.yaml is missing or invalid — startup aborts
                before anything else is created.
        """
        if self._started:
            logger.warning("Application.startup() called but already started; ignoring.")
            return

        # 1. Config first — nothing else can configure itself without it.
        self._config_loader = ConfigLoader(self._config_path)
        self.config = self._config_loader.load()

        # 2. Logging — reconfigure using the now-known settings (the
        #    fail-safe default in utils.logger may already have configured
        #    once with defaults if something logged before this point).
        configure_logging(
            log_dir=self.config.logging.log_dir,
            console_level=self.config.logging.console_level,
            file_level=self.config.logging.file_level,
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
        )
        logger.info("Starting {} v{}", self.config.app.name, self.config.app.version)

        # 3. Event bus.
        self.event_bus = EventBus(isolate_subscriber_errors=self.config.event_bus.isolate_subscriber_errors)

        # 4. Service registry — register the two foundation services so
        #    downstream modules resolve them instead of receiving them via
        #    hand-threaded constructor arguments.
        self.registry = ServiceRegistry()
        self.registry.register(SaraConfig, self.config)
        self.registry.register(EventBus, self.event_bus)

        # 5. Context Engine — first real subsystem, built on top of 1-4.
        if self.config.context_engine.enabled:
            self.context_engine = ContextEngine(
                event_bus=self.event_bus,
                config=self.config.context_engine,
            )
            self.registry.register(ContextEngine, self.context_engine)
            self.context_engine.start()
        else:
            logger.info("Context Engine disabled via config; skipping startup.")

        # 6a. LLM Provider — constructing this only stores config (host,
        #     model, timeout); it makes no network call, so it's safe to
        #     build even if Ollama isn't running yet. ConversationAgent is
        #     the only consumer; nothing else resolves LLMProvider.
        self.llm_provider = OllamaProvider(
            host=self.config.llm.host,
            model=self.config.llm.model,
            request_timeout_seconds=self.config.llm.request_timeout_seconds,
            default_temperature=self.config.llm.temperature,
        )
        self.registry.register(LLMProvider, self.llm_provider)

        # 6b. Executive Agent Framework — built on top of the event bus (3)
        #    and, when available, the Context Engine (5) for its Context
        #    Gathering stage. Memory/Notification remain Phase 2
        #    placeholders (log and return success, no real domain work).
        #    Conversation routes through the LLM Provider built in 6a;
        #    DesktopAutomation routes through a platform backend selected
        #    automatically for the running OS. Conversation also gets a
        #    reference back to the Executive Agent itself so it can
        #    re-delegate recognized desktop-action requests — see
        #    agents.conversation_agent.ConversationAgent's docstring.
        self.executive_agent = ExecutiveAgent(
            event_bus=self.event_bus,
            context_engine=self.context_engine,
        )
        self.registry.register(ExecutiveAgent, self.executive_agent)
        self.executive_agent.register_agent(MemoryAgent(event_bus=self.event_bus))
        self.executive_agent.register_agent(NotificationAgent(event_bus=self.event_bus))

        if self.config.automation.enabled:
            self.desktop_platform = get_platform(
                screenshot_directory=self.config.automation.screenshot_directory
            )
            self.registry.register(DesktopPlatform, self.desktop_platform)
            self.executive_agent.register_agent(
                DesktopAutomationAgent(event_bus=self.event_bus, platform=self.desktop_platform)
            )
        else:
            logger.info("Desktop automation disabled via config; DesktopAutomationAgent not registered.")

        self.executive_agent.register_agent(
            ConversationAgent(
                event_bus=self.event_bus,
                llm_provider=self.llm_provider,
                executive=self.executive_agent,
                desktop_timeout_seconds=self.config.automation.action_timeout_seconds,
            )
        )
        logger.info(
            "Executive Agent ready with {} registered agent(s), {} known capabilities",
            len(self.executive_agent.registered_agent_names),
            len(self.executive_agent.known_capabilities),
        )

        self._started = True
        self.event_bus.publish(ApplicationStarted(source="core.app.Application"))
        logger.info("S.A.R.A. core initialized")

    def shutdown(self) -> None:
        """Gracefully stop every running subsystem, in reverse startup order."""
        if not self._started:
            return

        logger.info("Shutting down S.A.R.A.")
        if self.event_bus is not None:
            self.event_bus.publish(ApplicationShuttingDown(source="core.app.Application"))

        if self.context_engine is not None:
            self.context_engine.stop()

        if self.executive_agent is not None:
            for agent_name in list(self.executive_agent.registered_agent_names):
                self.executive_agent.unregister_agent(agent_name)

        self._started = False
        logger.info("Shutdown complete")

    def __enter__(self) -> "Application":
        self.startup()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.shutdown()
