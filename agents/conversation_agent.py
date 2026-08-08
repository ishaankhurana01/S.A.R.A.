"""
Conversation Agent.

Owns exactly one capability, ``conversation.chat``, but as of Phase 4 it
makes one decision before answering: does this prompt describe a desktop
action ("open VS Code", "take a screenshot"), or is it ordinary
conversation? Recognition is delegated to
``agents.desktop_intent.recognize_desktop_intent`` (simple rule-based
matching — see that module's docstring for why); this class only decides
what to do with the answer.

Requirement #7/#8 shape this precisely:
    - A desktop-looking request is *never* executed here. This class has
      no reference to ``automation.desktop_platform.DesktopPlatform`` and
      never will — it only knows how to ask the Executive Agent to route
      the request to whichever agent owns that capability
      (``agents.desktop_automation_agent.DesktopAutomationAgent``, in
      practice, but ``ConversationAgent`` doesn't know or care).
    - Everything else still goes to the LLM exactly as in Phase 3.

Delegating back through the Executive Agent (rather than, say, publishing
an event directly) means this class needs *some* reference back to the
coordinator. That reference is typed against ``TaskDelegator`` — a
two-method Protocol, not the concrete ``ExecutiveAgent`` class — so this
module has zero import-time coupling to ``agents.executive.*`` and stays
just as easy to unit test as it was in Phase 3 (a stub satisfying the
Protocol is enough; see ``tests/unit/test_conversation_agent.py``). The
``executive`` parameter is optional and defaults to ``None`` specifically
so every existing Phase 3 call site and test — none of which pass it —
keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agents.base_agent import BaseAgent
from agents.desktop_intent import DesktopIntent, recognize_desktop_intent
from agents.desktop_response import describe_desktop_result
from core.interfaces import LLMProvider
from utils.logger import get_logger

logger = get_logger(__name__)

_CAPABILITIES: tuple[str, ...] = ("conversation.chat",)
_DEFAULT_DESKTOP_TIMEOUT_SECONDS = 10.0


@runtime_checkable
class TaskDelegator(Protocol):
    """The one Executive Agent method ConversationAgent needs: submit_task.

    Satisfied by ``agents.executive.executive_agent.ExecutiveAgent`` in
    production, and trivially by a hand-written stub in tests — see
    ``core.app.Application`` for the real wiring.
    """

    def submit_task(
        self,
        capability: str,
        description: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any: ...


class ConversationAgent(BaseAgent):
    """Worker Agent that answers conversational requests and routes desktop-action ones.

    Example:
        agent = ConversationAgent(
            event_bus=bus,
            llm_provider=OllamaProvider(...),
            executive=executive_agent,  # enables desktop-action delegation
        )
        executive_agent.register_agent(agent)
        executive_agent.submit_task("conversation.chat", "Open VS Code")
        # -> internally re-delegates to desktop.open_application, and the
        #    caller still just sees one TaskResult back.
    """

    def __init__(
        self,
        event_bus: Any,
        *,
        llm_provider: LLMProvider,
        executive: TaskDelegator | None = None,
        desktop_timeout_seconds: float = _DEFAULT_DESKTOP_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(event_bus)
        self._llm_provider = llm_provider
        self._executive = executive
        self._desktop_timeout_seconds = desktop_timeout_seconds

    @property
    def name(self) -> str:
        return "conversation_agent"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return _CAPABILITIES

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        """Answer ``task_description`` — conversationally, or by delegating a desktop action.

        Args:
            task_description: The user's prompt, verbatim.

        Returns:
            A result dict. For ordinary conversation, carries the LLM's
            text under ``"response"`` (unchanged from Phase 3). For a
            recognized desktop action, also carries the desktop
            action's own structured result under ``"desktop_result"``
            (see ``agents.desktop_automation_agent.DesktopAutomationAgent``).

        Raises:
            utils.exceptions.LLMError: For ordinary conversation, any
                provider failure propagates unchanged — see Phase 3's
                docstring for why this method doesn't catch it.
        """
        intent = recognize_desktop_intent(task_description)
        if intent is not None:
            return self._delegate_desktop_action(intent)

        logger.info("[ConversationAgent] generating response for prompt ({} chars)", len(task_description))
        response_text = self._llm_provider.generate(task_description)
        return {
            "status": "success",
            "agent": self.name,
            "response": response_text,
        }

    def _delegate_desktop_action(self, intent: DesktopIntent) -> dict[str, Any]:
        """Re-submit a recognized desktop action to the Executive Agent and summarize the outcome.

        This method never touches the OS itself — it only calls
        ``self._executive.submit_task(...)``, satisfying requirement #7
        structurally rather than just by convention.
        """
        if self._executive is None:
            logger.warning(
                "[ConversationAgent] recognized desktop intent '{}' but no Executive delegator "
                "is configured; cannot perform it.",
                intent.capability,
            )
            return {
                "status": "success",
                "agent": self.name,
                "response": (
                    "I recognized that as a desktop action, but I'm not currently "
                    "connected to the Executive Agent to carry it out."
                ),
            }

        logger.info(
            "[ConversationAgent] delegating desktop intent -> capability='{}' argument={!r}",
            intent.capability,
            intent.argument,
        )
        result = self._executive.submit_task(
            intent.capability,
            intent.argument,
            timeout_seconds=self._desktop_timeout_seconds,
        )

        desktop_result = result.result if isinstance(result.result, dict) else None
        response_text = describe_desktop_result(
            intent,
            success=result.success,
            desktop_result=desktop_result,
            error_message=result.error_message,
        )
        task_succeeded = result.success and bool(desktop_result and desktop_result.get("success"))

        return {
            "status": "success" if task_succeeded else "error",
            "agent": self.name,
            "response": response_text,
            "desktop_result": desktop_result,
        }
