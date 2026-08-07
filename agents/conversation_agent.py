"""
Conversation Agent.

The first Worker Agent that does real work rather than logging a
placeholder. It owns exactly one responsibility: take a task description,
hand it to its ``core.interfaces.LLMProvider`` (in practice,
``llm.providers.ollama_provider.OllamaProvider``), and return the
generated text. Everything else — event bus plumbing, filtering
delegations addressed to it, turning exceptions into ``TaskFailed`` — is
inherited from ``agents.base_agent.BaseAgent`` unchanged, exactly like the
Phase 2 placeholder agents.

Per requirement #5/#6, no other agent holds a reference to an
``LLMProvider``: this class is the single seam between the Executive
Agent Framework and the LLM. ``core.app.Application`` is the only other
place an ``LLMProvider`` instance gets constructed, and it's threaded in
here via the constructor — never resolved ad hoc elsewhere.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from core.interfaces import LLMProvider
from utils.logger import get_logger

logger = get_logger(__name__)

_CAPABILITIES: tuple[str, ...] = ("conversation.chat",)


class ConversationAgent(BaseAgent):
    """Worker Agent that answers conversational requests via an LLMProvider.

    Example:
        agent = ConversationAgent(event_bus=bus, llm_provider=OllamaProvider(...))
        executive.register_agent(agent)
        result = executive.submit_task("conversation.chat", "What's the weather like on Mars?")
    """

    def __init__(self, event_bus: Any, *, llm_provider: LLMProvider) -> None:
        super().__init__(event_bus)
        self._llm_provider = llm_provider

    @property
    def name(self) -> str:
        return "conversation_agent"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return _CAPABILITIES

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        """Generate a response to ``task_description`` via the LLM provider.

        Args:
            task_description: The user's prompt, verbatim — this phase
                does no prompt engineering, memory injection, or context
                assembly (those are later-phase concerns; see
                ``llm.prompt_builder`` in the architecture doc, not yet
                implemented). The World Model snapshot is available on
                ``context["_context"]`` if present, but intentionally
                unused here for the same reason.

        Returns:
            A result dict carrying the generated text under ``"response"``.

        Raises:
            utils.exceptions.LLMError: Any provider failure (unavailable,
                timeout, model not found, invalid response) propagates
                unchanged. This method does not catch it — ``BaseAgent``
                already converts any exception raised here into a
                ``TaskFailed`` event, so adding a try/except here would
                only duplicate that handling without adding information.
        """
        logger.info("[ConversationAgent] generating response for prompt ({} chars)", len(task_description))
        response_text = self._llm_provider.generate(task_description)
        return {
            "status": "success",
            "agent": self.name,
            "response": response_text,
        }
