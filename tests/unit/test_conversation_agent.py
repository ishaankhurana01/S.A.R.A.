"""
Tests for ``agents.conversation_agent.ConversationAgent``.

Two layers, matching the pattern established in Phase 2's tests:
  - Agent-level: publish ``AgentDelegated`` directly, inspect the
    resulting ``TaskCompleted``/``TaskFailed`` — proves the agent's own
    behavior in isolation from routing.
  - Executive-level: go through ``ExecutiveAgent.submit_task`` for a full
    Task -> Executive -> Capability Registry -> ConversationAgent ->
    LLMProvider -> Executive -> Result round trip — proves requirement #6's
    exact pipeline shape end-to-end.

Both layers use a hand-written ``_StubLLMProvider`` test double rather
than mocking ``requests`` again — ``OllamaProvider`` already has its own
dedicated tests in ``test_ollama_provider.py``; these tests are about the
agent and routing behavior, not the HTTP layer.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.conversation_agent import ConversationAgent
from agents.desktop_agent import DesktopAgent
from agents.executive.executive_agent import ExecutiveAgent
from core.event_bus import EventBus
from core.interfaces import LLMProvider
from events.event_types import AgentDelegated, TaskCompleted, TaskFailed
from utils.exceptions import LLMModelNotFoundError, LLMProviderUnavailableError


class _StubLLMProvider(LLMProvider):
    """Test double: returns a canned response, or raises a canned exception."""

    def __init__(self, *, response_text: str | None = None, raises: Exception | None = None) -> None:
        self._response_text = response_text
        self._raises = raises
        self.received_prompts: list[str] = []

    def generate(self, prompt: str, *, temperature: float = 0.7) -> str:
        self.received_prompts.append(prompt)
        if self._raises is not None:
            raise self._raises
        assert self._response_text is not None
        return self._response_text

    def is_available(self) -> bool:
        return self._raises is None


# --------------------------------------------------------------------------- #
# ConversationAgent behavior (agent-level, isolated from the Executive Agent)
# --------------------------------------------------------------------------- #
def test_agent_declares_conversation_capability() -> None:
    bus = EventBus()
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="hi"))

    assert agent.name == "conversation_agent"
    assert agent.capabilities == ("conversation.chat",)


def test_handle_returns_provider_response() -> None:
    bus = EventBus()
    provider = _StubLLMProvider(response_text="The sky is blue due to Rayleigh scattering.")
    agent = ConversationAgent(bus, llm_provider=provider)

    result = agent.handle("Why is the sky blue?", context={})

    assert result["status"] == "success"
    assert result["agent"] == "conversation_agent"
    assert result["response"] == "The sky is blue due to Rayleigh scattering."
    assert provider.received_prompts == ["Why is the sky blue?"]


def test_handle_propagates_provider_exception() -> None:
    bus = EventBus()
    provider = _StubLLMProvider(raises=LLMProviderUnavailableError("Ollama is not running"))
    agent = ConversationAgent(bus, llm_provider=provider)

    with pytest.raises(LLMProviderUnavailableError):
        agent.handle("hello", context={})


def test_agent_publishes_task_completed_via_bus() -> None:
    bus = EventBus()
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="hello!"))
    agent.start()

    completed: list[TaskCompleted] = []
    bus.subscribe(TaskCompleted, completed.append)

    bus.publish(
        AgentDelegated(
            task_id="conv-1",
            agent_name="conversation_agent",
            capability="conversation.chat",
            task_description="hi there",
        )
    )

    assert len(completed) == 1
    assert completed[0].result["response"] == "hello!"
    agent.stop()


def test_agent_publishes_task_failed_when_provider_raises() -> None:
    bus = EventBus()
    agent = ConversationAgent(
        bus, llm_provider=_StubLLMProvider(raises=LLMModelNotFoundError("model not found"))
    )
    agent.start()

    failed: list[TaskFailed] = []
    bus.subscribe(TaskFailed, failed.append)

    bus.publish(
        AgentDelegated(
            task_id="conv-2",
            agent_name="conversation_agent",
            capability="conversation.chat",
            task_description="hi there",
        )
    )

    assert len(failed) == 1
    assert failed[0].reason == "agent_exception"
    assert "model not found" in failed[0].error_message
    agent.stop()


# --------------------------------------------------------------------------- #
# Executive routing: Task -> Executive -> Capability Registry ->
# ConversationAgent -> LLMProvider -> Executive -> Result
# --------------------------------------------------------------------------- #
def test_successful_conversation_end_to_end_through_executive() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    provider = _StubLLMProvider(response_text="Paris is the capital of France.")
    executive.register_agent(ConversationAgent(bus, llm_provider=provider))

    result = executive.submit_task("conversation.chat", "What is the capital of France?")

    assert result.success is True
    assert result.result["response"] == "Paris is the capital of France."
    assert provider.received_prompts == ["What is the capital of France?"]


def test_executive_routes_conversation_capability_to_conversation_agent_only() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    provider = _StubLLMProvider(response_text="hi!")
    executive.register_agent(DesktopAgent(event_bus=bus))
    executive.register_agent(ConversationAgent(bus, llm_provider=provider))

    assert executive.capability_registry.resolve("conversation.chat") == "conversation_agent"

    result = executive.submit_task("conversation.chat", "hello")
    assert result.result["agent"] == "conversation_agent"


def test_unavailable_ollama_surfaces_as_failed_task_result() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    provider = _StubLLMProvider(raises=LLMProviderUnavailableError("Could not reach Ollama"))
    executive.register_agent(ConversationAgent(bus, llm_provider=provider))

    result = executive.submit_task("conversation.chat", "hello?")

    assert result.success is False
    assert result.reason == "agent_exception"
    assert "Could not reach Ollama" in result.error_message


def test_unknown_model_surfaces_as_failed_task_result() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    provider = _StubLLMProvider(raises=LLMModelNotFoundError("Model 'ghost' is not available"))
    executive.register_agent(ConversationAgent(bus, llm_provider=provider))

    result = executive.submit_task("conversation.chat", "hello?")

    assert result.success is False
    assert "not available" in result.error_message


def test_provider_failure_does_not_crash_executive_or_other_agents() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(DesktopAgent(event_bus=bus))
    executive.register_agent(
        ConversationAgent(bus, llm_provider=_StubLLMProvider(raises=RuntimeError("boom")))
    )

    conv_result = executive.submit_task("conversation.chat", "hello?")
    assert conv_result.success is False

    # The Executive Agent and event bus are still fully functional for
    # other capabilities after a Conversation Agent failure.
    desktop_result = executive.submit_task("desktop.open_application", "Open Notepad")
    assert desktop_result.success is True


def test_conversation_agent_registered_alongside_placeholders_without_conflict() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(DesktopAgent(event_bus=bus))
    executive.register_agent(ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="ok")))

    assert set(executive.registered_agent_names) == {"desktop_agent", "conversation_agent"}
    assert "conversation.chat" in executive.known_capabilities
    assert "desktop.open_application" in executive.known_capabilities
