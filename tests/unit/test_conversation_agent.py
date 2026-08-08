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


# --------------------------------------------------------------------------- #
# Phase 4: desktop intent recognition + delegation back through the Executive
# --------------------------------------------------------------------------- #
class _StubExecutive:
    """Test double satisfying agents.conversation_agent.TaskDelegator."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[tuple[str, str, float | None]] = []

    def submit_task(self, capability, description, *, payload=None, timeout_seconds=None):
        self.calls.append((capability, description, timeout_seconds))
        return self._result


class _FakeTaskResult:
    def __init__(self, *, success: bool, result: Any = None, error_message: str | None = None) -> None:
        self.success = success
        self.result = result
        self.error_message = error_message


def test_desktop_looking_prompt_does_not_call_llm() -> None:
    bus = EventBus()
    provider = _StubLLMProvider(response_text="should not be used")
    stub_executive = _StubExecutive(
        _FakeTaskResult(success=True, result={"success": True, "action": "open_application", "details": "Launched 'VS Code'"})
    )
    agent = ConversationAgent(bus, llm_provider=provider, executive=stub_executive)

    result = agent.handle("Open VS Code", context={})

    assert provider.received_prompts == []  # LLM never touched
    assert result["desktop_result"]["action"] == "open_application"


def test_desktop_delegation_submits_correct_capability_and_argument() -> None:
    bus = EventBus()
    stub_executive = _StubExecutive(
        _FakeTaskResult(success=True, result={"success": True, "action": "open_application", "details": "Launched 'VS Code'"})
    )
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="unused"), executive=stub_executive)

    agent.handle("Open VS Code", context={})

    assert stub_executive.calls == [("desktop.open_application", "VS Code", 10.0)]


def test_desktop_delegation_uses_configured_timeout() -> None:
    bus = EventBus()
    stub_executive = _StubExecutive(_FakeTaskResult(success=True, result={"details": "ok"}))
    agent = ConversationAgent(
        bus,
        llm_provider=_StubLLMProvider(response_text="unused"),
        executive=stub_executive,
        desktop_timeout_seconds=25.0,
    )

    agent.handle("close notepad", context={})

    assert stub_executive.calls[0][2] == 25.0


def test_desktop_delegation_success_produces_friendly_response() -> None:
    bus = EventBus()
    stub_executive = _StubExecutive(
        _FakeTaskResult(
            success=True,
            result={"success": True, "action": "take_screenshot", "details": "Screenshot saved to shot.png"},
        )
    )
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="unused"), executive=stub_executive)

    result = agent.handle("take a screenshot", context={})

    assert result["status"] == "success"
    assert "Screenshot saved to shot.png" in result["response"]


def test_desktop_delegation_failure_produces_error_response() -> None:
    bus = EventBus()
    stub_executive = _StubExecutive(_FakeTaskResult(success=False, error_message="No agent for capability"))
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="unused"), executive=stub_executive)

    result = agent.handle("open GhostApp", context={})

    assert result["status"] == "error"
    assert "No agent for capability" in result["response"]


def test_desktop_intent_without_executive_configured_degrades_gracefully() -> None:
    bus = EventBus()
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="unused"))  # no executive=

    result = agent.handle("Open VS Code", context={})

    assert result["status"] == "success"
    assert "not currently connected" in result["response"]


def test_ordinary_conversation_still_goes_to_llm_when_executive_is_configured() -> None:
    bus = EventBus()
    provider = _StubLLMProvider(response_text="The Eiffel Tower is in Paris.")
    stub_executive = _StubExecutive(_FakeTaskResult(success=True, result={}))
    agent = ConversationAgent(bus, llm_provider=provider, executive=stub_executive)

    result = agent.handle("Where is the Eiffel Tower?", context={})

    assert stub_executive.calls == []  # never delegated
    assert result["response"] == "The Eiffel Tower is in Paris."


def test_backward_compatible_construction_without_executive_kwarg() -> None:
    # Every Phase 3 call site constructs ConversationAgent without
    # `executive=` — this must keep working unchanged.
    bus = EventBus()
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="hi"))
    result = agent.handle("hello", context={})
    assert result["response"] == "hi"


# --------------------------------------------------------------------------- #
# End-to-end: real ExecutiveAgent + real DesktopAutomationAgent + ConversationAgent
# --------------------------------------------------------------------------- #
def test_full_pipeline_open_application_via_conversation_agent() -> None:
    from agents.desktop_automation_agent import DesktopAutomationAgent
    from automation.desktop_platform import DesktopPlatform

    class _StubPlatform(DesktopPlatform):
        def open_application(self, name: str) -> str:
            return f"Launched '{name}' for real"

    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(DesktopAutomationAgent(bus, platform=_StubPlatform(screenshot_directory="unused")))
    executive.register_agent(
        ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="unused"), executive=executive)
    )

    result = executive.submit_task("conversation.chat", "Open VS Code")

    assert result.success is True
    # Natural, user-facing phrasing — not the platform's technical details.
    assert result.result["response"] == "VS Code opened successfully."
    assert result.result["desktop_result"]["action"] == "open_application"
    assert result.result["desktop_result"]["details"] == "Launched 'VS Code' for real"


# --------------------------------------------------------------------------- #
# Phase 4 polish: sequential/mixed/repeated commands never leak into each other
# --------------------------------------------------------------------------- #
class _ScriptedExecutive:
    """Records every submit_task call and returns a canned per-capability result."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def submit_task(self, capability, description, *, payload=None, timeout_seconds=None):
        self.calls.append((capability, description))
        return _FakeTaskResult(
            success=True,
            result={"success": True, "action": capability.split(".", 1)[1], "details": f"did {capability}({description!r})"},
        )


def test_sequential_desktop_commands_never_leak_arguments() -> None:
    bus = EventBus()
    executive = _ScriptedExecutive()
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="unused"), executive=executive)

    agent.handle("Open VS Code", context={})
    agent.handle("what is the current directory", context={})
    agent.handle("Close VS Code", context={})
    agent.handle("open Notepad", context={})

    assert executive.calls == [
        ("desktop.open_application", "VS Code"),
        ("desktop.current_directory", ""),
        ("desktop.close_application", "VS Code"),
        ("desktop.open_application", "Notepad"),
    ]


def test_sequential_conversational_commands_never_leak() -> None:
    bus = EventBus()
    provider = _StubLLMProvider(response_text="answer")
    agent = ConversationAgent(bus, llm_provider=provider, executive=_ScriptedExecutive())

    agent.handle("What is the capital of France?", context={})
    agent.handle("How do airplanes fly?", context={})
    agent.handle("Tell me a joke", context={})

    assert provider.received_prompts == [
        "What is the capital of France?",
        "How do airplanes fly?",
        "Tell me a joke",
    ]


def test_mixed_conversation_and_desktop_commands_route_independently() -> None:
    bus = EventBus()
    provider = _StubLLMProvider(response_text="Paris is the capital of France.")
    executive = _ScriptedExecutive()
    agent = ConversationAgent(bus, llm_provider=provider, executive=executive)

    r1 = agent.handle("What is the capital of France?", context={})
    r2 = agent.handle("Open VS Code", context={})
    r3 = agent.handle("How do airplanes fly?", context={})
    r4 = agent.handle("what is the current directory", context={})

    assert provider.received_prompts == ["What is the capital of France?", "How do airplanes fly?"]
    assert executive.calls == [("desktop.open_application", "VS Code"), ("desktop.current_directory", "")]
    assert r1["response"] == "Paris is the capital of France."
    assert r2["response"] == "VS Code opened successfully."
    assert r4["desktop_result"]["action"] == "current_directory"


def test_repeated_identical_desktop_command_produces_identical_independent_calls() -> None:
    bus = EventBus()
    executive = _ScriptedExecutive()
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="unused"), executive=executive)

    for _ in range(3):
        agent.handle("open VS Code", context={})

    assert executive.calls == [("desktop.open_application", "VS Code")] * 3


def test_no_prior_argument_bleeds_into_argumentless_command() -> None:
    """Directly targets the reported symptom: an app-name argument from a
    prior "open X" call must never prefix a later argument-less command's
    result.
    """
    bus = EventBus()
    executive = _ScriptedExecutive()
    agent = ConversationAgent(bus, llm_provider=_StubLLMProvider(response_text="unused"), executive=executive)

    agent.handle("open VS Code", context={})
    agent.handle("what is the current directory", context={})

    # The second call's description must be exactly "" (current_directory
    # takes no argument) — not "VS Code" and not "VS Codewhat is the
    # current directory" or any other combination of the two calls.
    assert executive.calls[1] == ("desktop.current_directory", "")
