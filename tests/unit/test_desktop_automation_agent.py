"""
Tests for ``agents.desktop_automation_agent.DesktopAutomationAgent`` and
``agents.desktop_intent.recognize_desktop_intent``.

Mirrors the two-layer pattern used for ``test_conversation_agent.py``:
agent-level tests publish ``AgentDelegated`` directly against a stub
``DesktopPlatform``; executive-level tests go through
``ExecutiveAgent.submit_task`` for the full routing path. Neither layer
touches a real ``DesktopPlatform`` implementation — that's
``test_desktop_platform.py`` / ``test_desktop_platform_impls.py``'s job.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.desktop_automation_agent import DesktopAutomationAgent
from agents.desktop_intent import DesktopIntent, recognize_desktop_intent
from agents.executive.executive_agent import ExecutiveAgent
from automation.desktop_platform import DesktopPlatform
from core.event_bus import EventBus
from events.event_types import AgentDelegated, TaskCompleted, TaskFailed
from utils.exceptions import ApplicationLaunchError, InvalidDesktopTargetError


class _StubDesktopPlatform(DesktopPlatform):
    """Test double recording calls and returning/raising canned results."""

    def __init__(self) -> None:
        super().__init__(screenshot_directory="unused")
        self.calls: list[tuple[str, tuple]] = []
        self._raises: dict[str, Exception] = {}

    def fail_next(self, method_name: str, exc: Exception) -> None:
        self._raises[method_name] = exc

    def _record(self, method_name: str, *args) -> None:
        self.calls.append((method_name, args))
        if method_name in self._raises:
            raise self._raises.pop(method_name)

    def open_application(self, name: str) -> str:
        self._record("open_application", name)
        return f"Launched '{name}'"

    def close_application(self, name: str) -> str:
        self._record("close_application", name)
        return f"Closed '{name}'"

    def open_url(self, url: str) -> str:
        self._record("open_url", url)
        return f"Opened '{url}'"

    def list_processes(self, *, limit: int = 40) -> str:
        self._record("list_processes")
        return "3 running application(s): a, b, c"

    def current_directory(self) -> str:
        self._record("current_directory")
        return "Current working directory: /home/user"

    def take_screenshot(self, save_path: str | None = None) -> str:
        self._record("take_screenshot", save_path)
        return f"Screenshot saved to {save_path or 'default.png'}"


# --------------------------------------------------------------------------- #
# Agent-level: capabilities, dispatch, structured result shape
# --------------------------------------------------------------------------- #
def test_agent_declares_all_six_capabilities() -> None:
    bus = EventBus()
    agent = DesktopAutomationAgent(bus, platform=_StubDesktopPlatform())

    assert agent.name == "desktop_automation_agent"
    assert set(agent.capabilities) == {
        "desktop.open_application",
        "desktop.close_application",
        "desktop.open_url",
        "desktop.list_processes",
        "desktop.current_directory",
        "desktop.take_screenshot",
    }


@pytest.mark.parametrize(
    "capability,argument,expected_action",
    [
        ("desktop.open_application", "VS Code", "open_application"),
        ("desktop.close_application", "Chrome", "close_application"),
        ("desktop.open_url", "https://example.com", "open_url"),
        ("desktop.list_processes", "", "list_processes"),
        ("desktop.current_directory", "", "current_directory"),
        ("desktop.take_screenshot", "", "take_screenshot"),
    ],
)
def test_handle_returns_structured_result_for_each_capability(
    capability: str, argument: str, expected_action: str
) -> None:
    bus = EventBus()
    platform = _StubDesktopPlatform()
    agent = DesktopAutomationAgent(bus, platform=platform)

    result = agent.handle(argument, context={"_capability": capability})

    assert result["success"] is True
    assert result["action"] == expected_action
    assert isinstance(result["details"], str)
    assert isinstance(result["duration_ms"], float)
    assert result["duration_ms"] >= 0


def test_handle_unknown_capability_raises_value_error() -> None:
    bus = EventBus()
    agent = DesktopAutomationAgent(bus, platform=_StubDesktopPlatform())

    with pytest.raises(ValueError):
        agent.handle("something", context={"_capability": "desktop.unknown_thing"})


def test_handle_returns_failure_result_on_platform_error_without_raising() -> None:
    bus = EventBus()
    platform = _StubDesktopPlatform()
    platform.fail_next("open_application", ApplicationLaunchError("app not found"))
    agent = DesktopAutomationAgent(bus, platform=platform)

    result = agent.handle("GhostApp", context={"_capability": "desktop.open_application"})

    assert result["success"] is False
    assert result["action"] == "open_application"
    assert "app not found" in result["details"]


def test_handle_propagates_unexpected_exceptions() -> None:
    bus = EventBus()
    platform = _StubDesktopPlatform()
    platform.fail_next("open_application", RuntimeError("totally unexpected bug"))
    agent = DesktopAutomationAgent(bus, platform=platform)

    with pytest.raises(RuntimeError):
        agent.handle("GhostApp", context={"_capability": "desktop.open_application"})


def test_no_argument_actions_ignore_task_description() -> None:
    bus = EventBus()
    platform = _StubDesktopPlatform()
    agent = DesktopAutomationAgent(bus, platform=platform)

    agent.handle("this text is irrelevant", context={"_capability": "desktop.list_processes"})

    assert platform.calls == [("list_processes", ())]


# --------------------------------------------------------------------------- #
# Agent-level via the event bus (proves BaseAgent's "_capability" context passthrough)
# --------------------------------------------------------------------------- #
def test_agent_publishes_task_completed_with_structured_result_via_bus() -> None:
    bus = EventBus()
    agent = DesktopAutomationAgent(bus, platform=_StubDesktopPlatform())
    agent.start()

    completed: list[TaskCompleted] = []
    bus.subscribe(TaskCompleted, completed.append)

    bus.publish(
        AgentDelegated(
            task_id="d-1",
            agent_name="desktop_automation_agent",
            capability="desktop.open_application",
            task_description="Notepad",
        )
    )

    assert len(completed) == 1
    assert completed[0].result["success"] is True
    assert completed[0].result["action"] == "open_application"
    agent.stop()


def test_agent_publishes_task_failed_on_unexpected_error_via_bus() -> None:
    bus = EventBus()
    platform = _StubDesktopPlatform()
    platform.fail_next("current_directory", RuntimeError("bug"))
    agent = DesktopAutomationAgent(bus, platform=platform)
    agent.start()

    failed: list[TaskFailed] = []
    bus.subscribe(TaskFailed, failed.append)

    bus.publish(
        AgentDelegated(
            task_id="d-2",
            agent_name="desktop_automation_agent",
            capability="desktop.current_directory",
            task_description="",
        )
    )

    assert len(failed) == 1
    assert failed[0].reason == "agent_exception"
    agent.stop()


# --------------------------------------------------------------------------- #
# Executive-level: full routing for each capability
# --------------------------------------------------------------------------- #
def test_executive_routes_each_desktop_capability_correctly() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    platform = _StubDesktopPlatform()
    executive.register_agent(DesktopAutomationAgent(bus, platform=platform))

    open_result = executive.submit_task("desktop.open_application", "VS Code")
    close_result = executive.submit_task("desktop.close_application", "VS Code")
    url_result = executive.submit_task("desktop.open_url", "https://example.com")
    procs_result = executive.submit_task("desktop.list_processes", "")
    cwd_result = executive.submit_task("desktop.current_directory", "")
    shot_result = executive.submit_task("desktop.take_screenshot", "")

    for result in (open_result, close_result, url_result, procs_result, cwd_result, shot_result):
        assert result.success is True
        assert result.result["success"] is True

    assert [c[0] for c in platform.calls] == [
        "open_application",
        "close_application",
        "open_url",
        "list_processes",
        "current_directory",
        "take_screenshot",
    ]


def test_unsupported_capability_via_executive_fails_cleanly() -> None:
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    executive.register_agent(DesktopAutomationAgent(bus, platform=_StubDesktopPlatform()))

    result = executive.submit_task("desktop.reboot_computer", "")

    assert result.success is False
    assert result.reason == "unknown_capability"


def test_platform_level_failure_via_executive_still_reports_task_success() -> None:
    """A *desktop action* failing (app not found) is not a *task pipeline* failure.

    The TaskResult itself succeeds (the agent didn't raise); the failure
    is encoded in the structured result's "success" field, per
    requirement #9.
    """
    bus = EventBus()
    executive = ExecutiveAgent(event_bus=bus)
    platform = _StubDesktopPlatform()
    platform.fail_next("open_application", ApplicationLaunchError("not installed"))
    executive.register_agent(DesktopAutomationAgent(bus, platform=platform))

    result = executive.submit_task("desktop.open_application", "GhostApp")

    assert result.success is True  # the pipeline itself succeeded
    assert result.result["success"] is False  # the desktop action did not
    assert "not installed" in result.result["details"]


# --------------------------------------------------------------------------- #
# Desktop intent recognition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected_capability,expected_argument",
    [
        ("Open VS Code", "desktop.open_application", "VS Code"),
        ("open notepad", "desktop.open_application", "notepad"),
        ("launch spotify", "desktop.open_application", "spotify"),
        ("start Calculator", "desktop.open_application", "Calculator"),
        ("Close Chrome", "desktop.close_application", "Chrome"),
        ("quit spotify", "desktop.close_application", "spotify"),
        ("exit notepad", "desktop.close_application", "notepad"),
        ("open https://example.com", "desktop.open_url", "https://example.com"),
        ("go to google.com", "desktop.open_url", "google.com"),
        ("visit www.wikipedia.org", "desktop.open_url", "www.wikipedia.org"),
        ("take a screenshot", "desktop.take_screenshot", ""),
        ("screenshot", "desktop.take_screenshot", ""),
        ("what is the current directory", "desktop.current_directory", ""),
        ("pwd", "desktop.current_directory", ""),
        ("list running processes", "desktop.list_processes", ""),
        ("what processes are running", "desktop.list_processes", ""),
    ],
)
def test_recognize_desktop_intent_matches_expected(
    text: str, expected_capability: str, expected_argument: str
) -> None:
    intent = recognize_desktop_intent(text)
    assert intent == DesktopIntent(expected_capability, expected_argument)


@pytest.mark.parametrize(
    "text",
    [
        "What is the capital of France?",
        "Tell me a joke",
        "How does photosynthesis work?",
        "",
        "   ",
        "Can you help me write an essay about open source software?",
    ],
)
def test_recognize_desktop_intent_returns_none_for_conversation(text: str) -> None:
    assert recognize_desktop_intent(text) is None


# --------------------------------------------------------------------------- #
# Phase 4 polish: categories, politeness stripping, malformed/repeated commands
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "capability,expected_category",
    [
        ("desktop.open_application", "action"),
        ("desktop.close_application", "action"),
        ("desktop.open_url", "action"),
        ("desktop.list_processes", "information"),
        ("desktop.current_directory", "information"),
        ("desktop.take_screenshot", "information"),
    ],
)
def test_desktop_intent_category_assignment(capability: str, expected_category: str) -> None:
    intent = DesktopIntent(capability, "arg")
    assert intent.category == expected_category


def test_desktop_intent_category_not_required_at_construction() -> None:
    # category is auto-derived (init=False) — existing 2-arg construction
    # throughout the test suite must keep working unchanged.
    intent = DesktopIntent("desktop.open_application", "Notepad")
    assert intent.category == "action"


@pytest.mark.parametrize(
    "text,expected_capability,expected_argument",
    [
        ("please open VS Code", "desktop.open_application", "VS Code"),
        ("could you open VS Code", "desktop.open_application", "VS Code"),
        ("can you close Chrome", "desktop.close_application", "Chrome"),
        ("would you take a screenshot", "desktop.take_screenshot", ""),
        ("Sara, open notepad", "desktop.open_application", "notepad"),
        ("hey sara open notepad", "desktop.open_application", "notepad"),
    ],
)
def test_recognize_desktop_intent_strips_politeness(
    text: str, expected_capability: str, expected_argument: str
) -> None:
    intent = recognize_desktop_intent(text)
    assert intent is not None
    assert intent.capability == expected_capability
    assert intent.argument == expected_argument


@pytest.mark.parametrize(
    "text",
    [
        "open",
        "open   ",
        "close",
        "launch",
        "start",
        "",
        "   ",
    ],
)
def test_recognize_desktop_intent_malformed_commands_fall_through_to_conversation(text: str) -> None:
    assert recognize_desktop_intent(text) is None


def test_recognize_desktop_intent_repeated_identical_calls_are_independent() -> None:
    results = [recognize_desktop_intent("open VS Code") for _ in range(5)]
    assert all(r == DesktopIntent("desktop.open_application", "VS Code") for r in results)
    # Every call produced its own object — nothing is cached/shared in a
    # way that could let mutation of one leak into another.
    assert len({id(r) for r in results}) == 5


def test_recognize_desktop_intent_sequential_calls_never_leak_between_each_other() -> None:
    """Regression test for the reported cross-prompt leakage symptom.

    Feeds a scripted sequence of alternating desktop/conversational
    inputs through the recognizer one at a time and asserts each result
    depends only on that call's own input — never on what was recognized
    immediately before it.
    """
    sequence = [
        ("open VS Code", DesktopIntent("desktop.open_application", "VS Code")),
        ("what is the current directory", DesktopIntent("desktop.current_directory", "")),
        ("How are you today?", None),
        ("close VS Code", DesktopIntent("desktop.close_application", "VS Code")),
        ("list running processes", DesktopIntent("desktop.list_processes", "")),
        ("open VS Code", DesktopIntent("desktop.open_application", "VS Code")),  # repeat
        ("Tell me a joke", None),
        ("take a screenshot", DesktopIntent("desktop.take_screenshot", "")),
    ]
    for text, expected in sequence:
        result = recognize_desktop_intent(text)
        assert result == expected, f"input {text!r} produced {result!r}, expected {expected!r}"
        if expected is not None:
            # The argument must be exactly what this call's text implies —
            # never a previous call's argument, and never a concatenation
            # of the two.
            assert text.split()[-1].lower() in (result.argument.lower() or text.lower())
