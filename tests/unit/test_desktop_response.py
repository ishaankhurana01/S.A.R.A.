"""Unit tests for agents.desktop_response.describe_desktop_result."""

from __future__ import annotations

from agents.desktop_intent import DesktopIntent
from agents.desktop_response import describe_desktop_result


def _result(action: str, success: bool, details: str) -> dict:
    return {"success": success, "action": action, "details": details, "duration_ms": 1.0}


# --------------------------------------------------------------------------- #
# Action category — templated, natural phrasing (no implementation detail)
# --------------------------------------------------------------------------- #
def test_open_application_success_is_natural() -> None:
    intent = DesktopIntent("desktop.open_application", "Calculator")
    text = describe_desktop_result(
        intent,
        success=True,
        desktop_result=_result("open_application", True, "Launched 'Calculator' via macOS 'open -a'"),
        error_message=None,
    )
    assert text == "Calculator opened successfully."
    assert "open -a" not in text
    assert "via" not in text


def test_close_application_success_is_natural() -> None:
    intent = DesktopIntent("desktop.close_application", "Chrome")
    text = describe_desktop_result(
        intent,
        success=True,
        desktop_result=_result("close_application", True, "Terminated 1 process(es) matching 'Chrome': chrome"),
        error_message=None,
    )
    assert text == "Chrome closed successfully."


def test_open_url_success_is_natural() -> None:
    intent = DesktopIntent("desktop.open_url", "https://example.com")
    text = describe_desktop_result(
        intent,
        success=True,
        desktop_result=_result("open_url", True, "Opened URL: https://example.com"),
        error_message=None,
    )
    assert text == "Opened https://example.com in your browser."


def test_open_application_failure_mentions_argument_and_reason() -> None:
    intent = DesktopIntent("desktop.open_application", "GhostApp")
    text = describe_desktop_result(
        intent,
        success=True,  # pipeline succeeded; the *action* failed
        desktop_result=_result("open_application", False, "Could not launch application 'GhostApp'"),
        error_message=None,
    )
    assert "couldn't open GhostApp" in text
    assert "Could not launch application 'GhostApp'" in text


def test_action_failure_from_pipeline_level_uses_error_message() -> None:
    intent = DesktopIntent("desktop.open_application", "GhostApp")
    text = describe_desktop_result(
        intent,
        success=False,
        desktop_result=None,
        error_message="No agent registered for capability 'desktop.open_application'",
    )
    assert "couldn't open GhostApp" in text
    assert "No agent registered" in text


def test_close_application_failure_verb_is_close_not_open() -> None:
    intent = DesktopIntent("desktop.close_application", "Chrome")
    text = describe_desktop_result(
        intent,
        success=True,
        desktop_result=_result("close_application", False, "No running process matching 'Chrome' was found"),
        error_message=None,
    )
    assert text.startswith("I couldn't close Chrome:")


# --------------------------------------------------------------------------- #
# Information category — details ARE the answer, passed through as-is
# --------------------------------------------------------------------------- #
def test_information_success_passes_through_details() -> None:
    intent = DesktopIntent("desktop.current_directory", "")
    text = describe_desktop_result(
        intent,
        success=True,
        desktop_result=_result("current_directory", True, "Current working directory: /home/user"),
        error_message=None,
    )
    assert text == "Current working directory: /home/user"


def test_information_failure_reports_reason() -> None:
    intent = DesktopIntent("desktop.take_screenshot", "")
    text = describe_desktop_result(
        intent,
        success=True,
        desktop_result=_result("take_screenshot", False, "Failed to capture screenshot: no display"),
        error_message=None,
    )
    assert "couldn't do that" in text
    assert "Failed to capture screenshot: no display" in text
    assert "context=" not in text  # the exc.message fix — no leaked context dict


def test_information_pipeline_failure_uses_error_message() -> None:
    intent = DesktopIntent("desktop.list_processes", "")
    text = describe_desktop_result(
        intent,
        success=False,
        desktop_result=None,
        error_message="Task timed out",
    )
    assert "couldn't do that" in text
    assert "Task timed out" in text
