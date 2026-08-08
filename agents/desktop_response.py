"""
Natural-language phrasing for desktop action results.

``agents.desktop_automation_agent.DesktopAutomationAgent`` returns a
structured, technical result — that's correct for logs and for the
structured result schema (requirement #9), but showing it verbatim to the
user leaked implementation detail ("Launched 'calculator' via macOS
'open -a'", or a raw exception's "| context={...}" suffix). This module
is the one place that translates a structured desktop result into
something a person should actually read.

Only "action" category results (open/close application, open URL) get
templated phrasing — those are the ones whose technical ``details``
described *how* something was done. "information" category results
(current directory, running processes, screenshot path) keep using
``details`` as-is, because for those the technical detail *is* the
answer the user asked for, not an implementation leak.
"""

from __future__ import annotations

from typing import Any

from agents.desktop_intent import DesktopIntent

_ACTION_NAME = {
    "desktop.open_application": "open_application",
    "desktop.close_application": "close_application",
    "desktop.open_url": "open_url",
}

_SUCCESS_TEMPLATES: dict[str, str] = {
    "open_application": "{argument} opened successfully.",
    "close_application": "{argument} closed successfully.",
    "open_url": "Opened {argument} in your browser.",
}

_FAILURE_VERBS: dict[str, str] = {
    "open_application": "open",
    "close_application": "close",
    "open_url": "open",
}


def describe_desktop_result(
    intent: DesktopIntent,
    *,
    success: bool,
    desktop_result: dict[str, Any] | None,
    error_message: str | None,
) -> str:
    """Turn a desktop action's structured result into a natural sentence.

    Args:
        intent: The recognized intent that triggered this action —
            supplies the capability (for template lookup) and the
            original argument (app name / URL) for phrasing.
        success: Whether the task pipeline itself succeeded (a
            ``TaskResult.success`` — note this can be True even when the
            desktop action *itself* failed; see ``desktop_result["success"]``).
        desktop_result: The structured result dict from
            ``DesktopAutomationAgent`` (``{"success", "action", "details",
            "duration_ms"}``), or ``None`` if the task pipeline itself
            failed before producing one (e.g. a timeout).
        error_message: The task-level error, if ``success`` is False.

    Returns:
        A natural-language sentence suitable for showing to the user
        directly.
    """
    action_succeeded = bool(desktop_result and desktop_result.get("success"))

    if intent.category == "information":
        if success and desktop_result and action_succeeded:
            return str(desktop_result.get("details", "Done."))
        reason = (desktop_result or {}).get("details") or error_message or "an unknown error occurred"
        return f"I couldn't do that: {reason}"

    # category == "action"
    action_name = _ACTION_NAME.get(intent.capability, "")
    argument = intent.argument or "It"

    if success and action_succeeded:
        template = _SUCCESS_TEMPLATES.get(action_name)
        if template:
            return template.format(argument=argument)
        return str((desktop_result or {}).get("details", "Done."))

    reason = (desktop_result or {}).get("details") or error_message or "an unknown error occurred"
    verb = _FAILURE_VERBS.get(action_name, "do that")
    subject = f" {intent.argument}" if intent.argument else ""
    return f"I couldn't {verb}{subject}: {reason}"
