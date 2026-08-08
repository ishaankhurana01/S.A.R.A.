"""
Desktop Automation Agent.

Owns six desktop capabilities (open/close application, open URL, list
processes, current directory, take screenshot) and delegates the actual
OS interaction to a ``automation.desktop_platform.DesktopPlatform``
implementation, chosen automatically for the running OS by
``automation.platform_factory.get_platform``.

Like ``agents.conversation_agent.ConversationAgent`` (the only other
"real work" agent so far), this class is a thin adapter: event bus
plumbing, filtering, and exception-to-``TaskFailed`` conversion all come
from ``agents.base_agent.BaseAgent`` unchanged. What this class adds is
the capability -> platform-method dispatch (using the ``_capability`` key
``BaseAgent`` now includes in ``context``, since this agent — unlike
``ConversationAgent`` — owns more than one capability and needs to know
which one a given delegation is for) and the structured
``{success, action, details, duration_ms}`` result shape requirement #9
asks for.

Per requirement #6, nothing in this class (or anything it calls) ever
executes a shell string — see ``automation.desktop_platform`` for where
that's enforced.
"""

from __future__ import annotations

import time
from typing import Any

from agents.base_agent import BaseAgent
from automation.desktop_platform import DesktopPlatform
from utils.exceptions import DesktopAutomationError
from utils.logger import get_logger

logger = get_logger(__name__)

_CAPABILITIES: tuple[str, ...] = (
    "desktop.open_application",
    "desktop.close_application",
    "desktop.open_url",
    "desktop.list_processes",
    "desktop.current_directory",
    "desktop.take_screenshot",
)

# Maps each capability to the DesktopPlatform method name it dispatches
# to. Adding a seventh desktop capability later means adding one line
# here (and to _CAPABILITIES above) — the dispatch logic itself doesn't
# change.
_CAPABILITY_TO_ACTION: dict[str, str] = {
    "desktop.open_application": "open_application",
    "desktop.close_application": "close_application",
    "desktop.open_url": "open_url",
    "desktop.list_processes": "list_processes",
    "desktop.current_directory": "current_directory",
    "desktop.take_screenshot": "take_screenshot",
}

# Capabilities whose platform method takes no argument — task_description
# is ignored for these (it's typically empty for them anyway; see
# agents.desktop_intent.recognize_desktop_intent).
_NO_ARGUMENT_ACTIONS = frozenset({"list_processes", "current_directory"})


class DesktopAutomationAgent(BaseAgent):
    """Worker Agent executing OS-level desktop actions via a platform backend.

    Example:
        agent = DesktopAutomationAgent(event_bus=bus, platform=get_platform())
        executive.register_agent(agent)
        result = executive.submit_task("desktop.open_application", "VS Code")
        # result.result == {"success": True, "action": "open_application",
        #                    "details": "Launched 'VS Code' ...", "duration_ms": 42.1}
    """

    def __init__(self, event_bus: Any, *, platform: DesktopPlatform) -> None:
        super().__init__(event_bus)
        self._platform = platform

    @property
    def name(self) -> str:
        return "desktop_automation_agent"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return _CAPABILITIES

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        """Dispatch to the correct platform action based on the delegated capability.

        Args:
            task_description: The action's argument (an application name
                or URL). Ignored for argument-less actions.
            context: Provided by ``BaseAgent``; must include
                ``"_capability"`` (always true for a real delegation —
                see ``BaseAgent._on_agent_delegated``).

        Returns:
            A structured result dict: ``{"success": bool, "action": str,
            "details": str, "duration_ms": float}``. Note this method
            *returns* a failure result (``success: False``) for expected
            platform-level problems (app not found, invalid target, ...)
            rather than raising — those are normal outcomes of a desktop
            action, not agent bugs. An exception raised here means
            something unexpected happened and correctly becomes a
            ``TaskFailed`` event via ``BaseAgent``.

        Raises:
            ValueError: ``context`` is missing ``"_capability"`` or names
                a capability this agent doesn't own — indicates a
                misconfigured ``CapabilityRegistry`` entry, not a normal
                runtime condition, so it's allowed to propagate.
        """
        capability = context.get("_capability")
        action = _CAPABILITY_TO_ACTION.get(capability)
        if action is None:
            raise ValueError(
                f"DesktopAutomationAgent received a delegation for unknown capability '{capability}'"
            )

        logger.info("[DesktopAutomationAgent] executing '{}' (arg={!r})", action, task_description)
        start = time.monotonic()
        try:
            details = self._dispatch(action, task_description)
            success = True
        except DesktopAutomationError as exc:
            # exc.message (not str(exc)): SaraError.__str__ appends
            # " | context={...}" for logging purposes, which leaked raw
            # internal details (file paths, tried-candidate lists) into
            # the structured result's "details" field. The full str(exc)
            # — context included — still goes to the log line below,
            # where that detail belongs.
            details = exc.message
            success = False
            logger.warning("[DesktopAutomationAgent] action '{}' failed: {}", action, exc)

        duration_ms = round((time.monotonic() - start) * 1000, 2)
        result = {
            "success": success,
            "action": action,
            "details": details,
            "duration_ms": duration_ms,
        }
        logger.info(
            "[DesktopAutomationAgent] '{}' completed success={} in {}ms", action, success, duration_ms
        )
        return result

    def _dispatch(self, action: str, task_description: str) -> str:
        """Call the appropriate ``DesktopPlatform`` method for ``action``."""
        if action in _NO_ARGUMENT_ACTIONS:
            method = getattr(self._platform, action)
            return method()
        if action == "take_screenshot":
            save_path = task_description.strip() or None
            return self._platform.take_screenshot(save_path)
        # open_application / close_application / open_url all take the
        # single argument string.
        method = getattr(self._platform, action)
        return method(task_description)
