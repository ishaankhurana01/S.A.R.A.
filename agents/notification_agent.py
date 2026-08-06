"""
Notification Agent (placeholder).

Will eventually surface proactive suggestions (from ``behavior/``),
permission prompts (from ``permissions/``), and other user-facing alerts
as real desktop notifications. For this phase it only proves the
delegation path works: it logs what it received and reports success, with
no real notification shown.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)

_CAPABILITIES: tuple[str, ...] = ("notify.send",)


class NotificationAgent(BaseAgent):
    """Placeholder Worker Agent for surfacing notifications to the user."""

    @property
    def name(self) -> str:
        return "notification_agent"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return _CAPABILITIES

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        logger.info("[NotificationAgent] (placeholder) received: {}", task_description)
        return {
            "status": "success",
            "agent": self.name,
            "note": "placeholder — no real notification sent",
            "received_task": task_description,
        }
