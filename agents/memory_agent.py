"""
Memory Agent (placeholder).

Will eventually be the single front-end onto the four memory tiers
(``memory/working``, ``episodic``, ``semantic``, ``procedural`` — see the
architecture doc's Memory Architecture section), so other agents never
touch storage directly. For this phase it only proves the delegation path
works: it logs what it received and reports success, with no real reads
or writes to any store.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)

_CAPABILITIES: tuple[str, ...] = (
    "memory.recall",
    "memory.store",
)


class MemoryAgent(BaseAgent):
    """Placeholder Worker Agent for memory read/write tasks."""

    @property
    def name(self) -> str:
        return "memory_agent"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return _CAPABILITIES

    def handle(self, task_description: str, *, context: dict[str, Any]) -> Any:
        logger.info("[MemoryAgent] (placeholder) received: {}", task_description)
        return {
            "status": "success",
            "agent": self.name,
            "note": "placeholder — no real memory read/write performed",
            "received_task": task_description,
        }
