"""
Process context collector: which user-facing applications are currently running.

Enumerates processes via ``psutil`` and applies a light filter to reduce
noise from background services — the goal is "what apps is the user
running" (VS Code, Chrome, Slack, ...), not a full raw process table.
"""

from __future__ import annotations

from typing import Any

import psutil

from core.interfaces import ContextCollector
from utils.exceptions import CollectorError
from utils.logger import get_logger

logger = get_logger(__name__)

# Common background/system process names to exclude so the running-apps
# list reflects user-facing applications rather than OS plumbing. This is
# a starting heuristic, not exhaustive — it can be extended via config in
# a later phase without changing the collector's interface.
_NOISE_PROCESS_NAMES = frozenset(
    {
        "svchost.exe",
        "system idle process",
        "registry",
        "systemd",
        "kthreadd",
        "dbus-daemon",
        "wineserver",
    }
)


class ProcessCollector(ContextCollector):
    """Collects the set of currently running, user-facing application names."""

    @property
    def name(self) -> str:
        return "process"

    def collect(self) -> dict[str, Any]:
        try:
            names: set[str] = set()
            for proc in psutil.process_iter(attrs=["name"]):
                proc_name = (proc.info.get("name") or "").strip()
                if not proc_name:
                    continue
                if proc_name.lower() in _NOISE_PROCESS_NAMES:
                    continue
                names.add(proc_name)
        except Exception as exc:  # noqa: BLE001
            raise CollectorError(
                f"process collector failed: {exc}",
                context={"collector": self.name},
            ) from exc

        return {"running_applications": tuple(sorted(names))}
