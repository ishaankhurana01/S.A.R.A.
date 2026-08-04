"""
System-level context collector: current time, battery, network status.

Uses ``psutil`` exclusively, so this collector works identically on
Windows, macOS, and Linux — unlike ``window_collector`` and
``clipboard_collector``, which rely on platform-specific APIs.
"""

from __future__ import annotations

import datetime
import socket
from typing import Any

import psutil

from core.interfaces import ContextCollector
from utils.exceptions import CollectorError
from utils.logger import get_logger

logger = get_logger(__name__)


class SystemCollector(ContextCollector):
    """Collects wall-clock time, battery state, and network reachability."""

    @property
    def name(self) -> str:
        return "system"

    def collect(self) -> dict[str, Any]:
        try:
            now_iso = datetime.datetime.now().isoformat(timespec="seconds")
            battery_percent, battery_charging = self._read_battery()
            network_connected = self._check_network()
        except Exception as exc:  # noqa: BLE001 - any failure here becomes a CollectorError
            raise CollectorError(
                f"system collector failed: {exc}",
                context={"collector": self.name},
            ) from exc

        return {
            "current_time_iso": now_iso,
            "battery_percent": battery_percent,
            "battery_is_charging": battery_charging,
            "network_connected": network_connected,
        }

    @staticmethod
    def _read_battery() -> tuple[float | None, bool | None]:
        """Return (percent, is_charging), or (None, None) on desktops with no battery."""
        try:
            battery = psutil.sensors_battery()
        except Exception:  # noqa: BLE001 - platform may not expose battery sensors at all
            return None, None
        if battery is None:
            return None, None
        return float(battery.percent), bool(battery.power_plugged)

    @staticmethod
    def _check_network(host: str = "1.1.1.1", port: int = 53, timeout: float = 1.0) -> bool:
        """Best-effort connectivity check via a fast TCP handshake (Cloudflare DNS)."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False
