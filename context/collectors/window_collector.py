"""
Active-window context collector.

S.A.R.A. targets Windows (per the project's launch-on-startup and
system-tray requirements), so this collector uses ``pywin32`` to read the
foreground window's title and owning process name. On any platform where
``pywin32`` is unavailable (e.g. this collector being unit-tested on
Linux/macOS), it degrades to returning ``None`` values rather than raising
— platform unavailability is expected, not an error condition, so it is
handled here rather than surfacing as a ``CollectorError`` on every poll.
"""

from __future__ import annotations

from typing import Any

from core.interfaces import ContextCollector
from utils.exceptions import CollectorError
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import win32gui  # type: ignore[import-not-found]
    import win32process  # type: ignore[import-not-found]
    import psutil

    _PYWIN32_AVAILABLE = True
except ImportError:
    _PYWIN32_AVAILABLE = False


class WindowCollector(ContextCollector):
    """Collects the active window's title and owning application name (Windows only)."""

    def __init__(self) -> None:
        if not _PYWIN32_AVAILABLE:
            logger.warning(
                "pywin32 not available — window collector will report unknown values. "
                "This is expected on non-Windows platforms."
            )

    @property
    def name(self) -> str:
        return "window"

    def collect(self) -> dict[str, Any]:
        if not _PYWIN32_AVAILABLE:
            return {"active_window_title": None, "active_application_name": None}

        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or None

            app_name: str | None = None
            if hwnd:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    app_name = psutil.Process(pid).name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    app_name = None
        except Exception as exc:  # noqa: BLE001
            raise CollectorError(
                f"window collector failed: {exc}",
                context={"collector": self.name},
            ) from exc

        return {"active_window_title": title, "active_application_name": app_name}
