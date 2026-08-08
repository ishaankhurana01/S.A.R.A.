"""
Desktop platform abstraction.

``DesktopPlatform`` is the strategy interface
``agents.desktop_automation_agent.DesktopAutomationAgent`` delegates to;
``automation.platform_factory.get_platform`` selects the concrete
subclass (Windows/macOS/Linux) automatically based on the running OS.

Which methods are actually platform-specific
----------------------------------------------
Only ``open_application`` is declared abstract. Launching an application
by a human-friendly name (`"VS Code"`, `"Calculator"`) genuinely requires
OS-specific mechanics — there's no portable way to do it, which is why
``automation/platforms/windows_platform.py``, ``macos_platform.py``, and
``linux_platform.py`` each implement it independently.

The other five capabilities have a single well-tested cross-platform
implementation that lives here and is shared by every subclass, because
forcing artificial per-OS variants would just duplicate the exact same
logic three times:
    - ``close_application`` / ``list_processes`` — ``psutil`` is already
      cross-platform; there is no OS-specific behavior to abstract.
    - ``open_url`` — the stdlib ``webbrowser`` module already does the
      right OS-specific thing internally.
    - ``current_directory`` — ``os.getcwd()`` is portable by definition.
    - ``take_screenshot`` — ``mss`` handles Windows/macOS/Linux (including
      X11) without any OS-specific code on our side.
This is the Template Method pattern: the base class owns shared behavior,
subclasses override only what truly differs.

Safety (requirement #6 — never allow arbitrary shell execution)
-----------------------------------------------------------------
Every subprocess call anywhere in ``automation/`` is invoked with an
explicit argv list and ``shell=False`` — there is no code path in this
package that passes a string through a shell interpreter. As defense in
depth on top of that (shell=False already prevents injection),
``_validate_target``/``_validate_url`` below reject empty input, input
over a sane length, and a short list of characters/sequences associated
with shell metacharacters, even though they can't reach a shell here.
"""

from __future__ import annotations

import os
import re
import webbrowser
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import psutil

from utils.exceptions import (
    ApplicationCloseError,
    ApplicationLaunchError,
    InvalidDesktopTargetError,
    ScreenshotCaptureError,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_TARGET_LENGTH = 300
_DISALLOWED_SEQUENCES = (";", "|", "`", "$(", "&&", "||", "\n", "\r", "<", ">")
_URL_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$")


def _validate_target(value: str, *, field_name: str, max_length: int = _MAX_TARGET_LENGTH) -> str:
    """Validate and normalize a user-supplied target string (app name, path fragment, etc.).

    Raises:
        InvalidDesktopTargetError: If ``value`` is empty, too long, or
            contains a disallowed character sequence.
    """
    if value is None:
        raise InvalidDesktopTargetError(f"{field_name} must not be empty")
    cleaned = value.strip()
    if not cleaned:
        raise InvalidDesktopTargetError(f"{field_name} must not be empty")
    if len(cleaned) > max_length:
        raise InvalidDesktopTargetError(
            f"{field_name} exceeds the maximum allowed length of {max_length} characters",
            context={"field_name": field_name, "length": len(cleaned)},
        )
    for token in _DISALLOWED_SEQUENCES:
        if token in cleaned:
            raise InvalidDesktopTargetError(
                f"{field_name} contains a disallowed character sequence ('{token}')",
                context={"field_name": field_name, "value": cleaned},
            )
    return cleaned


def _validate_url(value: str) -> str:
    """Validate and normalize a URL target, rejecting local-file access.

    A missing scheme is treated as ``https://`` if the input otherwise
    looks like a bare domain (contains a dot, no whitespace) — this is
    what lets ``recognize desktop intent`` accept "open google.com"
    without the user needing to type the scheme.

    Raises:
        InvalidDesktopTargetError: If the value fails general target
            validation, uses a ``file://`` scheme, doesn't look like a
            URL, or contains characters outside the permitted URL set.
    """
    cleaned = _validate_target(value, field_name="URL")
    lowered = cleaned.lower()

    if lowered.startswith("file://"):
        raise InvalidDesktopTargetError(
            "Opening local file:// URLs is not permitted", context={"value": cleaned}
        )

    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        if "." in cleaned and " " not in cleaned:
            cleaned = f"https://{cleaned}"
        else:
            raise InvalidDesktopTargetError(f"'{value}' does not look like a valid URL", context={"value": value})

    if not _URL_ALLOWED_CHARS.match(cleaned):
        raise InvalidDesktopTargetError(
            f"URL contains characters outside the permitted set: '{cleaned}'",
            context={"value": cleaned},
        )
    return cleaned


class DesktopPlatform(ABC):
    """Strategy interface for OS-level desktop actions.

    Example:
        platform = get_platform()  # picks the right subclass for this OS
        platform.open_application("VS Code")
        platform.current_directory()
    """

    def __init__(self, *, screenshot_directory: str | Path = "data/screenshots") -> None:
        self._screenshot_directory = Path(screenshot_directory)

    # ------------------------------------------------------------------ #
    # Genuinely platform-specific — implemented per OS
    # ------------------------------------------------------------------ #
    @abstractmethod
    def open_application(self, name: str) -> str:
        """Launch an application by its common name.

        Args:
            name: A human-friendly application name (e.g. ``"VS Code"``).

        Returns:
            A short human-readable description of what was launched.

        Raises:
            InvalidDesktopTargetError: ``name`` failed validation.
            ApplicationLaunchError: The application could not be found
                or launched.
        """

    # ------------------------------------------------------------------ #
    # Shared, OS-independent — implemented once here
    # ------------------------------------------------------------------ #
    def close_application(self, name: str) -> str:
        """Terminate every running process whose name contains ``name`` (case-insensitive).

        Uses ``psutil`` process termination directly rather than shelling
        out to ``taskkill``/``pkill`` — this is both simpler and avoids
        adding another subprocess invocation to audit for shell safety.

        Raises:
            InvalidDesktopTargetError: ``name`` failed validation.
            ApplicationCloseError: No matching process was found, or
                every matching process could not be terminated
                (permission denied, already exited, etc.).
        """
        target = _validate_target(name, field_name="application name")
        needle = target.lower()

        matches = [
            proc
            for proc in psutil.process_iter(attrs=["name"])
            if needle in (proc.info.get("name") or "").lower()
        ]
        if not matches:
            raise ApplicationCloseError(
                f"No running process matching '{target}' was found",
                context={"name": target},
            )

        terminated: list[str] = []
        for proc in matches:
            try:
                proc.terminate()
                terminated.append(proc.info.get("name") or str(proc.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                logger.warning("Could not terminate process {} ({}): {}", proc.pid, proc.info.get("name"), exc)

        if not terminated:
            raise ApplicationCloseError(
                f"Found {len(matches)} process(es) matching '{target}' but could not terminate any "
                "(permission denied or already exited)",
                context={"name": target, "matched_count": len(matches)},
            )
        return f"Terminated {len(terminated)} process(es) matching '{target}': {', '.join(terminated)}"

    def open_url(self, url: str) -> str:
        """Open ``url`` in the user's default browser via the stdlib ``webbrowser`` module.

        Raises:
            InvalidDesktopTargetError: ``url`` failed validation.
            ApplicationLaunchError: No browser handler was available, or
                opening it raised.
        """
        normalized = _validate_url(url)
        try:
            opened = webbrowser.open(normalized)
        except Exception as exc:  # noqa: BLE001 - webbrowser backends raise inconsistent types
            raise ApplicationLaunchError(
                f"Failed to open URL '{normalized}': {exc}", context={"url": normalized}
            ) from exc
        if not opened:
            raise ApplicationLaunchError(
                f"No browser handler is available to open '{normalized}' on this system",
                context={"url": normalized},
            )
        return f"Opened URL: {normalized}"

    def list_processes(self, *, limit: int = 40) -> str:
        """Return a human-readable summary of currently running application names."""
        names = sorted(
            {proc.info.get("name") for proc in psutil.process_iter(attrs=["name"]) if proc.info.get("name")}
        )
        shown = names[:limit]
        suffix = f" (+{len(names) - limit} more not shown)" if len(names) > limit else ""
        return f"{len(names)} running application(s): {', '.join(shown)}{suffix}"

    def current_directory(self) -> str:
        """Return the process's current working directory."""
        return f"Current working directory: {os.getcwd()}"

    def take_screenshot(self, save_path: str | None = None) -> str:
        """Capture the full screen (all monitors) and save it as a PNG.

        Args:
            save_path: Explicit output path. Defaults to a timestamped
                file under this platform's configured screenshot
                directory (``config.config_schema.AutomationConfig.screenshot_directory``).

        Raises:
            ScreenshotCaptureError: Capture or save failed (e.g. no
                display available — expected on a headless server).
        """
        import mss
        import mss.tools

        target = Path(save_path) if save_path else self._default_screenshot_path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # index 0 = union of all monitors
                shot = sct.grab(monitor)
                mss.tools.to_png(shot.rgb, shot.size, output=str(target))
        except Exception as exc:  # noqa: BLE001 - mss/backends raise varied, backend-specific errors
            raise ScreenshotCaptureError(
                f"Failed to capture screenshot: {exc}", context={"path": str(target)}
            ) from exc
        return f"Screenshot saved to {target}"

    def _default_screenshot_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self._screenshot_directory / f"screenshot_{timestamp}.png"
