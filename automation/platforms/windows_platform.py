"""
Windows desktop platform.

Only ``open_application`` needs OS-specific logic here — see
``automation.desktop_platform.DesktopPlatform`` for why the other five
capabilities are inherited unchanged.
"""

from __future__ import annotations

import subprocess

from automation.app_aliases import resolve_alias
from automation.desktop_platform import DesktopPlatform, _validate_target
from utils.exceptions import ApplicationLaunchError
from utils.logger import get_logger

logger = get_logger(__name__)


class WindowsDesktopPlatform(DesktopPlatform):
    """Desktop actions for Windows, selected automatically by ``automation.platform_factory``."""

    def open_application(self, name: str) -> str:
        target = _validate_target(name, field_name="application name")
        resolved = resolve_alias(target, os_key="Windows")

        candidates: list[str] = [resolved]
        if target != resolved:
            candidates.append(target)  # fall back to the raw name if the alias doesn't resolve

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                # Single argv token, shell=False: never interpreted by a
                # shell, resolved only via PATH / registered App Paths.
                subprocess.Popen([candidate], shell=False)
                logger.info("Launched '{}' via candidate command '{}'", target, candidate)
                return f"Launched '{target}' (resolved as '{candidate}')"
            except (FileNotFoundError, OSError) as exc:
                logger.debug("Candidate '{}' failed for '{}': {}", candidate, target, exc)
                last_error = exc
                continue

        raise ApplicationLaunchError(
            f"Could not launch application '{target}'. Tried: {candidates}. "
            "It may not be installed, or its executable name isn't on PATH — "
            "consider adding it to automation.app_aliases's alias table.",
            context={"name": target, "tried": candidates},
        ) from last_error
