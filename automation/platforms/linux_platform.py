"""
Linux desktop platform.

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


class LinuxDesktopPlatform(DesktopPlatform):
    """Desktop actions for Linux, selected automatically by ``automation.platform_factory``."""

    def open_application(self, name: str) -> str:
        target = _validate_target(name, field_name="application name")
        alias = resolve_alias(target, os_key="Linux")
        # If no alias matched, resolve_alias returns the name unchanged —
        # fall back to a lowercased, hyphenated guess as a last resort.
        candidate = alias if alias.lower() != target.lower() else target.lower().replace(" ", "-")

        try:
            # Single argv token, shell=False: never interpreted by a
            # shell, resolved only via PATH.
            subprocess.Popen([candidate], shell=False)
        except (FileNotFoundError, OSError) as exc:
            raise ApplicationLaunchError(
                f"Could not launch application '{target}' (tried command '{candidate}'). "
                "It may not be installed, or its executable name isn't on PATH — "
                "consider adding it to automation.app_aliases's alias table.",
                context={"name": target, "tried": candidate},
            ) from exc

        logger.info("Launched '{}' via candidate command '{}'", target, candidate)
        return f"Launched '{target}' (resolved as '{candidate}')"
