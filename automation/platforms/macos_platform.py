"""
macOS desktop platform.

Only ``open_application`` needs OS-specific logic here — see
``automation.desktop_platform.DesktopPlatform`` for why the other five
capabilities are inherited unchanged.

macOS's ``open -a <name>`` resolves human-friendly application display
names (via Launch Services / Spotlight metadata) fairly well on its own,
but not perfectly — "Chrome" won't resolve to "Google Chrome.app" the way
"VS Code" won't resolve to "Visual Studio Code.app". ``automation.app_aliases``
covers exactly that gap, shared with the other two platforms.
"""

from __future__ import annotations

import subprocess

from automation.app_aliases import resolve_alias
from automation.desktop_platform import DesktopPlatform, _validate_target
from utils.exceptions import ApplicationLaunchError
from utils.logger import get_logger

logger = get_logger(__name__)

_OPEN_TIMEOUT_SECONDS = 15


class MacDesktopPlatform(DesktopPlatform):
    """Desktop actions for macOS, selected automatically by ``automation.platform_factory``."""

    def open_application(self, name: str) -> str:
        target = _validate_target(name, field_name="application name")
        resolved = resolve_alias(target, os_key="Darwin")
        try:
            # argv list, shell=False: "open" and "-a" are fixed, only the
            # validated, alias-resolved app name varies — never
            # shell-interpreted.
            subprocess.run(
                ["open", "-a", resolved],
                shell=False,
                check=True,
                capture_output=True,
                timeout=_OPEN_TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="ignore").strip() if exc.stderr else str(exc)
            raise ApplicationLaunchError(
                f"macOS could not open application '{target}' (tried '{resolved}'): {stderr}",
                context={"name": target, "resolved": resolved, "returncode": exc.returncode},
            ) from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ApplicationLaunchError(
                f"Failed to launch '{target}' (tried '{resolved}'): {exc}",
                context={"name": target, "resolved": resolved},
            ) from exc

        logger.info("Launched '{}' via 'open -a {}'", target, resolved)
        return f"Launched '{target}' (resolved as '{resolved}') via macOS 'open -a'"
