"""
Selects the correct ``DesktopPlatform`` implementation for the running OS.

This is the only place in ``automation/`` that inspects ``platform.system()``
— everything else (``DesktopAutomationAgent``, tests, callers) works
against the ``DesktopPlatform`` interface and never needs to know or care
which concrete subclass it got.
"""

from __future__ import annotations

import platform as _platform_module

from automation.desktop_platform import DesktopPlatform
from automation.platforms.linux_platform import LinuxDesktopPlatform
from automation.platforms.macos_platform import MacDesktopPlatform
from automation.platforms.windows_platform import WindowsDesktopPlatform
from utils.exceptions import UnsupportedPlatformError
from utils.logger import get_logger

logger = get_logger(__name__)

_PLATFORM_MAP: dict[str, type[DesktopPlatform]] = {
    "Windows": WindowsDesktopPlatform,
    "Darwin": MacDesktopPlatform,
    "Linux": LinuxDesktopPlatform,
}


def get_platform(*, screenshot_directory: str = "data/screenshots") -> DesktopPlatform:
    """Return a ``DesktopPlatform`` instance appropriate for the current OS.

    Args:
        screenshot_directory: Passed through to the selected platform,
            used as the default directory for ``take_screenshot`` output.

    Returns:
        A ``WindowsDesktopPlatform``, ``MacDesktopPlatform``, or
        ``LinuxDesktopPlatform`` instance, chosen via ``platform.system()``.

    Raises:
        UnsupportedPlatformError: The current OS has no implementation
            (e.g. an unrecognized ``platform.system()`` value).
    """
    system = _platform_module.system()
    platform_cls = _PLATFORM_MAP.get(system)
    if platform_cls is None:
        raise UnsupportedPlatformError(
            f"No DesktopPlatform implementation is available for OS '{system}'. "
            f"Supported: {list(_PLATFORM_MAP.keys())}",
            context={"system": system},
        )
    logger.info("Selected {} for OS '{}'", platform_cls.__name__, system)
    return platform_cls(screenshot_directory=screenshot_directory)
