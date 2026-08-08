"""
Centralized application-name aliases.

A human-friendly name ("VS Code") often isn't the string the OS actually
needs to launch it: Windows/Linux want a command-line-resolvable binary
name ("code"), macOS's ``open -a`` wants the app's real display name
("Visual Studio Code"). Before this module existed, that mapping was
duplicated as a local ``_ALIAS_MAP`` dict inside
``windows_platform.py``/``linux_platform.py`` (and didn't exist at all
for macOS, which was relying on ``open -a``'s fuzzy matching to paper
over the gap). This module is the single place that mapping lives now —
each platform's ``open_application`` calls ``resolve_alias`` instead of
maintaining its own table, so adding or correcting an alias is one edit
here rather than up to three edits scattered across platform files.

Unknown names are not an error: ``resolve_alias`` simply returns the
input unchanged (stripped), which is exactly the right fallback — the
platform then attempts to launch that name as-is, and
``ApplicationLaunchError`` (raised by the platform if that fails) is what
surfaces "not found" to the user, not this module.
"""

from __future__ import annotations

# Keys are lowercased, normalized (single spaces) friendly names. Values
# map each supported ``platform.system()`` string to the name that
# platform should actually use. Extend this table — never add
# application-specific branching inside the platform classes themselves.
_ALIASES: dict[str, dict[str, str]] = {
    "vs code": {"Windows": "code", "Darwin": "Visual Studio Code", "Linux": "code"},
    "visual studio code": {"Windows": "code", "Darwin": "Visual Studio Code", "Linux": "code"},
    "vscode": {"Windows": "code", "Darwin": "Visual Studio Code", "Linux": "code"},
    "chrome": {"Windows": "chrome", "Darwin": "Google Chrome", "Linux": "google-chrome"},
    "google chrome": {"Windows": "chrome", "Darwin": "Google Chrome", "Linux": "google-chrome"},
    "firefox": {"Windows": "firefox", "Darwin": "Firefox", "Linux": "firefox"},
    "edge": {"Windows": "msedge", "Darwin": "Microsoft Edge", "Linux": "microsoft-edge"},
    "microsoft edge": {"Windows": "msedge", "Darwin": "Microsoft Edge", "Linux": "microsoft-edge"},
    "terminal": {"Windows": "wt", "Darwin": "Terminal", "Linux": "x-terminal-emulator"},
    "command prompt": {"Windows": "cmd", "Darwin": "Terminal", "Linux": "x-terminal-emulator"},
    "powershell": {"Windows": "powershell", "Darwin": "Terminal", "Linux": "x-terminal-emulator"},
    "calculator": {"Windows": "calc", "Darwin": "Calculator", "Linux": "gnome-calculator"},
    "notepad": {"Windows": "notepad", "Darwin": "TextEdit", "Linux": "gedit"},
    "text editor": {"Windows": "notepad", "Darwin": "TextEdit", "Linux": "gedit"},
    "explorer": {"Windows": "explorer", "Darwin": "Finder", "Linux": "nautilus"},
    "file explorer": {"Windows": "explorer", "Darwin": "Finder", "Linux": "nautilus"},
    "files": {"Windows": "explorer", "Darwin": "Finder", "Linux": "nautilus"},
    "file manager": {"Windows": "explorer", "Darwin": "Finder", "Linux": "nautilus"},
    "finder": {"Windows": "explorer", "Darwin": "Finder", "Linux": "nautilus"},
    "spotify": {"Windows": "spotify", "Darwin": "Spotify", "Linux": "spotify"},
    "slack": {"Windows": "slack", "Darwin": "Slack", "Linux": "slack"},
    "outlook": {"Windows": "outlook", "Darwin": "Microsoft Outlook", "Linux": "thunderbird"},
    "word": {"Windows": "winword", "Darwin": "Microsoft Word", "Linux": "libreoffice"},
    "excel": {"Windows": "excel", "Darwin": "Microsoft Excel", "Linux": "libreoffice"},
}


def resolve_alias(name: str, *, os_key: str) -> str:
    """Return the platform-appropriate application name for a friendly ``name``.

    Args:
        name: A human-friendly application name (e.g. ``"VS Code"``),
            already validated (non-empty) by the caller.
        os_key: The exact ``platform.system()`` string for the target OS
            — ``"Windows"``, ``"Darwin"``, or ``"Linux"`` — matching the
            keys used by ``automation.platform_factory``.

    Returns:
        The resolved name for ``os_key`` if ``name`` (case-insensitively,
        whitespace-normalized) is a known alias; otherwise ``name``
        stripped and returned unchanged, so the caller can still attempt
        to launch it as given.
    """
    normalized = " ".join(name.strip().lower().split())
    entry = _ALIASES.get(normalized)
    if entry is None:
        return name.strip()
    return entry.get(os_key, name.strip())
