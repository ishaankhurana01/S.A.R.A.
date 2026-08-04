"""
Clipboard context collector.

Only ever exposes a short, truncated preview of clipboard text (see
``_MAX_PREVIEW_CHARS``) — never the full clipboard body, and never binary
content (images, files). This is a deliberate privacy boundary: the
World Model should know "the user recently copied something that looks
like a URL / code / an error message" without S.A.R.A. silently hoovering
up everything the user copies (a password manager entry, for instance).

Uses ``pyperclip`` for cross-platform clipboard access, with graceful
degradation if the underlying platform clipboard mechanism isn't
available (e.g. a headless Linux container with no clipboard utility
installed).
"""

from __future__ import annotations

from typing import Any

from core.interfaces import ContextCollector
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_PREVIEW_CHARS = 200

try:
    import pyperclip

    _PYPERCLIP_AVAILABLE = True
except ImportError:
    _PYPERCLIP_AVAILABLE = False


class ClipboardCollector(ContextCollector):
    """Collects a truncated preview of the current clipboard text contents."""

    def __init__(self) -> None:
        self._warned_unavailable = False
        if not _PYPERCLIP_AVAILABLE:
            logger.warning("pyperclip not installed — clipboard collector will report None.")

    @property
    def name(self) -> str:
        return "clipboard"

    def collect(self) -> dict[str, Any]:
        if not _PYPERCLIP_AVAILABLE:
            return {"clipboard_text_preview": None}

        try:
            text = pyperclip.paste()
        except Exception as exc:  # noqa: BLE001
            # A missing clipboard backend on headless Linux raises
            # pyperclip.PyperclipException; degrade gracefully instead of
            # treating "no clipboard mechanism on this machine" as fatal.
            if not self._warned_unavailable:
                logger.warning("Clipboard unavailable on this platform: {}", exc)
                self._warned_unavailable = True
            return {"clipboard_text_preview": None}

        if not text:
            return {"clipboard_text_preview": None}

        preview = text[:_MAX_PREVIEW_CHARS]
        if len(text) > _MAX_PREVIEW_CHARS:
            preview += "..."
        return {"clipboard_text_preview": preview}
