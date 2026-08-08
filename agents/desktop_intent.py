"""
Desktop intent recognition for ``agents.conversation_agent.ConversationAgent``.

This is deliberately simple, rule-based pattern matching — a handful of
regexes covering common phrasings ("open X", "close X", "take a
screenshot", ...). It is explicitly *not* an attempt at general NLU or
intent classification; the architecture's Behavior Learning Engine
(``behavior/``) is where a smarter, learned approach to this belongs in a
later phase, and this phase's requirements explicitly exclude
self-learning. This module exists so that boundary is a single, obvious,
swappable seam: replace ``recognize_desktop_intent`` with a smarter
implementation later without touching ``ConversationAgent`` at all.

Recognition failure is not an error — most user input is genuinely
conversational, and ``recognize_desktop_intent`` returning ``None`` is the
normal, expected outcome that sends a request to the LLM as before.

Statelessness (important): every call is fully independent. There is no
module-level or instance-level mutable state anywhere in this file — the
returned ``DesktopIntent`` is built entirely from the ``text`` argument
passed to that one call. Nothing here (or in ``ConversationAgent``, which
is a long-lived singleton across many requests) accumulates or retains
any part of a previous call's input, so one prompt can never leak into
the next. See ``tests/unit/test_conversation_agent.py``'s sequential-call
tests and ``tests/unit/test_desktop_automation_agent.py``'s
recognizer tests for the regression coverage proving this.

Categories: every recognized intent is tagged "information" (a read-only
query — current directory, running processes, a screenshot) or "action"
(something that changes system state — opening/closing an application,
opening a URL). ``agents.desktop_response`` uses this to phrase results
naturally: an information result's ``details`` *is* the answer and gets
shown as-is; an action result gets a templated confirmation instead of
exposing how it was carried out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

IntentCategory = Literal["information", "action"]

# Which category each desktop capability belongs to. Adding a seventh
# desktop capability later means adding one line here.
_CAPABILITY_CATEGORY: dict[str, IntentCategory] = {
    "desktop.open_application": "action",
    "desktop.close_application": "action",
    "desktop.open_url": "action",
    "desktop.list_processes": "information",
    "desktop.current_directory": "information",
    "desktop.take_screenshot": "information",
}


@dataclass(frozen=True)
class DesktopIntent:
    """A recognized desktop action request.

    Attributes:
        capability: The exact capability string to submit to the
            Executive Agent (e.g. ``"desktop.open_application"``).
        argument: The action's single argument (an application name or
            URL), or ``""`` for argument-less actions like
            ``desktop.list_processes``.
        category: Auto-derived from ``capability`` — ``"information"``
            for read-only queries, ``"action"`` for state-changing
            requests. Not a constructor argument (computed in
            ``__post_init__``), so existing 2-argument construction
            (``DesktopIntent(capability, argument)``) keeps working
            unchanged.
    """

    capability: str
    argument: str
    category: IntentCategory = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _CAPABILITY_CATEGORY.get(self.capability, "action"))


# Order matters: more specific patterns are checked first so, e.g.,
# "open google.com" matches the URL pattern rather than being treated as
# an application named "google.com".
_SCREENSHOT_RE = re.compile(
    r"^(take\s+a\s+screenshot|screenshot|capture\s+(the\s+)?screen)[.!]?$", re.IGNORECASE
)
_CURRENT_DIR_RE = re.compile(
    r"^(what(?:'s| is)\s+(?:the\s+)?current\s+(?:directory|folder)|pwd|where\s+am\s+i)\??[.!]?$",
    re.IGNORECASE,
)
_LIST_PROCESSES_RE = re.compile(
    r"^(?:list|show)\s+(?:running\s+)?(?:processes|applications|apps)\??[.!]?$"
    r"|^what\s+processes\s+are\s+running\??[.!]?$",
    re.IGNORECASE,
)
_OPEN_URL_RE = re.compile(
    r"^(?:open|go\s+to|navigate\s+to|visit)\s+"
    r"(https?://\S+|www\.\S+|[^\s]+\.[a-z]{2,}(?:/\S*)?)$",
    re.IGNORECASE,
)
_CLOSE_APP_RE = re.compile(r"^(?:close|quit|exit)\s+(.+)$", re.IGNORECASE)
_OPEN_APP_RE = re.compile(r"^(?:open|launch|start)\s+(.+)$", re.IGNORECASE)

# Polite/conversational lead-ins stripped before matching, so "could you
# open VS Code" recognizes the same as "open VS Code". This is plain
# string trimming, not NLU — it only removes a fixed set of known
# prefixes, repeatedly, from the front of the string.
_POLITE_PREFIX_RE = re.compile(
    r"^(please|could you|can you|would you|will you|hey sara|sara)[,.]?\s+",
    re.IGNORECASE,
)


def _strip_polite_prefixes(text: str) -> str:
    """Repeatedly strip known polite lead-ins from the front of ``text``."""
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _POLITE_PREFIX_RE.sub("", current).strip()
    return current


def recognize_desktop_intent(text: str) -> DesktopIntent | None:
    """Return a ``DesktopIntent`` if ``text`` looks like a desktop action request, else ``None``.

    Args:
        text: The raw user prompt, exactly as it would otherwise be sent
            to the LLM. This single call is entirely self-contained —
            nothing about a prior or future call affects it.

    Returns:
        A ``DesktopIntent`` naming the capability + argument to delegate,
        or ``None`` if the text should be treated as ordinary
        conversation (this includes plain conversational questions and
        malformed/argument-less desktop-looking commands like a bare
        "open" with nothing after it — those fall through to the LLM
        rather than being treated as valid actions).
    """
    cleaned = _strip_polite_prefixes((text or "").strip())
    if not cleaned:
        return None

    if _SCREENSHOT_RE.match(cleaned):
        return DesktopIntent("desktop.take_screenshot", "")

    if _CURRENT_DIR_RE.match(cleaned):
        return DesktopIntent("desktop.current_directory", "")

    if _LIST_PROCESSES_RE.match(cleaned):
        return DesktopIntent("desktop.list_processes", "")

    match = _OPEN_URL_RE.match(cleaned)
    if match:
        argument = match.group(1).strip()
        if argument:
            return DesktopIntent("desktop.open_url", argument)

    match = _CLOSE_APP_RE.match(cleaned)
    if match:
        argument = match.group(1).strip()
        if argument:
            return DesktopIntent("desktop.close_application", argument)

    match = _OPEN_APP_RE.match(cleaned)
    if match:
        argument = match.group(1).strip()
        if argument:
            return DesktopIntent("desktop.open_application", argument)

    return None
