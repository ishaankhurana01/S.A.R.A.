"""
Typed schema for S.A.R.A.'s World Model.

``ContextSnapshot`` is the single, structured representation of "what is
happening on this machine right now." Every collector in
``context/collectors/`` contributes some subset of these fields;
``context.context_engine.ContextEngine`` merges collector output into one
snapshot and publishes it as ``events.event_types.ContextUpdated``.

Every field is Optional with a None default: a collector being disabled,
unavailable on this platform, or mid-failure should degrade to "we don't
know" rather than crashing snapshot construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class ContextSnapshot:
    """A point-in-time picture of the user's environment.

    Field ownership (which collector populates what) is documented per
    field below so it's obvious where to look when a value is missing or
    wrong.
    """

    captured_at: float = field(default_factory=time.time)

    # --- system_collector.py ---
    current_time_iso: str | None = None
    battery_percent: float | None = None
    battery_is_charging: bool | None = None
    network_connected: bool | None = None

    # --- process_collector.py ---
    running_applications: tuple[str, ...] | None = None

    # --- window_collector.py ---
    active_window_title: str | None = None
    active_application_name: str | None = None

    # --- clipboard_collector.py ---
    clipboard_text_preview: str | None = None  # truncated; never the full clipboard body

    # --- project_collector.py (reserved) ---
    current_project_path: str | None = None

    # --- browser_collector.py (reserved) ---
    browser_active_tab_title: str | None = None
    browser_active_tab_url: str | None = None

    # --- assistant/agents (reserved — set by the Executive Agent, not a collector) ---
    current_task_description: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the snapshot as a plain dict, e.g. for prompt injection or logging."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def diff_fields(self, other: "ContextSnapshot") -> tuple[str, ...]:
        """Return field names whose value differs between this snapshot and ``other``.

        Used by the Context Engine to populate
        ``events.event_types.ContextUpdated.changed_fields`` so subscribers
        can cheaply ignore updates irrelevant to them. ``captured_at`` is
        excluded since it always differs and is not a meaningful signal.
        """
        changed = []
        for f in fields(self):
            if f.name == "captured_at":
                continue
            if getattr(self, f.name) != getattr(other, f.name):
                changed.append(f.name)
        return tuple(changed)
