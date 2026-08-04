"""
Thread-safe holder for the latest World Model snapshot.

``ContextStore`` is deliberately minimal: it holds the most recent
``ContextSnapshot`` and nothing else (no history, no persistence — that's
what Episodic Memory is for in Phase 5). Agents and other modules read the
current snapshot via ``ContextEngine.get_snapshot()``, which delegates
here; nothing outside ``context/`` should hold a reference to this store
directly.
"""

from __future__ import annotations

import threading

from context.context_model import ContextSnapshot


class ContextStore:
    """Holds the latest ``ContextSnapshot``, safe for concurrent reads/writes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = ContextSnapshot()

    def get(self) -> ContextSnapshot:
        """Return the current snapshot."""
        with self._lock:
            return self._snapshot

    def set(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        """Replace the current snapshot and return the previous one.

        Args:
            snapshot: The new snapshot to store.

        Returns:
            The snapshot that was current *before* this call, so the
            caller (``ContextEngine``) can diff old vs. new to compute
            changed fields without a separate read-then-write race.
        """
        with self._lock:
            previous = self._snapshot
            self._snapshot = snapshot
            return previous
