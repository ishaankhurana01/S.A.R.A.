"""
Context Engine — S.A.R.A.'s World Model.

Owns a background polling loop that runs every enabled
``core.interfaces.ContextCollector``, merges their output into one
``context.context_model.ContextSnapshot``, stores it in
``context.context_store.ContextStore``, and publishes
``events.event_types.ContextUpdated`` on the event bus whenever anything
changed.

This is deliberately the first real subsystem built on top of Phase 1's
foundation (event bus + service registry + config), per the architecture
doc: almost every future agent's "Context Gathering" step depends on it.

Failure isolation
------------------
A single collector raising ``CollectorError`` does not stop the poll
cycle or the engine. The engine logs it, publishes
``events.event_types.CollectorFailed``, and keeps that collector's
previous contribution to the snapshot rather than blanking it out — a
transient failure (e.g. a one-off permission hiccup reading the clipboard)
shouldn't make the World Model forget what it knew a moment ago.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any

from config.config_schema import ContextEngineConfig
from context.collectors.clipboard_collector import ClipboardCollector
from context.collectors.process_collector import ProcessCollector
from context.collectors.system_collector import SystemCollector
from context.collectors.window_collector import WindowCollector
from context.context_model import ContextSnapshot
from context.context_store import ContextStore
from core.event_bus import EventBus
from core.interfaces import ContextCollector
from events.event_types import CollectorFailed, ContextUpdated
from utils.exceptions import CollectorError
from utils.logger import get_logger

logger = get_logger(__name__)

# Maps the collector names used in settings.yaml's
# context_engine.enabled_collectors list to their implementation class.
# Adding a new collector (e.g. project_collector, browser_collector) means
# adding one line here — nothing else in the engine changes.
_COLLECTOR_REGISTRY: dict[str, type[ContextCollector]] = {
    "system": SystemCollector,
    "process": ProcessCollector,
    "window": WindowCollector,
    "clipboard": ClipboardCollector,
}


class ContextEngine:
    """Polls context collectors on a background thread and publishes updates.

    Example:
        engine = ContextEngine(event_bus=bus, config=config.context_engine)
        engine.start()
        ...
        snapshot = engine.get_snapshot()
        ...
        engine.stop()
    """

    def __init__(self, *, event_bus: EventBus, config: ContextEngineConfig) -> None:
        self._event_bus = event_bus
        self._config = config
        self._store = ContextStore()
        self._collectors = self._build_collectors(config.enabled_collectors)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _build_collectors(enabled_names: list[str]) -> list[ContextCollector]:
        collectors: list[ContextCollector] = []
        for name in enabled_names:
            collector_cls = _COLLECTOR_REGISTRY.get(name)
            if collector_cls is None:
                logger.warning(
                    "Unknown collector '{}' in context_engine.enabled_collectors; skipping. "
                    "Known collectors: {}",
                    name,
                    list(_COLLECTOR_REGISTRY.keys()),
                )
                continue
            collectors.append(collector_cls())
        return collectors

    def start(self) -> None:
        """Start the background polling loop. Safe to call only once per instance."""
        if self._thread is not None:
            logger.warning("ContextEngine.start() called but already running; ignoring.")
            return

        # Populate an initial snapshot synchronously so get_snapshot() has
        # real data immediately, rather than an all-None snapshot until the
        # first poll interval elapses.
        self._poll_once()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="ContextEngine", daemon=True)
        self._thread.start()
        logger.info(
            "Context Engine started with collectors {} (poll interval {}s)",
            [c.name for c in self._collectors],
            self._config.poll_interval_seconds,
        )

    def stop(self) -> None:
        """Stop the background polling loop and wait for it to exit."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._config.poll_interval_seconds + 2.0)
        self._thread = None
        logger.info("Context Engine stopped")

    def get_snapshot(self) -> ContextSnapshot:
        """Return the most recently captured ``ContextSnapshot``."""
        return self._store.get()

    def _run_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._config.poll_interval_seconds):
            self._poll_once()

    def _poll_once(self) -> None:
        previous = self._store.get()
        merged_fields: dict[str, Any] = dict(previous.as_dict())
        merged_fields.pop("captured_at", None)

        for collector in self._collectors:
            try:
                result = collector.collect()
            except CollectorError as exc:
                logger.error("Collector '{}' failed: {}", collector.name, exc)
                self._event_bus.publish(
                    CollectorFailed(
                        source="context.context_engine.ContextEngine",
                        collector_name=collector.name,
                        error_message=str(exc),
                    )
                )
                continue  # keep this collector's previous contribution
            except Exception as exc:  # noqa: BLE001 - a collector bug must not kill the engine
                logger.error("Collector '{}' raised an unexpected error: {}", collector.name, exc)
                self._event_bus.publish(
                    CollectorFailed(
                        source="context.context_engine.ContextEngine",
                        collector_name=collector.name,
                        error_message=str(exc),
                    )
                )
                continue

            merged_fields.update(result)

        # Filter to only fields ContextSnapshot actually declares, so a
        # collector returning an unexpected key fails loudly in tests
        # rather than being silently dropped or causing a TypeError here.
        valid_field_names = {f.name for f in dataclasses.fields(ContextSnapshot)}
        unknown_keys = set(merged_fields) - valid_field_names
        if unknown_keys:
            logger.warning("Discarding unknown context fields from collectors: {}", unknown_keys)
            for key in unknown_keys:
                merged_fields.pop(key, None)

        new_snapshot = ContextSnapshot(**merged_fields)
        self._store.set(new_snapshot)

        changed = previous.diff_fields(new_snapshot)
        if changed:
            self._event_bus.publish(
                ContextUpdated(
                    source="context.context_engine.ContextEngine",
                    snapshot=new_snapshot,
                    changed_fields=changed,
                )
            )
