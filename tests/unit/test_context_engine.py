from __future__ import annotations

import time
from typing import Any

from config.config_schema import ContextEngineConfig
from context.context_engine import ContextEngine, _COLLECTOR_REGISTRY
from core.event_bus import EventBus
from core.interfaces import ContextCollector
from events.event_types import CollectorFailed, ContextUpdated
from utils.exceptions import CollectorError


class _StaticCollector(ContextCollector):
    """A test collector returning a fixed, changeable value."""

    def __init__(self, value: str = "initial") -> None:
        self.value = value

    @property
    def name(self) -> str:
        return "system"  # reuse a real field-owning name for simplicity in tests

    def collect(self) -> dict[str, Any]:
        return {"active_window_title": self.value}


class _FailingCollector(ContextCollector):
    @property
    def name(self) -> str:
        return "process"

    def collect(self) -> dict[str, Any]:
        raise CollectorError("simulated failure", context={"collector": "process"})


def _make_engine(monkeypatch, collectors: list[ContextCollector], poll_interval: float = 0.05) -> tuple[ContextEngine, EventBus]:
    bus = EventBus()
    config = ContextEngineConfig(enabled=True, poll_interval_seconds=poll_interval, enabled_collectors=[])
    engine = ContextEngine(event_bus=bus, config=config)
    # Inject test collectors directly, bypassing the name->class registry,
    # since these are test doubles rather than real config-referenced collectors.
    engine._collectors = collectors  # type: ignore[attr-defined]
    return engine, bus


def test_initial_snapshot_populated_on_start(monkeypatch) -> None:
    engine, _bus = _make_engine(monkeypatch, [_StaticCollector("VS Code")])
    engine.start()
    try:
        snapshot = engine.get_snapshot()
        assert snapshot.active_window_title == "VS Code"
    finally:
        engine.stop()


def test_context_updated_published_on_change(monkeypatch) -> None:
    collector = _StaticCollector("VS Code")
    engine, bus = _make_engine(monkeypatch, [collector], poll_interval=0.05)

    received: list[ContextUpdated] = []
    bus.subscribe(ContextUpdated, received.append)

    engine.start()
    try:
        collector.value = "Chrome"
        time.sleep(0.2)  # allow at least one more poll cycle
        assert any(e.snapshot.active_window_title == "Chrome" for e in received)
    finally:
        engine.stop()


def test_no_event_published_when_nothing_changes(monkeypatch) -> None:
    collector = _StaticCollector("VS Code")
    engine, bus = _make_engine(monkeypatch, [collector], poll_interval=0.05)

    engine.start()
    time.sleep(0.15)  # let a couple of poll cycles happen with a stable value

    received: list[ContextUpdated] = []
    bus.subscribe(ContextUpdated, received.append)
    time.sleep(0.15)
    engine.stop()

    assert received == []


def test_failing_collector_does_not_crash_engine_and_publishes_failure_event(monkeypatch) -> None:
    good = _StaticCollector("VS Code")
    bad = _FailingCollector()
    engine, bus = _make_engine(monkeypatch, [good, bad], poll_interval=0.05)

    failures: list[CollectorFailed] = []
    bus.subscribe(CollectorFailed, failures.append)

    engine.start()
    try:
        time.sleep(0.15)
        snapshot = engine.get_snapshot()
        # Good collector's data still made it into the snapshot.
        assert snapshot.active_window_title == "VS Code"
        # Failure was reported rather than silently swallowed or crashing the loop.
        assert any(f.collector_name == "process" for f in failures)
    finally:
        engine.stop()


def test_known_collector_names_are_registered() -> None:
    for expected in ("system", "process", "window", "clipboard"):
        assert expected in _COLLECTOR_REGISTRY
