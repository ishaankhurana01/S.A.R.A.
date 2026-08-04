from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.event_bus import EventBus
from events.event_types import Event
from utils.exceptions import HandlerExecutionError


@dataclass(frozen=True)
class _SampleEvent(Event):
    payload: str = ""


def test_publish_delivers_to_subscriber() -> None:
    bus = EventBus()
    received: list[_SampleEvent] = []

    bus.subscribe(_SampleEvent, received.append)
    bus.publish(_SampleEvent(payload="hello"))

    assert len(received) == 1
    assert received[0].payload == "hello"


def test_publish_with_no_subscribers_does_not_raise() -> None:
    bus = EventBus()
    bus.publish(_SampleEvent(payload="nobody listening"))  # should not raise


def test_multiple_subscribers_all_receive_event() -> None:
    bus = EventBus()
    received_a: list[_SampleEvent] = []
    received_b: list[_SampleEvent] = []

    bus.subscribe(_SampleEvent, received_a.append)
    bus.subscribe(_SampleEvent, received_b.append)
    bus.publish(_SampleEvent(payload="broadcast"))

    assert len(received_a) == 1
    assert len(received_b) == 1


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    received: list[_SampleEvent] = []

    subscription = bus.subscribe(_SampleEvent, received.append)
    bus.unsubscribe(subscription)
    bus.publish(_SampleEvent(payload="should not arrive"))

    assert received == []


def test_unsubscribe_unknown_subscription_is_noop() -> None:
    bus = EventBus()

    def handler(event: _SampleEvent) -> None:
        pass

    # Never subscribed — must not raise.
    bus.unsubscribe((_SampleEvent, handler))


def test_isolated_subscriber_error_does_not_block_others() -> None:
    bus = EventBus(isolate_subscriber_errors=True)
    received: list[_SampleEvent] = []

    def failing_handler(event: _SampleEvent) -> None:
        raise RuntimeError("boom")

    bus.subscribe(_SampleEvent, failing_handler)
    bus.subscribe(_SampleEvent, received.append)

    bus.publish(_SampleEvent(payload="still delivered"))

    assert len(received) == 1


def test_non_isolated_subscriber_error_propagates() -> None:
    bus = EventBus(isolate_subscriber_errors=False)

    def failing_handler(event: _SampleEvent) -> None:
        raise RuntimeError("boom")

    bus.subscribe(_SampleEvent, failing_handler)

    with pytest.raises(HandlerExecutionError):
        bus.publish(_SampleEvent(payload="will raise"))


def test_subscriber_count() -> None:
    bus = EventBus()
    assert bus.subscriber_count(_SampleEvent) == 0

    bus.subscribe(_SampleEvent, lambda e: None)
    assert bus.subscriber_count(_SampleEvent) == 1
