"""
Event bus — the communication backbone of S.A.R.A.

Architectural rule (binding across every phase): modules and agents never
import or call each other directly. They publish typed ``events.event_types.Event``
instances onto this bus, and subscribe to the event types they care about.
This is what lets a new agent, collector, or plugin be added without
touching the code of anything that already exists.

Design notes
------------
- **Type-keyed subscriptions.** Subscribers register against an event
  *class*, not a string topic name — this gives IDE/type-checker support
  and rules out typo'd topic strings as a class of bug.
- **Synchronous dispatch, isolated failures.** Phase 1 dispatch is
  synchronous (a publish call invokes all subscribers before returning).
  This is intentionally simple for now; the interface is written so a
  later async dispatch mode could be added without changing subscriber
  code. Whether one failing subscriber blocks the others is controlled by
  ``EventBusConfig.isolate_subscriber_errors`` (default: isolated).
- **Thread safety.** A ``threading.RLock`` guards the subscriber registry
  since context collectors, voice input, and future agents will run on
  different threads.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable, TypeVar

from events.event_types import Event
from utils.exceptions import HandlerExecutionError
from utils.logger import get_logger

logger = get_logger(__name__)

TEvent = TypeVar("TEvent", bound=Event)
EventHandler = Callable[[TEvent], None]


class EventBus:
    """A synchronous, type-keyed publish/subscribe event bus.

    Example:
        bus = EventBus()

        def on_context_updated(event: ContextUpdated) -> None:
            print(event.changed_fields)

        subscription = bus.subscribe(ContextUpdated, on_context_updated)
        bus.publish(ContextUpdated(changed_fields=("battery",)))
        bus.unsubscribe(subscription)
    """

    def __init__(self, *, isolate_subscriber_errors: bool = True) -> None:
        """
        Args:
            isolate_subscriber_errors: If True (default), an exception
                raised by one subscriber is logged and does not prevent
                other subscribers of the same event from running, and does
                not propagate to the publisher. If False, the first
                failing subscriber's exception propagates immediately.
        """
        self._subscribers: dict[type[Event], list[EventHandler]] = defaultdict(list)
        self._lock = threading.RLock()
        self._isolate_subscriber_errors = isolate_subscriber_errors

    def subscribe(self, event_type: type[TEvent], handler: EventHandler) -> tuple[type[TEvent], EventHandler]:
        """Register ``handler`` to be called whenever ``event_type`` is published.

        Args:
            event_type: The exact ``Event`` subclass to listen for (not
                inherited — subscribing to ``Event`` does not receive
                ``ContextUpdated`` instances; subscribe to each type you
                need explicitly).
            handler: A callable accepting one argument: the published event.

        Returns:
            A subscription token — pass this to ``unsubscribe`` to remove
            this specific handler registration.
        """
        with self._lock:
            self._subscribers[event_type].append(handler)
        logger.debug(
            "Subscribed {} to {}", getattr(handler, "__qualname__", repr(handler)), event_type.__name__
        )
        return (event_type, handler)

    def unsubscribe(self, subscription: tuple[type[Event], EventHandler]) -> None:
        """Remove a previously registered subscription.

        Safe to call with a subscription that was already removed or never
        existed — this is a no-op in that case rather than raising, since
        shutdown code paths often unsubscribe defensively.
        """
        event_type, handler = subscription
        with self._lock:
            handlers = self._subscribers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)
                logger.debug(
                    "Unsubscribed {} from {}",
                    getattr(handler, "__qualname__", repr(handler)),
                    event_type.__name__,
                )

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers of its exact type.

        Args:
            event: The event instance to dispatch.

        Raises:
            HandlerExecutionError: If ``isolate_subscriber_errors`` is
                False and a subscriber raises during handling.
        """
        event_type = type(event)
        with self._lock:
            # Copy the list so a handler that subscribes/unsubscribes
            # during dispatch can't mutate the list we're iterating.
            handlers = list(self._subscribers.get(event_type, ()))

        if not handlers:
            logger.debug("Published {} with no subscribers", event_type.__name__)
            return

        logger.debug(
            "Publishing {} (id={}) to {} subscriber(s)",
            event_type.__name__,
            event.event_id,
            len(handlers),
        )

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 - deliberate: isolate arbitrary subscriber failures
                handler_name = getattr(handler, "__qualname__", repr(handler))
                if self._isolate_subscriber_errors:
                    logger.error(
                        "Subscriber {} raised while handling {}: {}",
                        handler_name,
                        event_type.__name__,
                        exc,
                    )
                    continue
                raise HandlerExecutionError(
                    f"Subscriber {handler_name} failed while handling {event_type.__name__}: {exc}",
                    context={"event_id": event.event_id, "event_type": event_type.__name__},
                ) from exc

    def subscriber_count(self, event_type: type[Event]) -> int:
        """Return how many handlers are currently subscribed to ``event_type``."""
        with self._lock:
            return len(self._subscribers.get(event_type, ()))

    def clear(self) -> None:
        """Remove all subscriptions. Intended for test teardown, not runtime use."""
        with self._lock:
            self._subscribers.clear()
