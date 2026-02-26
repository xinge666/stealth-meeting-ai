"""
Unit tests for the async EventBus.
"""

import asyncio
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.event_bus import EventBus, EventType, Event, speech_event, intent_event


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    """Helper to run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestEventBus:
    """Tests for EventBus publish/subscribe mechanism."""

    def test_single_subscriber(self):
        """A single subscriber should receive published events."""
        async def _test():
            bus = EventBus()
            received = []

            async def handler(event: Event):
                received.append(event)

            bus.subscribe(EventType.SPEECH_TEXT, handler)
            await bus.start()

            event = speech_event("hello world")
            await bus.publish(event)

            # Give dispatcher time to process
            await asyncio.sleep(0.1)
            await bus.stop()

            assert len(received) == 1
            assert received[0].data["text"] == "hello world"

        run(_test())

    def test_multiple_subscribers(self):
        """Multiple subscribers for the same event type should all receive it."""
        async def _test():
            bus = EventBus()
            received_a = []
            received_b = []

            async def handler_a(event: Event):
                received_a.append(event)

            async def handler_b(event: Event):
                received_b.append(event)

            bus.subscribe(EventType.SPEECH_TEXT, handler_a)
            bus.subscribe(EventType.SPEECH_TEXT, handler_b)
            await bus.start()

            await bus.publish(speech_event("test"))
            await asyncio.sleep(0.1)
            await bus.stop()

            assert len(received_a) == 1
            assert len(received_b) == 1

        run(_test())

    def test_event_type_filtering(self):
        """Subscribers should only receive events of their subscribed type."""
        async def _test():
            bus = EventBus()
            speech_received = []
            intent_received = []

            async def speech_handler(event: Event):
                speech_received.append(event)

            async def intent_handler(event: Event):
                intent_received.append(event)

            bus.subscribe(EventType.SPEECH_TEXT, speech_handler)
            bus.subscribe(EventType.INTENT_QUESTION, intent_handler)
            await bus.start()

            await bus.publish(speech_event("hello"))
            await bus.publish(intent_event("what is X?", 0.9))
            await asyncio.sleep(0.1)
            await bus.stop()

            assert len(speech_received) == 1
            assert len(intent_received) == 1
            assert speech_received[0].data["text"] == "hello"
            assert intent_received[0].data["text"] == "what is X?"

        run(_test())

    def test_multiple_events(self):
        """Multiple events should all be delivered in order."""
        async def _test():
            bus = EventBus()
            received = []

            async def handler(event: Event):
                received.append(event.data["text"])

            bus.subscribe(EventType.SPEECH_TEXT, handler)
            await bus.start()

            for i in range(5):
                await bus.publish(speech_event(f"msg_{i}"))

            await asyncio.sleep(0.2)
            await bus.stop()

            assert received == ["msg_0", "msg_1", "msg_2", "msg_3", "msg_4"]

        run(_test())

    def test_no_subscribers(self):
        """Publishing with no subscribers should not raise errors."""
        async def _test():
            bus = EventBus()
            await bus.start()
            await bus.publish(speech_event("nobody listening"))
            await asyncio.sleep(0.05)
            await bus.stop()

        run(_test())

    def test_event_constructors(self):
        """Convenience event constructors should produce correct events."""
        e1 = speech_event("hello", is_self=True)
        assert e1.type == EventType.SPEECH_TEXT
        assert e1.data["text"] == "hello"
        assert e1.data["is_self"] is True
        assert e1.source == "audio"

        e2 = intent_event("what?", 0.85)
        assert e2.type == EventType.INTENT_QUESTION
        assert e2.data["confidence"] == 0.85

    def test_handler_exception_does_not_crash_bus(self):
        """A handler that raises should not prevent other handlers from running."""
        async def _test():
            bus = EventBus()
            received_good = []

            async def bad_handler(event: Event):
                raise ValueError("intentional error")

            async def good_handler(event: Event):
                received_good.append(event)

            bus.subscribe(EventType.SPEECH_TEXT, bad_handler)
            bus.subscribe(EventType.SPEECH_TEXT, good_handler)
            await bus.start()

            await bus.publish(speech_event("test error handling"))
            await asyncio.sleep(0.2)
            await bus.stop()

            # Good handler should still receive the event
            assert len(received_good) == 1
            assert received_good[0].data["text"] == "test error handling"

        run(_test())

    def test_queue_full_drops_event(self):
        """When a subscriber queue is full, events should be dropped without error."""
        async def _test():
            bus = EventBus(maxsize=2)
            received = []

            async def slow_handler(event: Event):
                await asyncio.sleep(0.5)  # Deliberately slow
                received.append(event)

            bus.subscribe(EventType.SPEECH_TEXT, slow_handler)
            await bus.start()

            # Publish more events than the queue can hold
            for i in range(5):
                await bus.publish(speech_event(f"msg_{i}"))

            await asyncio.sleep(1.0)
            await bus.stop()

            # Should have received at most 2 (queue size)
            assert len(received) <= 2

        run(_test())

    def test_start_stop_lifecycle(self):
        """Bus should handle start/stop gracefully even with no subscribers."""
        async def _test():
            bus = EventBus()
            await bus.start()
            assert bus._running is True
            await bus.stop()
            assert bus._running is False

        run(_test())
