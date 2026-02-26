"""
Unit tests for the Intent Router / Question Classifier.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.intelligence.intent_router import IntentRouter
from src.event_bus import EventBus, EventType, Event, speech_event


class MockEventBus:
    """Minimal mock for testing IntentRouter without full EventBus."""
    def __init__(self):
        self._handlers = {}

    def subscribe(self, event_type, handler):
        self._handlers[event_type] = handler


def run(coro):
    """Helper to run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestIntentRouter:
    """Tests for question classification heuristics."""

    def setup_method(self):
        self.bus = MockEventBus()
        self.router = IntentRouter(self.bus)

    def test_chinese_question_with_mark(self):
        """Chinese question with '？' should score highly."""
        score = self.router._classify("请问这个系统的架构是怎么设计的？")
        assert score > 0.5

    def test_chinese_question_words(self):
        """Chinese question keywords should trigger detection."""
        score = self.router._classify("什么是微服务架构")
        assert score > 0.1  # Single keyword match without "?" scores ~0.2

    def test_english_question(self):
        """English questions should be detected."""
        score = self.router._classify("What is the difference between TCP and UDP?")
        assert score > 0.5

    def test_greeting_filtered(self):
        """Greetings should score low."""
        score = self.router._classify("你好大家好")
        assert score < 0.4

    def test_filler_filtered(self):
        """Filler words should score low."""
        score = self.router._classify("嗯嗯好的")
        assert score < 0.4

    def test_statement_filtered(self):
        """Plain statements should score lower than questions."""
        q_score = self.router._classify("什么是分布式系统？")
        s_score = self.router._classify("这是一个分布式系统")
        assert q_score > s_score

    def test_short_text_filtered(self):
        """Very short text below min_length should not pass."""
        score = self.router._classify("嗯")
        assert score < 0.4

    def test_how_to_question(self):
        """'如何' pattern should be detected."""
        score = self.router._classify("如何在Python中实现异步编程")
        assert score > 0.3

    def test_explain_request(self):
        """'请解释' request should be detected."""
        score = self.router._classify("请解释一下什么是事件驱动架构")
        assert score > 0.4

    def test_comparison_question(self):
        """Comparison questions should be detected."""
        score = self.router._classify("Redis和Memcached有什么区别")
        assert score > 0.4


class TestIntentRouterIntegration:
    """Integration tests: IntentRouter hooked to a real EventBus."""

    def test_question_published_to_bus(self):
        """A question speech event should produce an INTENT_QUESTION event."""
        async def _test():
            bus = EventBus()
            intent_events = []

            async def capture_intent(event: Event):
                intent_events.append(event)

            router = IntentRouter(bus)
            bus.subscribe(EventType.INTENT_QUESTION, capture_intent)
            await bus.start()

            # Publish a clear question
            await bus.publish(speech_event("请问什么是事件驱动架构？", is_self=False))
            await asyncio.sleep(0.3)
            await bus.stop()

            assert len(intent_events) == 1
            assert "事件驱动架构" in intent_events[0].data["text"]

        run(_test())

    def test_self_speech_not_published(self):
        """Speech marked as 'self' should not trigger INTENT_QUESTION."""
        async def _test():
            bus = EventBus()
            intent_events = []

            async def capture_intent(event: Event):
                intent_events.append(event)

            router = IntentRouter(bus)
            bus.subscribe(EventType.INTENT_QUESTION, capture_intent)
            await bus.start()

            await bus.publish(speech_event("什么是微服务？", is_self=True))
            await asyncio.sleep(0.2)
            await bus.stop()

            assert len(intent_events) == 0

        run(_test())

    def test_noise_not_published(self):
        """Non-question text should not produce INTENT_QUESTION events."""
        async def _test():
            bus = EventBus()
            intent_events = []

            async def capture_intent(event: Event):
                intent_events.append(event)

            router = IntentRouter(bus)
            bus.subscribe(EventType.INTENT_QUESTION, capture_intent)
            await bus.start()

            await bus.publish(speech_event("好的没问题", is_self=False))
            await asyncio.sleep(0.2)
            await bus.stop()

            assert len(intent_events) == 0

        run(_test())

    def test_multiple_questions_all_published(self):
        """Multiple questions should each produce an intent event."""
        async def _test():
            bus = EventBus()
            intent_events = []

            async def capture_intent(event: Event):
                intent_events.append(event)

            router = IntentRouter(bus)
            bus.subscribe(EventType.INTENT_QUESTION, capture_intent)
            await bus.start()

            await bus.publish(speech_event("什么是Docker？", is_self=False))
            await bus.publish(speech_event("如何使用Kubernetes？", is_self=False))
            await asyncio.sleep(0.3)
            await bus.stop()

            assert len(intent_events) == 2

        run(_test())

