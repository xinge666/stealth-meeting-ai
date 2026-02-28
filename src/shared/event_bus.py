"""
Async event bus for inter-component communication.
Uses asyncio.Queue with typed events and pub/sub pattern.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

class EventType(Enum):
    """All event types flowing through the system."""
    SPEECH_TEXT = auto()        # ASR recognized text
    INTENT_QUESTION = auto()   # Classified as a valid question
    SCREEN_CONTEXT = auto()    # OCR text from screen change
    LLM_RESPONSE_CHUNK = auto()  # Streaming LLM answer chunk
    LLM_RESPONSE_DONE = auto()   # LLM answer complete
    SYSTEM_STATUS = auto()     # System status updates


@dataclass
class Event:
    """Base event structure."""
    type: EventType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""


# ---------------------------------------------------------------------------
# Convenience Event Constructors
# ---------------------------------------------------------------------------

def speech_event(text: str, is_self: bool = False) -> Event:
    """Create a speech recognition event."""
    return Event(
        type=EventType.SPEECH_TEXT,
        data={"text": text, "is_self": is_self},
        source="audio"
    )


def intent_event(text: str, confidence: float) -> Event:
    """Create a question intent event."""
    return Event(
        type=EventType.INTENT_QUESTION,
        data={"text": text, "confidence": confidence},
        source="intent_router"
    )


def screen_event(text: str) -> Event:
    """Create a screen context event."""
    return Event(
        type=EventType.SCREEN_CONTEXT,
        data={"text": text},
        source="vision"
    )


def llm_chunk_event(chunk: str, is_done: bool = False) -> Event:
    """Create an LLM response chunk event."""
    return Event(
        type=EventType.LLM_RESPONSE_DONE if is_done else EventType.LLM_RESPONSE_CHUNK,
        data={"chunk": chunk},
        source="llm"
    )


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

# Type alias for subscriber callbacks
Subscriber = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Async publish/subscribe event bus.

    Components publish typed events; interested subscribers receive them
    asynchronously. Each subscriber gets its own queue to avoid blocking.
    """

    def __init__(self, maxsize: int = 256):
        self._subscribers: Dict[EventType, List[asyncio.Queue]] = {}
        self._handlers: Dict[EventType, List[Subscriber]] = {}
        self._maxsize = maxsize
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def subscribe(self, event_type: EventType, handler: Subscriber) -> None:
        """Register an async handler for a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
            self._handlers[event_type] = []
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers[event_type].append(q)
        self._handlers[event_type].append(handler)
        logger.debug("Subscriber registered for %s", event_type.name)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers of its type."""
        queues = self._subscribers.get(event.type, [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "Queue full for %s subscriber, dropping event", event.type.name
                )

    async def start(self) -> None:
        """Start dispatcher loops for all registered subscribers."""
        self._running = True
        for event_type, queues in self._subscribers.items():
            handlers = self._handlers[event_type]
            for q, handler in zip(queues, handlers):
                task = asyncio.create_task(
                    self._dispatch_loop(q, handler, event_type.name)
                )
                self._tasks.append(task)
        logger.info("EventBus started with %d dispatch loops", len(self._tasks))

    async def stop(self) -> None:
        """Stop all dispatcher loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("EventBus stopped")

    async def _dispatch_loop(
        self, queue: asyncio.Queue, handler: Subscriber, name: str
    ) -> None:
        """Continuously dispatch events from a queue to its handler."""
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                try:
                    await handler(event)
                except Exception:
                    logger.exception("Error in handler for %s", name)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
