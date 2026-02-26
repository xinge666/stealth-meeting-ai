"""
Context Aggregator — merges audio and visual context into
a structured prompt for the core LLM.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""
    speaker: str  # "other" or "self"
    text: str
    timestamp: datetime = field(default_factory=datetime.now)


class ContextAggregator:
    """
    Maintains a sliding window of conversation history and the latest
    screen context. Builds prompts for the core LLM.
    """

    def __init__(self, event_bus: EventBus, max_history: int = 20):
        self.bus = event_bus
        self.max_history = max_history
        self.conversation_history: deque[ConversationTurn] = deque(
            maxlen=max_history
        )
        self.latest_screen_context: str = ""
        self._on_question_callback = None

        # Subscribe to events
        self.bus.subscribe(EventType.SPEECH_TEXT, self._handle_speech)
        self.bus.subscribe(EventType.SCREEN_CONTEXT, self._handle_screen)
        self.bus.subscribe(EventType.INTENT_QUESTION, self._handle_question)

    def set_question_callback(self, callback):
        """Set the async callback to invoke when a question needs LLM answer."""
        self._on_question_callback = callback

    async def _handle_speech(self, event: Event):
        """Record all speech into conversation history."""
        text = event.data.get("text", "").strip()
        is_self = event.data.get("is_self", False)
        if text:
            self.conversation_history.append(ConversationTurn(
                speaker="self" if is_self else "other",
                text=text
            ))

    async def _handle_screen(self, event: Event):
        """Update the latest screen context."""
        text = event.data.get("text", "").strip()
        if text:
            self.latest_screen_context = text
            logger.debug("Screen context updated (%d chars)", len(text))

    async def _handle_question(self, event: Event):
        """Build a full prompt when a question is detected and trigger LLM."""
        question = event.data.get("text", "")
        prompt = self.build_prompt(question)
        if self._on_question_callback:
            await self._on_question_callback(prompt, question)

    def build_prompt(self, latest_question: str) -> str:
        """
        Build the full LLM prompt following the design doc template.
        """
        # Format conversation history
        history_lines = []
        for turn in self.conversation_history:
            role = "【我】" if turn.speaker == "self" else "【对方】"
            history_lines.append(f"{role}: {turn.text}")
        chat_history = "\n".join(history_lines[-10:])  # Last 10 turns max

        # Screen context
        visual_ctx = self.latest_screen_context or "(无屏幕上下文)"

        prompt = f"""[System]
你是一个专家级的会议智囊与技术对讲决策大脑。
请基于所提供【最新屏幕聚焦状态】以及【已过滤的对话有效历史】，仅针对对方的最新提问直接输出极精简的回答提示。

[Context Sync]
近期屏幕关键点:
{visual_ctx}

[Conversation Flow]
{chat_history}

[Action Required]
对方提问: "{latest_question}"

务必做到：直接输出答案要点，拒绝寒暄说明，采用纯粹硬核的技术术语或一两句话阐述，使用分点结构如：1. xxx 2. xxx。"""

        return prompt
