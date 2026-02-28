import logging
from typing import List
from collections import deque
from .schema import ConversationTurn
from ...shared.event_bus import Event, EventBus, EventType
from ..prompts.templates import ANSWERING_PROMPT

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Maintains a sliding window of conversation history and the latest
    screen context.
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
        self.bus.subscribe(EventType.INTENT_QUESTION, self._handle_intent)
        self.bus.subscribe(EventType.LLM_RESPONSE_CHUNK, self._handle_llm_chunk)
        self.bus.subscribe(EventType.LLM_RESPONSE_DONE, self._handle_llm_done)

    def set_question_callback(self, callback):
        """Set the async callback to invoke when a question needs LLM answer."""
        self._on_question_callback = callback

    async def _handle_speech(self, event: Event):
        """Record human speech into history."""
        text = event.data.get("text", "").strip()
        is_self = event.data.get("is_self", False)
        if text:
            turn = ConversationTurn(
                speaker="self" if is_self else "other",
                text=text
            )
            self.conversation_history.append(turn)

    async def _handle_screen(self, event: Event):
        """Update the latest screen context."""
        text = event.data.get("text", "").strip()
        if text:
            self.latest_screen_context = text

    async def _handle_intent(self, event: Event):
        """Triggered when a valid question is detected."""
        question = event.data.get("text", "")
        if self._on_question_callback:
            prompt = self.get_answering_prompt(question)
            await self._on_question_callback(prompt, question)

    async def _handle_llm_chunk(self, event: Event):
        """Buffer streaming LLM response chunks."""
        chunk = event.data.get("chunk", "")
        if chunk:
            self._current_ai_buffer += chunk

    async def _handle_llm_done(self, event: Event):
        """Record the completed AI answer into history."""
        if self._current_ai_buffer.strip():
            self.conversation_history.append(ConversationTurn(
                speaker="ai",
                text=self._current_ai_buffer.strip()
            ))
        self._current_ai_buffer = ""

    def get_full_history(self) -> List[ConversationTurn]:
        """Return the entire captured conversation turn by turn."""
        return list(self.conversation_history)

    def get_recent_history(self, limit: int = 5) -> str:
        """Get a concise string of the last few turns for intent recognition."""
        turns = list(self.conversation_history)[-limit:]
        lines = []
        for turn in turns:
            role = "我" if turn.speaker == "self" else "对方"
            lines.append(f"{role}: {turn.text}")
        return "\n".join(lines)

    def get_answering_prompt(self, latest_question: str) -> str:
        """Build the full prompt for the final answer generation."""
        history_lines = []
        for turn in self.conversation_history:
            role = "【我】" if turn.speaker == "self" else "【对方】"
            history_lines.append(f"{role}: {turn.text}")
        chat_history = "\n".join(history_lines[-10:])

        visual_ctx = self.latest_screen_context or "(无屏幕上下文)"

        prompt = ANSWERING_PROMPT.format(
            visual_ctx=visual_ctx,
            chat_history=chat_history,
            latest_question=latest_question
        )
        return prompt
