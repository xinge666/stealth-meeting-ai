import logging
from collections import deque
from .schema import ConversationTurn
from ..event_bus import Event, EventBus, EventType

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
        """Buffer and record AI answers once they are finished."""
        text = event.data.get("text", "")
        is_done = event.data.get("is_done", False)
        
        # We need a place to buffer. For simplicity, let's use a temporary attribute.
        if not hasattr(self, "_current_ai_buffer"):
            self._current_ai_buffer = ""
        
        self._current_ai_buffer += text
        
        if is_done and self._current_ai_buffer.strip():
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

        prompt = f"""[System]
你是一个专家级的会议智囊与技术对讲决策大脑。
请基于所提供【最新屏幕聚焦状态】以及【对话历史】，直接输出针对当前问题的极精简回答要点。

[Context Sync]
近期屏幕内容:
{visual_ctx}

[Conversation Flow]
{chat_history}

[Action Required]
最新提问: "{latest_question}"

务必做到：直接输出答案，拒绝废话，使用分点结构（1. 2. 3.）。"""
        return prompt
