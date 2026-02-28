"""
Intent Router / Question Classifier.
Filters ASR text to identify genuine questions, blocking noise and chatter.
Phase 1: Keyword/heuristic-based classifier.
Phase 2 (future): LLM micro-classifier or fine-tuned TinyBERT.
"""

import logging
import re

from ..event_bus import Event, EventBus, EventType, intent_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic question detection patterns
# ---------------------------------------------------------------------------

# Chinese question indicators
_ZH_QUESTION_WORDS = [
    "什么", "怎么", "如何", "为什么", "为何", "哪个", "哪些", "哪里", "哪儿",
    "谁", "几个", "几种", "多少", "是否", "能否", "可以吗", "对吗", "吗",
    "呢", "请问", "请说", "请介绍", "请解释", "请讲", "请描述",
    "区别", "优缺点", "优势", "劣势", "差异", "对比", "比较",
    "举例", "说一下", "讲一下", "聊一下", "谈谈", "解释一下", "什么是", "聊聊", "说说", "的作用", "的原理",
]

# English question indicators
_EN_QUESTION_WORDS = [
    "what", "how", "why", "when", "where", "which", "who", "whom",
    "could you", "can you", "would you", "will you", "do you",
    "is it", "are there", "have you", "please explain", "describe",
    "tell me", "what's", "how's",
]

# Patterns that indicate non-questions (greetings, filler)
_NOISE_PATTERNS = [
    r"^(嗯|哦|啊|好的|好|ok|okay|是的|对|没错|行|嗯嗯|哈哈|呵呵)",
    r"^(hello|hi|hey|你好|大家好|各位好|谢谢|感谢|辛苦了)",
    r"^(我[觉认]得|我[想要]说|其实|所以说|然后|接下来|那个)",
]

_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]


class IntentRouter:
    """
    Classifies incoming ASR text as question or noise.
    Uses a two-phase approach:
      Phase 1: Fast heuristic pre-filter to skip obvious noise (saves LLM API calls).
      Phase 2: LLM micro-classifier for precise intent detection and question extraction.
    Subscribes to SPEECH_TEXT events and publishes INTENT_QUESTION events.
    """

    def __init__(self, event_bus: EventBus, llm_client: "LLMClient", context_manager: "ContextManager", min_length: int = 4):
        self.bus = event_bus
        self.llm = llm_client
        self.context = context_manager
        self.min_length = min_length
        # Register as subscriber
        self.bus.subscribe(EventType.SPEECH_TEXT, self._handle_speech)

    def _is_obvious_noise(self, text: str) -> bool:
        """
        Fast heuristic check: returns True if the text is clearly noise
        (pure greetings, filler words, etc.) and should skip LLM classification.
        Only filters very short, obvious non-questions to avoid false negatives.
        """
        # Very short text that matches noise patterns is almost certainly not a question
        if len(text) < 10:
            for pattern in _NOISE_RE:
                if pattern.match(text):
                    return True
        return False

    async def _handle_speech(self, event: Event):
        """Evaluate speech text and publish if it's a question."""
        text = event.data.get("text", "").strip()
        is_self = event.data.get("is_self", False)

        # Skip our own speech
        if is_self:
            return

        # Skip very short text
        if len(text) < self.min_length:
            return

        # Phase 1: Fast heuristic pre-filter
        if self._is_obvious_noise(text):
            logger.debug("⚡ [Heuristic] Skipped obvious noise: %s", text[:40])
            return

        # Phase 2: Use LLM for precise classification and extraction
        # Get recent history for coreference resolution
        history = self.context.get_recent_history(limit=5)

        result = await self.llm.analyze_intent(text, history=history)
        
        is_question = result.get("is_question", False)
        extracted_text = result.get("extracted_question", "").strip()
        confidence = result.get("confidence", 0.0)

        if is_question and extracted_text and confidence >= 0.6:
            logger.info(
                "✅ [LLM Intent] Question detected (conf=%.2f): %s", 
                confidence, extracted_text[:80]
            )
            # Publish the CLEANED question text
            await self.bus.publish(intent_event(extracted_text, confidence))
        else:
            logger.debug(
                "❌ [LLM Intent] Filtered (conf=%.2f): %s", 
                confidence, text[:80]
            )

