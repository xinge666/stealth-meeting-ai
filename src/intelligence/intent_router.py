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
    "举例", "说一下", "讲一下", "聊一下", "谈谈", "解释一下",
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
    Subscribes to SPEECH_TEXT events and publishes INTENT_QUESTION events.
    """

    def __init__(self, event_bus: EventBus, min_length: int = 4):
        self.bus = event_bus
        self.min_length = min_length
        # Register as subscriber
        self.bus.subscribe(EventType.SPEECH_TEXT, self._handle_speech)

    async def _handle_speech(self, event: Event):
        """Evaluate speech text and publish if it's a question."""
        text = event.data.get("text", "").strip()
        is_self = event.data.get("is_self", False)

        # Skip our own speech
        if is_self:
            return

        # Skip very short text
        if len(text) < self.min_length:
            logger.debug("Text too short, skipping: '%s'", text)
            return

        # Check if it's a question
        confidence = self._classify(text)

        if confidence > 0.4:
            logger.info(
                "✅ Question detected (conf=%.2f): %s", confidence, text[:80]
            )
            await self.bus.publish(intent_event(text, confidence))
        else:
            logger.debug(
                "❌ Filtered as non-question (conf=%.2f): %s", confidence, text[:80]
            )

    def _classify(self, text: str) -> float:
        """
        Score how likely the text is a question (0.0 to 1.0).
        Uses keyword matching and structural heuristics.
        """
        score = 0.0
        text_lower = text.lower()

        # Check noise patterns first — strong negative signal
        for pattern in _NOISE_RE:
            if pattern.search(text_lower):
                score -= 0.3

        # Check for question mark (strong positive signal)
        if "？" in text or "?" in text:
            score += 0.5

        # Check Chinese question words
        zh_matches = sum(1 for w in _ZH_QUESTION_WORDS if w in text)
        score += min(zh_matches * 0.2, 0.5)

        # Check English question words
        en_matches = sum(1 for w in _EN_QUESTION_WORDS if w in text_lower)
        score += min(en_matches * 0.2, 0.5)

        # Length bonus: longer sentences are more likely real questions
        if len(text) > 15:
            score += 0.1
        if len(text) > 30:
            score += 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
