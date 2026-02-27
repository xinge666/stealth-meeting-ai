#!/usr/bin/env python3
"""
Test script for IntentRouter.
Verifies if various sentences are correctly classified as questions.
"""

import asyncio
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event_bus import EventBus, EventType, speech_event
from src.intelligence.intent_router import IntentRouter
from src.context import ContextManager

TEST_CASES = [
    # Positive (Questions)
    ("Transformer 的 Self-Attention 为什么要除以根号 d_k？", True),
    ("相比于传统的 RNN，Transformer 有什么优势？", True),
    ("请解释一下残差连接的作用。", True),
    ("跟我聊聊什么是 Layer Norm。", True),
    ("你能介绍一下 BERT 的原理吗？", True),
    ("区别是什么呢？", True),
    ("这个算法的优缺点有哪些？", True),
    
    # Negative (Noise / Filler)
    ("大家下午好，今天我们来开个会。", False),
    ("嗯，我觉得你说得对。", False),
    ("哈哈，那太有意思了。", False),
    ("好的，我知道了。", False),
    ("接下来我们看下一张幻灯片。", False),
    ("其实，我觉得吧，这个项目还是挺不错的。", False),
    ("哦哦，没问题。", False),
    ("你好，请问你是谁？", True), # Should be a question
    ("谢谢大家的收看。", False),
]

async def run_test():
    config = AppConfig.from_env()
    bus = EventBus()
    llm = LLMClient(config.llm, bus) # Primary
    await llm.initialize()

    flash_llm = LLMClient(config.flash_llm, bus)
    await flash_llm.initialize()
    
    context_mgr = ContextManager(bus)
    router = IntentRouter(bus, flash_llm, context_mgr)
    
    captured_questions = []

    async def on_question(event):
        captured_questions.append(event.data.get("text"))

    bus.subscribe(EventType.INTENT_QUESTION, on_question)
    await bus.start()

    print("\n" + "="*50)
    print("🎯 Intent Router Classification Test")
    print("="*50 + "\n")

    passed = 0
    for text, expected_is_question in TEST_CASES:
        # Clear captured list
        captured_questions.clear()
        
        # Publish speech event
        await bus.publish(speech_event(text))
        
        # Give a small window for the dispatcher loop
        await asyncio.sleep(0.05)
        
        is_question = len(captured_questions) > 0
        status = "✅ PASS" if is_question == expected_is_question else "❌ FAIL"
        if is_question == expected_is_question:
            passed += 1
            
        indicator = "[Q]" if is_question else "[N]"
        expected_indicator = "[Q]" if expected_is_question else "[N]"
        
        print(f"{status} | Expected: {expected_indicator} | Found: {indicator} | Text: {text}")

    print("\n" + "="*50)
    print(f"Overall Result: {passed}/{len(TEST_CASES)} Passed")
    print("="*50 + "\n")

    await bus.stop()

if __name__ == "__main__":
    asyncio.run(run_test())
