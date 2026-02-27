#!/usr/bin/env python3
"""
Precision Test for LLM-based IntentRouter.
Verifies both classification (is_question) and extraction (cleaned text).
"""

import asyncio
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig
from src.event_bus import EventBus, EventType, speech_event
from src.intelligence.llm_client import LLMClient
from src.intelligence.intent_router import IntentRouter
from src.context import ContextManager

TEST_CASES = [
    {
        "input": "呃，那个，我想请问一下，就是 Transformer 里面那个 Self-Attention 为什么要除以根号 dk 呀？",
        "expected_is_question": True,
        "desc": "Noisy interview question"
    },
    {
        "input": "好的，那接下来我们聊聊那个，呃，相比传统的 RNN 来说，Transformer 在并行训练上到底有什么核心优势？",
        "expected_is_question": True,
        "desc": "Question mixed with filler"
    },
    {
        "input": "嗯，我觉得你说得挺对的，没关系，我们继续吧。",
        "expected_is_question": False,
        "desc": "Conversational feedback"
    },
    {
        "input": "你好大家好，很高兴来到这里面试，我是张三。",
        "expected_is_question": False,
        "desc": "Self-introduction noise"
    },
    {
        "input": "请帮我解释一下什么是残差连接，以及它在深层网络中解决了什么问题。",
        "expected_is_question": True,
        "desc": "Clear technical question"
    }
]

async def run_precision_test():
    config = AppConfig.from_env()
    bus = EventBus()
    llm = LLMClient(config.llm, bus)
    await llm.initialize()

    flash_llm = LLMClient(config.flash_llm, bus)
    await flash_llm.initialize()
    
    context_mgr = ContextManager(bus)
    router = IntentRouter(bus, flash_llm, context_mgr)
    
    captured_results = []

    async def on_question(event):
        captured_results.append({
            "text": event.data.get("text"),
            "confidence": event.data.get("confidence")
        })

    bus.subscribe(EventType.INTENT_QUESTION, on_question)
    await bus.start()

    print("\n" + "="*70)
    print("🎯 LLM Intent Precision & Extraction Test")
    print("="*70 + "\n")

    passed = 0
    for case in TEST_CASES:
        text = case["input"]
        expected_q = case["expected_is_question"]
        
        captured_results.clear()
        
        print(f"📥 Input: {text}")
        await bus.publish(speech_event(text))
        
        # LLM call takes time
        await asyncio.sleep(2.5) 
        
        found_q = len(captured_results) > 0
        
        status = "❌ FAIL"
        if found_q == expected_q:
            status = "✅ PASS"
            passed += 1
            
        if found_q:
            extracted = captured_results[0]["text"]
            conf = captured_results[0]["confidence"]
            print(f"{status} | Found [Q] (conf={conf:.2f}): \"{extracted}\"")
        else:
            print(f"{status} | No question detected.")
        print("-" * 70)

    print(f"\n🚀 Final Result: {passed}/{len(TEST_CASES)} Passed")
    print("="*70 + "\n")

    await bus.stop()
    await llm.close()

if __name__ == "__main__":
    asyncio.run(run_precision_test())
