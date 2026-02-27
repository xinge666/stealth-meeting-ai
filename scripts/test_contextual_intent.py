#!/usr/bin/env python3
"""
Contextual Intent Test.
Verifies if the LLM can resolve pronouns based on conversation history.
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

async def run_context_test():
    config = AppConfig.from_env()
    bus = EventBus()
    llm = LLMClient(config.llm, bus)
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

    print("\n" + "="*70)
    print("🧠 Multi-turn Contextual Intent Test")
    print("="*70 + "\n")

    # Turn 1: Establish Context
    print("💬 [Turn 1] ASR: \"什么是 Transformer？\"")
    await bus.publish(speech_event("什么是 Transformer？"))
    await asyncio.sleep(2.0)
    if captured_questions:
        print(f"✅ Found: \"{captured_questions[-1]}\"")
    
    # Turn 2: Follow-up with pronoun "它" (It)
    print("\n💬 [Turn 2] ASR: \"那它的优势是什么？\"")
    await bus.publish(speech_event("那它的优势是什么？"))
    await asyncio.sleep(2.5)
    
    if len(captured_questions) > 1:
        extracted = captured_questions[-1]
        print(f"✅ Found: \"{extracted}\"")
        if "Transformer" in extracted or "优势" in extracted:
            print("🌟 [Success] Pronoun '它' was correctly resolved to 'Transformer'!")
        else:
            print("⚠️ [Warning] Question detected but extraction might be missing context.")
    else:
        print("❌ [Fail] Follow-up question not detected.")

    print("\n" + "="*70 + "\n")

    await bus.stop()
    await llm.close()

if __name__ == "__main__":
    asyncio.run(run_context_test())
