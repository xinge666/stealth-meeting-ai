#!/usr/bin/env python3
"""
Simulate an interview by injecting "Transformer 八股文" questions 
directly into the EventBus.
"""

import asyncio
import logging
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig
from src.event_bus import EventBus, speech_event
from src.intelligence.intent_router import IntentRouter
from src.context import ContextManager
from src.intelligence.llm_client import LLMClient
from src.presentation.server import WebSocketServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulation")

QUESTIONS = [
    "Transformer 的 Self-Attention 为什么要除以根号 d_k？",
    "相比于传统的 RNN，Transformer 在并行训练上有什么核心优势？",
    "为什么 Transformer 需要使用多头注意力 (Multi-head Attention) 机制？",
    "Transformer 的位置编码 (Positional Encoding) 是为了解决什么问题？",
    "简单解释一下 Transformer 中的残差连接 (Residual Connection) 和层归一化 (Layer Norm) 的作用。"
]

async def run_simulation():
    config = AppConfig.from_env()
    bus = EventBus()

    llm = LLMClient(config.llm, bus)
    await llm.initialize()

    flash_llm = LLMClient(config.flash_llm, bus)
    await flash_llm.initialize()

    context_mgr = ContextManager(bus, max_history=config.max_conversation_history)
    intent_router = IntentRouter(bus, flash_llm, context_mgr)

    # Wire Intelligence
    async def on_question(prompt: str, question: str):
        logger.info("🧠 Simulation: Sending to LLM -> %s", question)
        await llm.ask(prompt, question)

    context_mgr.set_question_callback(on_question)

    # 2. Initialize WebSocket Server
    ws_server = WebSocketServer(config.server, bus)

    # 3. Start Bus and Components
    await bus.start()
    ws_task = asyncio.create_task(ws_server.start())

    print("\n" + "="*50)
    print("🚀 Simulation Started")
    print(f"📱 UI URL: http://0.0.0.0:{config.server.port}")
    print("="*50)
    print("\n[Action] Please open the URL on your device now.")
    print("[Action] Waiting 10 seconds for you to connect...")
    await asyncio.sleep(10)

    try:
        for i, q in enumerate(QUESTIONS):
            print(f"\n📢 [Simulating Speech {i+1}/{len(QUESTIONS)}]: {q}")
            # Inject speech event
            await bus.publish(speech_event(q))
            
            # Wait for LLM to process and broadcast
            # We wait a bit between questions to allow the UI to scroll and page
            wait_time = 5
            print(f"⏳ Waiting {wait_time}s for response and paging...")
            await asyncio.sleep(wait_time)
            
        print("\n✅ Simulation Complete!")
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Cleaning up...")
        ws_task.cancel()
        await ws_server.stop()
        await bus.stop()
        await llm.close()
        print("Bye!")

if __name__ == "__main__":
    asyncio.run(run_simulation())
