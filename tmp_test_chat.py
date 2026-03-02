import asyncio
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(r"c:\Users\67091\Desktop\gjx_work\stealth-meeting-ai")
sys.path.append(str(project_root.absolute()))

from src.config import AppConfig
from src.event_bus import EventBus, EventType
from src.intelligence.llm_client import LLMClient

async def handle_chunk(event):
    if event.type == EventType.LLM_RESPONSE_CHUNK:
        print(event.data["chunk"], end="", flush=True)
    elif event.type == EventType.LLM_RESPONSE_DONE:
        print("\n\n[Response Done]")

async def test_chat():
    config = AppConfig.from_env()
    bus = EventBus()
    
    # Subscribe to LLM chunks to see streaming output
    bus.subscribe(EventType.LLM_RESPONSE_CHUNK, handle_chunk)
    bus.subscribe(EventType.LLM_RESPONSE_DONE, handle_chunk)
    
    client = LLMClient(config.llm, bus)
    
    print(f"Testing Model: {config.llm.model}")
    print(f"Base URL: {config.llm.base_url}")
    print("-" * 30)
    print("User: hello")
    print("Assistant: ", end="", flush=True)
    
    await bus.start()
    await client.initialize()
    
    try:
        response = await client.ask("hello")
    finally:
        await client.close()
        await bus.stop()

if __name__ == "__main__":
    # Ensure logs don't clutter output
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    asyncio.run(test_chat())
