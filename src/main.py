"""
Main orchestrator — wires all components together and runs the async event loop.
Entry point: python -m src.main
"""

import asyncio
import logging
import signal
import sys
import os

from .config import AppConfig
from .event_bus import EventBus, speech_event, screen_event
from .audio.capture import AudioCapture
from .audio.capture import AudioCapture
from .intelligence.intent_router import IntentRouter
from .context import ContextManager
from .analytics.meeting_analyzer import MeetingAnalyzer
from .intelligence.llm_client import LLMClient
from .intelligence.rag import RAGEngine
from .vision.screen_capture import ScreenCapture
from .presentation.server import WebSocketServer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("meeting_assistant")


async def main():
    """Bootstrap and run all system components."""
    config = AppConfig.from_env()
    bus = EventBus()

    logger.info("=" * 60)
    logger.info("  AI Meeting Assistant — Starting Up")
    logger.info("=" * 60)

    # ── 1. Initialize engines ──────────────────────────────────
    from .audio.factory import create_asr_engine
    asr = create_asr_engine(config.audio)
    logger.info("Initializing ASR engine...")
    await asr.initialize()

    from .vision.factory import create_vision_engine
    vision = create_vision_engine(config.vision)
    logger.info("Initializing Vision engine...")
    await vision.initialize()

    llm = LLMClient(config.llm, bus)
    await llm.initialize()

    flash_llm = LLMClient(config.flash_llm, bus)
    await flash_llm.initialize()

    # Initialize RAG Engine
    rag_engine = RAGEngine(docs_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "docs"))
    await rag_engine.initialize()

    # ── 2. Wire up intelligence layer ──────────────────────────
    context_mgr = ContextManager(bus, max_history=config.max_conversation_history)
    intent_router = IntentRouter(bus, flash_llm, context_mgr)

    # When a question is detected → ask the LLM
    async def on_question(prompt: str, question: str):
        logger.info("🧠 Sending question to RAG & LLM: %s", question[:60])
        
        # 1. Retrieve RAG Context
        retrieved_docs = await rag_engine.search(question, top_k=3)
        
        if retrieved_docs:
            logger.info("📚 Retrieved %d relevant documents", len(retrieved_docs))
            from .event_bus import rag_event
            await bus.publish(rag_event(retrieved_docs))
            
            # Augment the prompt with RAG context
            rag_context_str = "\n".join([f"- [{doc['source']}]: {doc['text']}" for doc in retrieved_docs])
            augmented_prompt = prompt + f"\n\n[RAG Retrieved Context]\n{rag_context_str}"
        else:
            augmented_prompt = prompt
            
        # 2. Ask LLM
        await llm.ask(augmented_prompt, question)

    context_mgr.set_question_callback(on_question)

    # Setup Review/Analysis
    analyzer = MeetingAnalyzer(llm)

    async def on_finish():
        logger.info("🏁 Meeting finished. Generating review report...")
        history = context_mgr.get_full_history()
        report = await analyzer.analyze(history)
        logger.info("✅ Report generated successfully.")

    ws_server = WebSocketServer(config.server, bus)
    ws_server.set_on_finish_callback(on_finish)

    # ── 3. Wire up audio pipeline ──────────────────────────────
    async def on_speech_segment(audio_data):
        logger.info("🎤 Speech segment received (%d samples)", len(audio_data))
        text = await asr.transcribe(audio_data)
        if text:
            logger.info("📝 ASR: %s", text[:80])
            await bus.publish(speech_event(text, is_self=False))

    audio = AudioCapture(config.audio, on_speech_segment)

    # ── 4. Wire up vision pipeline ─────────────────────────────
    async def on_screen_change(frame):
        text = await vision.extract_context(frame)
        if text and len(text.strip()) > 5:
            logger.info("🖥️  Screen text: %s", text[:60])
            await bus.publish(screen_event(text))

    screen = ScreenCapture(config.vision, on_screen_change)

    # ── 5. Start WebSocket server (instance created above) ──────────────────

    # ── 6. Start the event bus ─────────────────────────────────
    await bus.start()

    # ── 7. Graceful shutdown handling ──────────────────────────
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows

    # ── 8. Run all tasks concurrently ──────────────────────────
    logger.info("🚀 All components initializing...")
    logger.info(
        "📱 Open http://0.0.0.0:%d on your phone to view answers",
        config.server.port
    )

    # Start audio (non-blocking, uses thread internally)
    await audio.start()

    # Create tasks for blocking services
    screen_task = asyncio.create_task(screen.start())
    ws_task = asyncio.create_task(ws_server.start())

    try:
        # Wait for shutdown signal
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down...")
        
        # Stop background tasks
        screen_task.cancel()
        ws_task.cancel()
        
        # Call explicit stop methods
        await audio.stop()
        await screen.stop()
        await ws_server.stop()
        await bus.stop()
        await llm.close()
        
        # Wait for tasks to clean up
        await asyncio.gather(screen_task, ws_task, return_exceptions=True)
        logger.info("Bye! 👋")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

