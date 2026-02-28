import asyncio
import logging
from .shared.event_bus import EventBus
from .server.intelligence.llm_client import LLMClient
from .server.intelligence.intent_router import IntentRouter
from .server.context.manager import ContextManager
from .server.analytics.meeting_analyzer import MeetingAnalyzer
from .server.presentation.server import WebSocketServer
from .server.audio.factory import create_asr_engine
from .server.audio_processing.processor import ASRProcessor
from .shared.config import AppConfig

logger = logging.getLogger("MeetingServer")

async def main():
    """Main orchestrator for the heavy-compute Server."""
    config = AppConfig.from_env()
    bus = EventBus()

    # ── 1. Initialize Intelligence & Context ───────────────────
    llm = LLMClient(config.llm, bus)
    flash_llm = LLMClient(config.flash_llm, bus)
    await llm.initialize()
    await flash_llm.initialize()

    context_mgr = ContextManager(bus, max_history=config.max_conversation_history)
    intent_router = IntentRouter(bus, flash_llm, context_mgr)

    # ── 2. Initialize ASR Processing (Server Side) ─────────────
    asr_engine = create_asr_engine(config.audio)
    await asr_engine.initialize()
    asr_processor = ASRProcessor(bus, asr_engine)

    # ── 3. Initialize Presentation (Gateway) ───────────────────
    ws_server = WebSocketServer(config.server, bus)
    
    analyzer = MeetingAnalyzer(llm)

    # Question callback
    async def on_question(prompt: str, question: str):
        logger.info("🧠 Server processing question: %s", question[:60])
        await llm.ask(prompt, question)

    context_mgr.set_question_callback(on_question)

    # Finish callback
    async def on_finish():
        logger.info("🏁 Meeting finished. Generating review report...")
        history = context_mgr.get_full_history()
        report = await analyzer.analyze(history)
        logger.info("✅ Report generated.")

    ws_server.set_on_finish_callback(on_finish)

    # ── 4. Start ──────────────────────────────────────────────
    await bus.start()
    
    try:
        logger.info("🚀 Server is up and running.")
        await ws_server.start()
    except asyncio.CancelledError:
        logger.info("Server shutting down...")
    finally:
        await bus.stop()
        await llm.close()
        await flash_llm.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
