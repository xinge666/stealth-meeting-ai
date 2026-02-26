"""
Main orchestrator — wires all components together and runs the async event loop.
Entry point: python -m src.main
"""

import asyncio
import logging
import signal
import sys

from .config import AppConfig
from .event_bus import EventBus, speech_event, screen_event
from .audio.capture import AudioCapture
from .audio.asr import ASREngine
from .intelligence.intent_router import IntentRouter
from .intelligence.context_aggregator import ContextAggregator
from .intelligence.llm_client import LLMClient
from .vision.screen_capture import ScreenCapture
from .vision.ocr import OCREngine
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
    asr = ASREngine(config.audio)
    logger.info("Initializing ASR engine...")
    await asr.initialize()

    ocr = OCREngine()
    logger.info("Initializing OCR engine...")
    await ocr.initialize()

    llm = LLMClient(config.llm, bus)
    await llm.initialize()

    # ── 2. Wire up intelligence layer ──────────────────────────
    intent_router = IntentRouter(bus)
    context_agg = ContextAggregator(bus, max_history=config.max_conversation_history)

    # When a question is detected → ask the LLM
    async def on_question(prompt: str, question: str):
        logger.info("🧠 Sending to LLM: %s", question[:60])
        await llm.ask(prompt, question)

    context_agg.set_question_callback(on_question)

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
        text = await ocr.extract_text(frame)
        if text and len(text.strip()) > 5:
            logger.info("🖥️  Screen text: %s", text[:60])
            await bus.publish(screen_event(text))

    screen = ScreenCapture(config.vision, on_screen_change)

    # ── 5. Start WebSocket server ──────────────────────────────
    ws_server = WebSocketServer(config.server, bus)

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

    try:
        await asyncio.gather(
            audio.start(),
            screen.start(),
            ws_server.start(),
            shutdown_event.wait(),
        )
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down...")
        await audio.stop()
        await screen.stop()
        await ws_server.stop()
        await bus.stop()
        await llm.close()
        logger.info("Bye! 👋")


if __name__ == "__main__":
    asyncio.run(main())
