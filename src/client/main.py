import asyncio
import logging
import signal
import sys
from .audio.capture import AudioCapture
from .vision.screen_capture import ScreenCapture
from .vision.ocr import RapidOCREngine
from .connection import ClientConnection
from ..shared.config import AppConfig
from ..shared.protocol import ClientMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EdgeClient")

async def main():
    """Main entry point for the Edge Client."""
    config = AppConfig.from_env()
    
    # 1. Initialize Server Connection
    conn = ClientConnection(config.server.ws_server_url)
    
    # 2. Audio Callbacks
    async def on_speech_chunk(audio_data):
        # audio_data is np.ndarray (float32)
        message = ClientMessage.audio_chunk(audio_data.tobytes())
        await conn.send(message)

    async def on_speech_end():
        logger.info("🎤 Speech segment ended.")
        await conn.send(ClientMessage.audio_speech_end())

    audio = AudioCapture(config.audio, on_speech_chunk, on_speech_end)

    # 3. Vision Callbacks
    ocr_engine = RapidOCREngine(config.vision)
    await ocr_engine.initialize()

    async def on_screen_change(frame):
        # Extract text using CPU OCR locally
        text = await ocr_engine.extract_context(frame)
        if text and len(text.strip()) > 5:
            logger.info("🖥️  Screen text extracted (length=%d)", len(text))
            await conn.send(ClientMessage.screen_text(text))

    vision = ScreenCapture(config.vision, on_screen_change)

    # 4. Starting everything
    logger.info("🚀 Starting Edge Client components...")
    
    try:
        await conn.start()
        await audio.start()
        # Vision capture runs as a background loop
        asyncio.create_task(vision.start())
        
        logger.info("✅ Edge Client is running. Press Ctrl+C to stop.")
        
        # Keep the main loop running
        while True:
            await asyncio.sleep(1.0)
            
    except asyncio.CancelledError:
        logger.info("Stopping client...")
    except Exception as e:
        logger.exception("Client runtime error")
    finally:
        await audio.stop()
        await vision.stop()
        await conn.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
