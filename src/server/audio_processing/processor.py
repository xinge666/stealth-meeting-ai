import logging
from ..shared.event_bus import Event, EventBus, EventType, speech_event
from ..audio.base_asr import BaseASREngine

logger = logging.getLogger(__name__)

class ASRProcessor:
    """
    Subscribes to raw AUDIO_SEGMENT events, transcribes them using ASR,
    and publishes the resulting SPEECH_TEXT events back to the bus.
    """
    def __init__(self, event_bus: EventBus, asr_engine: BaseASREngine):
        self.bus = event_bus
        self.asr = asr_engine
        # Subscribe to raw audio segments from clients
        self.bus.subscribe(EventType.AUDIO_SEGMENT, self._handle_audio)

    async def _handle_audio(self, event: Event):
        """Process a raw audio segment."""
        audio_data = event.data.get("audio")
        if audio_data is None:
            return

        logger.info(f"🎙️  Server ASR: Transcribing segment ({len(audio_data)} samples)...")
        text = await self.asr.transcribe(audio_data)
        
        if text:
            logger.info(f"📝 Transcribed: {text[:80]}")
            # Publish as standard speech text for IntentRouter
            await self.bus.publish(speech_event(text, is_self=False))
        else:
            logger.debug("ASR returned empty text or hallucination.")
