"""
ASR (Automatic Speech Recognition) module using faster-whisper.
Runs transcription in a thread executor to avoid blocking the async loop.
"""

import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ASREngine:
    """
    Wraps faster-whisper for non-blocking speech-to-text.
    """

    def __init__(self, config):
        """
        Args:
            config: AudioConfig instance.
        """
        self.config = config
        self._model = None

    async def initialize(self):
        """Load the whisper model (run in executor since it's heavy)."""
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        logger.info("ASR engine initialized (model=%s)", self.config.whisper_model)

    def _load_model(self):
        from faster_whisper import WhisperModel
        return WhisperModel(
            self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type
        )

    async def transcribe(self, audio_segment: np.ndarray) -> str:
        """
        Transcribe an audio segment to text.

        Args:
            audio_segment: numpy float32 array of audio samples.

        Returns:
            Recognized text string.
        """
        if self._model is None:
            raise RuntimeError("ASR engine not initialized. Call initialize() first.")

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._do_transcribe, audio_segment)
        return text

    def _do_transcribe(self, audio_data: np.ndarray) -> str:
        """Synchronous transcription (runs in thread pool)."""
        segments, info = self._model.transcribe(
            audio_data,
            beam_size=5,
            language="zh",  # Optimize for Chinese; remove for auto-detect
            vad_filter=False  # We already do VAD upstream
        )
        text = "".join(seg.text for seg in segments).strip()
        return text
