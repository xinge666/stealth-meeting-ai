"""
ASR module using Aliyun DashScope Qwen Audio API (sensevoice-v1).
"""

import asyncio
import logging
import numpy as np
import io
import soundfile as sf
import dashscope

from .base_asr import BaseASREngine
from ..config import AudioConfig

logger = logging.getLogger(__name__)


class QwenASREngine(BaseASREngine):
    """
    ASR implementation using DashScope's sensevoice-v1 API.
    """

    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self._initialized = False

    async def initialize(self) -> None:
        """Validate API key and set up client."""
        if not self.config.qwen_api_key:
            logger.warning("Qwen API key is not set. Transcription will fail.")
        else:
            dashscope.api_key = self.config.qwen_api_key
            self._initialized = True
            logger.info("Qwen ASR engine initialized (model=%s)", self.config.qwen_api_model)

    async def transcribe(self, audio_segment: np.ndarray) -> str:
        """
        Transcribe audio via Qwen API.

        Args:
            audio_segment: float32 NumPy array.

        Returns:
            Transcribed text.
        """
        if not self._initialized:
            raise RuntimeError("Qwen ASR engine not initialized properly (check API key).")
        
        # Audio length check: avoid sending completely empty sound
        if len(audio_segment) < self.config.sample_rate * 0.1: # Less than 100ms
            return ""

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._do_transcribe, audio_segment)
        return text

    def _do_transcribe(self, audio_data: np.ndarray) -> str:
        """Synchronous API call. Converts float32 -> WAV bytes in memory."""
        try:
            # Create in-memory wav file for the API
            with io.BytesIO() as wav_io:
                sf.write(wav_io, audio_data, self.config.sample_rate, format='WAV', subtype='PCM_16')
                wav_io.seek(0)
                wav_bytes = wav_io.read()

            # SenseVoice currently only supports files in the DashScope SDK, 
            # so we'll route through universal API
            # For this MVP, we save it locally to a temp file, as the Python SDK 
            # might not support pure bytes for dashscope.audio.asr.Transcription.call
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav.write(wav_bytes)
                temp_path = temp_wav.name

            try:
                task_response = dashscope.audio.asr.Transcription.call(
                    model=self.config.qwen_model,
                    file_urls=[f"file://{temp_path}"],
                )
                
                # In actual implementation, this API might be async or require polling.
                # Assuming simple return here for structural MVP.
                text = ""
                if task_response.status_code == 200:
                    results = task_response.get('output', {}).get('results', [])
                    if results:
                        text = results[0].get('text', "")
                else:
                    logger.error("Qwen API Error: %s", task_response.message)
                return text.strip()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        except Exception as e:
            logger.exception("Qwen ASR API error")
            return ""
