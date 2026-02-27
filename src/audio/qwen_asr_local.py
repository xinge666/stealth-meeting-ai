"""
Local ASR module using Qwen3-ASR.
"""

import asyncio
import logging
import numpy as np
import sys
import os

from .base_asr import BaseASREngine
from ..config import AudioConfig

logger = logging.getLogger(__name__)

class QwenLocalASREngine(BaseASREngine):
    """
    Local implementation of Qwen ASR using transformers.
    """
    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self._model = None

    async def initialize(self) -> None:
        """Load the Qwen3-ASR model."""
        # Add local Qwen3-ASR path to sys.path if it exists
        local_repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "Qwen3-ASR")
        if os.path.exists(local_repo_path) and local_repo_path not in sys.path:
            sys.path.insert(0, local_repo_path)
            logger.info("Added %s to sys.path for Qwen3-ASR", local_repo_path)

        loop = asyncio.get_running_loop()
        try:
            self._model = await loop.run_in_executor(None, self._load_model)
            logger.info("Qwen Local ASR engine initialized (path=%s)", self.config.qwen_local_model_path)
        except Exception as e:
            logger.error("Failed to initialize Qwen Local ASR: %s", e)
            # We don't raise here to allow the system to start even if ASR fails
            # but maybe we should for a clean failure.
            raise

    def _load_model(self):
        import torch
        # Note: qwen-asr package needs to be discoverable
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            logger.error("qwen-asr package not found. Please install it or ensure Qwen3-ASR is in the path.")
            raise
        
        # Determine device and dtype
        device = self.config.qwen_local_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
            
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        
        return Qwen3ASRModel.from_pretrained(
            self.config.qwen_local_model_path,
            dtype=dtype,
            device_map=device,
        )

    async def transcribe(self, audio_segment: np.ndarray) -> str:
        """
        Transcribe audio using local Qwen model.
        """
        if self._model is None:
            return ""
        
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._do_transcribe, audio_segment)
            return text
        except Exception as e:
            logger.error("Qwen Local Transcription error: %s", e)
            return ""

    def _do_transcribe(self, audio_data: np.ndarray) -> str:
        """Synchronous transcription."""
        # Qwen3-ASR transcribe takes (np.ndarray, sr)
        results = self._model.transcribe(
            audio=(audio_data, self.config.sample_rate),
            language=None, # Auto-detect
        )
        if results and len(results) > 0:
            return results[0].text
        return ""
