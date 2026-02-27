"""
Base abstract classes and factory for ASR engines.
"""

from abc import ABC, abstractmethod
import numpy as np

from ..config import AudioConfig


class BaseASREngine(ABC):
    """
    Abstract base class for all speech-to-text engines.
    """
    def __init__(self, config: AudioConfig):
        self.config = config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize models, API clients, or allocate resources."""
        pass

    @abstractmethod
    async def transcribe(self, audio_segment: np.ndarray) -> str:
        """
        Transcribe audio synchronously or asynchronously.
        
        Args:
            audio_segment: float32 NumPy array of shape (samples,) representing 
                           a single channel of audio at the configured sampling rate.
                           
        Returns:
            The transcribed text. Returns empty string on failure.
        """
        pass
