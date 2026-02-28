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
    # Common hallucination patterns for ASR models like Whisper/Qwen
    HALLUCINATION_PATTERNS = [
        "字幕by",
        "字幕由",
        "谢谢收看",
        "请收看",
        "索兰娅",
        "感谢您的观看",
        "由索兰娅提供",
    ]

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

    def _clean_text(self, text: str) -> str:
        """
        Post-process transcribed text to remove hallucinations or unwanted watermarks.
        """
        if not text:
            return ""
            
        cleaned = text.strip()
        
        # Check if the text matches any hallucination patterns
        for pattern in self.HALLUCINATION_PATTERNS:
            if pattern in cleaned:
                # If the text is exactly the pattern or very similar, discard it
                if len(cleaned) < len(pattern) + 5:
                    return ""
                # Otherwise, just remove the pattern or keep it if it's part of a longer sentence?
                # Usually these patterns are the ENTIRE output during silence.
                
        return cleaned
