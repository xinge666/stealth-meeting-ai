"""
Base abstract classes and factory for Vision engines.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ..config import VisionConfig


class BaseVisionEngine(ABC):
    """
    Abstract base class for extracting context from screen captures.
    """
    def __init__(self, config: VisionConfig):
        self.config = config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize models, API clients, or allocate resources."""
        pass

    @abstractmethod
    async def extract_context(self, frame: np.ndarray) -> str:
        """
        Extract meaningful text context from a BGR/BGRA frame.
        
        Args:
            frame: NumPy array representing the image frame from the screen.
            
        Returns:
            The contextual text (e.g. from OCR or a VLM description).
        """
        pass
