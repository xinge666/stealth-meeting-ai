"""
Factory for creating Vision engine instances based on configuration.
"""

from .base_vision import BaseVisionEngine
from .ocr import RapidOCREngine
from ..config import VisionConfig

def create_vision_engine(config: VisionConfig) -> BaseVisionEngine:
    """
    Factory method to instantiate the correct Vision engine based on config.
    """
    engine_type = config.engine_type.lower()
    
    if engine_type == "ocr":
        return RapidOCREngine(config)
    elif engine_type == "vlm_api":
        # Placeholder for visual language model API engine (e.g., GPT-4o vision)
        # from .vlm_engine import VLMEngine
        # return VLMEngine(config)
        raise NotImplementedError("vlm_api engine not yet implemented")
    else:
        raise ValueError(f"Unknown Vision engine type: {engine_type}")
