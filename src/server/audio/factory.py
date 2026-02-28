"""
Factory for creating ASR engine instances based on configuration.
"""

from .base_asr import BaseASREngine
from .asr import WhisperASREngine
from ..config import AudioConfig

def create_asr_engine(config: AudioConfig) -> BaseASREngine:
    """
    Factory method to instantiate the correct ASR engine based on config.
    """
    engine_type = config.engine_type.lower()
    
    if engine_type == "whisper":
        return WhisperASREngine(config)
    elif engine_type == "qwen_api":
        from .qwen_asr import QwenASREngine
        return QwenASREngine(config)
    elif engine_type == "qwen_local":
        from .qwen_asr_local import QwenLocalASREngine
        return QwenLocalASREngine(config)
    else:
        raise ValueError(f"Unknown ASR engine type: {engine_type}")
