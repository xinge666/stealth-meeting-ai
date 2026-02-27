"""
Centralized configuration for the Meeting Assistant system.
All settings loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    engine_type: str = "whisper"  # "whisper", "qwen_api", etc.
    sample_rate: int = 16000
    chunk_duration_ms: int = 32  # 512 samples at 16kHz = 32ms
    vad_threshold: float = 0.5
    silence_timeout: float = 1.5  # seconds of silence to finalize segment
    device_name: str = ""  # empty = default mic; set to "BlackHole" for system audio
    
    # Whisper specific
    whisper_model: str = "small"
    whisper_device: str = "cpu"  # "cpu" or "cuda"
    whisper_compute_type: str = "int8"
    
    # Qwen Local specific
    qwen_local_model_path: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_local_device: str = "cuda"  # or "cpu"
    
    # Qwen API specific
    qwen_api_key: str = ""
    qwen_api_model: str = "sensevoice-v1"


@dataclass
class VisionConfig:
    """Screen capture configuration."""
    engine_type: str = "ocr"  # "ocr", "vlm_api", etc.
    capture_interval: float = 1.5  # seconds between screenshots
    diff_threshold: float = 0.05  # minimum SSIM diff to trigger OCR
    monitor_index: int = 1  # mss monitor index (1 = primary)


@dataclass
class LLMConfig:
    """LLM API configuration."""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    max_tokens: int = 512
    temperature: float = 0.3
    timeout: float = 30.0


@dataclass
class ServerConfig:
    """WebSocket server configuration."""
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass
class AppConfig:
    """Top-level application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    max_conversation_history: int = 20  # sliding window size

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls()

        # LLM config
        config.llm.api_key = os.getenv("LLM_API_KEY", config.llm.api_key)
        config.llm.base_url = os.getenv("LLM_BASE_URL", config.llm.base_url)
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)

        # Audio config
        config.audio.engine_type = os.getenv("ASR_ENGINE", config.audio.engine_type)
        config.audio.device_name = os.getenv("AUDIO_DEVICE", config.audio.device_name)
        config.audio.whisper_model = os.getenv("WHISPER_MODEL", config.audio.whisper_model)
        config.audio.qwen_api_key = os.getenv("QWEN_API_KEY", config.audio.qwen_api_key)
        config.audio.qwen_api_model = os.getenv("QWEN_API_MODEL", config.audio.qwen_api_model)
        config.audio.qwen_local_model_path = os.getenv("QWEN_LOCAL_MODEL_PATH", config.audio.qwen_local_model_path)
        config.audio.qwen_local_device = os.getenv("QWEN_LOCAL_DEVICE", config.audio.qwen_local_device)

        # Vision config
        config.vision.engine_type = os.getenv("VISION_ENGINE", config.vision.engine_type)
        interval = os.getenv("SCREEN_CAPTURE_INTERVAL")
        if interval:
            config.vision.capture_interval = float(interval)
        threshold = os.getenv("SCREEN_DIFF_THRESHOLD")
        if threshold:
            config.vision.diff_threshold = float(threshold)

        # Server config
        config.server.host = os.getenv("SERVER_HOST", config.server.host)
        port = os.getenv("SERVER_PORT")
        if port:
            config.server.port = int(port)

        return config
