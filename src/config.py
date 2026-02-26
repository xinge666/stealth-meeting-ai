"""
Centralized configuration for the Meeting Assistant system.
All settings loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = 16000
    chunk_duration_ms: int = 32  # 512 samples at 16kHz = 32ms
    vad_threshold: float = 0.5
    silence_timeout: float = 1.5  # seconds of silence to finalize segment
    device_name: str = ""  # empty = default mic; set to "BlackHole" for system audio
    whisper_model: str = "small"
    whisper_device: str = "cpu"  # "cpu" or "cuda"
    whisper_compute_type: str = "int8"


@dataclass
class VisionConfig:
    """Screen capture configuration."""
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
        config.audio.device_name = os.getenv("AUDIO_DEVICE", config.audio.device_name)
        config.audio.whisper_model = os.getenv("WHISPER_MODEL", config.audio.whisper_model)

        # Vision config
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
