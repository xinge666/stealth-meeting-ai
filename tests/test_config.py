"""
Unit tests for AppConfig environment variable loading.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import AppConfig


class TestAppConfig:
    """Tests for AppConfig.from_env()."""

    def test_default_values(self):
        """Config should have sensible defaults without env vars."""
        config = AppConfig()
        assert config.audio.sample_rate == 16000
        assert config.audio.vad_threshold == 0.5
        assert config.vision.capture_interval == 1.5
        assert config.llm.model == "deepseek-chat"
        assert config.server.port == 8765

    def test_from_env_llm(self, monkeypatch):
        """LLM config should load from environment variables."""
        monkeypatch.setenv("LLM_API_KEY", "test-key-123")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.test.com/v1")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        config = AppConfig.from_env()
        assert config.llm.api_key == "test-key-123"
        assert config.llm.base_url == "https://api.test.com/v1"
        assert config.llm.model == "gpt-4o"

    def test_from_env_audio(self, monkeypatch):
        """Audio config should load from environment variables."""
        monkeypatch.setenv("AUDIO_DEVICE", "BlackHole")
        monkeypatch.setenv("WHISPER_MODEL", "large-v2")
        config = AppConfig.from_env()
        assert config.audio.device_name == "BlackHole"
        assert config.audio.whisper_model == "large-v2"

    def test_from_env_server(self, monkeypatch):
        """Server config should load from environment variables."""
        monkeypatch.setenv("SERVER_HOST", "127.0.0.1")
        monkeypatch.setenv("SERVER_PORT", "9999")
        config = AppConfig.from_env()
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9999

    def test_from_env_vision(self, monkeypatch):
        """Vision config should load from environment variables."""
        monkeypatch.setenv("SCREEN_CAPTURE_INTERVAL", "2.5")
        monkeypatch.setenv("SCREEN_DIFF_THRESHOLD", "0.1")
        config = AppConfig.from_env()
        assert config.vision.capture_interval == 2.5
        assert config.vision.diff_threshold == 0.1

    def test_from_env_defaults_without_env(self):
        """Without env vars set, from_env should return defaults."""
        config = AppConfig.from_env()
        assert config.audio.device_name == ""
        assert config.server.host == "0.0.0.0"
        assert config.llm.base_url == "https://api.deepseek.com/v1"
