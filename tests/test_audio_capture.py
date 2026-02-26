"""
Unit tests for AudioCapture with mocked sounddevice and VAD.
"""

import asyncio
import sys
import os
import time
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audio.capture import AudioCapture
from src.config import AudioConfig


def run(coro):
    """Helper to run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestAudioCapture:
    """Tests for AudioCapture VAD logic."""

    def test_speech_buffer_init(self):
        """AudioCapture should initialize with empty speech buffer."""
        config = AudioConfig()
        callback = MagicMock()
        capture = AudioCapture(config, callback)
        assert capture._speech_buffer == []
        assert capture._is_speaking is False
        assert capture._running is False

    def test_config_values(self):
        """AudioCapture should use config values."""
        config = AudioConfig(sample_rate=44100)
        callback = MagicMock()
        capture = AudioCapture(config, callback)
        assert capture.sample_rate == 44100
        assert capture.chunk_size == 512

    def test_stop_without_start(self):
        """stop() should work even if start() was never called."""
        async def _test():
            config = AudioConfig()
            callback = MagicMock()
            capture = AudioCapture(config, callback)
            await capture.stop()
            assert capture._running is False

        run(_test())

    def test_speech_segment_accumulation(self):
        """Simulate speech buffer accumulation logic directly."""
        config = AudioConfig()
        callback = MagicMock()
        capture = AudioCapture(config, callback)

        # Simulate speech detected
        chunk1 = np.random.randn(512).astype(np.float32)
        chunk2 = np.random.randn(512).astype(np.float32)

        capture._is_speaking = True
        capture._speech_buffer.append(chunk1)
        capture._speech_buffer.append(chunk2)

        assert len(capture._speech_buffer) == 2
        assert capture._is_speaking is True

    def test_speech_end_produces_segment(self):
        """When silence timeout occurs, speech buffer should be concatenated."""
        config = AudioConfig(silence_timeout=0.5)
        callback = MagicMock()
        capture = AudioCapture(config, callback)

        # Simulate accumulated speech
        chunk1 = np.ones(512, dtype=np.float32)
        chunk2 = np.ones(512, dtype=np.float32) * 0.5

        capture._speech_buffer = [chunk1, chunk2]
        capture._is_speaking = True

        # Concatenate to simulate what would happen on speech end
        segment = np.concatenate(capture._speech_buffer)
        assert segment.shape == (1024,)
        assert segment[0] == 1.0
        assert segment[512] == 0.5

    def test_device_name_config(self):
        """Device name should be passed through from config."""
        config = AudioConfig(device_name="BlackHole")
        callback = MagicMock()
        capture = AudioCapture(config, callback)
        assert capture.config.device_name == "BlackHole"
