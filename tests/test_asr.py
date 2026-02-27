"""
Unit tests for WhisperASREngine with mocked faster-whisper.
"""

import asyncio
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audio.factory import create_asr_engine
from src.audio.asr import WhisperASREngine
from src.audio.qwen_asr_local import QwenLocalASREngine
from src.config import AudioConfig


def run(coro):
    """Helper to run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestASREngine:
    """Tests for ASR engine with mocked WhisperModel."""

    def _make_mock_model(self, text="这是一段测试文本"):
        """Create a mock WhisperModel that returns preset text."""
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = text
        mock_info = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        return mock_model

    def test_factory_creates_whisper(self):
        """Factory should return WhisperASREngine by default."""
        config = AudioConfig()
        engine = create_asr_engine(config)
        assert isinstance(engine, WhisperASREngine)

    def test_factory_creates_qwen_local(self):
        """Factory should return QwenLocalASREngine when configured."""
        config = AudioConfig(engine_type="qwen_local")
        engine = create_asr_engine(config)
        assert isinstance(engine, QwenLocalASREngine)

    def test_initialize_loads_model(self):
        """initialize() should load the whisper model."""
        async def _test():
            config = AudioConfig()
            engine = create_asr_engine(config)

            mock_model = self._make_mock_model()
            with patch.object(engine, '_load_model', return_value=mock_model):
                await engine.initialize()
                assert engine._model is not None

        run(_test())

    def test_transcribe_returns_text(self):
        """transcribe() should return recognized text from audio."""
        async def _test():
            config = AudioConfig()
            engine = create_asr_engine(config)

            mock_model = self._make_mock_model("你好世界")
            with patch.object(engine, '_load_model', return_value=mock_model):
                await engine.initialize()

            # Create a dummy audio segment
            audio = np.random.randn(16000).astype(np.float32)
            text = await engine.transcribe(audio)
            assert text == "你好世界"

        run(_test())

    def test_transcribe_empty_audio(self):
        """transcribe() on empty result should return empty string."""
        async def _test():
            config = AudioConfig()
            engine = create_asr_engine(config)

            mock_model = MagicMock()
            mock_info = MagicMock()
            mock_model.transcribe.return_value = ([], mock_info)

            with patch.object(engine, '_load_model', return_value=mock_model):
                await engine.initialize()

            audio = np.random.randn(16000).astype(np.float32)
            text = await engine.transcribe(audio)
            assert text == ""

        run(_test())

    def test_transcribe_without_init_raises(self):
        """transcribe() without initialize() should raise RuntimeError."""
        async def _test():
            config = AudioConfig()
            engine = create_asr_engine(config)
            audio = np.random.randn(16000).astype(np.float32)
            try:
                await engine.transcribe(audio)
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "not initialized" in str(e)

        run(_test())


class TestQwenLocalASREngine:
    """Tests for the local Qwen ASR engine."""

    def test_initialize_loads_model(self):
        """Should load the QwenLocalASR model via transformers."""
        async def _test():
            config = AudioConfig(engine_type="qwen_local")
            engine = create_asr_engine(config)

            mock_model = MagicMock()
            with patch.object(engine, '_load_model', return_value=mock_model):
                await engine.initialize()
                assert engine._model == mock_model

        run(_test())

    def test_transcribe_calls_qwen_api(self):
        """Should call the local Qwen model transcribe method."""
        async def _test():
            config = AudioConfig(engine_type="qwen_local")
            engine = create_asr_engine(config)

            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_result.text = "Qwen识别结果"
            mock_model.transcribe.return_value = [mock_result]
            
            engine._model = mock_model
            
            audio = np.random.randn(16000).astype(np.float32)
            text = await engine.transcribe(audio)
            
            assert text == "Qwen识别结果"
            # Verify it passed the correct sample rate
            args, kwargs = mock_model.transcribe.call_args
            assert kwargs['audio'][1] == 16000

        run(_test())

    def test_uninitialized_returns_empty(self):
        """Should return empty string if not initialized."""
        async def _test():
            config = AudioConfig(engine_type="qwen_local")
            engine = QwenLocalASREngine(config)
            audio = np.random.randn(16000).astype(np.float32)
            text = await engine.transcribe(audio)
            assert text == ""

        run(_test())

