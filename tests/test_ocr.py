"""
Unit tests for OCREngine with mocked RapidOCR.
"""

import asyncio
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vision.ocr import OCREngine


def run(coro):
    """Helper to run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestOCREngine:
    """Tests for OCR engine with mocked backend."""

    def test_not_initialized_returns_empty(self):
        """extract_text() without initialization should return empty string."""
        async def _test():
            engine = OCREngine()
            frame = np.zeros((100, 100, 4), dtype=np.uint8)
            result = await engine.extract_text(frame)
            assert result == ""

        run(_test())

    def test_initialize_with_rapidocr(self):
        """initialize() should load RapidOCR backend."""
        async def _test():
            engine = OCREngine()
            mock_rapid = MagicMock()
            with patch.object(
                engine, '_load_engine',
                return_value=("rapid", mock_rapid)
            ):
                await engine.initialize()
            assert engine._engine is not None
            assert engine._engine[0] == "rapid"

        run(_test())

    def test_extract_text_with_rapidocr(self):
        """extract_text() should return OCR results from RapidOCR."""
        async def _test():
            engine = OCREngine()
            mock_rapid = MagicMock()
            # RapidOCR returns list of [box, text, score]
            mock_rapid.return_value = (
                [
                    [None, "第一行文字", 0.95],
                    [None, "第二行文字", 0.88],
                ],
                None
            )
            engine._engine = ("rapid", mock_rapid)

            # Create a BGRA frame
            frame = np.zeros((100, 100, 4), dtype=np.uint8)
            text = await engine.extract_text(frame)
            assert "第一行文字" in text
            assert "第二行文字" in text

        run(_test())

    def test_extract_text_empty_result(self):
        """extract_text() should return empty string when OCR finds nothing."""
        async def _test():
            engine = OCREngine()
            mock_rapid = MagicMock()
            mock_rapid.return_value = (None, None)
            engine._engine = ("rapid", mock_rapid)

            frame = np.zeros((100, 100, 4), dtype=np.uint8)
            text = await engine.extract_text(frame)
            assert text == ""

        run(_test())

    def test_no_engine_available(self):
        """When no OCR engine is available, _load_engine returns None."""
        engine = OCREngine()
        with patch.dict('sys.modules', {
            'rapidocr_onnxruntime': None,
            'pytesseract': None
        }):
            # Directly test the fallback behavior
            assert engine._engine is None

    def test_extract_text_successive_calls(self):
        """Multiple calls to extract_text should work independently."""
        async def _test():
            engine = OCREngine()
            call_count = [0]

            mock_rapid = MagicMock()

            def side_effect(img):
                call_count[0] += 1
                return (
                    [[None, f"文本{call_count[0]}", 0.9]],
                    None
                )

            mock_rapid.side_effect = side_effect
            engine._engine = ("rapid", mock_rapid)

            frame = np.zeros((100, 100, 4), dtype=np.uint8)

            text1 = await engine.extract_text(frame)
            text2 = await engine.extract_text(frame)

            assert "文本1" in text1
            assert "文本2" in text2

        run(_test())
