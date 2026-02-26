"""
Unit tests for ScreenCapture with mocked mss and OpenCV.
"""

import asyncio
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vision.screen_capture import ScreenCapture
from src.config import VisionConfig


def run(coro):
    """Helper to run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestScreenCapture:
    """Tests for ScreenCapture frame diff and capture logic."""

    def test_initialization(self):
        """ScreenCapture should initialize with no previous frame."""
        config = VisionConfig()
        callback = MagicMock()
        capture = ScreenCapture(config, callback)
        assert capture._previous_frame is None
        assert capture._running is False

    def test_stop(self):
        """stop() should set _running to False."""
        async def _test():
            config = VisionConfig()
            callback = MagicMock()
            capture = ScreenCapture(config, callback)
            capture._running = True
            await capture.stop()
            assert capture._running is False

        run(_test())

    def test_compute_diff_identical_frames(self):
        """Identical frames should produce a diff score near 0."""
        config = VisionConfig()
        callback = MagicMock()
        capture = ScreenCapture(config, callback)

        # Create identical BGRA frames
        frame = np.full((100, 100, 4), 128, dtype=np.uint8)
        diff = capture._compute_diff(frame, frame)
        assert diff < 0.001

    def test_compute_diff_different_frames(self):
        """Completely different frames should produce a high diff score."""
        config = VisionConfig()
        callback = MagicMock()
        capture = ScreenCapture(config, callback)

        frame_black = np.zeros((100, 100, 4), dtype=np.uint8)
        frame_white = np.full((100, 100, 4), 255, dtype=np.uint8)
        diff = capture._compute_diff(frame_black, frame_white)
        assert diff > 0.5

    def test_compute_diff_partial_change(self):
        """Partial change should produce an intermediate diff score."""
        config = VisionConfig()
        callback = MagicMock()
        capture = ScreenCapture(config, callback)

        frame_a = np.zeros((100, 100, 4), dtype=np.uint8)
        frame_b = frame_a.copy()
        # Change top half
        frame_b[:50, :, :] = 255
        diff = capture._compute_diff(frame_a, frame_b)
        assert 0.1 < diff < 0.9

    def test_first_frame_always_triggers(self):
        """First capture cycle should always call the callback."""
        async def _test():
            config = VisionConfig()
            received = []

            async def on_change(frame):
                received.append(frame)

            capture = ScreenCapture(config, on_change)

            # Mock _grab_screen to return a frame
            test_frame = np.zeros((100, 100, 4), dtype=np.uint8)
            with patch.object(capture, '_grab_screen', return_value=test_frame):
                await capture._capture_cycle()

            assert len(received) == 1
            assert capture._previous_frame is not None

        run(_test())

    def test_no_change_does_not_trigger(self):
        """When frames are identical, callback should not be called."""
        async def _test():
            config = VisionConfig(diff_threshold=0.05)
            received = []

            async def on_change(frame):
                received.append(frame)

            capture = ScreenCapture(config, on_change)
            test_frame = np.full((100, 100, 4), 128, dtype=np.uint8)
            capture._previous_frame = test_frame.copy()

            with patch.object(capture, '_grab_screen', return_value=test_frame):
                await capture._capture_cycle()

            assert len(received) == 0

        run(_test())

    def test_significant_change_triggers(self):
        """When frames differ significantly, callback should be called."""
        async def _test():
            config = VisionConfig(diff_threshold=0.05)
            received = []

            async def on_change(frame):
                received.append(frame)

            capture = ScreenCapture(config, on_change)
            capture._previous_frame = np.zeros((100, 100, 4), dtype=np.uint8)

            new_frame = np.full((100, 100, 4), 200, dtype=np.uint8)
            with patch.object(capture, '_grab_screen', return_value=new_frame):
                await capture._capture_cycle()

            assert len(received) == 1

        run(_test())

    def test_grab_screen_none_skips(self):
        """If _grab_screen returns None, cycle should be a no-op."""
        async def _test():
            config = VisionConfig()
            received = []

            async def on_change(frame):
                received.append(frame)

            capture = ScreenCapture(config, on_change)

            with patch.object(capture, '_grab_screen', return_value=None):
                await capture._capture_cycle()

            assert len(received) == 0
            assert capture._previous_frame is None

        run(_test())
