"""
Screen capture module with change detection.
Uses mss for fast screenshots and OpenCV for frame diff.
Only triggers OCR when significant screen changes are detected.
"""

import asyncio
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    Periodically captures screenshots, computes frame-diff,
    and triggers OCR when significant changes are detected.
    """

    def __init__(self, config, on_screen_change):
        """
        Args:
            config: VisionConfig instance.
            on_screen_change: async callback(np.ndarray) called when
                              a significant screen change is detected.
        """
        self.config = config
        self.on_screen_change = on_screen_change
        self._previous_frame: Optional[np.ndarray] = None
        self._running = False

    async def start(self):
        """Start the screen capture loop."""
        self._running = True
        logger.info(
            "ScreenCapture started (interval=%.1fs, threshold=%.3f)",
            self.config.capture_interval,
            self.config.diff_threshold
        )
        while self._running:
            try:
                await self._capture_cycle()
            except Exception:
                logger.exception("Screen capture cycle error")
            await asyncio.sleep(self.config.capture_interval)

    async def stop(self):
        """Stop the capture loop."""
        self._running = False
        logger.info("ScreenCapture stopped.")

    async def _capture_cycle(self):
        """Take a screenshot, compare, trigger callback if changed."""
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, self._grab_screen)
        if frame is None:
            return

        if self._previous_frame is not None:
            diff = await loop.run_in_executor(
                None, self._compute_diff, self._previous_frame, frame
            )
            if diff > self.config.diff_threshold:
                logger.info("Screen change detected (diff=%.4f)", diff)
                await self.on_screen_change(frame)
        else:
            # First frame — always process
            await self.on_screen_change(frame)

        self._previous_frame = frame

    def _grab_screen(self) -> Optional[np.ndarray]:
        """Capture the screen using mss."""
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[self.config.monitor_index]
                shot = sct.grab(monitor)
                # Convert to numpy array (BGRA)
                frame = np.array(shot)
                return frame
        except Exception:
            logger.exception("Failed to grab screen")
            return None

    def _compute_diff(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> float:
        """
        Compute a simple normalized diff score between two frames.
        Returns a float in [0, 1] where 0 = identical, 1 = completely different.
        """
        try:
            import cv2

            # Convert to grayscale for faster comparison
            gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGRA2GRAY)
            gray_curr = cv2.cvtColor(curr, cv2.COLOR_BGRA2GRAY)

            # Resize for speed if frames are large
            h, w = gray_prev.shape
            if w > 800:
                scale = 800 / w
                dim = (800, int(h * scale))
                gray_prev = cv2.resize(gray_prev, dim)
                gray_curr = cv2.resize(gray_curr, dim)

            # Mean Absolute Error normalized
            diff = np.mean(np.abs(
                gray_prev.astype(np.float32) - gray_curr.astype(np.float32)
            )) / 255.0

            return float(diff)
        except Exception:
            logger.exception("Diff computation error")
            return 0.0
