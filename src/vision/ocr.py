"""
OCR module for extracting text from screen captures.
Uses rapidocr-onnxruntime for efficient local OCR (supports Chinese+English).
Falls back to a simpler approach if not available.
"""

import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Extracts text from screen capture frames.
    """

    def __init__(self):
        self._engine = None

    async def initialize(self):
        """Initialize the OCR engine."""
        loop = asyncio.get_running_loop()
        self._engine = await loop.run_in_executor(None, self._load_engine)

    def _load_engine(self):
        """Try to load RapidOCR, fall back to pytesseract."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            logger.info("OCR engine: RapidOCR (ONNX)")
            return ("rapid", engine)
        except ImportError:
            pass
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            logger.info("OCR engine: pytesseract")
            return ("tesseract", pytesseract)
        except Exception:
            pass
        logger.warning("No OCR engine available. Screen text extraction disabled.")
        return None

    async def extract_text(self, frame: np.ndarray) -> str:
        """
        Extract text from a screen frame.

        Args:
            frame: BGRA numpy array from mss.

        Returns:
            Extracted text string.
        """
        if self._engine is None:
            return ""

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._do_ocr, frame)
        return text

    def _do_ocr(self, frame: np.ndarray) -> str:
        """Synchronous OCR (runs in thread pool)."""
        try:
            import cv2
            # Convert BGRA to BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            engine_type, engine = self._engine

            if engine_type == "rapid":
                result, _ = engine(bgr)
                if result:
                    lines = [item[1] for item in result]
                    return "\n".join(lines)
                return ""

            elif engine_type == "tesseract":
                # Convert to RGB for pytesseract
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_img = Image.fromarray(rgb)
                text = engine.image_to_string(pil_img, lang='chi_sim+eng')
                return text.strip()

        except Exception:
            logger.exception("OCR extraction error")
            return ""
