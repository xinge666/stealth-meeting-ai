"""
Audio capture module with Silero VAD integration.
Captures audio from microphone or system loopback (BlackHole on Mac),
detects speech segments, and publishes them for ASR processing.
"""

import asyncio
import logging
import numpy as np
import threading
import time

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Captures audio from an input device, applies VAD (Silero),
    and emits completed speech segments via callback.
    """

    def __init__(self, config, on_speech_chunk, on_speech_end):
        """
        Args:
            config: AudioConfig instance.
            on_speech_chunk: async callback(np.ndarray) called for every 
                             audio chunk while speech is detected.
            on_speech_end: async callback() called when silence is detected 
                           after a speech segment.
        """
        self.config = config
        self.on_speech_chunk = on_speech_chunk
        self.on_speech_end = on_speech_end
        self.sample_rate = config.sample_rate
        self.chunk_size = 512  # Silero VAD requires 512/1024/1536 at 16kHz

        # VAD state
        self._is_speaking = False
        self._last_speech_time = 0.0

        # Control
        self._running = False
        self._loop = None
        self._stream = None

    async def start(self):
        """Start audio capture and VAD processing."""
        import sounddevice as sd
        import torch

        # Load Silero VAD
        logger.info("Loading Silero VAD model...")
        self._vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        logger.info("Silero VAD loaded.")

        self._loop = asyncio.get_running_loop()
        self._running = True

        # Find the audio device
        device_idx = None
        if self.config.device_name:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if self.config.device_name.lower() in d['name'].lower():
                    device_idx = i
                    logger.info("Using audio device: %s (idx=%d)", d['name'], i)
                    break
            if device_idx is None:
                logger.warning(
                    "Device '%s' not found, using default", self.config.device_name
                )

        # Start the audio stream in a background thread
        def _audio_thread():
            import torch as _torch

            def callback(indata, frames, time_info, status):
                if status:
                    logger.warning("Audio status: %s", status)
                audio_f32 = indata[:, 0].copy()  # mono, float32
                # VAD inference
                tensor = _torch.from_numpy(audio_f32)
                speech_prob = self._vad_model(tensor, self.sample_rate).item()
                is_speech = speech_prob > self.config.vad_threshold

                if is_speech:
                    self._is_speaking = True
                    self._last_speech_time = time.time()
                    # Schedule async callback for the chunk
                    if self._loop and self._running:
                        asyncio.run_coroutine_threadsafe(
                            self.on_speech_chunk(audio_f32), self._loop
                        )
                else:
                    if self._is_speaking:
                        if time.time() - self._last_speech_time > self.config.silence_timeout:
                            self._is_speaking = False
                            # Schedule end-of-speech async callback
                            if self._loop and self._running:
                                asyncio.run_coroutine_threadsafe(
                                    self.on_speech_end(), self._loop
                                )

            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=device_idx,
                channels=1,
                dtype='float32',
                callback=callback
            ):
                logger.info("Audio stream started.")
                while self._running:
                    time.sleep(0.1)

        self._thread = threading.Thread(target=_audio_thread, daemon=True)
        self._thread.start()
        logger.info("AudioCapture started.")

    async def stop(self):
        """Stop audio capture."""
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=3.0)
        logger.info("AudioCapture stopped.")
