import pyaudio
import numpy as np
import torch
import queue
import threading
import time
from faster_whisper import WhisperModel

class AudioPipeline:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Silero VAD requires exact chunk sizes: 512, 1024, or 1536 samples for 16000Hz
        self.chunk_size = 512
        self.audio_format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        
        # VAD Parameters
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

        # ASR Parameters (using small model for speed in real-time)
        print("Loading Whisper model...")
        self.asr_model = WhisperModel("small", device="cpu", compute_type="int8") # Use "cuda" if GPU is available
        print("Whisper model loaded.")
        
        self.vad_iterator = self.VADIterator(self.model)
        self.audio_queue = queue.Queue()
        self.is_recording = False

        self.speech_buffer = []
        self.silence_threshold = 1.0 # 1 second of silence to trigger ASR
        self.last_speech_time = time.time()
        self.is_speaking = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        # Decode PyAudio Int16 stream into float32 array (-1.0 to 1.0)
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_recording = True
        self.stream.start_stream()
        print("Started Recording & VAD...")
        
        # Start processing thread
        threading.Thread(target=self._process_stream, daemon=True).start()

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Stopped Recording.")

    def _process_stream(self):
        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                self._process_chunk(chunk)
            except queue.Empty:
                continue
                
    def _process_chunk(self, chunk):
        # Prepare chunk for Silero VAD (requires torch tensor)
        tensor_chunk = torch.from_numpy(chunk)
        
        # Check if speech is detected
        speech_prob = self.model(tensor_chunk, self.sample_rate).item()
        is_speech = speech_prob > 0.5 # Threshold

        if is_speech:
            self.is_speaking = True
            self.last_speech_time = time.time()
            self.speech_buffer.append(chunk)
        else:
            if self.is_speaking:
                self.speech_buffer.append(chunk)
                # Check for silence duration
                if time.time() - self.last_speech_time > self.silence_threshold:
                    self._on_speech_end()

    def _on_speech_end(self):
        self.is_speaking = False
        if len(self.speech_buffer) > 0:
            print("[VAD] Silence detected. Triggering ASR...")
            full_audio = np.concatenate(self.speech_buffer)
            self.speech_buffer = [] # Reset buffer
            
            # Start ASR in a separate thread so it doesn't block VAD capturing
            threading.Thread(target=self._transcribe_audio, args=(full_audio,), daemon=True).start()

    def _transcribe_audio(self, audio_data):
        print("[ASR] Transcribing...")
        # Faster-whisper expects a 1D numpy array of float32 for audio directly.
        segments, info = self.asr_model.transcribe(audio_data, beam_size=5)
        text = "".join([segment.text for segment in segments])
        if text.strip():
            print(f"\n✅ [识别结果]: {text.strip()}\n")

if __name__ == "__main__":
    pipeline = AudioPipeline()
    try:
        pipeline.start_recording()
        print("Running for 5 seconds to test initialization...")
        time.sleep(5)
        pipeline.stop_recording()
        print("Test complete.")
    except KeyboardInterrupt:
        pipeline.stop_recording()
