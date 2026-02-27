# AI Meeting Intelligence Assistant

An AI-powered real-time meeting assistant that acts as your private co-pilot during online meetings, interviews, or exams. The system continuously listens to system audio and monitors your screen, automatically filtering out chatter to identify actual questions. It then uses large language models (LLMs) like DeepSeek or GPT-4o to provide concise, real-time answers directly to your mobile device.

**Key Features:**

* **Real-time Audio & Vision:** Captures mic/system audio and screen changes simultaneously.
* **Invisible UI:** The answers are streamed to a local web server accessed via your phone, ensuring your main screen stays completely clean (defeating screen-sharing or proctoring software checks).
* **Smart Intent Routing:** Uses local heuristics (and optionally local LLMs) to filter out 80% of meeting chatter and pleasantries, only triggering the expensive LLM when a real technical question is asked.
* **Asynchronous & Fast:** Built entirely on Python's `asyncio` and event-driven architecture to guarantee responses within a 3-5 second budget.

---

## 🏗️ Architecture Stack

* **Core:** Python 3.10+, `asyncio` Event Bus
* **Audio:** `sounddevice`, Silero VAD (Voice Activity Detection), `faster-whisper` (Streaming ASR)
* **Vision:** `mss` (Screen Capture), OpenCV (Frame Diff), `rapidocr-onnxruntime` (OCR)
* **Intelligence:** `httpx` (Async LLM streaming via OpenAI-compatible APIs)
* **Presentation:** FastAPI, WebSockets, Vanilla JS (Dark AMOLED mobile interface)

## 🚀 Getting Started

### 1. Prerequisites

1. **Python 3.10+**
2. **Virtual Environment (Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Internal Audio Capture (macOS only)**: Install [BlackHole](https://existential.audio/blackhole/) to capture system audio.

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone <repo_url>
cd meeting_assistant
pip install -r requirements.txt

#### ASR Models Download

1.  **Faster-Whisper**: The system automatically downloads models. To download manually:
    ```bash
    # Example: download 'small' model to local folder
    python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', download_root='./models/whisper')"
    ```
2.  **Qwen3-ASR (Local)**: Recommended to use `modelscope` for fast downloads in China:
    ```bash
    # Download Qwen3-ASR-1.7B
    modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
    ```
    Alternatively, using git:
    ```bash
    git clone https://huggingface.co/Qwen/Qwen3-ASR-1.7B ./models/Qwen3-ASR-1.7B
    ```
    Then update `QWEN_LOCAL_MODEL_PATH` in your `.env` to the local path.
```

*(Note: Depending on your system, `faster-whisper` and `torch` might require specific CUDA/Metal installations for GPU acceleration. By default, it runs on CPU).*

### 3. Configuration (Environment Variables)

The system uses environment variables for configuration. We provide a template file `.env.example`.

**Setup:**

1. Copy the example file: `cp .env.example .env`
2. Edit `.env` with your actual API keys and preferences.

**Key Configuration Highlights:**

* **Dual LLM Setup**: The system uses two LLM configurations:
  * `LLM_*`: High-reasoning model (e.g., DeepSeek, GPT-4o) for generating final answers.
  * `LLM_FLASH_*`: Fast/cheap model (e.g., GPT-4o-mini, Qwen-Turbo) used by the Intent Router to filter chatter.
* **ASR Engine**: Choose between `whisper` (local), `qwen_api` (DashScope), or `qwen_local`.
* **Audio Device**: Set `AUDIO_DEVICE` to "BlackHole" (Mac) or your specific device name.

```bash
# Example loading manually (or let the app load .env automatically)
export LLM_API_KEY="sk-..."
export LLM_FLASH_API_KEY="sk-..."
```

### 4. Running the Assistant

Start the orchestrator:

```bash
python -m src.main
```

1. Open the browser on your **smartphone** (connected to the same Wi-Fi network).
2. Navigate to `http://<your-computer-local-ip>:8765`
3. You will see the dark-themed waiting screen. Start the meeting and watch the answers stream in!

## 🧪 Testing

The project uses `pytest` for unit testing the event bus and intent routing layers:

```bash
python -m pytest tests/ -v
```

## 🗺️ Roadmap & Phases

* [x] Phase 0: Async Event Bus & Core config
* [x] Phase 1: Audio Capture (VAD + Whisper)
* [x] Phase 2: Intent Router (Keyword heuristics)
* [x] Phase 3: Screen Capture & OCR
* [x] Phase 4: Context Aggregation & LLM Client
* [x] Phase 5: FastAPI Mobile Web UI
* [ ] Phase 6: Upgrade Intent Router to quantized local LLM (e.g., Qwen-1.5B)
