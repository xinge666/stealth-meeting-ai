# ASR 实现分析：Stealth Meeting AI

## 概览

`stealth-meeting-ai` 当前的 ASR（自动语音识别）实现是**基于片段（Segment-based）**的，而非完全的实时流式（Streaming）。它利用语音活动检测（VAD）将音频流切分为独立的句子或短语，并在用户停止说话后对整个片段进行批量处理。

## 数据流管线

1.  **音频采集 (`src/audio/capture.py`)**
    *   **机制**：使用 `sounddevice` 库在后台线程中从麦克风或环回设备（如 Mac 上的 BlackHole）读取原始音频。
    *   **VAD (语音活动检测)**：集成了 `Silero VAD` 模型实时监控音频流。
    *   **缓冲策略**：
        *   当检测到语音（`is_speech=True`）时，音频帧会被累积到 `self._speech_buffer` 中。
        *   系统会等待静音时长超过 `config.silence_timeout`（静音阈值）。
    *   **触发**：一旦检测到足够长的静音，系统会将缓冲的音频拼接成一个完整的 `numpy` 数组（片段），并通过 `on_speech_segment` 回调函数发出。

2.  **编排层 (`src/main.py`)**
    *   `AudioCapture` 组件在初始化时注册了一个回调函数。
    *   `async def on_speech_segment(audio_data)`：
        *   接收完整的音频片段。
        *   调用 `asr.transcribe(audio_data)` 进行识别。
        *   将生成的文本发布到 `EventBus` 事件总线。

3.  **ASR 处理 (`src/audio/qwen_asr*.py`)**
    *   **接口**：所有引擎都继承自 `BaseASREngine` 并实现 `transcribe(audio_segment: np.ndarray) -> str` 方法。
    *   **Qwen 本地引擎 (`src/audio/qwen_asr_local.py`)**：
        *   接收 `numpy` 数组，直接传递给加载好的 `Qwen3ASRModel`。
        *   对整个片段执行同步推理。
    *   **Qwen API 引擎 (`src/audio/qwen_asr.py`)**：
        *   将 `numpy` 数组写入内存中的临时 `.wav` 文件。
        *   将文件发送至阿里云 DashScope API 进行转写。

## “分段式”与“真正流式”的对比

| 特性 | 当前实现（基于片段/VAD） | 真正流式 ASR (True Streaming) |
| :--- | :--- | :--- |
| **延迟** | 较高（一句话长度 + 静音超时 + 推理时间） | 极低（逐字或逐词实时显示） |
| **反馈** | 用户说完后才会看到文本 | 用户说话过程中文本即时出现 |
| **复杂度** | 较低（无状态 ASR 调用，处理逻辑简单） | 较高（需处理有状态解码、中间结果/修正） |
| **网络** | 每个短语发送一次请求 | 持续的 WebSocket 或 gRPC 流 |

## 结论

目前的系统实现的是**由 VAD 触发的批量转写**。这种方式虽然能有效模拟对话流，但无法提供真正流式 ASR 中常见的“即打即看”（逐字弹出）反馈。若要实现真正的流式，架构需要从 `on_speech_segment` 回调改为基于 Chunk（块）的生成器模式，持续不断地向 ASR 引擎输送数据并处理中间识别结果（Provisional Results）。