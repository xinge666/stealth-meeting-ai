# 智能会议问答辅助系统 (Stealth Meeting AI)

## 🌟 项目简介

这是一个为程序员和职场人打造的**端侧实时会议辅助系统**。它像一位隐形的“副驾驶”，在您参加远程会议、面试或技术研讨会时，实时监听语音流并监控屏幕动态。系统通过智能算法精准提取对方提出的问题，并利用 DeepSeek/GPT-4o 等顶级大模型即时生成参考答案。

**核心杀手锏：绝对的隐蔽性。** 答案不会显示在您的电脑主屏幕上，而是通过局域网流式推送到您的**手机**浏览器中。即便您正在进行屏幕共享或被监控软件录屏，对方也无从察觉。

---

## 🚀 核心特性

- **🌊 实时音视频感官**：
  - **听觉**：利用 Silero VAD (静音检测) 和 Faster-Whisper (流式语音转文字) 捕获系统或麦克风语音。
  - **视觉**：采用极速截屏与 OpenCV 帧差检测，配合轻量化本地 OCR (RapidOCR) 实时提取屏幕关键信息。
- **📱 物理隔离展示**：
  - 采用手机作为“第二屏幕”。通过手机浏览器访问局域网地址，即可像查看即时消息一样接收流式推送到答案。
- **🧠 智能意图路由 (Intent Router)**：
  - 内置过滤引擎。自动剔除会议中的寒暄、废话及背景噪音，仅在识识别到真正的“提问”意图时才触发 LLM 调用，既省电又省钱。
- **⚡ 极速异步响应**：
  - 基于 Python `asyncio` 全异步驱动。从感知到问题到给出第一行答案，延迟控制在秒级。
- **🛡️ 纯净宿主环境**：
  - 后台静默运行，无需安装复杂的 UI 交互界面，完美规避各类屏幕检测与作弊防范算法。

---

## 🏗️ 技术架构

- **后端核心**: Python 3.10+, `asyncio` 事件总线模式
- **音频引擎**: `sounddevice`, `silero-vad`, `faster-whisper`
- **视觉引擎**: `mss` (极速截屏), `opencv-python`, `rapidocr-onnxruntime`
- **大脑中枢**: DeepSeek / OpenAI API (支持流式 `SSE`)
- **展示层**: FastAPI + WebSockets + HTML5 (暗黑护眼模式)

---

## 📂 项目结构

```text
stealth-meeting-ai/
├── src/
│   ├── audio/        # 音频捕获与 ASR 识别
│   ├── vision/       # 屏幕采集与本地 OCR
│   ├── intelligence/ # 意图识别、上下文聚合与 LLM 交互
│   ├── presentation/ # WebSocket 服务器与手机端 Web 界面
│   └── main.py       # 系统编排总线
├── tests/            # 自动化单元测试
└── requirements.txt  # 依赖清单
```

---

## 💡 使用场景

1. **远程技术面试**：帮助您在压力环境下快速梳理技术要点，查漏补缺。
2. **大型在线会议**：实时捕捉会议中的关键提问，确保不错过任何一个需要您回答的细节。
3. **技术研讨交流**：当遇到知识盲区时，AI 即刻提供背景资料，助您应对自如。

---

## 🚀 快速开始

### 1. 环境准备

1. **Python 3.10+**
2. 建议使用虚拟环境：

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **捕获系统声音 (仅限 macOS)**：需要事先安装 [BlackHole](https://existential.audio/blackhole/)。Windows 用户可以直接依靠系统混音器。

### 2. 安装依赖

```bash
git clone <repo_url>
cd meeting_assistant
pip install -r requirements.txt
```

*(提示：为了获得最佳性能，`faster-whisper` 可根据您的硬件配置额外的 CUDA 环境)*

### 3. 配置环境变量

系统完全由环境变量驱动。在启动前，请配置：

```bash
# 必填: 你的 LLM API 密钥 (默认适配 DeepSeek API)
export LLM_API_KEY="sk-你的密钥"

# 可选: 切换供应商模型 (如切换到 OpenAI)
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"

# 可选: 指定拾音设备名称 
# Mac 用户如果使用 BlackHole 捕获系统开会声音：
export AUDIO_DEVICE="BlackHole" 

# 可选: Server 端口设置
export SERVER_PORT="8765"
export SERVER_HOST="0.0.0.0"
```

### 4. 运行系统

启动主编排引擎：

```bash
python -m src.main
```

1. 保持您的**手机**与电脑连接在同一个 Wi-Fi 局域网下。
2. 用手机浏览器访问控制台提示的地址，例如：`http://192.168.1.100:8765`
3. 手机屏幕将呈现极简的深色待机界面。开始会议，等待对方提问，答案将自动显现到您的屏幕上！

## 🧪 单元测试

核心总线调度及意图路由节点包含完善的 pytest 单元测试：

```bash
python -m pytest tests/ -v
```

---

> [!TIP]
> **提示**：本项目仅限个人技术交流与辅助学习使用，请遵守所在公司的相关保密协议与合规政策。
