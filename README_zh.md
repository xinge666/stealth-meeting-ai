# 智能会议问答辅助系统 (AI Meeting Intelligence Assistant)

这是一个强大的端侧 AI 会议/面试辅助工具。它能够实时监控电脑的音频输出（或麦克风）以及桌面屏幕变化，智能提取会议、面试中的“提问”意图，并依托大语言模型（如 DeepSeek、GPT-4o）实时生成辅助回答。

它的主要特点是**绝对的隐蔽性**：程序在宿主机后台静默运行，生成的答案通过局域网 WebSocket 推送到您的手机浏览器上。电脑主屏幕保持纯净，无惧任何屏幕共享或各类监控防作弊软件的检测。

**核心特性：**

* **实时多模态感知**：同时处理系统级音频流与屏幕视觉变化（依靠 OpenCV 进行无损 Diff 侦测，仅在画面变动时触发 OCR）。
* **硬件级隐蔽架构**：舍弃了传统的 PyQt/Tkinter 等基于句柄的弹窗，采用手机端作为物理隔离的“副屏”进行 Typewriter 流式展示。
* **智能意图防骚扰**：内置前置意图路由器（Intent Router），精准过滤 80% 的寒暄、噪音与陈述句，仅在对方真正抛出技术问题时，才触发高昂成本的云端大模型。
* **极速并发设计**：基于 `asyncio` 的纯异步事件总线驱动，各个管道独立运行，互不阻塞，在 3-5 秒内完成从听到写的过程响应。

---

## 🏗️ 架构与技术栈

* **核心框架:** Python 3.10+, 基于 `asyncio.Queue` 的事件母线
* **听觉管道:** `sounddevice`, Silero VAD (端点检测), `faster-whisper` (流式语音识别)
* **视觉管道:** `mss` (极速截屏), OpenCV (帧高比差分), `rapidocr-onnxruntime` (本地 OCR 模型)
* **大脑中枢:** `httpx` (异步并发 LLM 客户端), 支持全兼容 OpenAI 格式 API 请求
* **展示前端:** FastAPI, WebSockets, 原生 HTML/JS (AMOLED 暗黑沉浸式 UI)

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
