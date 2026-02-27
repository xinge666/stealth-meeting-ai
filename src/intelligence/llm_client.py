"""
LLM Client — async streaming client for OpenAI-compatible APIs.
Supports DeepSeek, OpenAI, and compatible endpoints.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

import httpx

from ..config import LLMConfig
from ..event_bus import EventBus, llm_chunk_event

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async streaming LLM client using httpx + SSE.
    Publishes response chunks to the event bus.
    """

    def __init__(self, config: LLMConfig, event_bus: EventBus):
        self.config = config
        self.bus = event_bus
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """Create the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.config.timeout, connect=10.0),
        )
        logger.info("LLM client initialized (model=%s)", self.config.model)

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()

    async def ask(self, prompt: str, question: str = "") -> str:
        """
        Send a prompt to the LLM and stream the response.
        Publishes LLM_RESPONSE_CHUNK events as chunks arrive.
        Returns the full completed response text.
        """
        if not self._client:
            raise RuntimeError("LLM client not initialized")
        if not self.config.api_key:
            logger.warning("No LLM API key configured — returning placeholder")
            placeholder = f"[LLM未配置] 收到提问: {question}"
            await self.bus.publish(llm_chunk_event(placeholder, is_done=True))
            return placeholder

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }

        full_response = []
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        chunk = delta.get("content", "")
                        if chunk:
                            full_response.append(chunk)
                            await self.bus.publish(llm_chunk_event(chunk))
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.debug("SSE parse skip: %s", e)
                        continue

        except httpx.HTTPStatusError as e:
            error_msg = f"[LLM Error] HTTP {e.response.status_code}"
            logger.error(error_msg)
            await self.bus.publish(llm_chunk_event(error_msg, is_done=True))
            return error_msg
        except Exception as e:
            error_msg = f"[LLM Error] {type(e).__name__}: {e}"
            logger.error(error_msg)
            await self.bus.publish(llm_chunk_event(error_msg, is_done=True))
            return error_msg

        # Publish done event
        complete_text = "".join(full_response)
        await self.bus.publish(llm_chunk_event("", is_done=True))

        logger.info("LLM response completed (%d chars)", len(complete_text))
        return complete_text

    async def analyze_intent(self, text: str, history: str = "") -> dict:
        """
        Analyze the intent of ASR text using a non-streaming LLM call.
        Returns a dict with classification and extraction results.
        Incorporates conversation history for context-aware extraction.
        """
        if not self._client or not self.config.api_key:
            return {"is_question": False, "extracted_question": "", "confidence": 0.0}

        history_block = f"\n[最近对话上下文]:\n{history}\n" if history else ""

        prompt = f"""
你是一个面试助手。请分析以下经过语音识别（ASR）的文本片段：
---
"{text}"
---
{history_block}
任务要求：
1. 判断这是否是一个需要你（AI）作为面试官或专家进行回答的实质性技术问题或业务问题。
2. 结合【最近对话上下文】，如果当前文本包含指代（如“它”、“那个”、“刚才说的”），请在提取时将其还原为具体的主体。
3. 如果包含“废话、口头禅（呃、那个、嗯）、单纯的寒暄（你好、大家好）”但同时也包含问题，请将问题提取出来并进行语言清洗（变得书面化、清晰）。
4. 如果这只是单纯的噪音、闲聊、废话或不完整的句子，请判定为非问题。

请严格返回 JSON 格式，不要有任何其他文字：
{{
  "is_question": boolean, // 是否是实质性问题
  "extracted_question": string, // 结合上下文还原并清洗后的完整问题文本
  "confidence": number // 0.0 到 1.0 的置信度
}}
"""
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.1,
            "stream": False,
        }

        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Extract JSON from potential markdown blocks
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_str)
            return result
        except Exception as e:
            logger.error("Intent analysis LLM error: %s", e)
            return {"is_question": False, "extracted_question": "", "confidence": 0.0}
