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
