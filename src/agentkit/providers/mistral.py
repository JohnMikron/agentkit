"""
Mistral AI provider for AgentKit.

Supports Mistral and Codestral models.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx

from agentkit.core.exceptions import MissingAPIKeyError, ProviderAuthenticationError, ProviderError, ProviderRateLimitError
from agentkit.core.types import (
    FinishReason,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)
from agentkit.providers.base import LLMProvider


class MistralProvider(LLMProvider):
    """
    Mistral AI API provider.

    Supports Mistral Large, Small, and Codestral models.
    Uses OpenAI-compatible API format.

    Example:
        provider = MistralProvider(
            model="mistral-large-latest",
            api_key="..."
        )
        response = await provider.acomplete([Message.user("Hello!")])
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key, None, temperature, max_tokens, **kwargs)

        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1"
        self.timeout = timeout

        if not self.api_key:
            raise MissingAPIKeyError("mistral")

        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request body (OpenAI-compatible format)."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api_format() for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if self.max_tokens or "max_tokens" in kwargs:
            body["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        if "top_p" in kwargs:
            body["top_p"] = kwargs["top_p"]

        if tools:
            body["tools"] = [t.to_api_format() for t in tools]
            body["tool_choice"] = kwargs.get("tool_choice", "auto")

        return body

    def _parse_response(self, response_data: Dict[str, Any]) -> LLMResponse:
        """Parse API response."""
        choices = response_data.get("choices", [])
        if not choices:
            return LLMResponse(content="", finish_reason=FinishReason.ERROR)

        choice = choices[0]
        message = choice.get("message", {})

        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", "{}"),
                )
                for tc in message["tool_calls"]
            ]

        usage_data = response_data.get("usage", {})
        usage = self._create_usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return LLMResponse(
            id=response_data.get("id", ""),
            content=message.get("content", "") or "",
            tool_calls=tool_calls,
            finish_reason=self._parse_finish_reason(choice.get("finish_reason")),
            usage=usage,
            model=response_data.get("model", self.model),
            raw_response=response_data,
        )

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion synchronously."""
        start_time = time.perf_counter()

        body = self._build_request_body(messages, tools, **kwargs)

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )

        if response.status_code == 401:
            raise ProviderAuthenticationError("mistral", "Invalid API key")
        elif response.status_code == 429:
            raise ProviderRateLimitError("mistral")
        elif response.status_code != 200:
            raise ProviderError(
                f"Mistral API error: {response.text}",
                provider="mistral",
                details={"status_code": response.status_code},
            )

        result = self._parse_response(response.json())
        result.latency_ms = self._measure_latency(start_time)
        return result

    async def acomplete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion asynchronously."""
        start_time = time.perf_counter()

        body = self._build_request_body(messages, tools, **kwargs)

        response = await self._async_client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )

        if response.status_code == 401:
            raise ProviderAuthenticationError("mistral", "Invalid API key")
        elif response.status_code == 429:
            raise ProviderRateLimitError("mistral")
        elif response.status_code != 200:
            raise ProviderError(
                f"Mistral API error: {response.text}",
                provider="mistral",
                details={"status_code": response.status_code},
            )

        result = self._parse_response(response.json())
        result.latency_ms = self._measure_latency(start_time)
        return result

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream completion synchronously."""
        body = self._build_request_body(messages, tools, **kwargs)
        body["stream"] = True

        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if content := delta.get("content"):
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def astream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion asynchronously."""
        body = self._build_request_body(messages, tools, **kwargs)
        body["stream"] = True

        async with self._async_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if content := delta.get("content"):
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def __del__(self) -> None:
        """Clean up HTTP clients."""
        self._client.close()
