"""
Google AI provider for AgentKit.

Supports Gemini models.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

import httpx

from agentkit.core.exceptions import (
    MissingAPIKeyError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
)
from agentkit.core.types import (
    FinishReason,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
)
from agentkit.providers.base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class GoogleProvider(LLMProvider):
    """
    Google AI API provider.

    Supports Gemini 2.0 and later models.

    Example:
        provider = GoogleProvider(
            model="gemini-2.0-flash",
            api_key="..."
        )
        response = await provider.acomplete([Message.user("Hello!")])
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key, None, temperature, max_tokens, **kwargs)

        self.api_key = (
            api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )
        self.timeout = timeout

        if not self.api_key:
            raise MissingAPIKeyError("google")

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _convert_messages(self, messages: list[Message]) -> dict[str, Any]:
        """Convert messages to Gemini format."""
        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role == Role.USER:
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": msg.content}],
                    }
                )
            elif msg.role == Role.ASSISTANT:
                parts: list[dict[str, Any]] = [{"text": msg.content}] if msg.content else []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        parts.append(
                            {
                                "functionCall": {
                                    "name": tc.name,
                                    "args": args,
                                }
                            }
                        )
                contents.append({"role": "model", "parts": parts})
            elif msg.role == Role.TOOL:
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.name,
                                    "response": {"result": msg.content},
                                }
                            }
                        ],
                    }
                )

        result: dict[str, Any] = {"contents": contents}
        if system_instruction:
            result["systemInstruction"] = system_instruction
        return result

    def _convert_tools(self, tools: list[ToolDefinition]) -> dict[str, Any]:
        """Convert tools to Gemini format."""
        return {
            "functionDeclarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
        }

    def _build_request_body(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request body for Gemini API."""
        body = self._convert_messages(messages)

        generation_config: dict[str, Any] = {}
        if "temperature" in kwargs or self.temperature is not None:
            generation_config["temperature"] = kwargs.get("temperature", self.temperature)
        if self.max_tokens or "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs.get("max_tokens", self.max_tokens)
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs["top_p"]
        if "stop" in kwargs:
            generation_config["stopSequences"] = kwargs["stop"]

        if generation_config:
            body["generationConfig"] = generation_config

        if tools:
            body["tools"] = [self._convert_tools(tools)]

        return body

    def _parse_response(self, response_data: dict[str, Any]) -> LLMResponse:
        """Parse Gemini API response."""
        candidates = response_data.get("candidates", [])
        if not candidates:
            return LLMResponse(content="", finish_reason=FinishReason.ERROR)

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])

        text_content = ""
        tool_calls = None

        for part in content_parts:
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                if tool_calls is None:
                    tool_calls = []
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        name=fc.get("name", ""),
                        arguments=json.dumps(fc.get("args", {})),
                    )
                )

        usage_data = response_data.get("usageMetadata", {})
        usage = self._create_usage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            cached_tokens=usage_data.get("cachedContentTokenCount", 0),
        )

        finish_reason_map = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER,
            "FUNCTION_CALL": FinishReason.TOOL_CALLS,
        }
        finish_reason = finish_reason_map.get(
            candidate.get("finishReason", "STOP"), FinishReason.STOP
        )

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
            raw_response=response_data,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion synchronously."""
        start_time = time.perf_counter()

        body = self._build_request_body(messages, tools, **kwargs)

        response = self._client.post(
            f"{self.base_url}/{self.model}:generateContent",
            params={"key": self.api_key},
            json=body,
        )

        if response.status_code == 401:
            raise ProviderAuthenticationError("google", "Invalid API key")
        elif response.status_code == 429:
            raise ProviderRateLimitError("google")
        elif response.status_code != 200:
            raise ProviderError(
                f"Google AI API error: {response.text}",
                provider="google",
                details={"status_code": response.status_code},
            )

        result = self._parse_response(response.json())
        result.latency_ms = self._measure_latency(start_time)
        return result

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion asynchronously."""
        start_time = time.perf_counter()

        body = self._build_request_body(messages, tools, **kwargs)

        response = await self._async_client.post(
            f"{self.base_url}/{self.model}:generateContent",
            params={"key": self.api_key},
            json=body,
        )

        if response.status_code == 401:
            raise ProviderAuthenticationError("google", "Invalid API key")
        elif response.status_code == 429:
            raise ProviderRateLimitError("google")
        elif response.status_code != 200:
            raise ProviderError(
                f"Google AI API error: {response.text}",
                provider="google",
                details={"status_code": response.status_code},
            )

        result = self._parse_response(response.json())
        result.latency_ms = self._measure_latency(start_time)
        return result

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream completion synchronously."""
        body = self._build_request_body(messages, tools, **kwargs)

        with self._client.stream(
            "POST",
            f"{self.base_url}/{self.model}:streamGenerateContent",
            params={"key": self.api_key},
            json=body,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "text" in part:
                                yield part["text"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def astream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion asynchronously."""
        body = self._build_request_body(messages, tools, **kwargs)

        async with self._async_client.stream(
            "POST",
            f"{self.base_url}/{self.model}:streamGenerateContent",
            params={"key": self.api_key},
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "text" in part:
                                yield part["text"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def __del__(self) -> None:
        """Clean up HTTP clients."""
        if hasattr(self, "_client"):
            self._client.close()
