"""
OpenAI provider for AgentKit.

Supports GPT-4, GPT-3.5, o1 models, and any OpenAI-compatible API.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

import httpx

from agentkit.core.exceptions import (
    MissingAPIKeyError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from agentkit.core.types import (
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)
from agentkit.providers.base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    Supports GPT-4, GPT-3.5-turbo, o1, and OpenAI-compatible APIs.
    Can be configured for Azure OpenAI or custom endpoints.

    Example:
        provider = OpenAIProvider(
            model="gpt-4o",
            api_key="sk-..."
        )
        response = await provider.acomplete([Message.user("Hello!")])
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key, base_url, temperature, max_tokens, **kwargs)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.organization = organization or os.environ.get("OPENAI_ORG_ID")
        self.timeout = timeout

        if not self.api_key:
            raise MissingAPIKeyError("openai")

        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _build_request_body(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request body for OpenAI API."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_to_api_format(m) for m in messages],
        }

        # Temperature (not supported by o1 models)
        if not self.model.startswith("o1") and not self.model.startswith("o3"):
            body["temperature"] = kwargs.get("temperature", self.temperature)

        # Max tokens
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens:
            if self.model.startswith("o1") or self.model.startswith("o3"):
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens

        # Top p
        if "top_p" in kwargs:
            body["top_p"] = kwargs["top_p"]

        # Stop sequences
        if "stop" in kwargs:
            body["stop"] = kwargs["stop"]

        # Tools
        if tools:
            body["tools"] = [t.to_api_format() for t in tools]
            body["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Response format
        if "response_format" in kwargs:
            body["response_format"] = kwargs["response_format"]

        # Extra parameters
        for key in ["presence_penalty", "frequency_penalty", "seed", "logprobs", "top_logprobs"]:
            if key in kwargs:
                body[key] = kwargs[key]

        return body

    def _message_to_api_format(self, message: Message) -> dict[str, Any]:
        """Convert Message to OpenAI API format."""
        result: dict[str, Any] = {"role": message.role.value}

        if message.content:
            result["content"] = message.content

        if message.name:
            result["name"] = message.name

        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id

        if message.tool_calls:
            result["tool_calls"] = [tc.to_api_format() for tc in message.tool_calls]

        return result

    def _parse_response(self, response_data: dict[str, Any]) -> LLMResponse:
        """Parse OpenAI API response."""
        choices = response_data.get("choices", [])
        if not choices:
            raise ProviderResponseError("openai", "No choices in response")

        choice = choices[0]
        message = choice.get("message", {})

        # Parse tool calls
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

        # Parse usage
        usage_data = response_data.get("usage", {})
        usage = self._create_usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            cached_tokens=usage_data.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        )

        # Parse finish reason
        finish_reason = self._parse_finish_reason(choice.get("finish_reason"))

        return LLMResponse(
            id=response_data.get("id", ""),
            content=message.get("content", "") or "",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=response_data.get("model", self.model),
            raw_response=response_data,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
            error_type = error_data.get("error", {}).get("type", "")
        except Exception:
            error_msg = response.text
            error_type = ""

        if response.status_code == 401:
            raise ProviderAuthenticationError("openai", error_msg)
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimitError(
                "openai",
                retry_after=int(retry_after) if retry_after else None,
            )
        else:
            raise ProviderError(
                f"OpenAI API error: {error_msg}",
                provider="openai",
                details={"status_code": response.status_code, "type": error_type},
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
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )

        if response.status_code != 200:
            self._handle_error(response)

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
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )

        if response.status_code != 200:
            self._handle_error(response)

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
        body["stream"] = True

        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        ) as response:
            if response.status_code != 200:
                self._handle_error(response)

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
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
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
            if response.status_code != 200:
                self._handle_error(response)

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
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass

        if hasattr(self, "_async_client"):
            try:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        loop.create_task(self._async_client.aclose())
                except RuntimeError:
                    # No running loop, try to use run_until_complete if possible
                    # or just rely on gc if we can't get a loop
                    pass
            except Exception:
                pass
