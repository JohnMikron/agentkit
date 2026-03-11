"""
Anthropic provider for AgentKit.

Supports Claude 3 and Claude 3.5 models.
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
)
from agentkit.core.types import (
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
)
from agentkit.providers.base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider.

    Example:
        provider = AnthropicProvider(
            model="claude-3-5-sonnet-latest",
            api_key="sk-ant-..."
        )
        response = await provider.acomplete([Message.user("Hello!")])
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key, base_url, temperature, max_tokens, **kwargs)

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url or "https://api.anthropic.com/v1"
        self.timeout = timeout

        if not self.api_key:
            raise MissingAPIKeyError("anthropic")

        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Anthropic uses a separate system prompt and messages list.
        """
        system_prompt = ""
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            elif msg.role == Role.USER:
                converted.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": args,
                            }
                        )
                converted.append({"role": "assistant", "content": content_blocks})
            elif msg.role == Role.TOOL:
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        return system_prompt, converted

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _build_request_body(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request body for Anthropic API."""
        system_prompt, converted_messages = self._convert_messages(messages)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens or 4096),
        }

        if system_prompt:
            body["system"] = system_prompt

        body["temperature"] = kwargs.get("temperature", self.temperature)

        if tools:
            body["tools"] = self._convert_tools(tools)

        if "top_p" in kwargs:
            body["top_p"] = kwargs["top_p"]

        if "stop" in kwargs:
            body["stop_sequences"] = kwargs["stop"]

        return body

    def _parse_response(self, response_data: dict[str, Any]) -> LLMResponse:
        """Parse Anthropic API response."""
        content_blocks = response_data.get("content", [])
        text_content = ""
        tool_calls = None

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input", {})),
                    )
                )

        usage_data = response_data.get("usage", {})
        usage = self._create_usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
        )

        finish_reason = self._parse_finish_reason(response_data.get("stop_reason"))

        return LLMResponse(
            id=response_data.get("id", ""),
            content=text_content,
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
            raise ProviderAuthenticationError("anthropic", error_msg)
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimitError(
                "anthropic",
                retry_after=int(retry_after) if retry_after else None,
            )
        else:
            raise ProviderError(
                f"Anthropic API error: {error_msg}",
                provider="anthropic",
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
            f"{self.base_url}/messages",
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
            f"{self.base_url}/messages",
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
            f"{self.base_url}/messages",
            headers=self._headers(),
            json=body,
        ) as response:
            if response.status_code != 200:
                self._handle_error(response)

            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta" and (
                            text := data.get("delta", {}).get("text")
                        ):
                            yield text
                    except (json.JSONDecodeError, KeyError):
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
            f"{self.base_url}/messages",
            headers=self._headers(),
            json=body,
        ) as response:
            if response.status_code != 200:
                self._handle_error(response)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta" and (
                            text := data.get("delta", {}).get("text")
                        ):
                            yield text
                    except (json.JSONDecodeError, KeyError):
                        continue

    def __del__(self) -> None:
        """Clean up HTTP clients."""
        if hasattr(self, "_client"):
            self._client.close()
