"""
Ollama provider for AgentKit.

Supports local models via Ollama.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

import httpx

from agentkit.core.exceptions import ProviderError
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


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local models.

    Supports any model available through Ollama including:
    - Llama 3.x
    - Mistral
    - Phi-3
    - Qwen 2.5
    - And many more

    Example:
        provider = OllamaProvider(model="llama3.2")
        response = await provider.acomplete([Message.user("Hello!")])

    Prerequisites:
        - Install Ollama: https://ollama.ai
        - Pull a model: ollama pull llama3.2
        - Run Ollama: ollama serve
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, None, base_url, temperature, max_tokens, **kwargs)

        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to Ollama format."""
        result = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                result.append({"role": "system", "content": msg.content})
            elif msg.role == Role.USER:
                result.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                ollama_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    ollama_msg["tool_calls"] = [
                        {
                            "function": {
                                "name": tc.name,
                                "arguments": json.loads(tc.arguments) if tc.arguments else {},
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(ollama_msg)
            elif msg.role == Role.TOOL:
                result.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                    }
                )

        return result

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tools to Ollama format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _build_request_body(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request body for Ollama API."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }

        if self.max_tokens or "max_tokens" in kwargs:
            body["options"]["num_predict"] = kwargs.get("max_tokens", self.max_tokens)

        if "top_p" in kwargs:
            body["options"]["top_p"] = kwargs["top_p"]

        if "stop" in kwargs:
            body["options"]["stop"] = kwargs["stop"]

        if "seed" in kwargs:
            body["options"]["seed"] = kwargs["seed"]

        if tools:
            body["tools"] = self._convert_tools(tools)

        return body

    def _parse_response(self, response_data: dict[str, Any]) -> LLMResponse:
        """Parse Ollama API response."""
        message = response_data.get("message", {})

        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=f"call_{i}_{tc.get('function', {}).get('name', '')}",
                    name=tc.get("function", {}).get("name", ""),
                    arguments=json.dumps(tc.get("function", {}).get("arguments", {})),
                )
                for i, tc in enumerate(message["tool_calls"])
            ]

        usage = self._create_usage(
            prompt_tokens=response_data.get("prompt_eval_count", 0),
            completion_tokens=response_data.get("eval_count", 0),
        )

        done_reason = response_data.get("done_reason", "stop")
        finish_reason_map = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
        }

        return LLMResponse(
            content=message.get("content", ""),
            tool_calls=tool_calls,
            finish_reason=finish_reason_map.get(done_reason, FinishReason.STOP),
            usage=usage,
            model=response_data.get("model", self.model),
            raw_response=response_data,
        )

    def _check_connection(self) -> None:
        """Check if Ollama is running."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ProviderError(
                    "Ollama is not responding. Make sure Ollama is running (ollama serve)",
                    provider="ollama",
                )
        except httpx.ConnectError as err:
            raise ProviderError(
                "Cannot connect to Ollama. Make sure Ollama is running:\n"
                "  1. Install: https://ollama.ai\n"
                "  2. Pull model: ollama pull llama3.2\n"
                "  3. Run: ollama serve",
                provider="ollama",
            ) from err

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion synchronously."""
        start_time = time.perf_counter()

        body = self._build_request_body(messages, tools, **kwargs)

        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=body,
            )

            if response.status_code != 200:
                raise ProviderError(
                    f"Ollama error: {response.text}",
                    provider="ollama",
                    details={"status_code": response.status_code},
                )

        except httpx.ConnectError as err:
            raise ProviderError(
                "Cannot connect to Ollama. Make sure Ollama is running:\n"
                "  1. Install: https://ollama.ai\n"
                "  2. Pull model: ollama pull llama3.2\n"
                "  3. Run: ollama serve",
                provider="ollama",
            ) from err

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

        try:
            response = await self._async_client.post(
                f"{self.base_url}/api/chat",
                json=body,
            )

            if response.status_code != 200:
                raise ProviderError(
                    f"Ollama error: {response.text}",
                    provider="ollama",
                    details={"status_code": response.status_code},
                )

        except httpx.ConnectError as err:
            raise ProviderError(
                "Cannot connect to Ollama. Make sure Ollama is running:\n"
                "  1. Install: https://ollama.ai\n"
                "  2. Pull model: ollama pull llama3.2\n"
                "  3. Run: ollama serve",
                provider="ollama",
            ) from err

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

        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=body,
            ) as response:
                for line in response.iter_lines():
                    try:
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content
                    except json.JSONDecodeError:
                        continue

        except httpx.ConnectError as err:
            raise ProviderError(
                "Cannot connect to Ollama. Make sure Ollama is running.",
                provider="ollama",
            ) from err

    async def astream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion asynchronously."""
        body = self._build_request_body(messages, tools, **kwargs)
        body["stream"] = True

        try:
            async with self._async_client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=body,
            ) as response:
                async for line in response.aiter_lines():
                    try:
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content
                    except json.JSONDecodeError:
                        continue

        except httpx.ConnectError as err:
            raise ProviderError(
                "Cannot connect to Ollama. Make sure Ollama is running.",
                provider="ollama",
            ) from err

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def __del__(self) -> None:
        """Clean up HTTP clients."""
        self._client.close()
