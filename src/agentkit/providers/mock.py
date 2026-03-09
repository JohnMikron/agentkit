"""
Mock provider for AgentKit (for testing and out-of-the-box experience).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from agentkit.core.types import FinishReason, LLMResponse, Message, Usage
from agentkit.providers.base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing and demos.

    Does not require API keys. Returns predefined responses or echoes the input.
    """

    def __init__(
        self,
        model: str = "mock-model",
        responses: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key="mock", **kwargs)
        self.responses = responses or []
        self._response_idx = 0

    def complete(
        self,
        messages: list[Message],
        tools: Any | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock completion."""
        time.sleep(0.1)  # Simulate some latency

        if self._response_idx < len(self.responses):
            content = self.responses[self._response_idx]
            self._response_idx += 1
        else:
            # Echo logic if no more responses
            last_msg = messages[-1].content if messages else ""
            content = f"Mock response to: {last_msg[:50]}..." if last_msg else "Hello, I am a mock agent!"

        return LLMResponse(
            id=f"mock-{time.time()}",
            content=content,
            model=self.model,
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason=FinishReason.STOP,
        )

    async def acomplete(
        self,
        messages: list[Message],
        tools: Any | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock completion asynchronously."""
        return self.complete(messages, tools, **kwargs)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream mock response."""
        resp = self.complete(messages, **kwargs)
        for word in resp.content.split():
            yield word + " "
            time.sleep(0.05)

    async def astream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        """Stream mock response asynchronously."""
        resp = await self.acomplete(messages, **kwargs)
        for word in resp.content.split():
            yield word + " "
            await asyncio.sleep(0.05)
