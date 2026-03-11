"""
Base LLM provider interface.

This module defines the abstract interface that all LLM providers must implement.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agentkit.core.types import (
    FinishReason,
    LLMResponse,
    Message,
    ToolDefinition,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (OpenAI, Anthropic, Google, Mistral, Ollama) must implement
    this interface to work with AgentKit.

    The interface provides both sync and async methods for:
    - Complete: Generate a full response
    - Stream: Stream response token by token
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LLM provider.

        Args:
            model: Model identifier
            api_key: API key for the provider
            base_url: Custom base URL (for proxies or self-hosted)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion synchronously.

        Args:
            messages: Conversation history
            tools: Optional list of tools the model can call
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with the completion result
        """
        pass

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion asynchronously.

        Args:
            messages: Conversation history
            tools: Optional list of tools the model can call
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with the completion result
        """
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream the completion token by token.

        Args:
            messages: Conversation history
            tools: Optional list of tools the model can call
            **kwargs: Additional provider-specific parameters

        Yields:
            Text chunks from the completion
        """
        pass

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream the completion asynchronously.

        Args:
            messages: Conversation history
            tools: Optional list of tools the model can call
            **kwargs: Additional provider-specific parameters

        Yields:
            Text chunks from the completion
        """
        pass

    def _measure_latency(self, start_time: float) -> float:
        """Calculate latency in milliseconds."""
        return (time.perf_counter() - start_time) * 1000

    def _create_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> Usage:
        """Create a Usage object."""
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_tokens=cached_tokens,
        )

    def _parse_finish_reason(self, reason: str | None) -> FinishReason | None:
        """Parse finish reason string to enum."""
        if not reason:
            return None

        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "tool_use": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
            "error": FinishReason.ERROR,
        }
        return mapping.get(reason.lower())
