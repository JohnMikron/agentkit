"""
Core types and data structures for AgentKit.

This module defines the fundamental types used throughout the framework,
including messages, tool definitions, and configuration structures.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Type variables for generics
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class FinishReason(str, Enum):
    """Reason for completion of LLM response."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class Message(BaseModel):
    """
    A message in the conversation.

    Represents a single message exchanged between user, assistant,
    system, or tools in a conversation.

    Attributes:
        id: Unique identifier for the message
        role: The role of the message sender
        content: The text content of the message
        name: Optional name for tool/function messages
        tool_call_id: Optional ID for tool call responses
        tool_calls: Optional list of tool calls made by the assistant
        metadata: Additional metadata attached to the message
        created_at: Timestamp when the message was created
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_api_format(self) -> dict[str, Any]:
        """Convert to API-compatible dictionary format."""
        result: dict[str, Any] = {"role": self.role.value}

        if self.content:
            result["content"] = self.content

        if self.name:
            result["name"] = self.name

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            result["tool_calls"] = [tc.to_api_format() for tc in self.tool_calls]

        return result

    @classmethod
    def system(cls, content: str, **kwargs: Any) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs: Any) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content, **kwargs)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        **kwargs: Any,
    ) -> Message:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls, **kwargs)

    @classmethod
    def tool_result(
        cls,
        content: str,
        tool_call_id: str,
        name: str,
        **kwargs: Any,
    ) -> Message:
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            **kwargs,
        )


class ToolCall(BaseModel):
    """
    A tool call request from the LLM.

    Attributes:
        id: Unique identifier for the tool call
        name: Name of the tool to call
        arguments: Arguments to pass to the tool (JSON string)
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    name: str
    arguments: str = "{}"

    def to_api_format(self) -> dict[str, Any]:
        """Convert to API-compatible format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    def parse_arguments(self) -> dict[str, Any]:
        """Parse arguments from JSON string."""
        import json

        try:
            from typing import cast

            return cast("dict[str, Any]", json.loads(self.arguments))
        except json.JSONDecodeError:
            return {}


class ToolDefinition(BaseModel):
    """
    Definition of a tool that can be called by the LLM.

    Attributes:
        name: The name of the tool
        description: What the tool does
        parameters: JSON Schema of the tool's parameters
        strict: Whether to use strict mode for parameter validation
    """

    model_config = ConfigDict(frozen=False)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool = False

    def to_api_format(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible tool format."""
        result: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        if self.strict:
            result["function"]["strict"] = True
        return result


class ToolResult(BaseModel):
    """
    Result of a tool execution.

    Attributes:
        tool_call_id: ID of the tool call this is a response to
        name: Name of the tool that was executed
        content: The result content
        is_error: Whether the execution resulted in an error
        execution_time_ms: Time taken to execute in milliseconds
    """

    model_config = ConfigDict(frozen=False)

    tool_call_id: str
    name: str
    content: str
    raw_result: Any = Field(default=None)
    is_error: bool = False
    execution_time_ms: float | None = None

    def to_message(self) -> Message:
        """Convert to a Message object."""
        return Message.tool_result(
            content=self.content,
            tool_call_id=self.tool_call_id,
            name=self.name,
            metadata={"is_error": self.is_error, "execution_time_ms": self.execution_time_ms},
        )


class Usage(BaseModel):
    """Token usage statistics."""

    model_config = ConfigDict(frozen=False)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Add two Usage objects together.

        The resulting total_tokens is recalculated from the summed
        prompt/completion counts instead of simply adding the existing
        `total_tokens` fields (which may be stale or zero).
        """
        prompt = self.prompt_tokens + other.prompt_tokens
        completion = self.completion_tokens + other.completion_tokens
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class LLMResponse(BaseModel):
    """
    Response from an LLM provider.

    Attributes:
        id: Unique identifier for the response
        content: The text content of the response
        tool_calls: Optional list of tool calls requested by the model
        finish_reason: Why the model stopped generating
        usage: Token usage statistics
        model: The model used for generation
        latency_ms: Response latency in milliseconds
        raw_response: The raw response from the provider for debugging
    """

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    finish_reason: FinishReason | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    latency_ms: float | None = None
    raw_response: Any | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)

    @property
    def is_empty(self) -> bool:
        """Check if response is empty."""
        return not self.content and not self.tool_calls


class AgentState(str, Enum):
    """State of an agent."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    EXECUTING_TOOLS = "executing_tools"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentResult(BaseModel):
    """
    Result of an agent execution.

    Attributes:
        success: Whether the execution was successful
        content: The final content/response
        tool_calls: All tool calls made during execution
        tool_results: All tool results from execution
        messages: Complete message history
        usage: Total token usage
        iterations: Number of iterations performed
        state: Final state of the agent
        data: Structured data returned from the execution (e.g. Pydantic model)
        error: Error message if failed
        latency_ms: Total execution time in milliseconds
        metadata: Additional metadata
    """

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    success: bool = True
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    iterations: int = 0
    state: AgentState = AgentState.COMPLETED
    data: Any | None = None
    error: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Check if execution failed."""
        return not self.success


# Event types for hooks and observability
class EventType(str, Enum):
    """Types of events in agent execution."""

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"
    MESSAGE_ADDED = "message_added"
    STATE_CHANGE = "state_change"
    AGENT_THOUGHT = "agent_thought"


@dataclass
class Event:
    """
    An event emitted during agent execution.

    Attributes:
        type: The type of event
        timestamp: When the event occurred
        agent_name: Name of the agent
        data: Event-specific data
        metadata: Additional metadata
    """

    type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_name: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# Hook function types
HookFunction = Callable[[Event], None]
AsyncHookFunction = Callable[[Event], Any]
ToolFunction = Callable[..., Any]
AsyncToolFunction = Callable[..., Any]


# Model identifiers for type-safe model selection
class ModelId(str, Enum):
    """Supported model identifiers."""

    # OpenAI
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O3_MINI = "o3-mini"

    # Anthropic
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-latest"

    # Google
    GEMINI_2_PRO = "gemini-2.5-pro"
    GEMINI_2_FLASH = "gemini-2.5-flash"

    # Mistral
    MISTRAL_LARGE = "mistral-large-latest"
    MISTRAL_SMALL = "mistral-small-latest"
    CODESTRAL = "codestral-latest"

    # Local (via Ollama)
    LLAMA_3_2 = "llama3.2"
    LLAMA_3_1 = "llama3.1"
    MISTRAL_7B = "mistral"
    PHI_3 = "phi3"
    QWEN_2_5 = "qwen2.5"
