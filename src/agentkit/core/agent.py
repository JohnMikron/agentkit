"""
Core Agent class for AgentKit.

This module provides the main Agent class that orchestrates LLM interactions,
tool execution, memory management, and observability in a simple, powerful API.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.config import Settings, get_settings
from agentkit.core.exceptions import (
    AgentCancelledError,
    AgentError,
    AgentMaxIterationsError,
    MissingAPIKeyError,
    ToolError,
)
from agentkit.core.memory import FileStorage, InMemoryStorage, Memory
from agentkit.core.tools import Tool, ToolRegistry, get_builtin_tools
from agentkit.core.types import (
    AgentResult,
    AgentState,
    Event,
    EventType,
    Message,
    ToolCall,
    ToolResult,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from agentkit.providers.base import LLMProvider

logger = structlog.get_logger()


class AgentHooks:
    """
    Container for agent event hooks.

    Each hook is called at a specific point in the agent's execution cycle.
    """

    def __init__(self) -> None:
        self._on_start: list[Callable[[Event], Any]] = []
        self._on_end: list[Callable[[Event], Any]] = []
        self._on_error: list[Callable[[Event], Any]] = []
        self._on_llm_request: list[Callable[[Event], Any]] = []
        self._on_llm_response: list[Callable[[Event], Any]] = []
        self._on_tool_call_start: list[Callable[[Event], Any]] = []
        self._on_tool_call_end: list[Callable[[Event], Any]] = []
        self._on_state_change: list[Callable[[Event], Any]] = []
        self._on_thought: list[Callable[[Event], Any]] = []
        self._on_message: list[Callable[[Event], Any]] = []

    def on_start(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for agent start."""
        self._on_start.append(func)
        return func

    def on_end(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for agent end."""
        self._on_end.append(func)
        return func

    def on_error(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for errors."""
        self._on_error.append(func)
        return func

    def on_llm_request(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for LLM requests."""
        self._on_llm_request.append(func)
        return func

    def on_llm_response(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for LLM responses."""
        self._on_llm_response.append(func)
        return func

    def on_tool_call_start(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for tool call start."""
        self._on_tool_call_start.append(func)
        return func

    def on_tool_call_end(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for tool call end."""
        self._on_tool_call_end.append(func)
        return func

    def on_state_change(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for state changes."""
        self._on_state_change.append(func)
        return func

    def on_thought(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for agent thoughts."""
        self._on_thought.append(func)
        return func

    def on_message(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for message additions."""
        self._on_message.append(func)
        return func

    async def emit(self, event: Event) -> None:
        """Emit an event to relevant hooks."""
        hook_map = {
            EventType.AGENT_START: self._on_start,
            EventType.AGENT_END: self._on_end,
            EventType.AGENT_ERROR: self._on_error,
            EventType.LLM_REQUEST: self._on_llm_request,
            EventType.LLM_RESPONSE: self._on_llm_response,
            EventType.TOOL_CALL_START: self._on_tool_call_start,
            EventType.TOOL_CALL_END: self._on_tool_call_end,
            EventType.STATE_CHANGE: self._on_state_change,
            EventType.MESSAGE_ADDED: self._on_message,
        }

        hooks = hook_map.get(event.type, [])
        for hook in hooks:
            try:
                result = hook(event)
                if asyncio.iscoroutine(result):
                    # Await coroutine to run synchronously with execution (Middleware pattern)
                    await result
            except Exception as e:
                logger.warning("Hook error", hook=hook.__name__, error=str(e))


class AgentConfig(BaseModel):
    """
    Configuration for an Agent.

    Attributes:
        name: Name of the agent
        model: Model identifier
        provider: Provider to use (auto-detected if not specified)
        system_prompt: System prompt for the agent
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        memory_enabled: Whether to enable memory
        memory_max_entries: Maximum entries in memory
        memory_file: Optional file path for persistent memory
        max_iterations: Maximum tool call iterations
        max_tool_calls: Maximum total tool calls
        timeout: Request timeout in seconds
        tools: List of tool names to include from built-ins
        hooks_enabled: Whether to enable event hooks
        streaming: Whether to enable streaming by default
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="agent", min_length=1, max_length=100)
    model: str = Field(default="gpt-5.3-chat-latest")
    provider: str = Field(default="openai")
    system_prompt: str | None = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_tokens_limit: int | None = Field(default=None, ge=1)
    memory_enabled: bool = Field(default=False)
    memory_max_entries: int | None = Field(default=100)
    memory_file: str | None = Field(default=None)
    max_iterations: int = Field(default=10, ge=1)
    max_tool_calls: int = Field(default=50, ge=1)
    timeout: float = Field(default=60.0, ge=1.0)
    tools: list[str] | None = Field(default=None)
    hooks_enabled: bool = Field(default=True)
    streaming: bool = Field(default=True)


class Agent:
    """
    A powerful AI agent with tool calling, memory, and observability.

    The Agent class is the main entry point for AgentKit. It provides
    a simple yet powerful API for building AI agents with:

    - Tool calling with automatic validation
    - Conversation memory (in-memory or persistent)
    - Event hooks for debugging and monitoring
    - Streaming support
    - Multiple LLM providers
    - Rate limiting and retries

    Example:
        agent = Agent("assistant", model="gpt-4o")

        @agent.tool
        def search(query: str) -> str:
            '''Search the web'''
            return "results..."

        result = await agent.run("Search for Python news")

    Features:
        - Simple @agent.tool decorator
        - Automatic memory management
        - Works with OpenAI, Anthropic, Google, Mistral, local models
        - Debugging hooks for monitoring
        - Streaming support
        - Both sync and async APIs
    """

    def __init__(
        self,
        name: str = "agent",
        model: str | None = None,
        memory: bool = False,
        system_prompt: str | None = None,
        config: AgentConfig | None = None,
        settings: Settings | None = None,
        approval_handler: Callable[[ToolCall], bool] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Create a new Agent.

        Args:
            name: Name of the agent
            model: Model to use (e.g., "gpt-4o", "claude-3-5-sonnet", "local:llama3")
            memory: Enable conversation memory
            system_prompt: System prompt for the agent
            config: Full configuration object (overrides other params)
            settings: Global settings object
            approval_handler: Optional callback for Human-in-the-Loop tool approval
            **kwargs: Additional configuration options
        """
        # Use provided config or create from parameters
        if config:
            self.config = config
        else:
            self.config = AgentConfig(
                name=name,
                model=model or kwargs.get("model", "gpt-4o-mini"),
                system_prompt=system_prompt or kwargs.get("system_prompt"),
                memory_enabled=memory or kwargs.get("memory_enabled", False),
                **{k: v for k, v in kwargs.items() if k in AgentConfig.model_fields},
            )

        self.settings = settings or get_settings()

        # Components
        self._tools = ToolRegistry()
        self._hooks = AgentHooks()
        self._provider = None
        self._state = AgentState.IDLE
        self._cancelled = False
        self._approval_handler = approval_handler

        # Initialize memory
        if self.config.memory_enabled:
            if self.config.memory_file:
                storage = FileStorage(
                    self.config.memory_file,
                    max_entries=self.config.memory_max_entries,
                )
            else:
                storage = InMemoryStorage(max_entries=self.config.memory_max_entries)
            self._memory = Memory(storage=storage, system_prompt=self.config.system_prompt)
        else:
            self._memory = None

        # Add built-in tools if specified
        if self.config.tools:
            builtin = get_builtin_tools(include=self.config.tools)
            for tool in builtin:
                self._tools.add(tool)

        # Token usage tracking
        self._total_usage = Usage()

        logger.info(
            "Agent initialized",
            name=self.name,
            model=self.config.model,
            memory=self.config.memory_enabled,
        )

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self.config.name

    @property
    def state(self) -> AgentState:
        """Get the current agent state."""
        return self._state

    @property
    def tools(self) -> list[Tool]:
        """Get all registered tools."""
        return self._tools.list_tools()

    @property
    def total_usage(self) -> Usage:
        """Get total token usage."""
        return self._total_usage

    @property
    def provider(self) -> LLMProvider:
        """Get or create the LLM provider."""
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider

    def _create_provider(self) -> LLMProvider:
        """Create the appropriate LLM provider based on explicit configuration."""
        import os

        from agentkit.providers.mock import MockProvider

        model_str = self.config.model
        provider_name = self.config.provider
        model_name = model_str

        # Parse from model string if prefix exists (e.g. "anthropic:claude-3")
        if ":" in model_str:
            prefix, rest = model_str.split(":", 1)
            provider_name = prefix
            model_name = rest

        if not provider_name:
            # This should technically not happen with the new default "openai"
            # but we keep it for safety if someone passes None explicitly.
            raise ValueError(
                f"No provider specified for model '{model_str}'. "
                "Please use 'provider:model' format or specify 'provider' in configuration."
            )

        provider_name = provider_name.lower()

        # Helper to decide if we should fallback to mock
        def get_fallback_provider(p_type: str, err: Exception) -> LLMProvider:
            if os.environ.get("AGENTKIT_DEMO_MODE") == "true":
                logger.warning(f"{p_type} API key missing, falling back to MockProvider")
                return MockProvider(model=f"mock-{model_name}")
            raise err

        try:
            if provider_name == "openai":
                from agentkit.providers.openai import OpenAIProvider
                return OpenAIProvider(model=model_name, api_key=self.settings.llm.openai_api_key, **self._provider_kwargs())

            if provider_name in ("anthropic", "claude"):
                from agentkit.providers.anthropic import AnthropicProvider
                return AnthropicProvider(model=model_name, api_key=self.settings.llm.anthropic_api_key, **self._provider_kwargs())

            if provider_name in ("google", "gemini"):
                from agentkit.providers.google import GoogleProvider
                return GoogleProvider(model=model_name, api_key=self.settings.llm.google_api_key, **self._provider_kwargs())

            if provider_name == "mistral":
                from agentkit.providers.mistral import MistralProvider
                return MistralProvider(model=model_name, api_key=self.settings.llm.mistral_api_key, **self._provider_kwargs())

            if provider_name in ("local", "ollama"):
                from agentkit.providers.ollama import OllamaProvider
                return OllamaProvider(model=model_name, **self._provider_kwargs())

            if provider_name == "mock":
                return MockProvider(model=model_name, **self._provider_kwargs())

        except MissingAPIKeyError as e:
            return get_fallback_provider(provider_name.capitalize(), e)

        raise ValueError(f"Unknown provider or unsupported model configuration: {provider_name}")

    def _provider_kwargs(self) -> dict[str, Any]:
        """Get standard provider kwargs."""
        return {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

    async def _aset_state(self, new_state: AgentState) -> None:
        """Update agent state and emit event asynchronously."""
        old_state = self._state
        self._state = new_state

        if self.config.hooks_enabled:
            event = Event(
                type=EventType.STATE_CHANGE,
                agent_name=self.name,
                data={"old_state": old_state.value, "new_state": new_state.value},
            )
            await self._hooks.emit(event)

    async def _aemit(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """Emit an event asynchronously."""
        if self.config.hooks_enabled:
            event = Event(
                type=event_type,
                agent_name=self.name,
                data=data or {},
            )
            await self._hooks.emit(event)

    # =========================================================================
    # Tool Management
    # =========================================================================

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
    ) -> Tool | Callable[[Callable[..., Any]], Tool]:
        """
        Decorator to add a tool to the agent.

        Can be used with or without arguments:
            @agent.tool
            def my_func(x: str) -> str:
                '''Does something'''
                return x

            @agent.tool(name="custom_name")
            def my_func(x: str) -> str:
                return x

        Args:
            func: The function to wrap
            name: Optional custom name
            description: Optional custom description
            strict: Whether to use strict validation

        Returns:
            Tool instance or decorator
        """

        def decorator(f: Callable[..., Any]) -> Tool:
            # Only pass description if we have something meaningful; otherwise
            # let Tool.__init__ infer it (avoids None validation issues).
            desc_val: str | None = None
            if description is not None:
                desc_val = description
            elif f.__doc__:
                desc_val = f.__doc__.strip().split("\n")[0]

            t = Tool(
                name=name or f.__name__,
                **({"description": desc_val} if desc_val is not None else {}),
                func=f,
                strict=strict,
            )
            self._tools.add(t)
            return t

        if func is not None:
            return decorator(func)
        return decorator

    def add_tool(self, tool: Tool) -> Agent:
        """
        Add an existing Tool instance.

        Returns self for chaining.
        """
        self._tools.add(tool)
        return self

    def add_tools(self, tools: list[Tool]) -> Agent:
        """
        Add multiple Tool instances.

        Returns self for chaining.
        """
        for tool in tools:
            self._tools.add(tool)
        return self

    # =========================================================================
    # Hook Registration
    # =========================================================================

    def on_start(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for agent start."""
        return self._hooks.on_start(func)

    def on_end(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for agent end."""
        return self._hooks.on_end(func)

    def on_error(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for errors."""
        return self._hooks.on_error(func)

    def on_llm_request(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for LLM requests."""
        return self._hooks.on_llm_request(func)

    def on_llm_response(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for LLM responses."""
        return self._hooks.on_llm_response(func)

    def on_tool_call(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for tool calls (start and end)."""
        self._hooks.on_tool_call_start(func)
        self._hooks.on_tool_call_end(func)
        return func

    def on_tool_call_start(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for tool call start."""
        return self._hooks.on_tool_call_start(func)

    def on_tool_call_end(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for tool call end."""
        return self._hooks.on_tool_call_end(func)

    def on_thought(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for agent thoughts."""
        return self._hooks.on_thought(func)

    def on_message(self, func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        """Register hook for message additions."""
        return self._hooks.on_message(func)

    # =========================================================================
    # Core Execution
    # =========================================================================

    def _build_messages(self, user_input: str) -> list[Message]:
        """Build message list for LLM."""
        messages = []

        # Add system prompt
        system = self.config.system_prompt or f"You are a helpful AI assistant named {self.name}."
        messages.append(Message.system(system))

        # Add memory history if enabled
        if self._memory is not None:
            history = self._memory.get_history(limit=self.config.max_iterations)
            messages.extend(history)

        # Add current input
        messages.append(Message.user(user_input))

        return messages


    async def _aexecute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls asynchronously."""
        results = []

        for tc in tool_calls:
            args = tc.parse_arguments()

            await self._aemit(
                EventType.TOOL_CALL_START,
                {"tool_name": tc.name, "arguments": args, "tool_call_id": tc.id},
            )

            if self._approval_handler and not self._approval_handler(tc):
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content="Error: Human approval denied",
                    is_error=True,
                ))
                await self._aemit(
                    EventType.TOOL_CALL_END,
                    {"tool_name": tc.name, "tool_call_id": tc.id, "is_error": True},
                )
                continue

            try:
                result = await self._tools.aexecute(tc.name, args)
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=result.content,
                    raw_result=result.raw_result,
                    is_error=result.is_error,
                    execution_time_ms=result.execution_time_ms,
                ))
            except ToolError as e:
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=f"Error: {e.message}",
                    is_error=True,
                ))

            await self._aemit(
                EventType.TOOL_CALL_END,
                {"tool_name": tc.name, "tool_call_id": tc.id, "is_error": results[-1].is_error},
            )

        return results

    def run(
        self,
        prompt: str,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the agent synchronously.

        Args:
            prompt: The user's prompt/input
            tools: Optional additional tools for this run only
            **kwargs: Additional parameters for the LLM

        Returns:
            The agent's response as a string
        """
        result = asyncio.run(self.arun(prompt, tools=tools, **kwargs))
        return result.content

    def run_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Run the agent synchronously and enforce a structured Pydantic output.
        """
        return asyncio.run(self.arun_structured(prompt, response_model, tools=tools, **kwargs))

    async def arun_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Run the agent asynchronously and enforce a structured Pydantic output.
        Falls back to JSON prompting and validation if the provider lacks native structured support.
        """
        import json
        import re

        # We append a structured output instruction to the prompt context.
        schema = response_model.model_json_schema()
        structured_prompt = f"{prompt}\n\nYou MUST respond entirely in valid JSON format matching this schema:\n{json.dumps(schema, indent=2)}"

        if self._memory and getattr(self._memory, "auto_summary", False) and getattr(self._memory, "max_messages", None) and len(self._memory) > self._memory.max_messages:
            await self._memory.asummarize(self.provider.acomplete)

        result = await self.arun(structured_prompt, tools=tools, **kwargs)

        # Robust JSON extraction using Regex
        content = result.content
        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
        if not match:
            raise AgentError(
                f"Could not extract JSON from agent response.\nRaw Response: {content}"
            )

        json_str = match.group(1)

        try:
            return response_model.model_validate_json(json_str)
        except Exception as e:
            raise AgentError(f"Failed to parse structured output: {e!s}\nExtracted JSON: {json_str}\nRaw Response: {content}") from e

    async def arun(
        self,
        prompt: str,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Run the agent asynchronously.

        Args:
            prompt: The user's prompt/input
            tools: Optional additional tools for this run only
            **kwargs: Additional parameters for the LLM

        Returns:
            AgentResult with full execution details
        """
        start_time = time.perf_counter()
        self._cancelled = False
        await self._aset_state(AgentState.RUNNING)

        # Emit start event
        await self._aemit(EventType.AGENT_START, {"prompt": prompt[:200]})

        all_tool_calls: list[ToolCall] = []
        all_tool_results: list[ToolResult] = []
        all_messages: list[Message] = []
        total_usage = Usage()
        iterations = 0
        error_msg: str | None = None

        try:
            # Auto-summarize memory if needed
            if self._memory and getattr(self._memory, "auto_summary", False) and getattr(self._memory, "max_messages", None) and len(self._memory) > self._memory.max_messages:
                await self._memory.asummarize(self.provider.acomplete)

            # Build initial messages
            messages = self._build_messages(prompt)
            all_messages = messages.copy()

            # Get tool definitions
            tool_defs = self._tools.get_definitions()
            if tools:
                tool_defs.extend([t.to_definition() for t in tools])

            # Main agent loop
            while iterations < self.config.max_iterations:
                if self._cancelled:
                    raise AgentCancelledError(self.name)

                iterations += 1

                # Emit LLM request event
                await self._aemit(EventType.LLM_REQUEST, {"iteration": iterations})

                # Call LLM with retry
                response = await self.provider.acomplete(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    **kwargs,
                )

                # Track usage
                total_usage = total_usage + response.usage

                # Check maximum budget
                if self.config.max_tokens_limit is not None and getattr(total_usage, "total_tokens", 0) > self.config.max_tokens_limit:
                    raise AgentError(f"Token limit exceeded: used {total_usage.total_tokens}, limit {self.config.max_tokens_limit}")

                # Emit LLM response event
                await self._aemit(
                    EventType.LLM_RESPONSE,
                    {
                        "has_tool_calls": response.has_tool_calls,
                        "finish_reason": response.finish_reason.value if response.finish_reason else None,
                    },
                )

                # Handle tool calls
                if response.has_tool_calls:
                    await self._aset_state(AgentState.EXECUTING_TOOLS)

                    # Add assistant message with tool calls
                    assistant_msg = Message.assistant(
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                    messages.append(assistant_msg)
                    all_messages.append(assistant_msg)

                    # Track tool calls
                    all_tool_calls.extend(response.tool_calls or [])

                    # Check max tool calls
                    if len(all_tool_calls) > self.config.max_tool_calls:
                        raise AgentMaxIterationsError(self.name, self.config.max_tool_calls)

                    # Execute tools
                    tool_results = await self._aexecute_tool_calls(response.tool_calls or [])
                    all_tool_results.extend(tool_results)

                    # Add tool results to messages
                    for tr in tool_results:
                        msg = tr.to_message()
                        messages.append(msg)
                        all_messages.append(msg)

                    await self._aset_state(AgentState.RUNNING)

                else:
                    # No tool calls - we have a final response
                    final_content = response.content

                    # Add final assistant message
                    final_msg = Message.assistant(content=final_content)
                    messages.append(final_msg)
                    all_messages.append(final_msg)

                    # Store in memory
                    if self._memory is not None:
                        self._memory.add_user_message(prompt)
                        self._memory.add_assistant_message(final_content)

                    self._total_usage = self._total_usage + total_usage
                    await self._aset_state(AgentState.COMPLETED)

                    latency = (time.perf_counter() - start_time) * 1000

                    result = AgentResult(
                        success=True,
                        content=final_content,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        messages=all_messages,
                        usage=total_usage,
                        iterations=iterations,
                        state=AgentState.COMPLETED,
                        latency_ms=latency,
                    )

                    await self._aemit(EventType.AGENT_END, {"success": True, "iterations": iterations})

                    return result

            # Max iterations reached
            raise AgentMaxIterationsError(self.name, self.config.max_iterations)

        except AgentCancelledError:
            await self._aset_state(AgentState.CANCELLED)
            error_msg = "Agent execution cancelled"
            raise

        except AgentMaxIterationsError:
            await self._aset_state(AgentState.FAILED)
            error_msg = f"Max iterations ({self.config.max_iterations}) exceeded"
            raise

        except Exception as e:
            await self._aset_state(AgentState.FAILED)
            error_msg = str(e)
            await self._aemit(EventType.AGENT_ERROR, {"error": error_msg, "error_type": type(e).__name__})
            raise AgentError(f"Agent execution failed: {error_msg}", agent_name=self.name) from e

        finally:
            latency = (time.perf_counter() - start_time) * 1000
            self._total_usage = self._total_usage + total_usage

    def stream(
        self,
        prompt: str,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream the agent's response token by token.

        Yields:
            Text chunks from the response
        """
        async def _collect():
            return [chunk async for chunk in self.astream(prompt, tools=tools, **kwargs)]

        yield from asyncio.run(_collect())

    async def astream(
        self,
        prompt: str,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream the agent's response asynchronously.

        Yields:
            Text chunks from the response
        """
        await self._aset_state(AgentState.RUNNING)

        try:
            if self._memory and getattr(self._memory, "auto_summary", False) and getattr(self._memory, "max_messages", None) and len(self._memory) > self._memory.max_messages:
                await self._memory.asummarize(self.provider.acomplete)

            messages = self._build_messages(prompt)
            tool_defs = self._tools.get_definitions()
            if tools:
                tool_defs.extend([t.to_definition() for t in tools])

            # For streaming, we need to handle tool calls differently
            # First check if we need tools
            response = await self.provider.acomplete(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                **kwargs,
            )

            if response.has_tool_calls:
                # Execute tools, then stream final response
                messages.append(Message.assistant(content=response.content, tool_calls=response.tool_calls))

                tool_results = await self._aexecute_tool_calls(response.tool_calls or [])
                for tr in tool_results:
                    messages.append(tr.to_message())

                # Stream final response
                async for chunk in self.provider.astream(messages=messages, **kwargs):
                    yield chunk
            else:
                # Stream directly
                async for chunk in self.provider.astream(messages=messages, **kwargs):
                    yield chunk

        finally:
            await self._aset_state(AgentState.COMPLETED)

    def cancel(self) -> None:
        """Cancel the current execution."""
        self._cancelled = True

    def clear_memory(self) -> None:
        """Clear the agent's conversation memory."""
        if self._memory:
            self._memory.clear()

    def get_memory(self) -> Memory | None:
        """Get the agent's memory object."""
        return self._memory

    def __repr__(self) -> str:
        """String representation."""
        return f"Agent(name={self.name!r}, model={self.config.model!r}, tools={len(self._tools)}, state={self.state.value})"

    def __enter__(self) -> Agent:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cancel()
