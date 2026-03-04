"""Tests for AgentKit core module."""

import pytest

from agentkit.core.types import (
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
    LLMResponse,
    AgentState,
    AgentResult,
)
from agentkit.core.tools import Tool, ToolRegistry, tool
from agentkit.core.exceptions import AgentKitError, ToolError


class TestMessage:
    """Tests for Message class."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello!")
        assert msg.role == Role.USER
        assert msg.content == "Hello!"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are helpful.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_call = ToolCall(id="call_1", name="search", arguments='{"query": "test"}')
        msg = Message.assistant(content="", tool_calls=[tool_call])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_to_api_format(self):
        """Test converting to API format."""
        msg = Message.user("Hello")
        data = msg.to_api_format()
        assert data["role"] == "user"
        assert data["content"] == "Hello"


class TestToolCall:
    """Tests for ToolCall class."""

    def test_parse_arguments(self):
        """Test parsing JSON arguments."""
        tc = ToolCall(id="1", name="test", arguments='{"a": 1, "b": 2}')
        args = tc.parse_arguments()
        assert args == {"a": 1, "b": 2}

    def test_parse_invalid_arguments(self):
        """Test parsing invalid JSON returns empty dict."""
        tc = ToolCall(id="1", name="test", arguments="not json")
        args = tc.parse_arguments()
        assert args == {}


class TestTool:
    """Tests for Tool class."""

    def test_create_tool(self):
        """Test creating a tool from a function."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert "Greet" in greet.description

    def test_tool_with_custom_name(self):
        """Test creating a tool with custom name."""

        @tool(name="custom_name")
        def my_func() -> str:
            return "ok"

        assert my_func.name == "custom_name"

    def test_tool_execution(self):
        """Test executing a tool."""

        @tool
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        result = add.execute({"a": 1, "b": 2})
        assert result.content == "3"
        assert not result.is_error

    def test_tool_schema_generation(self):
        """Test JSON Schema generation."""

        @tool
        def complex_func(
            text: str,
            number: int,
            flag: bool = False,
        ) -> str:
            """Complex function."""
            return "ok"

        props = complex_func.parameters["properties"]
        assert props["text"]["type"] == "string"
        assert props["number"]["type"] == "integer"
        assert props["flag"]["type"] == "boolean"


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_add_and_get(self):
        """Test adding and retrieving tools."""

        @tool
        def test_tool() -> str:
            return "ok"

        registry = ToolRegistry()
        registry.add(test_tool)

        assert registry.has("test_tool")
        assert registry.get("test_tool") == test_tool

    def test_execute(self):
        """Test executing tools from registry."""

        @tool
        def echo(text: str) -> str:
            """Echo text."""
            return text

        registry = ToolRegistry()
        registry.add(echo)

        result = registry.execute("echo", '{"text": "hello"}')
        assert result.content == "hello"

    def test_execute_with_dict(self):
        """Test executing with dict arguments."""

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        registry = ToolRegistry()
        registry.add(add)

        result = registry.execute("add", {"a": 1, "b": 2})
        assert result.content == "3"

    def test_contains_operator(self):
        """Test 'in' operator."""

        @tool
        def my_tool() -> str:
            return "ok"

        registry = ToolRegistry()
        registry.add(my_tool)

        assert "my_tool" in registry
        assert "other" not in registry


class TestUsage:
    """Tests for Usage class."""

    def test_usage_addition(self):
        """Test adding Usage objects."""
        u1 = Usage(prompt_tokens=10, completion_tokens=5)
        u2 = Usage(prompt_tokens=20, completion_tokens=10)

        total = u1 + u2

        assert total.prompt_tokens == 30
        assert total.completion_tokens == 15
        assert total.total_tokens == 45


class TestAgentResult:
    """Tests for AgentResult class."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = AgentResult(
            success=True,
            content="Done!",
            iterations=3,
        )

        assert result.success
        assert not result.failed
        assert result.content == "Done!"

    def test_failed_result(self):
        """Test creating a failed result."""
        result = AgentResult(
            success=False,
            error="Something went wrong",
        )

        assert not result.success
        assert result.failed


class TestExceptions:
    """Tests for custom exceptions."""

    def test_agentkit_error(self):
        """Test AgentKitError."""
        error = AgentKitError("Test error", code="TEST_ERROR")
        assert error.message == "Test error"
        assert error.code == "TEST_ERROR"

    def test_tool_error(self):
        """Test ToolError."""
        error = ToolError("Tool failed", tool_name="test_tool")
        assert error.tool_name == "test_tool"
        assert "test_tool" in str(error)
