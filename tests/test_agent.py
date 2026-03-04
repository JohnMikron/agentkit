"""Tests for AgentKit agent module."""

import pytest

from agentkit.core.agent import Agent, AgentConfig, AgentHooks
from agentkit.core.types import Event, EventType


class TestAgentCreation:
    """Tests for creating agents."""

    def test_basic_creation(self):
        """Test creating a basic agent."""
        agent = Agent("test")
        assert agent.name == "test"
        assert agent.config.model == "gpt-4o-mini"
        assert len(agent.tools) == 0

    def test_with_custom_model(self):
        """Test creating agent with custom model."""
        agent = Agent("test", model="gpt-4o")
        assert agent.config.model == "gpt-4o"

    def test_with_memory(self):
        """Test creating agent with memory."""
        agent = Agent("test", memory=True)
        assert agent.config.memory_enabled
        assert agent.get_memory() is not None

    def test_with_system_prompt(self):
        """Test creating agent with system prompt."""
        agent = Agent("test", system_prompt="You are helpful.")
        assert agent.config.system_prompt == "You are helpful."

    def test_with_config(self):
        """Test creating agent with config object."""
        config = AgentConfig(
            name="custom",
            model="claude-3-5-sonnet-latest",
            temperature=0.5,
            memory_enabled=True,
        )
        agent = Agent(config=config)
        assert agent.name == "custom"
        assert agent.config.model == "claude-3-5-sonnet-latest"
        assert agent.config.temperature == 0.5

    def test_repr(self):
        """Test string representation."""
        agent = Agent("my_agent")
        assert "my_agent" in repr(agent)


class TestAgentTools:
    """Tests for agent tool management."""

    def test_tool_decorator(self):
        """Test adding tool with decorator."""
        agent = Agent("test")

        @agent.tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "greet"

    def test_tool_decorator_with_name(self):
        """Test tool decorator with custom name."""
        agent = Agent("test")

        @agent.tool(name="custom_name")
        def my_func() -> str:
            return "ok"

        assert agent.tools[0].name == "custom_name"

    def test_add_tool_instance(self):
        """Test adding a Tool instance directly."""
        from agentkit.core.tools import Tool

        agent = Agent("test")

        @tool
        def my_tool() -> str:
            return "result"

        agent.add_tool(my_tool)
        assert len(agent.tools) == 1

    def test_add_tools_chaining(self):
        """Test adding multiple tools with chaining."""
        from agentkit.core.tools import Tool

        agent = Agent("test")

        @tool
        def tool1() -> str:
            return "1"

        @tool
        def tool2() -> str:
            return "2"

        agent.add_tool(tool1).add_tool(tool2)
        assert len(agent.tools) == 2

    def test_multiple_tools(self):
        """Test adding multiple tools via decorator."""
        agent = Agent("test")

        @agent.tool
        def tool1() -> str:
            return "1"

        @agent.tool
        def tool2() -> str:
            return "2"

        @agent.tool
        def tool3() -> str:
            return "3"

        assert len(agent.tools) == 3


class TestAgentHooks:
    """Tests for agent hooks."""

    def test_register_hooks(self):
        """Test registering event hooks."""
        agent = Agent("test")

        @agent.on_start
        def on_start(event):
            pass

        @agent.on_end
        def on_end(event):
            pass

        @agent.on_tool_call
        def on_tool(event):
            pass

        # Hooks should be registered
        assert len(agent._hooks._on_start) == 1
        assert len(agent._hooks._on_end) == 1
        assert len(agent._hooks._on_tool_call_start) == 1


class TestAgentMemory:
    """Tests for agent memory management."""

    def test_memory_disabled_by_default(self):
        """Test that memory is disabled by default."""
        agent = Agent("test")
        assert agent.get_memory() is None

    def test_enable_memory(self):
        """Test enabling memory."""
        agent = Agent("test", memory=True)
        assert agent.get_memory() is not None

    def test_clear_memory(self):
        """Test clearing memory."""
        agent = Agent("test", memory=True)
        agent.get_memory().add_user_message("Test")
        agent.clear_memory()
        assert len(agent.get_memory()) == 0


class TestAgentConfig:
    """Tests for agent configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.name == "agent"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_iterations == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentConfig(
            name="custom",
            model="claude-3-opus",
            temperature=0.3,
            max_tokens=1000,
            max_iterations=5,
        )
        assert config.name == "custom"
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.3
        assert config.max_tokens == 1000
        assert config.max_iterations == 5


class TestAgentContext:
    """Tests for context manager."""

    def test_context_manager(self):
        """Test using agent as context manager."""
        with Agent("test") as agent:
            assert agent.name == "test"
