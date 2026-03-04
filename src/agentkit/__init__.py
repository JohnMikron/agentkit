"""
AgentKit - Enterprise-grade AI Agent Framework.

Build AI agents in 5 lines of code, not 500.
The most powerful, simple, and production-ready Python agent framework for 2026.

Example:
    from agentkit import Agent

    agent = Agent("assistant")

    @agent.tool
    def search(query: str) -> str:
        '''Search the web'''
        return "results..."

    result = await agent.run("Search for Python news")
"""

from agentkit.core.agent import Agent, AgentConfig, AgentResult
from agentkit.core.types import Message, Role, ToolDefinition, ToolResult
from agentkit.core.tools import Tool, tool
from agentkit.core.memory import Memory, InMemoryStorage
from agentkit.core.exceptions import AgentKitError
from agentkit.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    MistralProvider,
    OllamaProvider,
)

__version__ = "1.0.0"
__author__ = "AgentKit Team"
__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Message",
    "Role",
    "ToolDefinition",
    "ToolResult",
    "Tool",
    "tool",
    "Memory",
    "InMemoryStorage",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MistralProvider",
    "OllamaProvider",
    # Exceptions
    "AgentKitError",
    # Version
    "__version__",
]
