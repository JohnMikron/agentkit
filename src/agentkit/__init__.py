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
from agentkit.core.exceptions import AgentKitError
from agentkit.core.memory import InMemoryStorage, Memory, SQLiteStorage, VectorStorage
from agentkit.core.tools import Tool, tool
from agentkit.core.types import Message, Role, ToolDefinition, ToolResult
from agentkit.orchestration.web import WebAgent
from agentkit.providers import (
    AnthropicProvider,
    GoogleProvider,
    LLMProvider,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
)
from agentkit.providers.mock import MockProvider

__version__ = "1.2.0"
__author__ = "AgentKit Team"
__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    # Exceptions
    "AgentKitError",
    "AgentResult",
    "AnthropicProvider",
    "GoogleProvider",
    "InMemoryStorage",
    # Providers
    "LLMProvider",
    "Memory",
    "Message",
    "MistralProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "Role",
    "SQLiteStorage",
    "Tool",
    "ToolDefinition",
    "ToolResult",
    "VectorStorage",
    # Agents
    "WebAgent",
    # Version
    "__version__",
    "tool",
]
