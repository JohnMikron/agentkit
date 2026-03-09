#!/usr/bin/env python3
"""
Basic AgentKit example - The simplest way to build an AI agent.

This example demonstrates:
- Creating an agent with a single line
- Adding tools with the @agent.tool decorator
- Running the agent
"""

import asyncio

    # Set your API key via agent config or environment var
from agentkit import Agent


async def main():
    """Run basic agent examples."""

    # ============================================================
    # Example 1: The simplest agent (literally 5 lines!)
    # ============================================================
    print("=" * 60)
    print("Example 1: Basic Agent")
    print("=" * 60)

    agent = Agent("assistant")

    @agent.tool
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}! Nice to meet you."

    # Run the agent using basic run commands.
    print("\nAgent created with 1 tool:")
    print(f"  - {agent.tools[0].name}: {agent.tools[0].description}")

    # ============================================================
    # Example 2: Multiple tools
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 2: Agent with Multiple Tools")
    print("=" * 60)

    agent2 = Agent("multi_tool_agent", model="gpt-4o-mini")

    @agent2.tool
    def calculate(expression: str) -> float:
        """
        Evaluate a mathematical expression.

        Args:
            expression: Math expression like "2 + 2" or "sqrt(16)"
        """
        import math

        allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "pi": math.pi}
        return eval(expression, {"__builtins__": {}}, allowed)

    @agent2.tool
    def get_time() -> str:
        """Get the current date and time."""
        from datetime import datetime

        return datetime.now().isoformat()

    @agent2.tool
    def word_count(text: str) -> int:
        """Count words in a text."""
        return len(text.split())

    print(f"\nAgent has {len(agent2.tools)} tools:")
    for tool in agent2.tools:
        print(f"  - {tool.name}")

    # ============================================================
    # Example 3: Agent with memory
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 3: Agent with Memory")
    print("=" * 60)

    memory_agent = Agent(
        "memory_agent",
        model="gpt-4o-mini",
        memory=True,
        system_prompt="You are a helpful assistant with perfect memory.",
    )

    # Simulate a conversation
    print("\nMemory enabled:", memory_agent.config.memory_enabled)
    print("Memory object:", memory_agent.get_memory())

    # Add some context to memory
    memory_agent.get_memory().add_user_message("My favorite color is blue.")
    memory_agent.get_memory().add_assistant_message("I'll remember that blue is your favorite color!")

    print(f"Messages in memory: {len(memory_agent.get_memory())}")

    # ============================================================
    # Example 4: Using local models (Ollama)
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 4: Local Models (Ollama)")
    print("=" * 60)

    # Just add "local:" prefix to use local models!
    local_agent = Agent(
        "local_assistant",
        model="local:llama3.2",  # or "local:mistral", "local:phi3"
    )

    print(f"\nLocal agent model: {local_agent.config.model}")
    print("This would use Ollama for inference (requires 'ollama serve')")

    # ============================================================
    # Example 5: Using different providers
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 5: Different Providers")
    print("=" * 60)

    providers = [
        ("OpenAI GPT-4", "gpt-4o"),
        ("OpenAI GPT-4 Mini", "gpt-4o-mini"),
        ("Anthropic Claude 3.5 Sonnet", "claude-3-5-sonnet-latest"),
        ("Anthropic Claude 3 Opus", "claude-3-opus-latest"),
        ("Google Gemini 2.0 Flash", "gemini-2.0-flash"),
        ("Mistral Large", "mistral-large-latest"),
        ("Local Llama 3.2", "local:llama3.2"),
    ]

    print("\nSupported providers and models:")
    for name, model in providers:
        print(f"  {name}: Agent(model='{model}')")


if __name__ == "__main__":
    asyncio.run(main())
