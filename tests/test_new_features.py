import os
import pytest
from agentkit import Agent
from agentkit.core.memory import Memory, VectorStorage
from agentkit.orchestration.web import WebAgent
from agentkit.providers.mock import MockProvider

def test_mock_provider():
    provider = MockProvider(responses=["Hello from mock!"])
    agent = Agent(name="test", model="mock:test")
    # Manually set provider to skip detection which might fail if no keys
    agent._provider = provider
    
    result = agent.run("Hi")
    assert result == "Hello from mock!"

def test_vector_memory():
    try:
        import chromadb
        import sentence_transformers
    except ImportError:
        pytest.skip("Vector memory dependencies not installed")

    storage = VectorStorage()
    memory = Memory(storage=storage)
    
    memory.add_user_message("My favorite color is blue")
    memory.add_assistant_message("I will remember that.")
    
    # Search should find the entry
    results = memory.search("What is my favorite color?")
    assert len(results) > 0
    assert "blue" in results[0].content.lower()

def test_web_agent_initialization():
    agent = WebAgent(name="webby", model="mock:web")
    assert "search_web" in [t.name for t in agent.tools]
    assert "scrape_url" in [t.name for t in agent.tools]

def test_demo_mode_fallback(monkeypatch):
    monkeypatch.setenv("AGENTKIT_DEMO_MODE", "true")
    # Even without keys, this should not raise MissingAPIKeyError
    agent = Agent(model="gpt-4o")
    # Accessing provider should trigger creation of MockProvider
    assert isinstance(agent.provider, MockProvider)
    assert agent.provider.model == "mock-gpt-4o"
