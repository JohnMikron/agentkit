import pytest
import os
from agentkit.core.types import Message, Role
from agentkit.providers import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    MistralProvider,
    MockProvider,
    OllamaProvider,
)
from agentkit.core.exceptions import MissingAPIKeyError

@pytest.fixture
def mock_messages():
    return [Message.user("Hello")]

class TestMockProvider:
    def test_complete(self, mock_messages):
        provider = MockProvider(responses=["Mocked response"])
        result = provider.complete(mock_messages)
        assert result.content == "Mocked response"
        assert result.usage.total_tokens == 20

    @pytest.mark.asyncio
    async def test_acomplete(self, mock_messages):
        provider = MockProvider(responses=["Async mocked"])
        result = await provider.acomplete(mock_messages)
        assert result.content == "Async mocked"

    def test_stream(self, mock_messages):
        provider = MockProvider(responses=["Word by word"])
        chunks = list(provider.stream(mock_messages))
        assert "Word " in chunks
        assert "by " in chunks

    @pytest.mark.asyncio
    async def test_astream(self, mock_messages):
        provider = MockProvider(responses=["Word by word"])
        chunks = []
        async for chunk in provider.astream(mock_messages):
            chunks.append(chunk)
        assert "Word " in chunks

class TestMissingKeys:
    def test_openai_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            OpenAIProvider()

    def test_anthropic_missing_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            AnthropicProvider()

    def test_google_missing_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            GoogleProvider()

    def test_mistral_missing_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            MistralProvider()

class TestOllamaProvider:
    def test_initialization(self):
        # Ollama doesn't require an API key
        provider = OllamaProvider(model="llama3.3")
        assert provider.model == "llama3.3"
        assert provider.base_url == "http://localhost:11434"

class TestProviderFormatters:
    # Testing the internal mappers for tools and messages
    def test_openai_message_format(self):
        provider = OpenAIProvider(api_key="test")
        msg = Message.user("Hi")
        formatted = provider._message_to_api_format(msg)
        assert formatted["role"] == "user"
        assert formatted["content"] == "Hi"

        sys_msg = Message.system("System prompt")
        sys_formatted = provider._message_to_api_format(sys_msg)
        assert sys_formatted["role"] == "system"

    def test_anthropic_message_format(self):
        provider = AnthropicProvider(api_key="test")
        msg = Message.user("Hi")
        # Anthropic converts a list of messages into (system_prompt, api_messages)
        system_prompt, converted = provider._convert_messages([msg])
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hi"
        
        # System messages are extracted out
        sys_msg = Message.system("System prompt")
        system_prompt, converted = provider._convert_messages([sys_msg])
        assert system_prompt == "System prompt"
        assert len(converted) == 0
