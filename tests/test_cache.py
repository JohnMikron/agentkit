import time
from unittest.mock import patch

import pytest

from agentkit.utils.cache import SemanticCache


@pytest.fixture
def mock_vector_storage():
    with patch('agentkit.core.memory.VectorStorage') as mock_storage_class:
        mock_instance = mock_storage_class.return_value
        yield mock_instance

def test_semantic_cache_set_get(mock_vector_storage):
    cache = SemanticCache(similarity_threshold=0.9)

    # Mock search to return a memory entry
    from agentkit.core.memory import MemoryEntry
    mock_entry = MemoryEntry(
        id="123",
        role="system",
        content="What is the capital of France?",
        metadata={"response": "Paris", "expiry": time.time() + 3600}
    )
    mock_vector_storage.search.return_value = [mock_entry]

    # Test get
    result = cache.get("What's the capital of France?")
    assert result == "Paris"
    mock_vector_storage.search.assert_called_with("What's the capital of France?", limit=1)

    # Test set
    cache.set("What is the capital of France?", "Paris")
    mock_vector_storage.save.assert_called_once()

def test_semantic_cache_expiry(mock_vector_storage):
    cache = SemanticCache()

    from agentkit.core.memory import MemoryEntry
    mock_entry = MemoryEntry(
        id="123",
        role="system",
        content="Hello",
        metadata={"response": "Hi", "expiry": time.time() - 3600} # Expired
    )
    mock_vector_storage.search.return_value = [mock_entry]

    result = cache.get("Hello")
    assert result is None
    mock_vector_storage.delete.assert_called_with("123")
