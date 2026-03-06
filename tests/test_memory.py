import os
import json
import pytest
import sqlite3
import tempfile
from agentkit.core.memory import (
    MemoryEntry,
    Memory,
    InMemoryStorage,
    FileStorage,
    SQLiteStorage,
)
from agentkit.core.types import Role

@pytest.fixture
def temp_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

class TestInMemoryStorage:
    def test_add_and_get(self):
        storage = InMemoryStorage()
        entry = MemoryEntry(role=Role.USER, content="Hello")
        storage.save(entry)
        
        entries = storage.load()
        assert len(entries) == 1
        assert entries[0].content == "Hello"

    def test_search_and_clear(self):
        storage = InMemoryStorage()
        storage.save(MemoryEntry(role=Role.USER, content="Hello World"))
        storage.save(MemoryEntry(role=Role.USER, content="Goodbye"))
        
        results = storage.search("World")
        assert len(results) == 1
        assert results[0].content == "Hello World"
        
        storage.clear()
        assert len(storage.load()) == 0

class TestFileStorage:
    def test_file_persistence(self, temp_file):
        storage = FileStorage(temp_file)
        storage.save(MemoryEntry(role=Role.USER, content="Persistent Hello"))
        
        # New instance should load from file
        storage2 = FileStorage(temp_file)
        entries = storage2.load()
        assert len(entries) == 1
        assert entries[0].content == "Persistent Hello"
        
    def test_clear_file(self, temp_file):
        storage = FileStorage(temp_file)
        storage.save(MemoryEntry(role=Role.USER, content="Test"))
        storage.clear()
        
        storage2 = FileStorage(temp_file)
        assert len(storage2.load()) == 0

class TestSQLiteStorage:
    def test_sqlite_persistence(self, temp_db):
        storage = SQLiteStorage(temp_db)
        storage.save(MemoryEntry(role=Role.USER, content="SQLite Hello"))
        
        storage2 = SQLiteStorage(temp_db)
        entries = storage2.load()
        assert len(entries) == 1
        assert entries[0].content == "SQLite Hello"

    def test_sqlite_search(self, temp_db):
        storage = SQLiteStorage(temp_db)
        storage.save(MemoryEntry(role=Role.USER, content="Find me if you can"))
        storage.save(MemoryEntry(role=Role.ASSISTANT, content="I am here"))
        
        results = storage.search("Find")
        assert len(results) == 1
        assert "Find me" in results[0].content
        
        results = storage.search("notpresent", limit=5)
        assert len(results) == 0

    def test_sqlite_clear(self, temp_db):
        storage = SQLiteStorage(temp_db)
        storage.save(MemoryEntry(role=Role.USER, content="Delete me"))
        storage.clear()
        
        assert len(storage.load()) == 0

class TestMemoryWrapper:
    def test_add_messages(self):
        memory = Memory() # defaults to InMemoryStorage
        memory.add_user_message("User says")
        memory.add_assistant_message("Assistant says")
        memory.add_system_message("System says")
        
        messages = memory.get_history()
        assert len(messages) == 3
        assert messages[0].role == Role.USER
        assert messages[1].role == Role.ASSISTANT
        assert messages[2].role == Role.SYSTEM

    def test_memory_clear(self):
        memory = Memory()
        memory.add_user_message("Hello")
        memory.clear()
        
        assert len(memory.get_history()) == 0

    @pytest.mark.asyncio
    async def test_memory_asummarize(self):
        memory = Memory(max_messages=3, auto_summary=True)
        memory.add_user_message("My name is John.")
        memory.add_assistant_message("Hello John.")
        memory.add_user_message("I am 30 years old.")
        memory.add_assistant_message("Got it.")
        
        async def mock_provider(messages):
            from agentkit.core.types import Message, Role
            return Message(role=Role.ASSISTANT, content="John is 30.")
            
        await memory.asummarize(mock_provider, keep_recent=2)
        history = memory.get_history()
        
        # Needs to be 1 summary system message + 2 retained recent messages = 3 messages total
        assert len(history) == 3
        assert "Summary of prior conversation: John is 30." in history[0].content
        assert history[1].content == "I am 30 years old."
        assert history[2].content == "Got it."

    def test_with_tool_calls(self):
        from agentkit.core.types import ToolCall, Message
        memory = Memory()
        memory.add_message(Message(role=Role.ASSISTANT, content="Thinking", tool_calls=[ToolCall(id="1", name="test", arguments="{}")]))
        memory.add_message(Message.tool_result(content="Result", tool_call_id="1", name="test"))
        
        messages = memory.get_history()
        assert len(messages) == 2
        assert messages[0].tool_calls is not None
        assert messages[1].tool_call_id == "1"
