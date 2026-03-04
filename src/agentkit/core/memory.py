"""
Memory systems for AgentKit.

This module provides multiple memory implementations:
- InMemoryStorage: Fast, ephemeral storage
- FileStorage: Persistent file-based storage
- RedisStorage: Distributed storage with Redis
- VectorMemory: Semantic search with embeddings
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from agentkit.core.types import Message, Role

T = TypeVar("T")


@dataclass
class MemoryEntry:
    """
    A single memory entry.

    Attributes:
        id: Unique identifier
        role: Who said this (user, assistant, system)
        content: The content of the memory
        timestamp: When this was stored
        metadata: Additional metadata
        embedding: Optional vector embedding for semantic search
    """

    id: str = field(default_factory=lambda: f"mem_{int(time.time() * 1000)}")
    role: str = "user"
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.utcnow()),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_message(cls, message: Message) -> "MemoryEntry":
        """Create from a Message object."""
        return cls(
            role=message.role.value,
            content=message.content,
            metadata=message.metadata,
        )

    def to_message(self) -> Message:
        """Convert to a Message object."""
        return Message(
            role=Role(self.role),
            content=self.content,
            metadata=self.metadata,
        )


class MemoryStorage(ABC, Generic[T]):
    """
    Abstract base class for memory storage backends.

    Defines the interface that all storage implementations must follow.
    """

    @abstractmethod
    def save(self, entry: MemoryEntry) -> str:
        """Save a memory entry and return its ID."""
        pass

    @abstractmethod
    def load(self, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Load memory entries, optionally limited to the most recent."""
        pass

    @abstractmethod
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID."""
        pass

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the number of stored entries."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search for entries containing the query."""
        pass


class InMemoryStorage(MemoryStorage[MemoryEntry]):
    """
    In-memory storage backend.

    Stores all memory entries in a list. Memory is lost when the
    Python process ends. Best for short-lived agents or testing.

    Example:
        storage = InMemoryStorage()
        storage.save(MemoryEntry(role="user", content="Hello!"))
        entries = storage.load()
    """

    def __init__(self, max_entries: Optional[int] = None) -> None:
        """
        Initialize in-memory storage.

        Args:
            max_entries: Maximum number of entries to keep (default: unlimited)
        """
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []
        self._by_id: Dict[str, MemoryEntry] = {}

    def save(self, entry: MemoryEntry) -> str:
        """Save a memory entry."""
        self._entries.append(entry)
        self._by_id[entry.id] = entry

        # Enforce max entries limit
        if self.max_entries and len(self._entries) > self.max_entries:
            removed = self._entries.pop(0)
            self._by_id.pop(removed.id, None)

        return entry.id

    def load(self, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Load memory entries."""
        if limit:
            return self._entries[-limit:]
        return self._entries.copy()

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID."""
        return self._by_id.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        if entry_id in self._by_id:
            entry = self._by_id.pop(entry_id)
            self._entries.remove(entry)
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._by_id.clear()

    def count(self) -> int:
        """Get the number of stored entries."""
        return len(self._entries)

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search for entries containing the query (case-insensitive)."""
        query_lower = query.lower()
        results = [e for e in self._entries if query_lower in e.content.lower()]
        return results[-limit:]


class FileStorage(MemoryStorage[MemoryEntry]):
    """
    File-based storage backend.

    Persists memory entries to a JSON file. Memory survives across
    Python process restarts. Best for long-running agents.

    Example:
        storage = FileStorage("agent_memory.json")
        storage.save(MemoryEntry(role="user", content="Hello!"))
    """

    def __init__(
        self,
        filepath: str,
        max_entries: Optional[int] = None,
        autosave: bool = True,
    ) -> None:
        """
        Initialize file-based storage.

        Args:
            filepath: Path to the JSON file for storing memory
            max_entries: Maximum number of entries to keep
            autosave: Whether to save after each change (default: True)
        """
        self.filepath = Path(filepath)
        self.max_entries = max_entries
        self.autosave = autosave
        self._entries: List[MemoryEntry] = []
        self._by_id: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()

        # Load existing entries
        self._load_from_file()

    def _load_from_file(self) -> None:
        """Load entries from the file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._entries = [MemoryEntry.from_dict(e) for e in data]
                    self._by_id = {e.id: e for e in self._entries}
            except (json.JSONDecodeError, KeyError):
                self._entries = []
                self._by_id = {}

    def _save_to_file(self) -> None:
        """Save entries to the file."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2)

    def save(self, entry: MemoryEntry) -> str:
        """Save a memory entry."""
        self._entries.append(entry)
        self._by_id[entry.id] = entry

        if self.max_entries and len(self._entries) > self.max_entries:
            removed = self._entries.pop(0)
            self._by_id.pop(removed.id, None)

        if self.autosave:
            self._save_to_file()

        return entry.id

    def load(self, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Load memory entries."""
        if limit:
            return self._entries[-limit:]
        return self._entries.copy()

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID."""
        return self._by_id.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        if entry_id in self._by_id:
            entry = self._by_id.pop(entry_id)
            self._entries.remove(entry)
            if self.autosave:
                self._save_to_file()
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._by_id.clear()
        if self.autosave:
            self._save_to_file()

    def count(self) -> int:
        """Get the number of stored entries."""
        return len(self._entries)

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search for entries containing the query."""
        query_lower = query.lower()
        results = [e for e in self._entries if query_lower in e.content.lower()]
        return results[-limit:]


class RedisStorage(MemoryStorage[MemoryEntry]):
    """
    Redis-based storage backend.

    Provides distributed memory storage with Redis. Best for
    multi-instance deployments and production use.

    Requires: pip install redis
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "agentkit:memory:",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Initialize Redis storage.

        Args:
            redis_url: Redis connection URL
            key_prefix: Key prefix for namespacing
            ttl: Optional TTL in seconds
        """
        try:
            import redis
        except ImportError:
            raise ImportError("Redis storage requires 'redis' package: pip install redis")

        self._redis = redis.from_url(redis_url)
        self.key_prefix = key_prefix
        self.ttl = ttl
        self._list_key = f"{key_prefix}entries"

    def _get_key(self, entry_id: str) -> str:
        """Get the Redis key for an entry."""
        return f"{self.key_prefix}entry:{entry_id}"

    def save(self, entry: MemoryEntry) -> str:
        """Save a memory entry."""
        key = self._get_key(entry.id)
        data = json.dumps(entry.to_dict())

        pipe = self._redis.pipeline()
        pipe.set(key, data)
        pipe.rpush(self._list_key, entry.id)

        if self.ttl:
            pipe.expire(key, self.ttl)

        pipe.execute()
        return entry.id

    def load(self, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Load memory entries."""
        # Get all entry IDs
        if limit:
            # Get the most recent entries
            total = self._redis.llen(self._list_key)
            start = max(0, total - limit)
            ids = self._redis.lrange(self._list_key, start, -1)
        else:
            ids = self._redis.lrange(self._list_key, 0, -1)

        # Batch get entries
        if not ids:
            return []

        keys = [self._get_key(id.decode()) for id in ids]
        entries_data = self._redis.mget(keys)

        entries = []
        for data in entries_data:
            if data:
                entries.append(MemoryEntry.from_dict(json.loads(data)))

        return entries

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID."""
        key = self._get_key(entry_id)
        data = self._redis.get(key)
        if data:
            return MemoryEntry.from_dict(json.loads(data))
        return None

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        key = self._get_key(entry_id)
        pipe = self._redis.pipeline()
        pipe.delete(key)
        pipe.lrem(self._list_key, 0, entry_id)
        results = pipe.execute()
        return results[0] > 0

    def clear(self) -> None:
        """Clear all entries."""
        # Get all entry IDs
        ids = self._redis.lrange(self._list_key, 0, -1)
        if ids:
            keys = [self._get_key(id.decode()) for id in ids]
            self._redis.delete(*keys)

        self._redis.delete(self._list_key)

    def count(self) -> int:
        """Get the number of stored entries."""
        return self._redis.llen(self._list_key)

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search for entries (basic implementation)."""
        # For better search, consider using Redisearch or vector search
        entries = self.load()
        query_lower = query.lower()
        results = [e for e in entries if query_lower in e.content.lower()]
        return results[-limit:]


class Memory:
    """
    High-level memory interface for agents.

    Provides a simple API for storing and retrieving conversation
    history, facts, and other information the agent should remember.

    Example:
        memory = Memory(storage=FileStorage("agent.json"))
        memory.add_user_message("My name is Alice")
        memory.add_assistant_message("Hello Alice!")
        history = memory.get_history()
    """

    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize memory with optional storage backend.

        Args:
            storage: Storage backend (default: InMemoryStorage)
            system_prompt: Optional system prompt to include in history
            max_tokens: Optional max tokens limit for context window
        """
        self.storage = storage or InMemoryStorage()
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def add_message(self, message: Message) -> str:
        """Add a message to memory."""
        entry = MemoryEntry.from_message(message)
        return self.storage.save(entry)

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a user message to memory."""
        return self.add_message(Message.user(content, metadata=metadata or {}))

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add an assistant message to memory."""
        return self.add_message(Message.assistant(content, metadata=metadata or {}))

    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a system message to memory."""
        return self.add_message(Message.system(content, metadata=metadata or {}))

    def get_history(self, limit: Optional[int] = None) -> List[Message]:
        """
        Get conversation history as Messages.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of messages, oldest first
        """
        entries = self.storage.load(limit)
        return [e.to_message() for e in entries]

    def get_last_n(self, n: int) -> List[Message]:
        """Get the last N messages."""
        return self.get_history(limit=n)

    def search(self, query: str, limit: int = 10) -> List[Message]:
        """
        Search memory for entries containing the query.

        Args:
            query: Search string
            limit: Maximum results

        Returns:
            List of matching messages
        """
        entries = self.storage.search(query, limit)
        return [e.to_message() for e in entries]

    def clear(self) -> None:
        """Clear all memory."""
        self.storage.clear()

    def count(self) -> int:
        """Get the number of stored entries."""
        return self.storage.count()

    def to_messages(self) -> List[Message]:
        """
        Convert memory to message list for LLM provider.

        Includes system prompt if set.
        """
        messages = []

        if self.system_prompt:
            messages.append(Message.system(self.system_prompt))

        messages.extend(self.get_history())
        return messages

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self.system_prompt = prompt

    def __len__(self) -> int:
        """Get the number of entries."""
        return self.storage.count()

    def __bool__(self) -> bool:
        """Check if there are any entries."""
        return self.storage.count() > 0
