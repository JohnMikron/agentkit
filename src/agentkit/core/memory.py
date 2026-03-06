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
import uuid
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from typing import Any, Dict, Generic, List, Optional, TypeVar, Union, Callable

from agentkit.core.types import Message, Role, ModelId

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
        meta = message.metadata.copy()
        if message.name:
            meta["name"] = message.name
        if message.tool_call_id:
            meta["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            meta["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
            
        return cls(
            role=message.role.value,
            content=message.content,
            metadata=meta,
        )

    def to_message(self) -> Message:
        """Convert to a Message object."""
        from agentkit.core.types import ToolCall
        meta = self.metadata.copy()
        name = meta.pop("name", None)
        tool_call_id = meta.pop("tool_call_id", None)
        tool_calls_data = meta.pop("tool_calls", None)
        
        tool_calls = None
        if tool_calls_data:
            tool_calls = [ToolCall(**tc) for tc in tool_calls_data]
            
        return Message(
            role=Role(self.role),
            content=self.content,
            name=name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
            metadata=meta,
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


class VectorStorage(MemoryStorage[MemoryEntry]):
    """
    Vector-based storage backend using ChromaDB.

    Provides semantic search capabilities using embeddings.
    Best for RAG (Retrieval Augmented Generation).

    Requires: pip install chromadb sentence-transformers
    """

    def __init__(
        self,
        collection_name: str = "agentkit_memory",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize Vector storage.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Optional directory for persistence
            embedding_model: Name of the sentence-transformers model
        """
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Vector storage requires 'chromadb' and 'sentence-transformers' packages: "
                "pip install chromadb sentence-transformers"
            )

        self._client = chromadb.PersistentClient(path=persist_directory) if persist_directory else chromadb.EphemeralClient()
        self._model = SentenceTransformer(embedding_model)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def save(self, entry: MemoryEntry) -> str:
        """Save a memory entry with its embedding."""
        embedding = self._model.encode(entry.content).tolist()
        
        self._collection.add(
            ids=[entry.id],
            embeddings=[embedding],
            documents=[entry.content],
            metadatas=[{**entry.metadata, "role": entry.role, "timestamp": entry.timestamp.isoformat()}],
        )
        return entry.id

    def load(self, limit: Optional[int] = None) -> List[MemoryEntry]:
        """Load memory entries (ordered by timestamp)."""
        results = self._collection.get()
        entries = []
        
        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            entries.append(MemoryEntry(
                id=results["ids"][i],
                role=metadata.get("role", "user"),
                content=results["documents"][i],
                timestamp=datetime.fromisoformat(metadata["timestamp"]),
                metadata={k: v for k, v in metadata.items() if k not in ("role", "timestamp")},
            ))
            
        # Sort by timestamp
        entries.sort(key=lambda x: x.timestamp)
        if limit:
            return entries[-limit:]
        return entries

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID."""
        result = self._collection.get(ids=[entry_id])
        if not result["ids"]:
            return None
            
        metadata = result["metadatas"][0]
        return MemoryEntry(
            id=result["ids"][0],
            role=metadata.get("role", "user"),
            content=result["documents"][0],
            timestamp=datetime.fromisoformat(metadata["timestamp"]),
            metadata={k: v for k, v in metadata.items() if k not in ("role", "timestamp")},
        )

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        self._collection.delete(ids=[entry_id])
        return True

    def clear(self) -> None:
        """Clear all entries."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.create_collection(self._collection.name)

    def count(self) -> int:
        """Get the number of stored entries."""
        return self._collection.count()

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search for entries by semantic similarity."""
        query_embedding = self._model.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )
        
        entries = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            entries.append(MemoryEntry(
                id=results["ids"][0][i],
                role=metadata.get("role", "user"),
                content=results["documents"][0][i],
                timestamp=datetime.fromisoformat(metadata["timestamp"]),
                metadata={k: v for k, v in metadata.items() if k not in ("role", "timestamp")},
            ))
        return entries


class SQLiteStorage(MemoryStorage[MemoryEntry]):
    """
    SQLite-based storage backend.

    Provides reliable, concurrent, zero-dependency persistent memory.
    Best for production agents that need persistent memory without external services.
    
    Example:
        storage = SQLiteStorage("agent_memory.db")
        memory = Memory(storage=storage)
        agent = Agent("assistant", memory=memory)
    """

    def __init__(self, db_path: str = "agent_memory.db") -> None:
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to the SQLite database file
        """
        import sqlite3
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        import sqlite3
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    metadata TEXT
                )
            ''')
            conn.commit()

    def save(self, entry: MemoryEntry) -> str:
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO memory_entries (id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                entry.id,
                entry.role,
                entry.content,
                entry.timestamp.isoformat(),
                json.dumps(entry.metadata)
            ))
            conn.commit()
        return entry.id

    def load(self, limit: Optional[int] = None) -> List[MemoryEntry]:
        query = "SELECT id, role, content, timestamp, metadata FROM memory_entries ORDER BY timestamp ASC"
        if limit is not None:
            # Get the last N rows, but keep them in ascending order
            query = f"SELECT * FROM (SELECT id, role, content, timestamp, metadata FROM memory_entries ORDER BY timestamp DESC LIMIT {limit}) ORDER BY timestamp ASC"
            
        entries = []
        with self._get_connection() as conn:
            cursor = conn.execute(query)
            for row in cursor:
                entries.append(MemoryEntry(
                    id=row[0],
                    role=row[1],
                    content=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4])
                ))
        return entries

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id, role, content, timestamp, metadata FROM memory_entries WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            if row:
                return MemoryEntry(
                    id=row[0],
                    role=row[1],
                    content=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4])
                )
        return None

    def delete(self, entry_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM memory_entries")
            conn.commit()

    def count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memory_entries")
            return cursor.fetchone()[0]

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        query_lower = f"%{query.lower()}%"
        sql = "SELECT id, role, content, timestamp, metadata FROM memory_entries WHERE LOWER(content) LIKE ? ORDER BY timestamp DESC LIMIT ?"
        entries = []
        with self._get_connection() as conn:
            cursor = conn.execute(sql, (query_lower, limit))
            for row in reversed(cursor.fetchall()):
                entries.append(MemoryEntry(
                    id=row[0],
                    role=row[1],
                    content=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4])
                ))
        return entries


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
        max_messages: Optional[int] = None,
        auto_summary: bool = False,
    ) -> None:
        """
        Initialize memory with optional storage backend.

        Args:
            storage: Storage backend (default: InMemoryStorage)
            system_prompt: Optional system prompt to include in history
            max_tokens: Optional max tokens limit for context window
            max_messages: Optional max messages limit before summarization is needed
            auto_summary: Whether agent should automatically summarize when limit is reached
        """
        self.storage = storage or InMemoryStorage()
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.auto_summary = auto_summary

    async def asummarize(self, provider_complete: Callable[[List[Message]], Message], keep_recent: int = 2) -> bool:
        """
        Summarize older messages asynchronously using a provider to save context.
        
        Args:
            provider_complete: An async callable that takes messages and returns a summary message.
            keep_recent: Number of recent messages to retain in full text.
            
        Returns:
            True if summarization occurred, False otherwise.
        """
        history = self.get_history()
        if len(history) <= keep_recent + 1:
            return False
            
        to_summarize = history[:-keep_recent]
        recent = history[-keep_recent:]
        
        prompt = "Summarize the following conversation concisely, focusing on key facts, decisions, and context:\n\n"
        for m in to_summarize:
            prompt += f"{m.role}: {m.content}\n"
            
        summary_msg = await provider_complete([Message.user(prompt)])
        
        self.clear()
        self.add_system_message(f"Summary of prior conversation: {summary_msg.content}")
        for m in recent:
            self.add_message(m)
            
        return True

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
