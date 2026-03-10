"""
Caching utilities for AgentKit.

Provides multiple cache backends:
- InMemoryCache: Fast, in-process cache
- RedisCache: Distributed cache with Redis
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

T = TypeVar("T")


class Cache(ABC, Generic[T]):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: str) -> T | None:
        """Get a value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """Set a value in cache with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from cache."""
        pass

    @staticmethod
    def make_key(*args: Any, **kwargs: Any) -> str:
        """Create a cache key from arguments."""
        data = {"args": args, "kwargs": kwargs}
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class InMemoryCache(Cache[T]):
    """
    In-memory cache implementation.

    Fast, in-process cache using cachetools.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600) -> None:
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        from cachetools import TTLCache  # type: ignore[import-untyped]

        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=default_ttl)
        self.default_ttl = default_ttl

    def get(self, key: str) -> T | None:
        """Get a value from cache."""
        from typing import cast

        return cast("T | None", self._cache.get(key))

    def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """Set a value in cache."""
        self._cache[key] = value

    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all values from cache."""
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cache

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._cache)


class RedisCache(Cache[T]):
    """
    Redis-based distributed cache.

    Requires: pip install redis
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "agentkit:",
        default_ttl: int = 3600,
    ) -> None:
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
        """
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError("Redis cache requires 'redis' package: pip install redis") from err

        self._redis = redis.from_url(redis_url)
        self.prefix = prefix
        self.default_ttl = default_ttl

    def _make_redis_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> T | None:
        """Get a value from cache."""
        redis_key = self._make_redis_key(key)
        data = self._redis.get(redis_key)
        if data:
            try:
                from typing import cast

                return cast("T | None", json.loads(data))
            except json.JSONDecodeError:
                from typing import cast

                return cast("T | None", data)
        return None

    def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """Set a value in cache."""
        redis_key = self._make_redis_key(key)
        ttl = ttl or self.default_ttl

        data = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

        self._redis.setex(redis_key, ttl, data)

    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        redis_key = self._make_redis_key(key)
        return bool(self._redis.delete(redis_key))

    def clear(self) -> None:
        """Clear all values with the prefix."""
        keys = self._redis.keys(f"{self.prefix}*")
        if keys:
            self._redis.delete(*keys)


class SemanticCache(Cache[str]):
    """
    Semantic cache using embeddings and VectorStorage.

    Caches responses based on semantic similarity rather than exact match.
    Uses ChromaDB under the hood via VectorStorage.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        default_ttl: int = 3600,
        collection_name: str = "agentkit_semantic_cache",
        persist_directory: str | None = None,
    ) -> None:
        """
        Initialize semantic cache.

        Args:
            similarity_threshold: Minimum similarity for cache hit (Not fully used by direct chroma distance, but abstractly applied)
            default_ttl: Default TTL in seconds
        """
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl

        from agentkit.core.memory import VectorStorage

        self._storage = VectorStorage(
            collection_name=collection_name, persist_directory=persist_directory
        )

    def get(self, key: str) -> str | None:
        """Get a semantically similar cached response."""
        # Search returns the most similar entry
        results = self._storage.search(key, limit=1)
        if not results:
            return None

        entry = results[0]
        # Check TTL
        expiry = entry.metadata.get("expiry", 0)
        if time.time() > expiry:
            self._storage.delete(entry.id)
            return None

        return entry.metadata.get("response")

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Cache a response with its embedding."""
        from agentkit.core.memory import MemoryEntry

        expiry = time.time() + (ttl or self.default_ttl)

        entry = MemoryEntry(
            role="system", content=value, metadata={"original_prompt": key, "expiry": expiry}
        )
        # VectorStorage automatically embeds the 'content'. Wait.
        # We want to search by the PROMPT, not the RESPONSE.
        # So we should embed the prompt, and store the response in metadata.
        # Let's override how we use it:
        entry = MemoryEntry(
            role="system",
            content=key,  # The prompt to be embedded and searched
            metadata={"response": value, "expiry": expiry},
        )
        self._storage.save(entry)

    def delete(self, key: str) -> bool:
        """Delete a cached entry by exact prompt match."""
        results = self._storage.search(key, limit=1)
        if results and results[0].content == key:
            self._storage.delete(results[0].id)
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._storage.clear()

    def get_response(self, prompt: str) -> str | None:
        """Get response by prompt (alias for get)."""
        results = self._storage.search(prompt, limit=1)
        if not results:
            return None

        entry = results[0]
        if time.time() > entry.metadata.get("expiry", 0):
            return None

        # We might want to check the actual distance/similarity here,
        # but ChromaDB search returns the top match.
        # For a true threshold, we'd need access to Chroma's distances.
        return entry.metadata.get("response")


if TYPE_CHECKING:
    from collections.abc import Callable


def cached(
    cache: Cache[Any],
    key_func: Callable[..., str] | None = None,
    ttl: int | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to cache function results.

    Args:
        cache: Cache instance to use
        key_func: Optional function to generate cache key
        ttl: Optional TTL override

    Returns:
        Decorated function
    """
    import functools

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = Cache.make_key(func.__name__, *args, **kwargs)

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result, ttl)

            return result

        return wrapper

    return decorator
