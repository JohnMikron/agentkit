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
from typing import Any, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


class Cache(ABC, Generic[T]):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get a value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
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
        from cachetools import TTLCache

        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=default_ttl)
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[T]:
        """Get a value from cache."""
        return self._cache.get(key)

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
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
            import redis
        except ImportError:
            raise ImportError("Redis cache requires 'redis' package: pip install redis")

        self._redis = redis.from_url(redis_url)
        self.prefix = prefix
        self.default_ttl = default_ttl

    def _make_redis_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[T]:
        """Get a value from cache."""
        redis_key = self._make_redis_key(key)
        data = self._redis.get(redis_key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return None

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        redis_key = self._make_redis_key(key)
        ttl = ttl or self.default_ttl

        if isinstance(value, (dict, list)):
            data = json.dumps(value)
        else:
            data = str(value)

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
    Semantic cache using embeddings.

    Caches responses based on semantic similarity rather than exact match.
    Requires vector database support.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        default_ttl: int = 3600,
    ) -> None:
        """
        Initialize semantic cache.

        Args:
            similarity_threshold: Minimum similarity for cache hit
            default_ttl: Default TTL in seconds
        """
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self._cache: Dict[str, tuple[list[float], str, float]] = {}
        self._embedder = None

    def _get_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError(
                    "Semantic cache requires 'sentence-transformers': "
                    "pip install sentence-transformers"
                )
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        """Get embedding for text."""
        embedder = self._get_embedder()
        return embedder.encode(text).tolist()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def get(self, key: str) -> Optional[str]:
        """Get a semantically similar cached response."""
        query_embedding = self._embed(key)
        current_time = time.time()

        for _, (cached_embedding, response, expiry) in self._cache.items():
            if current_time > expiry:
                continue

            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity >= self.similarity_threshold:
                return response

        return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Cache a response with its embedding."""
        embedding = self._embed(key)
        cache_key = Cache.make_key(key)
        expiry = time.time() + (ttl or self.default_ttl)
        self._cache[cache_key] = (embedding, value, expiry)

    def delete(self, key: str) -> bool:
        """Delete a cached entry."""
        cache_key = Cache.make_key(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


def cached(
    cache: Cache,
    key_func: Optional[callable] = None,
    ttl: Optional[int] = None,
):
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

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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
