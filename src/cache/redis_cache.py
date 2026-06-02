"""Caché Redis con soporte de similitud semántica para consultas RAG."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


class SemanticCache:
    """Cache with exact match + semantic similarity fallback.

    Features
    --------
    - Stores query results in Redis with a configurable TTL (default 1 hour).
    - Exact match by SHA-256 hash key (like the previous LRU cache).
    - Semantic match: encode query -> compare cosine similarity with cached
      query embeddings -> if similarity >= threshold, return cached result.
    - Falls back gracefully to in-memory ``OrderedDict`` when Redis is
      unavailable.
    - Uses JSON serialization for cached values.
    - Thread-safe (relies on Redis atomicity + GIL for in-memory dict).
    """

    def __init__(
        self,
        embedder=None,
        redis_url: str | None = None,
        ttl: int = 3600,
        max_memory_entries: int = 128,
        similarity_threshold: float = 0.95,
    ):
        self._embedder = embedder
        self._ttl = ttl
        self._similarity_threshold = similarity_threshold
        self._redis = None
        self._memory_cache: OrderedDict[str, dict] = OrderedDict()
        self._max_memory = max_memory_entries
        # Store embeddings for semantic lookup (in memory for fast comparison)
        self._query_embeddings: dict[str, np.ndarray] = {}

        redis_url = redis_url or os.getenv("REDIS_URL", "")
        if redis_url:
            try:
                import redis  # noqa: PLC0415

                self._redis = redis.Redis.from_url(
                    redis_url, decode_responses=True
                )
                self._redis.ping()
                logger.info("redis_cache_conectado", url=redis_url)
            except Exception as e:  # noqa: BLE001
                logger.warning("redis_no_disponible", error=str(e))
                self._redis = None

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(
        pregunta: str, top_k: int, alpha: float, reranking: bool
    ) -> str:
        """Build a deterministic cache key from query parameters."""
        raw = f"{pregunta.strip().lower()}|{top_k}|{alpha:.2f}|{reranking}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(self, key: str, query_text: str | None = None) -> dict | None:
        """Try exact match first, then semantic similarity.

        Parameters
        ----------
        key:
            The deterministic hash key produced by :meth:`cache_key`.
        query_text:
            Original query string used for semantic similarity lookup when
            the exact key is not found.
        """
        # 1. Exact match in Redis
        if self._redis:
            try:
                cached = self._redis.get(f"rag:query:{key}")
                if cached:
                    logger.info("cache_hit", tipo="redis_exact", key=key)
                    return json.loads(cached)
            except Exception as e:  # noqa: BLE001
                logger.warning("redis_get_error", error=str(e))

        # 2. Exact match in memory
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            logger.info("cache_hit", tipo="memory_exact", key=key)
            return self._memory_cache[key]

        # 3. Semantic similarity (if embedder available and query text given)
        if query_text and self._embedder and self._query_embeddings:
            try:
                query_emb = self._embedder.encode([query_text])[0]
                best_key, best_sim = None, 0.0
                for cached_key, cached_emb in self._query_embeddings.items():
                    sim = float(np.dot(query_emb, cached_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_key = cached_key

                if best_key and best_sim >= self._similarity_threshold:
                    result = self._memory_cache.get(best_key)
                    if result:
                        logger.info(
                            "cache_hit",
                            tipo="semantic",
                            similitud=round(best_sim, 4),
                            key=best_key,
                        )
                        return result
            except Exception as e:  # noqa: BLE001
                logger.warning("semantic_cache_error", error=str(e))

        return None

    # ------------------------------------------------------------------
    # Put
    # ------------------------------------------------------------------

    def put(
        self, key: str, value: dict, query_text: str | None = None
    ) -> None:
        """Store result in both Redis and memory.

        Parameters
        ----------
        key:
            The deterministic hash key.
        value:
            The response dict to cache.
        query_text:
            Original query string — its embedding is stored for future
            semantic comparisons.
        """
        # Redis
        if self._redis:
            try:
                self._redis.setex(
                    f"rag:query:{key}",
                    self._ttl,
                    json.dumps(value, ensure_ascii=False, default=str),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("redis_put_error", error=str(e))

        # Memory
        self._memory_cache[key] = value
        if len(self._memory_cache) > self._max_memory:
            evicted_key, _ = self._memory_cache.popitem(last=False)
            self._query_embeddings.pop(evicted_key, None)

        # Store embedding for semantic lookup
        if query_text and self._embedder:
            try:
                self._query_embeddings[key] = self._embedder.encode(
                    [query_text]
                )[0]
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Clear all caches (memory + Redis ``rag:query:*`` keys)."""
        self._memory_cache.clear()
        self._query_embeddings.clear()
        if self._redis:
            try:
                cursor: int | str = 0
                while True:
                    cursor, keys = self._redis.scan(
                        cursor, match="rag:query:*", count=100
                    )
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
                logger.info("redis_cache_invalidado")
            except Exception as e:  # noqa: BLE001
                logger.warning("redis_invalidate_error", error=str(e))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of entries in the in-memory cache."""
        return len(self._memory_cache)

    @property
    def redis_connected(self) -> bool:
        """Check whether the Redis connection is alive."""
        if self._redis:
            try:
                self._redis.ping()
                return True
            except Exception:  # noqa: BLE001
                return False
        return False
