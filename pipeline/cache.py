"""
pipeline/cache.py
=================
Redis-backed cache layer for RAG results.

Provides a simple get/set interface with JSON serialisation and configurable TTL.
Connection errors are caught and treated as cache misses so the pipeline
continues to work even if Redis is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Optional

import redis
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
DEFAULT_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

# Namespace prefix for all RAG cache keys so we don't collide with other apps
KEY_PREFIX = "spark_scholar:"


class ResultCache:
    """
    Redis-backed result cache for RAG pipeline outputs.

    All values are JSON-serialised. Failures (connection errors, decode errors)
    are logged as warnings and treated as cache misses so the pipeline degrades
    gracefully when Redis is unavailable.
    """

    def __init__(self, redis_url: Optional[str] = None, default_ttl: Optional[int] = None):
        """
        Parameters
        ----------
        redis_url : str, optional
            Redis connection URL. Defaults to REDIS_URL env var.
        default_ttl : int, optional
            Default TTL in seconds. Defaults to CACHE_TTL_SECONDS env var.
        """
        self._url = redis_url or REDIS_URL
        self._ttl = default_ttl if default_ttl is not None else DEFAULT_TTL
        self._client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self) -> None:
        """Attempt to create a Redis connection pool."""
        try:
            self._client = redis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
            )
            # Ping to verify connectivity
            self._client.ping()
            logger.info("ResultCache: connected to Redis at %s", self._url)
        except (redis.ConnectionError, redis.TimeoutError, Exception) as exc:
            logger.warning(
                "ResultCache: could not connect to Redis at %s — caching disabled (%s)",
                self._url,
                exc,
            )
            self._client = None

    @staticmethod
    def make_key(query: str, collections: list[str], extra: str = "") -> str:
        """
        Build a deterministic cache key from the query and target collections.

        The key is a hex digest of (query + sorted collections + extra) so that
        the same logical request always maps to the same key regardless of
        collection order.

        Parameters
        ----------
        query : str
            The search query string.
        collections : list[str]
            List of collection names that were searched.
        extra : str
            Optional extra discriminator (e.g. filter parameters).

        Returns
        -------
        str
            Redis key with the KEY_PREFIX namespace.
        """
        canonical = json.dumps(
            {"q": query.strip().lower(), "c": sorted(collections), "x": extra},
            sort_keys=True,
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:32]
        return f"{KEY_PREFIX}{digest}"

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.

        Parameters
        ----------
        key : str
            Cache key (from make_key).

        Returns
        -------
        Any or None
            Deserialised Python object, or None on miss or error.
        """
        if self._client is None:
            return None

        try:
            raw = self._client.get(key)
            if raw is None:
                logger.debug("Cache MISS: %s", key)
                return None
            value = json.loads(raw)
            logger.debug("Cache HIT: %s", key)
            return value
        except redis.ConnectionError as exc:
            logger.warning("Cache get failed (connection): %s", exc)
            self._client = None  # disable further attempts this session
            return None
        except json.JSONDecodeError as exc:
            logger.warning("Cache get failed (JSON decode): %s", exc)
            return None
        except Exception as exc:
            logger.warning("Cache get failed (unexpected): %s", exc)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store a value in the cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            JSON-serialisable Python object.
        ttl : int, optional
            Time-to-live in seconds. Defaults to self._ttl.

        Returns
        -------
        bool
            True if stored successfully, False otherwise.
        """
        if self._client is None:
            return False

        effective_ttl = ttl if ttl is not None else self._ttl

        try:
            raw = json.dumps(value, ensure_ascii=False)
            self._client.setex(name=key, time=effective_ttl, value=raw)
            logger.debug("Cache SET: %s (TTL=%ds)", key, effective_ttl)
            return True
        except redis.ConnectionError as exc:
            logger.warning("Cache set failed (connection): %s", exc)
            self._client = None
            return False
        except (TypeError, ValueError) as exc:
            logger.warning("Cache set failed (serialisation): %s", exc)
            return False
        except Exception as exc:
            logger.warning("Cache set failed (unexpected): %s", exc)
            return False

    def delete(self, key: str) -> bool:
        """Delete a single key from the cache."""
        if self._client is None:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception as exc:
            logger.warning("Cache delete failed: %s", exc)
            return False

    def flush_namespace(self) -> int:
        """Delete all keys in the spark_scholar namespace. Returns count deleted."""
        if self._client is None:
            return 0
        try:
            pattern = f"{KEY_PREFIX}*"
            keys = list(self._client.scan_iter(pattern))
            if keys:
                self._client.delete(*keys)
            logger.info("Cache flush: deleted %d keys", len(keys))
            return len(keys)
        except Exception as exc:
            logger.warning("Cache flush failed: %s", exc)
            return 0

    def is_available(self) -> bool:
        """Return True if Redis is reachable."""
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    def stats(self) -> dict:
        """Return basic stats from the Redis INFO command."""
        if self._client is None:
            return {"available": False}
        try:
            info = self._client.info(section="stats")
            return {
                "available": True,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "connected_clients": self._client.info("clients").get("connected_clients", 0),
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}


# Singleton instance for use across the pipeline
_cache_instance: Optional[ResultCache] = None


def get_cache() -> ResultCache:
    """Return the singleton ResultCache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResultCache()
    return _cache_instance


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)

    cache = ResultCache()
    print(f"Redis available: {cache.is_available()}")
    print(f"Stats: {cache.stats()}")

    key = cache.make_key("what is quantum entanglement", ["arxiv-quantph-grqc"])
    print(f"Key: {key}")

    test_value = {
        "response": "Quantum entanglement is...",
        "sources": ["2301.00001", "2302.00002"],
    }

    success = cache.set(key, test_value, ttl=60)
    print(f"Set success: {success}")

    retrieved = cache.get(key)
    print(f"Retrieved: {retrieved}")

    print(f"Match: {retrieved == test_value}")

    miss = cache.get("spark_scholar:nonexistent_key_abc123")
    print(f"Miss result: {miss}")
