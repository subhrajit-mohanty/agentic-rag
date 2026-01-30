"""
Redis Cache Module

Async Redis caching with automatic serialization, TTL management,
and cache invalidation patterns for the RAG platform.
"""

import hashlib
import json
import logging
from typing import Any, Optional, TypeVar, Union

import redis.asyncio as redis
from pydantic import BaseModel

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CacheManager:
    """
    Redis cache manager with async support.
    
    Features:
    - Automatic JSON serialization/deserialization
    - Pydantic model support
    - TTL management
    - Cache statistics tracking
    - Key namespacing
    """
    
    NAMESPACE = "rag"
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connected: bool = False
        self._hits: int = 0
        self._misses: int = 0
    
    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("Cache not initialized. Call connect() first.")
        return self._client
    
    async def connect(self) -> None:
        """Initialize Redis connection."""
        if self._connected:
            logger.warning("Cache already connected")
            return
        
        settings = get_settings()
        
        try:
            logger.info(f"Connecting to Redis at {settings.redis.host}:{settings.redis.port}")
            
            self._client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            
            # Verify connection
            await self._client.ping()
            
            self._connected = True
            logger.info("Connected to Redis successfully")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Don't raise - cache is optional, app can work without it
            self._client = None
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            logger.info("Closing Redis connection...")
            await self._client.close()
            self._client = None
            self._connected = False
            logger.info("Redis connection closed")
    
    def _make_key(self, *parts: str) -> str:
        """Generate namespaced cache key."""
        return f"{self.NAMESPACE}:{':'.join(parts)}"
    
    def _hash_key(self, data: str) -> str:
        """Generate hash for cache key from data."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self._client:
            return None
        
        try:
            full_key = self._make_key(key)
            value = await self._client.get(full_key)
            
            if value is not None:
                self._hits += 1
                return json.loads(value)
            
            self._misses += 1
            return None
            
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time-to-live in seconds (uses default if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._client:
            return False
        
        try:
            settings = get_settings()
            ttl = ttl or settings.redis.cache_ttl_seconds
            
            full_key = self._make_key(key)
            serialized = json.dumps(value, default=str)
            
            await self._client.setex(full_key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._client:
            return False
        
        try:
            full_key = self._make_key(key)
            await self._client.delete(full_key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self._client:
            return 0
        
        try:
            full_pattern = self._make_key(pattern)
            keys = []
            async for key in self._client.scan_iter(match=full_pattern):
                keys.append(key)
            
            if keys:
                await self._client.delete(*keys)
            
            return len(keys)
        except Exception as e:
            logger.warning(f"Cache delete pattern error for {pattern}: {e}")
            return 0
    
    async def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get value from cache or compute and cache it.
        
        Args:
            key: Cache key
            factory: Async callable to generate value if not cached
            ttl: Time-to-live in seconds
            
        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value
        
        # Compute value
        if callable(factory):
            import asyncio
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
        
        await self.set(key, value, ttl)
        return value
    
    def generate_query_key(
        self,
        query: str,
        framework: str = "default",
        persona_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate cache key for RAG queries.
        
        Args:
            query: User query
            framework: Agent framework
            persona_id: Optional persona identifier
            **kwargs: Additional parameters to include in key
            
        Returns:
            Unique cache key
        """
        params = {
            "query": query.lower().strip(),
            "framework": framework,
            "persona_id": persona_id or "none",
            **kwargs
        }
        params_str = json.dumps(params, sort_keys=True)
        key_hash = self._hash_key(params_str)
        return f"query:{key_hash}"
    
    async def health_check(self) -> dict:
        """Check Redis connection health."""
        try:
            if not self._client:
                return {"status": "unavailable", "message": "Not connected"}
            
            await self._client.ping()
            info = await self._client.info("memory")
            
            return {
                "status": "healthy",
                "message": "Connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{(self._hits / max(self._hits + self._misses, 1)) * 100:.1f}%"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e)
            }
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": (self._hits / max(total, 1)) * 100
        }


# Global cache manager instance
cache_manager = CacheManager()


async def get_cache() -> CacheManager:
    """Dependency for FastAPI to get cache instance."""
    return cache_manager


# Convenience functions
async def init_cache() -> None:
    """Initialize cache connection."""
    await cache_manager.connect()


async def close_cache() -> None:
    """Close cache connection."""
    await cache_manager.disconnect()
