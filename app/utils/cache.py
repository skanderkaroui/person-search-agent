import json
import time
import hashlib
from typing import Any, Dict, Optional
import redis
from app.core.config import settings

class Cache:
    """Simple cache implementation for web scraping results."""

    def __init__(self, use_redis: bool = True, expiration: int = 3600):
        """
        Initialize the cache.

        Args:
            use_redis: Whether to use Redis for caching.
            expiration: Cache expiration time in seconds (default: 1 hour).
        """
        self.use_redis = use_redis
        self.expiration = expiration
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize Redis connection if enabled
        if self.use_redis:
            try:
                self.redis = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    decode_responses=True
                )
                # Test connection
                self.redis.ping()
                print("Redis cache initialized successfully")
            except Exception as e:
                print(f"Error connecting to Redis: {str(e)}")
                self.use_redis = False
                print("Falling back to in-memory cache")

    def _generate_key(self, source: str, query: str) -> str:
        """
        Generate a cache key from source and query.

        Args:
            source: Source name (e.g., 'Wikipedia', 'Twitter').
            query: Search query.

        Returns:
            str: Cache key.
        """
        # Create a hash of the query to use as the key
        key = f"{source}:{hashlib.md5(query.encode()).hexdigest()}"
        return key

    def get(self, source: str, query: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            source: Source name.
            query: Search query.

        Returns:
            Optional[Any]: Cached value or None if not found.
        """
        key = self._generate_key(source, query)

        # Try Redis first if enabled
        if self.use_redis:
            try:
                cached_data = self.redis.get(key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                print(f"Error retrieving from Redis: {str(e)}")

        # Fall back to memory cache
        if key in self.memory_cache:
            cache_entry = self.memory_cache[key]
            # Check if the entry is expired
            if time.time() - cache_entry["timestamp"] < self.expiration:
                return cache_entry["data"]
            else:
                # Remove expired entry
                del self.memory_cache[key]

        return None

    def set(self, source: str, query: str, data: Any) -> None:
        """
        Set a value in the cache.

        Args:
            source: Source name.
            query: Search query.
            data: Data to cache.
        """
        key = self._generate_key(source, query)

        # Try Redis first if enabled
        if self.use_redis:
            try:
                self.redis.setex(
                    key,
                    self.expiration,
                    json.dumps(data, default=str)
                )
            except Exception as e:
                print(f"Error setting in Redis: {str(e)}")

        # Also set in memory cache as fallback
        self.memory_cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

    def clear(self, source: Optional[str] = None) -> None:
        """
        Clear the cache.

        Args:
            source: Optional source name to clear only that source's cache.
        """
        if source:
            # Clear only the specified source
            if self.use_redis:
                try:
                    for key in self.redis.keys(f"{source}:*"):
                        self.redis.delete(key)
                except Exception as e:
                    print(f"Error clearing Redis cache for {source}: {str(e)}")

            # Clear memory cache for the source
            keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(f"{source}:")]
            for key in keys_to_delete:
                del self.memory_cache[key]
        else:
            # Clear all cache
            if self.use_redis:
                try:
                    self.redis.flushdb()
                except Exception as e:
                    print(f"Error clearing Redis cache: {str(e)}")

            # Clear memory cache
            self.memory_cache.clear()

# Create a global cache instance
cache = Cache()