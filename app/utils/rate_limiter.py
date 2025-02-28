import time
from typing import Dict, Tuple
import redis
from app.core.config import settings

class RateLimiter:
    """Simple rate limiter to prevent abuse of web scraping."""

    def __init__(self, use_redis: bool = True, max_requests: int = 60, time_window: int = 60):
        """
        Initialize the rate limiter.

        Args:
            use_redis: Whether to use Redis for rate limiting.
            max_requests: Maximum number of requests allowed in the time window.
            time_window: Time window in seconds.
        """
        self.use_redis = use_redis
        self.max_requests = max_requests
        self.time_window = time_window
        self.memory_store: Dict[str, Tuple[int, float]] = {}  # {ip: (count, timestamp)}

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
                print("Redis rate limiter initialized successfully")
            except Exception as e:
                print(f"Error connecting to Redis for rate limiting: {str(e)}")
                self.use_redis = False
                print("Falling back to in-memory rate limiting")

    async def is_allowed(self, ip: str) -> bool:
        """
        Check if a request from the given IP is allowed.

        Args:
            ip: IP address of the requester.

        Returns:
            bool: True if the request is allowed, False otherwise.
        """
        # Try Redis first if enabled
        if self.use_redis:
            try:
                # Get the current count for this IP
                key = f"rate_limit:{ip}"
                count = self.redis.get(key)

                if count is None:
                    # First request from this IP
                    self.redis.setex(key, self.time_window, 1)
                    return True

                count = int(count)
                if count < self.max_requests:
                    # Increment the count
                    self.redis.incr(key)
                    return True

                return False
            except Exception as e:
                print(f"Error checking Redis rate limit: {str(e)}")
                # Fall back to memory store

        # Use memory store
        current_time = time.time()

        if ip in self.memory_store:
            count, timestamp = self.memory_store[ip]

            # Check if the time window has passed
            if current_time - timestamp > self.time_window:
                # Reset the counter
                self.memory_store[ip] = (1, current_time)
                return True

            if count < self.max_requests:
                # Increment the count
                self.memory_store[ip] = (count + 1, timestamp)
                return True

            return False
        else:
            # First request from this IP
            self.memory_store[ip] = (1, current_time)
            return True

    def get_remaining(self, ip: str) -> int:
        """
        Get the number of remaining requests allowed for the given IP.

        Args:
            ip: IP address of the requester.

        Returns:
            int: Number of remaining requests allowed.
        """
        # Try Redis first if enabled
        if self.use_redis:
            try:
                # Get the current count for this IP
                key = f"rate_limit:{ip}"
                count = self.redis.get(key)

                if count is None:
                    return self.max_requests

                count = int(count)
                return max(0, self.max_requests - count)
            except Exception as e:
                print(f"Error getting Redis rate limit: {str(e)}")
                # Fall back to memory store

        # Use memory store
        if ip in self.memory_store:
            count, timestamp = self.memory_store[ip]

            # Check if the time window has passed
            if time.time() - timestamp > self.time_window:
                return self.max_requests

            return max(0, self.max_requests - count)
        else:
            return self.max_requests

# Create a global rate limiter instance
rate_limiter = RateLimiter()