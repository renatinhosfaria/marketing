"""
Rate Limiting Middleware for FastAPI.

Implements token bucket algorithm with configurable limits per client.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
import threading

from shared.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10  # Max burst requests
    enabled: bool = True
    whitelist_paths: list[str] = None  # Paths to skip rate limiting

    def __post_init__(self):
        if self.whitelist_paths is None:
            self.whitelist_paths = ["/health", "/api/health", "/docs", "/openapi.json"]


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (burst limit)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Returns:
            True if tokens were consumed, False if rate limited
        """
        with self.lock:
            now = time.time()

            # Add tokens based on time elapsed
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_retry_after(self) -> int:
        """Get seconds until a token is available."""
        with self.lock:
            if self.tokens >= 1:
                return 0
            needed = 1 - self.tokens
            return int(needed / self.rate) + 1


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.

    Features:
    - Per-client rate limiting (by IP or API key)
    - Configurable limits per minute/hour
    - Burst handling
    - Whitelist for health checks and docs
    - Thread-safe
    """

    def __init__(
        self,
        app,
        config: Optional[RateLimitConfig] = None,
        get_client_id: Optional[Callable[[Request], str]] = None,
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: ASGI application
            config: Rate limit configuration
            get_client_id: Optional function to extract client identifier
        """
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.get_client_id = get_client_id or self._default_client_id

        # Per-client buckets
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                rate=self.config.requests_per_minute / 60.0,
                capacity=self.config.burst_limit
            )
        )
        self._bucket_lock = threading.Lock()

        # Hourly tracking
        self._hourly_counts: dict[str, int] = defaultdict(int)
        self._hourly_reset: dict[str, float] = {}

        # Cleanup old buckets periodically
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

        logger.info(
            "Rate limiter initialized",
            requests_per_minute=self.config.requests_per_minute,
            requests_per_hour=self.config.requests_per_hour,
            burst_limit=self.config.burst_limit,
            enabled=self.config.enabled
        )

    def _default_client_id(self, request: Request) -> str:
        """Extract client ID from request (default: IP address)."""
        # Check for API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:8]}"  # Use first 8 chars as identifier

        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        client = request.client
        return client.host if client else "unknown"

    def _is_whitelisted(self, path: str) -> bool:
        """Check if path is whitelisted from rate limiting."""
        return any(path.startswith(wp) for wp in self.config.whitelist_paths)

    def _check_hourly_limit(self, client_id: str) -> tuple[bool, int]:
        """
        Check hourly rate limit.

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()

        # Reset hourly counter if needed
        if client_id not in self._hourly_reset or now - self._hourly_reset[client_id] >= 3600:
            self._hourly_counts[client_id] = 0
            self._hourly_reset[client_id] = now

        if self._hourly_counts[client_id] >= self.config.requests_per_hour:
            retry_after = int(3600 - (now - self._hourly_reset[client_id]))
            return False, max(1, retry_after)

        self._hourly_counts[client_id] += 1
        return True, 0

    def _cleanup_old_buckets(self):
        """Remove buckets for clients that haven't been seen recently."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        with self._bucket_lock:
            stale_clients = [
                client_id for client_id, bucket in self._buckets.items()
                if now - bucket.last_update > 3600  # 1 hour inactive
            ]

            for client_id in stale_clients:
                del self._buckets[client_id]
                if client_id in self._hourly_counts:
                    del self._hourly_counts[client_id]
                if client_id in self._hourly_reset:
                    del self._hourly_reset[client_id]

            self._last_cleanup = now

            if stale_clients:
                logger.debug(f"Cleaned up {len(stale_clients)} stale rate limit buckets")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip if disabled
        if not self.config.enabled:
            return await call_next(request)

        # Skip whitelisted paths
        if self._is_whitelisted(request.url.path):
            return await call_next(request)

        # Get client identifier
        client_id = self.get_client_id(request)

        # Periodic cleanup
        self._cleanup_old_buckets()

        # Check hourly limit first
        hourly_allowed, hourly_retry = self._check_hourly_limit(client_id)
        if not hourly_allowed:
            logger.warning(
                "Hourly rate limit exceeded",
                client_id=client_id,
                path=request.url.path
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "error": "too_many_requests",
                    "retry_after": hourly_retry,
                    "limit_type": "hourly"
                },
                headers={"Retry-After": str(hourly_retry)}
            )

        # Check per-minute limit (token bucket)
        bucket = self._buckets[client_id]
        if not bucket.consume():
            retry_after = bucket.get_retry_after()
            logger.warning(
                "Per-minute rate limit exceeded",
                client_id=client_id,
                path=request.url.path
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "error": "too_many_requests",
                    "retry_after": retry_after,
                    "limit_type": "per_minute"
                },
                headers={"Retry-After": str(retry_after)}
            )

        # Add rate limit headers to response
        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))

        return response

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._bucket_lock:
            return {
                "active_clients": len(self._buckets),
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "burst_limit": self.config.burst_limit,
                    "enabled": self.config.enabled,
                },
                "clients": {
                    client_id: {
                        "tokens_available": int(bucket.tokens),
                        "hourly_count": self._hourly_counts.get(client_id, 0),
                    }
                    for client_id, bucket in list(self._buckets.items())[:10]  # Limit to first 10
                }
            }
