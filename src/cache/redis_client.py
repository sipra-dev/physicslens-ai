from __future__ import annotations

from typing import Any

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.exceptions import RedisError

from src.config import settings


class RedisClient:
    """
    Shared asynchronous Redis client.

    Used by async application infrastructure such as:
    - cache invalidation
    - rate limiting
    - health checks

    The connection URL comes from the central
    application settings.
    """

    _RATE_LIMIT_SCRIPT = """
    local current = redis.call(
        "INCR",
        KEYS[1]
    )

    local ttl = redis.call(
        "TTL",
        KEYS[1]
    )

    if current == 1 or ttl < 0 then
        redis.call(
            "EXPIRE",
            KEYS[1],
            ARGV[1]
        )

        ttl = redis.call(
            "TTL",
            KEYS[1]
        )
    end

    return {
        current,
        ttl
    }
    """

    def __init__(
        self,
    ) -> None:
        self.redis_url = (
            settings.redis_url
            or "redis://localhost:6379/0"
        )

        # Pool of reusable Redis connections.
        self.pool = (
            ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=50,
                socket_connect_timeout=2,
                socket_timeout=3,
                health_check_interval=30,
            )
        )

        self.client = redis.Redis(
            connection_pool=self.pool
        )

    async def health_check(
        self,
    ) -> bool:
        """
        Check whether Redis is reachable.
        """

        try:
            return bool(
                await self.client.ping()
            )

        except RedisError:
            return False

    async def get(
        self,
        key: str,
    ) -> str | None:
        """
        Read a string value from Redis.
        """

        try:
            return await self.client.get(
                key
            )

        except RedisError:
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Store a string value in Redis,
        optionally with an expiry.
        """

        try:
            await self.client.set(
                key,
                value,
                ex=ttl_seconds,
            )

            return True

        except RedisError:
            return False

    async def delete(
        self,
        key: str,
    ) -> bool:
        """
        Delete a Redis key.
        """

        try:
            await self.client.delete(
                key
            )

            return True

        except RedisError:
            return False

    async def increment(
        self,
        key: str,
    ) -> int | None:
        """
        Atomically increment a Redis integer.

        Generic helper retained for callers that
        only need an increment operation.
        """

        try:
            value = await self.client.incr(
                key
            )

            return int(
                value
            )

        except (
            RedisError,
            TypeError,
            ValueError,
        ):
            return None

    async def expire(
        self,
        key: str,
        ttl_seconds: int,
    ) -> bool:
        """
        Give an existing key an expiry time.
        """

        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be positive."
            )

        try:
            return bool(
                await self.client.expire(
                    key,
                    ttl_seconds,
                )
            )

        except RedisError:
            return False

    async def rate_limit_increment(
        self,
        *,
        key: str,
        window_seconds: int,
    ) -> tuple[int, int] | None:
        """
        Atomically increment one fixed-window
        rate-limit counter and guarantee that
        the counter has an expiry.

        Returns:
            (current_count, ttl_seconds)

        Example:
            (7, 42)

        means:
            - this is request number 7
            - the current window resets in
              approximately 42 seconds

        None means Redis could not perform the
        operation.
        """

        normalized_key = (
            key.strip()
        )

        if not normalized_key:
            raise ValueError(
                "Rate-limit key cannot be empty."
            )

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be positive."
            )

        try:
            result: Any = (
                await self.client.eval(
                    self._RATE_LIMIT_SCRIPT,
                    1,
                    normalized_key,
                    window_seconds,
                )
            )

            if not isinstance(
                result,
                (
                    list,
                    tuple,
                ),
            ):
                return None

            if len(result) != 2:
                return None

            current_count = int(
                result[0]
            )

            ttl_seconds = int(
                result[1]
            )

            # Defensive fallback for an expiry
            # occurring exactly while the result
            # is being returned.
            if ttl_seconds <= 0:
                ttl_seconds = 1

            return (
                current_count,
                ttl_seconds,
            )

        except (
            RedisError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            # Redis asyncio connections are tied to the
            # event loop that owns their transport.
            #
            # During test isolation, worker reloads, or a
            # broken Redis transport, redis-py can surface
            # lifecycle failures as RuntimeError /
            # AttributeError rather than RedisError.
            #
            # Rate limiting is intentionally fail-safe:
            # returning None makes the middleware use its
            # in-process fallback instead of returning 500.
            return None

    async def close(
        self,
    ) -> None:
        """
        Close Redis connections cleanly during
        application shutdown.
        """

        await self.client.aclose()
        await self.pool.aclose()


# One shared async Redis client for the
# whole application.
redis_client = RedisClient()
