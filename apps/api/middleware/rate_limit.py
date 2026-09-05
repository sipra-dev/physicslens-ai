from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from typing import Any

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.cache.redis_client import redis_client


class LocalRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis-backed API rate limiter with a small
    in-memory fallback.

    Primary path:
        all FastAPI instances share counters
        through Redis.

    Fallback path:
        if Redis is temporarily unavailable,
        this process still applies a local limit
        instead of removing rate limiting entirely.
    """

    WINDOW_SECONDS = 60

    EXCLUDED_PATHS = {
        "/",
        "/health/live",
        "/health/ready",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
    }

    def __init__(
        self,
        app: Any,
        *,
        default_requests_per_minute: int,
        upload_requests_per_minute: int,
    ) -> None:
        super().__init__(app)

        if default_requests_per_minute <= 0:
            raise ValueError(
                "default_requests_per_minute "
                "must be positive."
            )

        if upload_requests_per_minute <= 0:
            raise ValueError(
                "upload_requests_per_minute "
                "must be positive."
            )

        self.default_limit = (
            default_requests_per_minute
        )

        self.upload_limit = (
            upload_requests_per_minute
        )

        # -------------------------------------------------
        # LOCAL FALLBACK ONLY
        # -------------------------------------------------
        #
        # Redis is the primary shared rate limiter.
        #
        # This memory structure is used only if Redis
        # cannot perform the rate-limit operation.
        # -------------------------------------------------

        self.request_history: dict[
            str,
            deque[float],
        ] = defaultdict(deque)

        self.lock = asyncio.Lock()

    def _get_identifier(
        self,
        request: Request,
    ) -> str:
        """
        Determine who is making the request.

        Prefer the application user identity.
        Fall back to client IP when no user ID exists.
        """

        user_id_header = request.headers.get(
            "X-User-ID"
        )

        if (
            user_id_header
            and user_id_header.strip()
        ):
            return (
                "user:"
                f"{user_id_header.strip()}"
            )

        if request.client is not None:
            return (
                "ip:"
                f"{request.client.host}"
            )

        return "ip:unknown"

    def _get_route_limit(
        self,
        request: Request,
    ) -> int:
        """
        Uploads get their own stricter quota.
        Other API calls use the default quota.
        """

        is_upload_request = (
            request.method.upper() == "POST"
            and request.url.path.endswith(
                "/documents/upload"
            )
        )

        if is_upload_request:
            return self.upload_limit

        return self.default_limit

    @staticmethod
    def _build_rate_key(
        *,
        identifier: str,
        method: str,
        path: str,
    ) -> str:
        """
        Create a compact privacy-safe Redis key.

        Raw user IDs / IP addresses are not placed
        directly inside the Redis key.
        """

        raw_identity = "\x1f".join(
            (
                identifier,
                method.upper(),
                path,
            )
        ).encode(
            "utf-8"
        )

        digest = hashlib.sha256(
            raw_identity
        ).hexdigest()

        return (
            "phymentor:"
            "rate-limit:"
            f"{digest}"
        )

    async def _check_local_fallback(
        self,
        *,
        rate_key: str,
        route_limit: int,
    ) -> tuple[
        bool,
        int,
        int,
    ]:
        """
        Local fallback when Redis is unavailable.

        Returns:
            allowed
            remaining
            retry_after_seconds
        """

        current_time = (
            time.monotonic()
        )

        window_start = (
            current_time
            - self.WINDOW_SECONDS
        )

        async with self.lock:
            timestamps = (
                self.request_history[
                    rate_key
                ]
            )

            while (
                timestamps
                and timestamps[0]
                <= window_start
            ):
                timestamps.popleft()

            if (
                len(timestamps)
                >= route_limit
            ):
                retry_after = max(
                    1,
                    int(
                        self.WINDOW_SECONDS
                        - (
                            current_time
                            - timestamps[0]
                        )
                    ),
                )

                return (
                    False,
                    0,
                    retry_after,
                )

            timestamps.append(
                current_time
            )

            remaining = max(
                0,
                route_limit
                - len(timestamps),
            )

            return (
                True,
                remaining,
                self.WINDOW_SECONDS,
            )

    @staticmethod
    def _rate_limit_response(
        *,
        request: Request,
        route_limit: int,
        retry_after: int,
    ) -> JSONResponse:
        """
        Build the standard HTTP 429 response.
        """

        request_id = getattr(
            request.state,
            "request_id",
            None,
        )

        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    "Too many requests. "
                    "Please try again shortly."
                ),
                "request_id": request_id,
            },
            headers={
                "Retry-After": str(
                    max(
                        1,
                        retry_after,
                    )
                ),
                "X-RateLimit-Limit": str(
                    route_limit
                ),
                "X-RateLimit-Remaining": "0",
            },
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:

        # Browser CORS preflight requests should
        # never consume the user's API quota.
        if request.method.upper() == "OPTIONS":
            return await call_next(
                request
            )

        if (
            request.url.path
            in self.EXCLUDED_PATHS
        ):
            return await call_next(
                request
            )

        identifier = (
            self._get_identifier(
                request
            )
        )

        route_limit = (
            self._get_route_limit(
                request
            )
        )

        rate_key = (
            self._build_rate_key(
                identifier=identifier,
                method=request.method,
                path=request.url.path,
            )
        )

        # -------------------------------------------------
        # PRIMARY: SHARED REDIS RATE LIMIT
        # -------------------------------------------------

        redis_result = (
            await redis_client
            .rate_limit_increment(
                key=rate_key,
                window_seconds=(
                    self.WINDOW_SECONDS
                ),
            )
        )

        if redis_result is not None:
            (
                current_count,
                ttl_seconds,
            ) = redis_result

            if (
                current_count
                > route_limit
            ):
                return (
                    self._rate_limit_response(
                        request=request,
                        route_limit=(
                            route_limit
                        ),
                        retry_after=(
                            ttl_seconds
                        ),
                    )
                )

            remaining = max(
                0,
                route_limit
                - current_count,
            )

        # -------------------------------------------------
        # FALLBACK: LOCAL PROCESS MEMORY
        # -------------------------------------------------
        #
        # Redis failure should not silently remove
        # all protection.
        # -------------------------------------------------

        else:
            (
                allowed,
                remaining,
                retry_after,
            ) = await self._check_local_fallback(
                rate_key=rate_key,
                route_limit=route_limit,
            )

            if not allowed:
                return (
                    self._rate_limit_response(
                        request=request,
                        route_limit=(
                            route_limit
                        ),
                        retry_after=(
                            retry_after
                        ),
                    )
                )

        # -------------------------------------------------
        # REQUEST ALLOWED
        # -------------------------------------------------

        response = await call_next(
            request
        )

        response.headers[
            "X-RateLimit-Limit"
        ] = str(
            route_limit
        )

        response.headers[
            "X-RateLimit-Remaining"
        ] = str(
            remaining
        )

        return response