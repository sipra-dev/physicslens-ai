from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from apps.api.middleware import (
    LocalRateLimitMiddleware,
)
from src.cache.redis_client import (
    redis_client,
)


async def _dummy_app(
    scope: dict[str, Any],
    receive: Any,
    send: Any,
) -> None:
    pass


def _make_request(
    *,
    path: str,
    method: str = "GET",
    user_id: str = "test-user",
) -> Request:
    scope = {
        "type": "http",
        "asgi": {
            "version": "3.0",
        },
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(
            "utf-8"
        ),
        "query_string": b"",
        "headers": [
            (
                b"x-user-id",
                user_id.encode(
                    "utf-8"
                ),
            )
        ],
        "client": (
            "127.0.0.1",
            12345,
        ),
        "server": (
            "localhost",
            8000,
        ),
    }

    request = Request(
        scope
    )

    request.state.request_id = (
        "phase7-rate-limit-test"
    )

    return request


async def _call_next(
    request: Request,
) -> Response:
    return Response(
        content="OK",
        status_code=200,
    )


async def main() -> None:
    middleware = (
        LocalRateLimitMiddleware(
            _dummy_app,
            default_requests_per_minute=3,
            upload_requests_per_minute=1,
        )
    )

    original_rate_limit_increment = (
        redis_client.rate_limit_increment
    )

    try:
        # -------------------------------------------------
        # 1. ALLOWED REQUEST
        # -------------------------------------------------

        async def allow_request(
            *,
            key: str,
            window_seconds: int,
        ) -> tuple[int, int]:
            return (
                1,
                55,
            )

        redis_client.rate_limit_increment = (
            allow_request
        )

        allowed_request = (
            _make_request(
                path="/api/v1/chat",
                method="POST",
            )
        )

        allowed_response = (
            await middleware.dispatch(
                allowed_request,
                _call_next,
            )
        )

        allowed_ok = (
            allowed_response.status_code == 200
            and allowed_response.headers.get(
                "X-RateLimit-Limit"
            )
            == "3"
            and allowed_response.headers.get(
                "X-RateLimit-Remaining"
            )
            == "2"
        )

        print(
            "ALLOWED_REQUEST_OK="
            f"{allowed_ok}"
        )

        # -------------------------------------------------
        # 2. LIMIT EXCEEDED -> 429
        # -------------------------------------------------

        async def block_request(
            *,
            key: str,
            window_seconds: int,
        ) -> tuple[int, int]:
            return (
                4,
                42,
            )

        redis_client.rate_limit_increment = (
            block_request
        )

        blocked_request = (
            _make_request(
                path="/api/v1/chat",
                method="POST",
            )
        )

        blocked_response = (
            await middleware.dispatch(
                blocked_request,
                _call_next,
            )
        )

        blocked_ok = (
            blocked_response.status_code
            == 429
        )

        retry_after_ok = (
            blocked_response.headers.get(
                "Retry-After"
            )
            == "42"
        )

        remaining_zero_ok = (
            blocked_response.headers.get(
                "X-RateLimit-Remaining"
            )
            == "0"
        )

        print(
            "BLOCKED_REQUEST_429_OK="
            f"{blocked_ok}"
        )

        print(
            "RETRY_AFTER_OK="
            f"{retry_after_ok}"
        )

        print(
            "REMAINING_ZERO_OK="
            f"{remaining_zero_ok}"
        )

        # -------------------------------------------------
        # 3. UPLOAD-SPECIFIC LIMIT
        # -------------------------------------------------

        async def upload_limit_hit(
            *,
            key: str,
            window_seconds: int,
        ) -> tuple[int, int]:
            return (
                2,
                30,
            )

        redis_client.rate_limit_increment = (
            upload_limit_hit
        )

        upload_request = (
            _make_request(
                path=(
                    "/api/v1/documents/upload"
                ),
                method="POST",
            )
        )

        upload_response = (
            await middleware.dispatch(
                upload_request,
                _call_next,
            )
        )

        upload_limit_ok = (
            upload_response.status_code
            == 429
            and upload_response.headers.get(
                "X-RateLimit-Limit"
            )
            == "1"
        )

        print(
            "UPLOAD_LIMIT_OK="
            f"{upload_limit_ok}"
        )

        # -------------------------------------------------
        # 4. HEALTH ENDPOINT BYPASS
        # -------------------------------------------------

        async def should_not_be_called(
            *,
            key: str,
            window_seconds: int,
        ) -> tuple[int, int]:
            raise RuntimeError(
                "Redis should not be called "
                "for excluded paths."
            )

        redis_client.rate_limit_increment = (
            should_not_be_called
        )

        health_request = (
            _make_request(
                path="/health/live",
            )
        )

        health_response = (
            await middleware.dispatch(
                health_request,
                _call_next,
            )
        )

        health_bypass_ok = (
            health_response.status_code
            == 200
        )

        print(
            "HEALTH_BYPASS_OK="
            f"{health_bypass_ok}"
        )

        # -------------------------------------------------
        # FINAL RESULT
        # -------------------------------------------------

        all_ok = all(
            (
                allowed_ok,
                blocked_ok,
                retry_after_ok,
                remaining_zero_ok,
                upload_limit_ok,
                health_bypass_ok,
            )
        )

        print()

        if all_ok:
            print(
                "PHASE7_RATE_LIMIT_MIDDLEWARE_TEST=PASS"
            )
        else:
            print(
                "PHASE7_RATE_LIMIT_MIDDLEWARE_TEST=FAIL"
            )

    finally:
        redis_client.rate_limit_increment = (
            original_rate_limit_increment
        )


if __name__ == "__main__":
    asyncio.run(
        main()
    )