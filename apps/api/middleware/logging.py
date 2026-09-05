from __future__ import annotations

import logging
import time

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger(
    "phymentor.api"
)


def configure_logging(
    log_level: str,
) -> None:
    numeric_level = getattr(
        logging,
        log_level.upper(),
        logging.INFO,
    )

    logging.basicConfig(
        level=numeric_level,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log method, path, status and request latency.

    Raw user questions and file contents are intentionally not logged.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        started_at = time.perf_counter()

        request_id = getattr(
            request.state,
            "request_id",
            "unknown",
        )

        logger.info(
            "request_started request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)

        except Exception:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1000

            logger.exception(
                (
                    "request_failed request_id=%s "
                    "method=%s path=%s latency_ms=%.2f"
                ),
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )

            raise

        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000

        logger.info(
            (
                "request_completed request_id=%s "
                "method=%s path=%s status=%s "
                "latency_ms=%.2f"
            ),
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers[
            "X-Process-Time-Ms"
        ] = f"{duration_ms:.2f}"

        return response