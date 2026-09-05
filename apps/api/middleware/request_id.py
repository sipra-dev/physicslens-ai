from __future__ import annotations

from uuid import uuid4

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attach a unique request ID to every incoming request.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        incoming_request_id = request.headers.get(
            "X-Request-ID"
        )

        request_id = (
            incoming_request_id.strip()
            if incoming_request_id
            else uuid4().hex
        )

        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        return response