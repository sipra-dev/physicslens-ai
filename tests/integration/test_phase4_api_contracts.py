from __future__ import annotations

import sys
import unittest
from pathlib import Path

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from apps.api.middleware import (
    LocalRateLimitMiddleware,
    RequestIDMiddleware,
    register_exception_handlers,
)
from src.retrieval.service import (
    RetrievalServiceError,
)


class ValidationPayload(BaseModel):
    value: int = Field(
        gt=0
    )


def build_test_app() -> FastAPI:
    app = FastAPI(
        debug=False
    )

    @app.get(
        "/v1/test/ping"
    )
    async def ping():
        return {
            "status": "ok"
        }

    @app.post(
        "/v1/test/validation"
    )
    async def validation(
        payload: ValidationPayload,
    ):
        return {
            "value": payload.value
        }

    @app.get(
        "/v1/test/missing-index"
    )
    async def missing_index():
        raise RetrievalServiceError(
            "Document retrieval indexes "
            "are missing or incomplete."
        )

    @app.get(
        "/v1/test/invalid-retrieval"
    )
    async def invalid_retrieval():
        raise RetrievalServiceError(
            "dense_top_k must be positive."
        )

    @app.get(
        "/v1/test/retrieval-down"
    )
    async def retrieval_down():
        raise RetrievalServiceError(
            "Hybrid document retrieval failed."
        )

    @app.get(
        "/v1/test/value-error"
    )
    async def value_error():
        raise ValueError(
            "Example invalid value."
        )

    app.add_middleware(
        LocalRateLimitMiddleware,
        default_requests_per_minute=2,
        upload_requests_per_minute=1,
    )

    # Added last -> executes first.
    app.add_middleware(
        RequestIDMiddleware
    )

    register_exception_handlers(
        app
    )

    return app


class Phase4APIContractTests(
    unittest.IsolatedAsyncioTestCase
):

    async def asyncSetUp(
        self,
    ) -> None:
        self.app = build_test_app()

        transport = httpx.ASGITransport(
            app=self.app,
            raise_app_exceptions=False,
        )

        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        )

    async def asyncTearDown(
        self,
    ) -> None:
        await self.client.aclose()

    async def test_request_validation_maps_to_422(
        self,
    ) -> None:
        response = await self.client.post(
            "/v1/test/validation",
            json={
                "value": "not-an-integer"
            },
            headers={
                "X-User-ID": (
                    "validation-user"
                )
            },
        )

        self.assertEqual(
            response.status_code,
            422,
        )

        payload = response.json()

        self.assertEqual(
            payload.get("error_code"),
            "VALIDATION_ERROR",
        )

        self.assertTrue(
            payload.get("request_id")
        )

    async def test_unknown_route_maps_to_404(
        self,
    ) -> None:
        response = await self.client.get(
            "/v1/test/does-not-exist",
            headers={
                "X-User-ID": (
                    "unknown-user"
                )
            },
        )

        self.assertEqual(
            response.status_code,
            404,
        )

        payload = response.json()

        self.assertEqual(
            payload.get("error_code"),
            "HTTP_404",
        )

    async def test_missing_index_maps_to_404(
        self,
    ) -> None:
        response = await self.client.get(
            "/v1/test/missing-index",
            headers={
                "X-User-ID": (
                    "missing-user"
                )
            },
        )

        self.assertEqual(
            response.status_code,
            404,
        )

        payload = response.json()

        self.assertEqual(
            payload.get("error_code"),
            "DOCUMENT_INDEX_NOT_FOUND",
        )

    async def test_bad_retrieval_parameter_maps_to_400(
        self,
    ) -> None:
        response = await self.client.get(
            "/v1/test/invalid-retrieval",
            headers={
                "X-User-ID": (
                    "invalid-user"
                )
            },
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        payload = response.json()

        self.assertEqual(
            payload.get("error_code"),
            "INVALID_RETRIEVAL_REQUEST",
        )

    async def test_retrieval_failure_maps_to_503(
        self,
    ) -> None:
        response = await self.client.get(
            "/v1/test/retrieval-down",
            headers={
                "X-User-ID": (
                    "down-user"
                )
            },
        )

        self.assertEqual(
            response.status_code,
            503,
        )

        payload = response.json()

        self.assertEqual(
            payload.get("error_code"),
            (
                "RETRIEVAL_SERVICE_UNAVAILABLE"
            ),
        )

    async def test_value_error_maps_to_400(
        self,
    ) -> None:
        response = await self.client.get(
            "/v1/test/value-error",
            headers={
                "X-User-ID": (
                    "value-user"
                )
            },
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        payload = response.json()

        self.assertEqual(
            payload.get("error_code"),
            "INVALID_VALUE",
        )

    async def test_request_and_rate_headers_exist(
        self,
    ) -> None:
        response = await self.client.get(
            "/v1/test/ping",
            headers={
                "X-User-ID": (
                    "header-user"
                )
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertIn(
            "X-Request-ID",
            response.headers,
        )

        self.assertIn(
            "X-RateLimit-Limit",
            response.headers,
        )

        self.assertIn(
            "X-RateLimit-Remaining",
            response.headers,
        )

    async def test_rate_limit_and_user_isolation(
        self,
    ) -> None:
        headers = {
            "X-User-ID": "same-user"
        }

        first = await self.client.get(
            "/v1/test/ping",
            headers=headers,
        )

        second = await self.client.get(
            "/v1/test/ping",
            headers=headers,
        )

        third = await self.client.get(
            "/v1/test/ping",
            headers=headers,
        )

        self.assertEqual(
            first.status_code,
            200,
        )

        self.assertEqual(
            second.status_code,
            200,
        )

        self.assertEqual(
            third.status_code,
            429,
        )

        self.assertEqual(
            third.headers.get(
                "X-RateLimit-Remaining"
            ),
            "0",
        )

        self.assertIn(
            "Retry-After",
            third.headers,
        )

        self.assertTrue(
            third.json().get(
                "request_id"
            )
        )

        other_user = await self.client.get(
            "/v1/test/ping",
            headers={
                "X-User-ID": (
                    "different-user"
                )
            },
        )

        self.assertEqual(
            other_user.status_code,
            200,
        )


if __name__ == "__main__":
    unittest.main()