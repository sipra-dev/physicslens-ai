from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import settings


BASE_URL = "http://127.0.0.1:8000"

USER_ID = "local-user"

DOCUMENT_ID = (
    "13deb748922e4c1db47924380ae70c76"
)


async def main() -> None:

    print()
    print("=" * 70)
    print(
        "POINT 10 — API HARDENING + RATE LIMIT"
    )
    print("=" * 70)

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=30.0,
    ) as client:

        # ==================================================
        # 10A — SERVER HEALTH
        # ==================================================

        print("\n10A — SERVER HEALTH")

        response = await client.get("/")

        print(
            "Status:",
            response.status_code,
        )

        assert response.status_code == 200

        print(
            "10A Server Health PASS ✅"
        )

        # ==================================================
        # 10B — REQUEST VALIDATION ERROR
        # ==================================================

        print(
            "\n10B — REQUEST VALIDATION"
        )

        identity = (
            "validation-"
            + uuid.uuid4().hex
        )

        response = await client.post(
            "/v1/retrieval/search",
            content='{"user_id": ',
            headers={
                "Content-Type": (
                    "application/json"
                ),
                "X-User-ID": identity,
            },
        )

        print(
            "Status:",
            response.status_code,
        )

        print(
            response.text[:700]
        )

        assert (
            response.status_code == 422
        )

        payload = response.json()

        assert (
            payload.get("error_code")
            == "VALIDATION_ERROR"
        )

        assert (
            "request_id" in payload
        )

        print(
            "10B Validation Mapping PASS ✅"
        )

        # ==================================================
        # 10C — UNKNOWN ROUTE
        # ==================================================

        print(
            "\n10C — UNKNOWN ROUTE"
        )

        identity = (
            "unknown-route-"
            + uuid.uuid4().hex
        )

        response = await client.get(
            "/v1/this-route-does-not-exist",
            headers={
                "X-User-ID": identity,
            },
        )

        print(
            "Status:",
            response.status_code,
        )

        print(
            response.text[:500]
        )

        assert (
            response.status_code == 404
        )

        payload = response.json()

        assert (
            payload.get("error_code")
            == "HTTP_404"
        )

        print(
            "10C 404 Mapping PASS ✅"
        )

        # ==================================================
        # 10D — MISSING RETRIEVAL INDEX
        # ==================================================

        print(
            "\n10D — MISSING DOCUMENT INDEX"
        )

        identity = (
            "missing-document-"
            + uuid.uuid4().hex
        )

        response = await client.post(
            "/v1/retrieval/search",
            headers={
                "X-User-ID": identity,
            },
            json={
                "user_id": USER_ID,
                "document_id": (
                    "ffffffffffffffff"
                    "ffffffffffffffff"
                ),
                "query": (
                    "What is simple "
                    "harmonic motion?"
                ),
            },
        )

        print(
            "Status:",
            response.status_code,
        )

        print(
            response.text[:700]
        )

        assert (
            response.status_code == 404
        )

        payload = response.json()

        assert (
            payload.get("error_code")
            == "DOCUMENT_INDEX_NOT_FOUND"
        )

        assert (
            "request_id" in payload
        )

        print(
            "10D Retrieval Error Mapping PASS ✅"
        )

        # ==================================================
        # 10E — BURST RATE LIMIT
        #
        # Send more requests than the configured
        # default per-minute limit.
        # ==================================================

        print()
        print(
            "10E — BURST RATE LIMIT"
        )

        configured_limit = int(
            settings.default_rate_limit_per_minute
        )

        request_count = (
            configured_limit + 5
        )

        print(
            "Configured limit:",
            configured_limit,
        )

        print(
            "Burst requests:",
            request_count,
        )

        burst_identity = (
            "burst-"
            + uuid.uuid4().hex
        )

        status_path = (
            f"/v1/documents/"
            f"{DOCUMENT_ID}/status"
        )

        async def send_request():
            return await client.get(
                status_path,
                headers={
                    "X-User-ID": (
                        burst_identity
                    ),
                },
            )

        responses = await asyncio.gather(
            *[
                send_request()
                for _ in range(
                    request_count
                )
            ]
        )

        status_codes = [
            item.status_code
            for item in responses
        ]

        successful = sum(
            code == 200
            for code in status_codes
        )

        limited = sum(
            code == 429
            for code in status_codes
        )

        print(
            "200 responses:",
            successful,
        )

        print(
            "429 responses:",
            limited,
        )

        print(
            "Status codes:",
            status_codes,
        )

        assert successful > 0, (
            "10E FAILED: "
            "all requests were blocked."
        )

        assert limited > 0, (
            "10E FAILED: "
            "no 429 response was returned."
        )

        limited_response = next(
            response
            for response in responses
            if response.status_code == 429
        )

        limited_payload = (
            limited_response.json()
        )

        print(
            "429 body:",
            limited_payload,
        )

        print(
            "Retry-After:",
            limited_response.headers.get(
                "Retry-After"
            ),
        )

        assert (
            "Too many requests"
            in limited_payload.get(
                "detail",
                "",
            )
        )

        assert (
            "Retry-After"
            in limited_response.headers
        )

        assert (
            limited_response.headers.get(
                "X-RateLimit-Remaining"
            )
            == "0"
        )

        print(
            "10E Burst Rate Limit PASS ✅"
        )

        # ==================================================
        # 10F — DIFFERENT USER GETS OWN BUCKET
        # ==================================================

        print()
        print(
            "10F — PER-USER RATE ISOLATION"
        )

        second_identity = (
            "second-user-"
            + uuid.uuid4().hex
        )

        response = await client.get(
            status_path,
            headers={
                "X-User-ID": second_identity,
            },
        )

        print(
            "Second user status:",
            response.status_code,
        )

        assert (
            response.status_code == 200
        ), (
            "10F FAILED: a different "
            "user inherited another user's "
            "rate-limit bucket."
        )

        print(
            "10F Per-user Isolation PASS ✅"
        )

        # ==================================================
        # 10G — RATE LIMIT HEADERS
        # ==================================================

        print()
        print(
            "10G — RATE LIMIT HEADERS"
        )

        header_identity = (
            "header-test-"
            + uuid.uuid4().hex
        )

        response = await client.get(
            status_path,
            headers={
                "X-User-ID": header_identity,
            },
        )

        print(
            "Limit:",
            response.headers.get(
                "X-RateLimit-Limit"
            ),
        )

        print(
            "Remaining:",
            response.headers.get(
                "X-RateLimit-Remaining"
            ),
        )

        assert (
            "X-RateLimit-Limit"
            in response.headers
        )

        assert (
            "X-RateLimit-Remaining"
            in response.headers
        )

        print(
            "10G Rate Headers PASS ✅"
        )

        # ==================================================
        # 10H — SERVER STILL HEALTHY AFTER BURST
        # ==================================================

        print()
        print(
            "10H — SERVER SURVIVES BURST"
        )

        response = await client.get("/")

        print(
            "Status:",
            response.status_code,
        )

        assert (
            response.status_code == 200
        )

        print(
            "10H Server Survival PASS ✅"
        )

    print()
    print("=" * 70)
    print("POINT 10 COMPLETE ✅")
    print("=" * 70)

    print(
        "\n"
        "Server health ✅\n"
        "Validation mapping ✅\n"
        "404 mapping ✅\n"
        "Retrieval error mapping ✅\n"
        "Concurrent burst → 429 ✅\n"
        "Per-user rate isolation ✅\n"
        "Rate-limit headers ✅\n"
        "Server survives burst ✅"
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )