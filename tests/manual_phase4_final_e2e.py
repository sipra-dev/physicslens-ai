from __future__ import annotations

import asyncio
import re
import sys
import uuid
from pathlib import Path

import httpx


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import settings


BASE_URL = "http://127.0.0.1:8000"

USER_ID = "local-user"

SHM_DOCUMENT_ID = (
    "7e814389ea2142bcb7f9b6bfc5f9234b"
)

DIAGRAM_DOCUMENT_ID = (
    "13deb748922e4c1db47924380ae70c76"
)

ACCELERATION_DOCUMENT_ID = (
    "c027fd8b35bf40719d73c3ff008c4441"
)


def unique_identity(
    label: str,
) -> str:
    return (
        f"phase4-final-{label}-"
        f"{uuid.uuid4().hex}"
    )


def context_blob(
    payload: dict,
) -> str:
    parts: list[str] = []

    for item in (
        payload
        .get("context", {})
        .get("items", [])
    ):
        text = item.get(
            "text"
        )

        caption = item.get(
            "caption"
        )

        equations = item.get(
            "equations",
            [],
        )

        if text:
            parts.append(
                str(text)
            )

        if caption:
            parts.append(
                str(caption)
            )

        parts.extend(
            str(equation)
            for equation in equations
            if equation
        )

    return "\n".join(parts)


def compact_math(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        "",
        text.lower(),
    )


def assert_retrieval_contract(
    payload: dict,
    *,
    user_id: str,
    document_id: str,
) -> None:

    assert (
        payload.get(
            "evidence_found"
        )
        is True
    )

    assert (
        payload.get(
            "failure_reason"
        )
        is None
    )

    fused_hits = payload.get(
        "fused_hits",
        [],
    )

    reranked_hits = payload.get(
        "reranked_hits",
        [],
    )

    context = payload.get(
        "context",
        {},
    )

    context_items = context.get(
        "items",
        [],
    )

    assert (
        len(fused_hits)
        <= settings.hybrid_candidate_pool_size
    )

    assert (
        len(reranked_hits)
        <= settings.reranker_top_k
    )

    assert (
        len(context_items)
        <= settings.final_context_count
    )

    assert (
        context.get(
            "total_characters",
            0,
        )
        <= settings.max_context_characters
    )

    assert (
        context.get("user_id")
        == user_id
    )

    assert (
        context.get("document_id")
        == document_id
    )

    # Final context must never leak evidence
    # from another user/document.
    for item in context_items:
        assert (
            item.get("user_id")
            == user_id
        )

        assert (
            item.get("document_id")
            == document_id
        )

    for source_name in (
        "dense_hits",
        "bm25_hits",
    ):
        for hit in payload.get(
            source_name,
            [],
        ):
            assert (
                hit.get("user_id")
                == user_id
            )

            assert (
                hit.get("document_id")
                == document_id
            )

    for source_name in (
        "fused_hits",
        "reranked_hits",
    ):
        for wrapped in payload.get(
            source_name,
            [],
        ):
            hit = wrapped.get(
                "hit",
                {},
            )

            assert (
                hit.get("user_id")
                == user_id
            )

            assert (
                hit.get("document_id")
                == document_id
            )


async def retrieve(
    client: httpx.AsyncClient,
    *,
    document_id: str,
    query: str,
    label: str,
    user_id: str = USER_ID,
) -> tuple[
    httpx.Response,
    dict,
]:

    response = await client.post(
        "/v1/retrieval/search",
        headers={
            "X-User-ID": (
                unique_identity(label)
            ),
        },
        json={
            "user_id": user_id,
            "document_id": (
                document_id
            ),
            "query": query,
        },
    )

    print(
        "Status:",
        response.status_code,
    )

    if response.status_code != 200:
        print(
            response.text[:1000]
        )

    assert (
        response.status_code == 200
    )

    assert (
        "X-Request-ID"
        in response.headers
    )

    assert (
        "X-RateLimit-Limit"
        in response.headers
    )

    assert (
        "X-RateLimit-Remaining"
        in response.headers
    )

    payload = response.json()

    assert_retrieval_contract(
        payload,
        user_id=user_id,
        document_id=document_id,
    )

    return (
        response,
        payload,
    )


async def assert_document_ready(
    client: httpx.AsyncClient,
    document_id: str,
    label: str,
) -> None:

    response = await client.get(
        (
            f"/v1/documents/"
            f"{document_id}/status"
        ),
        headers={
            "X-User-ID": (
                unique_identity(label)
            ),
        },
    )

    assert (
        response.status_code == 200
    )

    payload = response.json()

    assert (
        payload.get("document_id")
        == document_id
    )

    assert (
        str(
            payload.get(
                "status",
                "",
            )
        ).upper()
        == "READY"
    )


async def main() -> None:

    print()
    print("=" * 72)
    print(
        "POINT 12 — FINAL PHASE-4 END-TO-END"
    )
    print("=" * 72)

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=120.0,
    ) as client:

        # ==================================================
        # 12A — SERVER + DOCUMENT READINESS
        # ==================================================

        print(
            "\n12A — SERVER + DOCUMENT READINESS"
        )

        live = await client.get(
            "/health/live"
        )

        ready = await client.get(
            "/health/ready"
        )

        assert (
            live.status_code == 200
        )

        assert (
            live.json().get("status")
            == "alive"
        )

        assert (
            ready.status_code == 200
        )

        assert (
            ready.json().get("status")
            == "ready"
        )

        await assert_document_ready(
            client,
            SHM_DOCUMENT_ID,
            "shm-ready",
        )

        await assert_document_ready(
            client,
            DIAGRAM_DOCUMENT_ID,
            "diagram-ready",
        )

        await assert_document_ready(
            client,
            ACCELERATION_DOCUMENT_ID,
            "acceleration-ready",
        )

        print(
            "12A Readiness PASS ✅"
        )

        # ==================================================
        # 12B — SHM DEFINITION
        # ==================================================

        print(
            "\n12B — SHM DEFINITION"
        )

        _, payload = await retrieve(
            client,
            document_id=(
                SHM_DOCUMENT_ID
            ),
            query=(
                "What is simple harmonic motion?"
            ),
            label="shm-definition",
        )

        blob = context_blob(
            payload
        ).lower()

        assert (
            "simple harmonic motion"
            in blob
        )

        assert (
            "restoring force"
            in blob
        )

        assert (
            "displacement"
            in blob
        )

        print(
            "12B SHM Definition PASS ✅"
        )

        # ==================================================
        # 12C — IDEAL SPRING EQUATION
        # ==================================================

        print(
            "\n12C — IDEAL SPRING EQUATION"
        )

        _, payload = await retrieve(
            client,
            document_id=(
                SHM_DOCUMENT_ID
            ),
            query=(
                "What is the angular frequency "
                "of an ideal spring?"
            ),
            label="spring-equation",
        )

        math_text = compact_math(
            context_blob(payload)
        )

        assert (
            "\\omega=\\sqrt{\\frac{k}{m}}"
            in math_text
        )

        print(
            "12C Spring Equation PASS ✅"
        )

        # ==================================================
        # 12D — SIMPLE PENDULUM EQUATION
        # ==================================================

        print(
            "\n12D — SIMPLE PENDULUM EQUATION"
        )

        _, payload = await retrieve(
            client,
            document_id=(
                SHM_DOCUMENT_ID
            ),
            query=(
                "What formula gives the period T "
                "of a simple pendulum in terms "
                "of L and g?"
            ),
            label="pendulum-equation",
        )

        math_text = compact_math(
            context_blob(payload)
        )

        assert (
            "simplependulum"
            in math_text
        )

        assert (
            (
                "2\\pi\\sqrt{\\frac{l}{g}}"
                in math_text
            )
            or (
                "\\sqrt{\\frac{l}{g}}"
                in math_text
            )
        )

        print(
            "12D Pendulum Equation PASS ✅"
        )

        # ==================================================
        # 12E — EXACT NUMERICAL EVIDENCE
        # ==================================================

        print(
            "\n12E — EXACT NUMERICAL EVIDENCE"
        )

        _, payload = await retrieve(
            client,
            document_id=(
                SHM_DOCUMENT_ID
            ),
            query=(
                "A body is attached to a spring "
                "with k = 120 N/m and frequency "
                "6.00 Hz. What mass is shown "
                "in the uploaded document?"
            ),
            label="numerical",
        )

        blob = context_blob(
            payload
        )

        assert (
            "0.0845"
            in blob
        )

        assert (
            "84.5"
            in blob
        )

        print(
            "12E Numerical Evidence PASS ✅"
        )

        # ==================================================
        # 12F — CROSS-USER ISOLATION
        # ==================================================

        print(
            "\n12F — CROSS-USER ISOLATION"
        )

        response = await client.post(
            "/v1/retrieval/search",
            headers={
                "X-User-ID": (
                    unique_identity(
                        "wrong-user"
                    )
                )
            },
            json={
                "user_id": (
                    "phase4-other-user"
                ),
                "document_id": (
                    SHM_DOCUMENT_ID
                ),
                "query": (
                    "What is simple "
                    "harmonic motion?"
                ),
            },
        )

        print(
            "Wrong-user status:",
            response.status_code,
        )

        assert (
            response.status_code == 404
        )

        payload = response.json()

        assert (
            payload.get("error_code")
            == "DOCUMENT_INDEX_NOT_FOUND"
        )

        print(
            "12F Cross-user Isolation PASS ✅"
        )

        # ==================================================
        # 12G — VISUAL DIAGRAM RETRIEVAL
        # ==================================================

        print(
            "\n12G — VISUAL DIAGRAM"
        )

        _, payload = await retrieve(
            client,
            document_id=(
                DIAGRAM_DOCUMENT_ID
            ),
            query=(
                "When does an object speed up "
                "and when does it slow down "
                "based on the signs of velocity "
                "and acceleration?"
            ),
            label="diagram",
        )

        blob = context_blob(
            payload
        ).lower()

        assert (
            "speeding up"
            in blob
        )

        assert (
            "slowing down"
            in blob
        )

        assert (
            "same signs"
            in blob
        )

        assert (
            "opposite signs"
            in blob
        )

        visual_items = [
            item
            for item in (
                payload
                .get("context", {})
                .get("items", [])
            )
            if (
                item.get("content_type")
                == "figure"
            )
        ]

        assert visual_items

        assert any(
            item.get("image_path")
            for item in visual_items
        )

        print(
            "12G Visual Diagram PASS ✅"
        )

        # ==================================================
        # 12H — HANDWRITTEN / IMAGE ACCELERATION
        # ==================================================

        print(
            "\n12H — ACCELERATION IMAGE"
        )

        _, payload = await retrieve(
            client,
            document_id=(
                ACCELERATION_DOCUMENT_ID
            ),
            query=(
                "What is acceleration according "
                "to the uploaded image, and what "
                "are its units?"
            ),
            label="acceleration",
        )

        blob = context_blob(
            payload
        ).lower()

        assert (
            "acceleration"
            in blob
        )

        assert (
            "change in velocity"
            in blob
        )

        assert (
            (
                "m/s^2"
                in blob
            )
            or (
                "m/s²"
                in blob
            )
        )

        visual_items = [
            item
            for item in (
                payload
                .get("context", {})
                .get("items", [])
            )
            if (
                item.get("content_type")
                == "figure"
            )
        ]

        assert visual_items

        print(
            "12H Acceleration Image PASS ✅"
        )

        # ==================================================
        # 12I — KNOWN EQUATION-CORRUPTION REGRESSION
        # ==================================================

        print(
            "\n12I — EQUATION CORRUPTION REGRESSION"
        )

        _, payload = await retrieve(
            client,
            document_id=(
                SHM_DOCUMENT_ID
            ),
            query=(
                "What is the angular frequency "
                "of an ideal spring?"
            ),
            label="corruption-check",
        )

        blob = context_blob(
            payload
        )

        assert (
            "s!!"
            not in blob
        )

        assert (
            "! kx!"
            not in blob
        )

        assert (
            "\\omega"
            in blob
        )

        print(
            "12I Equation Regression PASS ✅"
        )

        # ==================================================
        # 12J — SERVER STILL HEALTHY
        # ==================================================

        print(
            "\n12J — FINAL SERVER HEALTH"
        )

        response = await client.get(
            "/health/live"
        )

        assert (
            response.status_code == 200
        )

        assert (
            response.json().get(
                "status"
            )
            == "alive"
        )

        print(
            "12J Final Health PASS ✅"
        )

    print()
    print("=" * 72)
    print(
        "POINT 12 COMPLETE ✅"
    )
    print(
        "PHASE 4 FINAL VALIDATION COMPLETE ✅"
    )
    print("=" * 72)

    print(
        "\n"
        "API readiness ✅\n"
        "Document readiness ✅\n"
        "Dense + BM25 + RRF ✅\n"
        "CrossEncoder reranking ✅\n"
        "Parent context ✅\n"
        "Equation evidence ✅\n"
        "Exact numerical evidence ✅\n"
        "User/document isolation ✅\n"
        "Visual retrieval ✅\n"
        "Context limits ✅\n"
        "Regression protection ✅\n"
        "Server health ✅"
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
