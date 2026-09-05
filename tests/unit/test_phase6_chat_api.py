from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from apps.api.main import app
from src.cache.redis_client import redis_client
from src.models.contracts import (
    AnswerType,
    IntentDecision,
    LanguageCode,
    RequestIntent,
    TutorAnswer,
)


class _FakeChatGraph:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(
        self,
        graph_input: dict,
        config: dict | None = None,
    ) -> dict:
        self.calls.append(
            dict(graph_input)
        )

        intent = IntentDecision(
            intent=RequestIntent.GREETING,
            confidence=1.0,
            language=(
                graph_input.get(
                    "requested_language"
                )
                or LanguageCode.UNKNOWN
            ),
            estimated_grade=None,
            has_physics_request=False,
            is_follow_up=False,
            prefer_visual=False,
        )

        answer = TutorAnswer(
            answer_type=AnswerType.DIRECT_ANSWER,
            direct_answer="Hello from the fake graph.",
        )

        return {
            "request_id": (
                graph_input["request_id"]
            ),
            "session_id": (
                graph_input["session_id"]
            ),
            "active_document_id": (
                graph_input.get(
                    "explicit_document_id"
                )
            ),
            "intent": intent,
            "final_answer": answer,
            "verification_result": None,
            "terminal_action": None,
            "generation_attempts": 0,
            "retrieval_rounds": 0,
        }


async def _allow_rate_limit(
    *,
    key: str,
    window_seconds: int,
) -> tuple[int, int]:
    """
    Unit-test rate-limit stub.

    These Phase-6 chat API tests are testing
    request/response and graph wiring, not Redis.

    Returning count=1 keeps every request below
    the configured API limit without opening a
    real Redis connection.
    """

    return (
        1,
        window_seconds,
    )


class Phase6ChatAPITests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(
        self,
    ) -> None:
        # -------------------------------------------------
        # RATE LIMIT TEST ISOLATION
        # -------------------------------------------------
        #
        # The FastAPI application contains the real
        # Redis-backed rate-limit middleware.
        #
        # These tests use IsolatedAsyncioTestCase, which
        # creates a separate asyncio event loop for each
        # test. Reusing a real async Redis connection
        # across those loops can produce:
        #
        #     RuntimeError: Event loop is closed
        #
        # Rate limiting itself already has dedicated
        # Phase-7 tests, so here we replace only the
        # Redis counter operation with a deterministic
        # async stub.
        # -------------------------------------------------

        self.rate_limit_patcher = patch.object(
            redis_client,
            "rate_limit_increment",
            new=_allow_rate_limit,
        )

        self.rate_limit_patcher.start()

        self.transport = httpx.ASGITransport(
            app=app
        )

        self.client = httpx.AsyncClient(
            transport=self.transport,
            base_url="http://testserver",
        )

    async def asyncTearDown(
        self,
    ) -> None:
        await self.client.aclose()

        self.rate_limit_patcher.stop()

    async def test_chat_endpoint_is_registered_and_returns_response(
        self,
    ) -> None:
        fake_graph = _FakeChatGraph()

        with patch(
            "apps.api.routes.chat.chat_graph",
            fake_graph,
        ):
            response = await self.client.post(
                "/v1/chat",
                headers={
                    "X-User-ID": "phase6-user",
                },
                json={
                    "session_id": "session-1",
                    "query": "hello",
                },
            )

        self.assertEqual(
            response.status_code,
            200,
            response.text,
        )

        payload = response.json()

        self.assertEqual(
            payload["session_id"],
            "session-1",
        )

        self.assertEqual(
            payload["intent"],
            "GREETING",
        )

        self.assertEqual(
            payload["answer"][
                "direct_answer"
            ],
            "Hello from the fake graph.",
        )

        self.assertEqual(
            payload["generation_attempts"],
            0,
        )

        self.assertEqual(
            payload["retrieval_rounds"],
            0,
        )

        self.assertEqual(
            len(fake_graph.calls),
            1,
        )

    async def test_chat_maps_request_context_into_graph_state(
        self,
    ) -> None:
        fake_graph = _FakeChatGraph()

        with patch(
            "apps.api.routes.chat.chat_graph",
            fake_graph,
        ):
            response = await self.client.post(
                "/v1/chat",
                headers={
                    "X-User-ID": "student-42",
                },
                json={
                    "session_id": "session-42",
                    "query": (
                        "Explain this diagram."
                    ),
                    "document_id": "doc-42",
                    "selected_page": 5,
                    "selected_figure_id": (
                        "figure-5-1"
                    ),
                    "language": "bn_en",
                },
            )

        self.assertEqual(
            response.status_code,
            200,
            response.text,
        )

        call = fake_graph.calls[0]

        self.assertEqual(
            call["user_id"],
            "student-42",
        )

        self.assertEqual(
            call["session_id"],
            "session-42",
        )

        self.assertEqual(
            call["raw_query"],
            "Explain this diagram.",
        )

        self.assertEqual(
            call["explicit_document_id"],
            "doc-42",
        )

        self.assertEqual(
            call["selected_page"],
            5,
        )

        self.assertEqual(
            call["selected_figure_id"],
            "figure-5-1",
        )

        self.assertEqual(
            call["requested_language"],
            LanguageCode.BENGALI_ENGLISH_MIXED,
        )

        # Phase 6 must not pre-create a MemorySnapshot here.
        # Leaving it absent allows Phase 7's session_store hook
        # to load persistent memory inside load_session.
        self.assertNotIn(
            "memory",
            call,
        )

    async def test_chat_request_rejects_user_id_in_json_body(
        self,
    ) -> None:
        fake_graph = _FakeChatGraph()

        with patch(
            "apps.api.routes.chat.chat_graph",
            fake_graph,
        ):
            response = await self.client.post(
                "/v1/chat",
                json={
                    "session_id": "session-1",
                    "query": "hello",
                    "user_id": "must-not-be-here",
                },
            )

        self.assertEqual(
            response.status_code,
            422,
            response.text,
        )

        self.assertEqual(
            fake_graph.calls,
            [],
        )

    async def test_chat_request_rejects_invalid_selected_page(
        self,
    ) -> None:
        fake_graph = _FakeChatGraph()

        with patch(
            "apps.api.routes.chat.chat_graph",
            fake_graph,
        ):
            response = await self.client.post(
                "/v1/chat",
                json={
                    "session_id": "session-1",
                    "query": "Explain this page.",
                    "selected_page": 0,
                },
            )

        self.assertEqual(
            response.status_code,
            422,
            response.text,
        )

        self.assertEqual(
            fake_graph.calls,
            [],
        )


if __name__ == "__main__":
    unittest.main()