from __future__ import annotations

import unittest
from typing import Any

from src.config import settings
from src.models.contracts import (
    IntentDecision,
    LanguageCode,
    ModelTask,
    QueryRewriteResult,
    QueryScopeDecision,
    RequestIntent,
    ScopeStatus,
)
from src.models.fallback import (
    FailureKind,
    FallbackPolicy,
    RecoveryAction,
)
from src.models.gateway import LLMGateway
from src.models.routing import ModelRouter
from src.query.service import QueryUnderstandingService


class FakeStructuredRunner:
    """
    Provider-free test double.

    It lets us verify Phase-5 wiring without making
    any OpenAI/API call.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def generate_structured(
        self,
        *,
        route,
        system_prompt: str,
        user_prompt: str,
        response_model,
    ) -> Any:
        self.call_count += 1

        if response_model is IntentDecision:
            return IntentDecision(
                intent=RequestIntent.PHYSICS_QUESTION,
                confidence=0.99,
                language=LanguageCode.ENGLISH,
                estimated_grade=9,
                has_physics_request=True,
                is_follow_up=False,
                prefer_visual=False,
            )

        if response_model is QueryScopeDecision:
            return QueryScopeDecision(
                is_physics=True,
                school_level=True,
                supported=True,
                status=ScopeStatus.IN_SCOPE,
                estimated_grade_range=[8, 10],
                topics=["motion_and_kinematics"],
                confidence=0.99,
                reason="Supported school-level Physics.",
            )

        if response_model is QueryRewriteResult:
            return QueryRewriteResult(
                original_query="What is acceleration?",
                rewritten_query=(
                    "What is acceleration in "
                    "school-level kinematics?"
                ),
                retrieval_queries=[
                    (
                        "What is acceleration in "
                        "school-level kinematics?"
                    ),
                ],
                was_rewritten=True,
                prefer_visual=False,
                preferred_page_numbers=[],
                referenced_figure_id=None,
                use_hyde=False,
                hyde_text=None,
            )

        raise AssertionError(
            f"Unexpected response model: {response_model}"
        )


class Phase5WiringTests(unittest.TestCase):
    def _router(self) -> ModelRouter:
        return ModelRouter(
            classifier_model=(
                settings.phase5_classifier_model
            ),
            text_model=settings.tutor_text_model,
            multimodal_model=(
                settings.tutor_multimodal_model
            ),
            reasoning_model=(
                settings.tutor_reasoning_model
            ),
            verifier_model=settings.verifier_model,
            fallback_model=(
                settings.model_fallback_model
            ),
        )

    def test_phase5_settings_exist(self) -> None:
        self.assertTrue(
            settings.phase5_classifier_model
        )
        self.assertTrue(
            settings.tutor_text_model
        )
        self.assertTrue(
            settings.tutor_multimodal_model
        )
        self.assertTrue(
            settings.tutor_reasoning_model
        )
        self.assertTrue(
            settings.verifier_model
        )
        self.assertGreater(
            settings.model_gateway_timeout_seconds,
            0,
        )

    def test_model_router_routes_all_core_tasks(self) -> None:
        router = self._router()

        self.assertEqual(
            router.route_task(
                ModelTask.INTENT_CLASSIFICATION
            ).task,
            ModelTask.INTENT_CLASSIFICATION,
        )

        self.assertEqual(
            router.route_task(
                ModelTask.QUERY_SCOPE
            ).task,
            ModelTask.QUERY_SCOPE,
        )

        self.assertEqual(
            router.route_task(
                ModelTask.QUERY_REWRITE
            ).task,
            ModelTask.QUERY_REWRITE,
        )

        self.assertEqual(
            router.route_task(
                ModelTask.TUTOR_TEXT
            ).task,
            ModelTask.TUTOR_TEXT,
        )

        self.assertTrue(
            router.route_task(
                ModelTask.TUTOR_MULTIMODAL
            ).requires_vision
        )

        self.assertTrue(
            router.route_task(
                ModelTask.TUTOR_NUMERICAL
            ).reasoning_heavy
        )

        self.assertEqual(
            router.route_task(
                ModelTask.VERIFIER
            ).task,
            ModelTask.VERIFIER,
        )

    def test_fallback_policy_is_bounded(self) -> None:
        policy = FallbackPolicy()

        first = policy.decide(
            task=ModelTask.QUERY_SCOPE,
            failure_kind=FailureKind.TIMEOUT,
            primary_attempt_count=1,
            primary_model="primary-model",
            fallback_model="fallback-model",
            allow_fallback=True,
        )

        self.assertEqual(
            first.action,
            RecoveryAction.RETRY_PRIMARY,
        )

        second = policy.decide(
            task=ModelTask.QUERY_SCOPE,
            failure_kind=FailureKind.TIMEOUT,
            primary_attempt_count=2,
            primary_model="primary-model",
            fallback_model="fallback-model",
            allow_fallback=True,
        )

        self.assertEqual(
            second.action,
            RecoveryAction.USE_FALLBACK,
        )

        self.assertEqual(
            second.next_model,
            "fallback-model",
        )

    def test_gateway_can_be_constructed_without_network(self) -> None:
        router = self._router()

        gateway = LLMGateway(
            model_router=router,
            client=object(),
            timeout_seconds=(
                settings.model_gateway_timeout_seconds
            ),
        )

        self.assertTrue(
            callable(
                gateway.generate_structured
            )
        )

    def test_pure_greeting_uses_no_model_call(self) -> None:
        runner = FakeStructuredRunner()

        service = QueryUnderstandingService(
            model_runner=runner,
            model_router=self._router(),
        )

        result = service.understand(
            query="hello",
        )

        self.assertEqual(
            result.intent.intent,
            RequestIntent.GREETING,
        )

        self.assertIsNone(
            result.scope
        )

        self.assertIsNone(
            result.rewrite
        )

        self.assertEqual(
            runner.call_count,
            0,
        )

    def test_physics_query_flows_intent_scope_rewrite(self) -> None:
        runner = FakeStructuredRunner()

        service = QueryUnderstandingService(
            model_runner=runner,
            model_router=self._router(),
        )

        result = service.understand(
            query="What is acceleration?",
        )

        self.assertEqual(
            result.intent.intent,
            RequestIntent.PHYSICS_QUESTION,
        )

        self.assertIsNotNone(
            result.scope
        )

        self.assertEqual(
            result.scope.status,
            ScopeStatus.IN_SCOPE,
        )

        self.assertIsNotNone(
            result.rewrite
        )

        self.assertEqual(
            result.rewrite.rewritten_query,
            (
                "What is acceleration in "
                "school-level kinematics?"
            ),
        )

        self.assertEqual(
            runner.call_count,
            3,
        )


if __name__ == "__main__":
    unittest.main()