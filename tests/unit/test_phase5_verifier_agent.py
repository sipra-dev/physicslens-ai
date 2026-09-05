from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.agents.verifier_agent import (
    VerifierAgent,
    VerifierAgentError,
)
from src.config import settings
from src.models.contracts import (
    AnswerType,
    IntentDecision,
    LanguageCode,
    ModelTask,
    QueryScopeDecision,
    RequestIntent,
    ScopeStatus,
    SourceCitation,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)
from src.models.routing import ModelRouter
from src.retrieval.models import (
    ContextBundle,
    ContextItem,
)


class FakeVerifierGateway:
    """
    Provider-free Verifier gateway test double.

    No OpenAI/API call is made.
    """

    def __init__(
        self,
        *,
        response: VerificationResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response or VerificationResult(
            grounded=True,
            physics_correct=True,
            calculation_correct=True,
            units_correct=True,
            diagram_claims_supported=True,
            within_school_scope=True,
            citation_valid=True,
            issues=[],
            action=VerificationAction.PASS,
            confidence=0.99,
        )
        self.error = error

    def generate_structured(
        self,
        *,
        route,
        system_prompt: str,
        user_prompt: str,
        response_model,
        image_urls=None,
    ):
        self.calls.append(
            {
                "route": route,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
                "image_urls": tuple(image_urls or ()),
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


class Phase5VerifierAgentTests(unittest.TestCase):
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

    def _scope(
        self,
        *,
        in_scope: bool = True,
    ) -> QueryScopeDecision:
        if in_scope:
            return QueryScopeDecision(
                is_physics=True,
                school_level=True,
                supported=True,
                status=ScopeStatus.IN_SCOPE,
                estimated_grade_range=[8, 10],
                topics=[
                    "motion_and_kinematics"
                ],
                confidence=0.99,
                reason=(
                    "Supported school-level Physics."
                ),
            )

        return QueryScopeDecision(
            is_physics=False,
            school_level=False,
            supported=False,
            status=ScopeStatus.OUT_OF_SCOPE,
            estimated_grade_range=None,
            topics=[],
            confidence=0.99,
            reason="Not a Physics question.",
        )

    def _intent(
        self,
        request_intent: RequestIntent,
        *,
        prefer_visual: bool = False,
    ) -> IntentDecision:
        return IntentDecision(
            intent=request_intent,
            confidence=0.99,
            language=LanguageCode.ENGLISH,
            estimated_grade=9,
            has_physics_request=(
                request_intent
                not in {
                    RequestIntent.OUT_OF_SCOPE,
                    RequestIntent.UNSUPPORTED,
                }
            ),
            is_follow_up=(
                request_intent
                == RequestIntent.FOLLOW_UP
            ),
            prefer_visual=prefer_visual,
        )

    def _context(
        self,
        *,
        image_path: str | None = None,
    ) -> ContextBundle:
        text = (
            "Acceleration is the rate of change "
            "of velocity. Newton's second law "
            "uses F = ma."
        )

        return ContextBundle(
            query="What is acceleration?",
            user_id="local-user",
            document_id="doc-test",
            items=[
                ContextItem(
                    context_id="ctx-2",
                    user_id="local-user",
                    document_id="doc-test",
                    page_number=2,
                    source_chunk_ids=[
                        "chunk-2"
                    ],
                    parent_id="parent-2",
                    text=text,
                    content_type="text",
                    linked_figure_ids=[
                        "fig-2"
                    ],
                    equations=["F = ma"],
                    image_path=image_path,
                    caption=(
                        "A school Physics diagram."
                        if image_path
                        else None
                    ),
                    rerank_score=0.95,
                )
            ],
            total_characters=len(text),
            truncated=False,
        )

    def _concept_answer(
        self,
        *,
        invalid_citation: bool = False,
    ) -> TutorAnswer:
        return TutorAnswer(
            answer_type=(
                AnswerType.CONCEPT_EXPLANATION
            ),
            direct_answer=(
                "Acceleration is the rate "
                "of change of velocity."
            ),
            steps=[],
            formulae=[],
            diagram_explanation=None,
            common_mistake=None,
            final_result=None,
            source_pages=(
                [99]
                if invalid_citation
                else [2]
            ),
            citations=[
                SourceCitation(
                    page_number=(
                        99
                        if invalid_citation
                        else 2
                    ),
                    source_chunk_ids=[
                        (
                            "invented-chunk"
                            if invalid_citation
                            else "chunk-2"
                        )
                    ],
                    figure_id=None,
                )
            ],
        )

    def _agent(
        self,
        gateway: FakeVerifierGateway,
    ) -> VerifierAgent:
        return VerifierAgent(
            model_gateway=gateway,
            model_router=self._router(),
        )

    def test_valid_conceptual_answer_passes(
        self,
    ) -> None:
        gateway = FakeVerifierGateway()
        agent = self._agent(gateway)

        result = agent.verify(
            query="What is acceleration?",
            intent=self._intent(
                RequestIntent.PHYSICS_QUESTION
            ),
            scope=self._scope(),
            tutor_answer=(
                self._concept_answer()
            ),
            context=self._context(),
        )

        self.assertEqual(
            len(gateway.calls),
            1,
        )
        self.assertEqual(
            gateway.calls[0]["route"].task,
            ModelTask.VERIFIER,
        )
        self.assertEqual(
            gateway.calls[0]["image_urls"],
            (),
        )
        self.assertIs(
            gateway.calls[0]["response_model"],
            VerificationResult,
        )
        self.assertEqual(
            result.action,
            VerificationAction.PASS,
        )
        self.assertTrue(
            result.citation_valid
        )

    def test_wrong_numerical_overrides_model_pass(
        self,
    ) -> None:
        gateway = FakeVerifierGateway()
        agent = self._agent(gateway)

        bad_answer = TutorAnswer(
            answer_type=(
                AnswerType.NUMERICAL_SOLUTION
            ),
            direct_answer=(
                "The acceleration is 6 m/s^2."
            ),
            steps=[
                "a = 10 / 2 = 6"
            ],
            formulae=[],
            diagram_explanation=None,
            common_mistake=None,
            final_result="6 m/s^2",
            source_pages=[2],
            citations=[
                SourceCitation(
                    page_number=2,
                    source_chunk_ids=[
                        "chunk-2"
                    ],
                    figure_id=None,
                )
            ],
        )

        result = agent.verify(
            query=(
                "A 10 N force acts on a 2 kg "
                "body. Find the acceleration."
            ),
            intent=self._intent(
                RequestIntent.NUMERICAL_PROBLEM
            ),
            scope=self._scope(),
            tutor_answer=bad_answer,
            context=self._context(),
        )

        self.assertEqual(
            len(gateway.calls),
            1,
        )
        self.assertFalse(
            result.calculation_correct
        )
        self.assertEqual(
            result.action,
            VerificationAction.REGENERATE,
        )
        self.assertTrue(
            any(
                "Arithmetic mismatch"
                in issue
                for issue
                in result.issues
            )
        )

    def test_invalid_citation_overrides_model_pass(
        self,
    ) -> None:
        gateway = FakeVerifierGateway()
        agent = self._agent(gateway)

        result = agent.verify(
            query="What is acceleration?",
            intent=self._intent(
                RequestIntent.PHYSICS_QUESTION
            ),
            scope=self._scope(),
            tutor_answer=(
                self._concept_answer(
                    invalid_citation=True
                )
            ),
            context=self._context(),
        )

        self.assertFalse(
            result.citation_valid
        )
        self.assertEqual(
            result.action,
            VerificationAction.REGENERATE,
        )

    def test_diagram_without_image_requests_clearer_image(
        self,
    ) -> None:
        gateway = FakeVerifierGateway()
        agent = self._agent(gateway)

        diagram_answer = TutorAnswer(
            answer_type=(
                AnswerType.DIAGRAM_EXPLANATION
            ),
            direct_answer=(
                "The arrow represents the normal force."
            ),
            steps=[],
            formulae=[],
            diagram_explanation=(
                "The arrow points away from the surface."
            ),
            common_mistake=None,
            final_result=None,
            source_pages=[2],
            citations=[
                SourceCitation(
                    page_number=2,
                    source_chunk_ids=[
                        "chunk-2"
                    ],
                    figure_id="fig-2",
                )
            ],
        )

        result = agent.verify(
            query=(
                "Which arrow is the normal force?"
            ),
            intent=self._intent(
                RequestIntent.DIAGRAM_QUESTION,
                prefer_visual=True,
            ),
            scope=self._scope(),
            tutor_answer=diagram_answer,
            context=self._context(
                image_path=None
            ),
        )

        self.assertEqual(
            len(gateway.calls),
            0,
        )
        self.assertFalse(
            result.diagram_claims_supported
        )
        self.assertEqual(
            result.action,
            (
                VerificationAction
                .ASK_FOR_CLEARER_IMAGE
            ),
        )

    def test_diagram_with_image_uses_visual_verifier_route(
        self,
    ) -> None:
        gateway = FakeVerifierGateway()
        agent = self._agent(gateway)

        with tempfile.TemporaryDirectory() as directory:
            image_path = (
                Path(directory)
                / "figure.png"
            )

            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\nTEST"
            )

            diagram_answer = TutorAnswer(
                answer_type=(
                    AnswerType.DIAGRAM_EXPLANATION
                ),
                direct_answer=(
                    "The arrow represents "
                    "the normal force."
                ),
                steps=[],
                formulae=[],
                diagram_explanation=(
                    "The explanation uses "
                    "the supplied diagram."
                ),
                common_mistake=None,
                final_result=None,
                source_pages=[2],
                citations=[
                    SourceCitation(
                        page_number=2,
                        source_chunk_ids=[
                            "chunk-2"
                        ],
                        figure_id="fig-2",
                    )
                ],
            )

            result = agent.verify(
                query=(
                    "Which arrow is the "
                    "normal force?"
                ),
                intent=self._intent(
                    RequestIntent.DIAGRAM_QUESTION,
                    prefer_visual=True,
                ),
                scope=self._scope(),
                tutor_answer=diagram_answer,
                context=self._context(
                    image_path=str(
                        image_path
                    )
                ),
            )

        self.assertEqual(
            len(gateway.calls),
            1,
        )
        self.assertEqual(
            gateway.calls[0]["route"].task,
            ModelTask.VERIFIER,
        )
        self.assertTrue(
            gateway.calls[0][
                "route"
            ].requires_vision
        )

        sent_images = (
            gateway.calls[0][
                "image_urls"
            ]
        )

        self.assertEqual(
            len(sent_images),
            1,
        )
        self.assertTrue(
            sent_images[0].startswith(
                "data:image/png;base64,"
            )
        )
        self.assertEqual(
            result.action,
            VerificationAction.PASS,
        )

    def test_out_of_scope_never_calls_verifier_model(
        self,
    ) -> None:
        gateway = FakeVerifierGateway()
        agent = self._agent(gateway)

        result = agent.verify(
            query="Explain photosynthesis.",
            intent=self._intent(
                RequestIntent.OUT_OF_SCOPE
            ),
            scope=self._scope(
                in_scope=False
            ),
            tutor_answer=(
                self._concept_answer()
            ),
            context=self._context(),
        )

        self.assertEqual(
            len(gateway.calls),
            0,
        )
        self.assertFalse(
            result.within_school_scope
        )
        self.assertEqual(
            result.action,
            (
                VerificationAction
                .REJECT_OUT_OF_SCOPE
            ),
        )

    def test_correct_no_evidence_refusal_passes_without_model(
        self,
    ) -> None:
        gateway = FakeVerifierGateway()
        agent = self._agent(gateway)

        refusal = TutorAnswer(
            answer_type=(
                AnswerType.INSUFFICIENT_EVIDENCE
            ),
            direct_answer=(
                "I do not have enough reliable "
                "evidence to answer safely."
            ),
            steps=[],
            formulae=[],
            diagram_explanation=None,
            common_mistake=None,
            final_result=None,
            source_pages=[],
            citations=[],
        )

        result = agent.verify(
            query=(
                "What value is shown on page 7?"
            ),
            intent=self._intent(
                RequestIntent.PHYSICS_QUESTION
            ),
            scope=self._scope(),
            tutor_answer=refusal,
            context=None,
            strict_document_mode=True,
        )

        self.assertEqual(
            len(gateway.calls),
            0,
        )
        self.assertEqual(
            result.action,
            VerificationAction.PASS,
        )
        self.assertTrue(
            result.grounded
        )

    def test_gateway_failure_is_wrapped(
        self,
    ) -> None:
        gateway = FakeVerifierGateway(
            error=RuntimeError(
                "provider failed"
            )
        )
        agent = self._agent(gateway)

        with self.assertRaises(
            VerifierAgentError
        ):
            agent.verify(
                query="What is acceleration?",
                intent=self._intent(
                    RequestIntent.PHYSICS_QUESTION
                ),
                scope=self._scope(),
                tutor_answer=(
                    self._concept_answer()
                ),
                context=self._context(),
            )


if __name__ == "__main__":
    unittest.main()