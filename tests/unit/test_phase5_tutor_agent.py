from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.agents.tutor_agent import (
    TutorAgent,
    TutorAgentError,
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
)
from src.models.routing import ModelRouter
from src.retrieval.models import (
    ContextBundle,
    ContextItem,
)


class FakeTutorGateway:
    """
    Provider-free Tutor gateway test double.

    No OpenAI/API call is made.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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

        if route.task == ModelTask.TUTOR_NUMERICAL:
            return TutorAnswer(
                answer_type=AnswerType.NUMERICAL_SOLUTION,
                direct_answer="The acceleration is 5 m/s^2.",
                steps=[
                    "Given: F = 10 N and m = 2 kg.",
                    "Required: acceleration.",
                    "Relevant formula: F = ma.",
                    "Substitution: a = 10 / 2.",
                    "Calculation: a = 5.",
                    "Unit check: N/kg = m/s^2.",
                ],
                formulae=[],
                diagram_explanation=None,
                common_mistake=None,
                final_result="5 m/s^2",
                source_pages=[2],
                citations=[
                    SourceCitation(
                        page_number=2,
                        source_chunk_ids=["chunk-2"],
                        figure_id=None,
                    )
                ],
            )

        if route.task == ModelTask.TUTOR_MULTIMODAL:
            return TutorAnswer(
                answer_type=AnswerType.DIAGRAM_EXPLANATION,
                direct_answer="The diagram shows the requested force relationship.",
                steps=[],
                formulae=[],
                diagram_explanation=(
                    "The explanation uses only the supplied visual evidence."
                ),
                common_mistake=None,
                final_result=None,
                source_pages=[2],
                citations=[
                    SourceCitation(
                        page_number=2,
                        source_chunk_ids=["chunk-2"],
                        figure_id="fig-2",
                    )
                ],
            )

        return TutorAnswer(
            answer_type=AnswerType.CONCEPT_EXPLANATION,
            direct_answer="Acceleration is the rate of change of velocity.",
            steps=[],
            formulae=[],
            diagram_explanation=None,
            common_mistake=None,
            final_result=None,
            # Include invalid references intentionally.
            # TutorAgent must remove them deterministically.
            source_pages=[2, 99],
            citations=[
                SourceCitation(
                    page_number=2,
                    source_chunk_ids=[
                        "chunk-2",
                        "invented-chunk",
                    ],
                    figure_id="fig-2",
                ),
                SourceCitation(
                    page_number=99,
                    source_chunk_ids=[
                        "invented-chunk",
                    ],
                    figure_id="invented-figure",
                ),
            ],
        )


class Phase5TutorAgentTests(
    unittest.TestCase
):
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
            verifier_model=(
                settings.verifier_model
            ),
            fallback_model=(
                settings.model_fallback_model
            ),
        )

    def _scope(self) -> QueryScopeDecision:
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
            has_physics_request=True,
            is_follow_up=False,
            prefer_visual=prefer_visual,
        )

    def _context(
        self,
        *,
        image_path: str | None = None,
    ) -> ContextBundle:
        text = (
            "Acceleration is the rate of change of velocity. "
            "For force and mass, Newton's second law uses F = ma."
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
                        "chunk-2",
                    ],
                    parent_id="parent-2",
                    text=text,
                    content_type="text",
                    linked_figure_ids=[
                        "fig-2",
                    ],
                    equations=[
                        "F = ma",
                    ],
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

    def _agent(
        self,
        gateway: FakeTutorGateway,
    ) -> TutorAgent:
        return TutorAgent(
            model_gateway=gateway,
            model_router=self._router(),
        )

    def test_strict_mode_without_evidence_returns_insufficient(
        self,
    ) -> None:
        gateway = FakeTutorGateway()
        agent = self._agent(gateway)

        answer = agent.answer(
            query="What is acceleration?",
            intent=self._intent(
                RequestIntent.PHYSICS_QUESTION
            ),
            scope=self._scope(),
            context=None,
            strict_document_mode=True,
        )

        self.assertEqual(
            answer.answer_type,
            AnswerType.INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(
            len(gateway.calls),
            0,
        )
        self.assertEqual(
            answer.source_pages,
            [],
        )
        self.assertEqual(
            answer.citations,
            [],
        )

    def test_conceptual_question_uses_text_route_and_sanitizes_citations(
        self,
    ) -> None:
        gateway = FakeTutorGateway()
        agent = self._agent(gateway)

        answer = agent.answer(
            query="What is acceleration?",
            intent=self._intent(
                RequestIntent.PHYSICS_QUESTION
            ),
            scope=self._scope(),
            context=self._context(),
        )

        self.assertEqual(
            len(gateway.calls),
            1,
        )
        self.assertEqual(
            gateway.calls[0]["route"].task,
            ModelTask.TUTOR_TEXT,
        )
        self.assertEqual(
            gateway.calls[0]["image_urls"],
            (),
        )

        self.assertEqual(
            answer.source_pages,
            [2],
        )
        self.assertEqual(
            len(answer.citations),
            1,
        )
        self.assertEqual(
            answer.citations[0].page_number,
            2,
        )
        self.assertEqual(
            answer.citations[0].source_chunk_ids,
            ["chunk-2"],
        )
        self.assertEqual(
            answer.citations[0].figure_id,
            "fig-2",
        )

    def test_numerical_question_uses_reasoning_route(
        self,
    ) -> None:
        gateway = FakeTutorGateway()
        agent = self._agent(gateway)

        answer = agent.answer(
            query=(
                "A 10 N force acts on a 2 kg body. "
                "Find its acceleration."
            ),
            intent=self._intent(
                RequestIntent.NUMERICAL_PROBLEM
            ),
            scope=self._scope(),
            context=self._context(),
        )

        self.assertEqual(
            len(gateway.calls),
            1,
        )
        self.assertEqual(
            gateway.calls[0]["route"].task,
            ModelTask.TUTOR_NUMERICAL,
        )
        self.assertEqual(
            answer.answer_type,
            AnswerType.NUMERICAL_SOLUTION,
        )
        self.assertEqual(
            answer.final_result,
            "5 m/s^2",
        )

    def test_diagram_question_uses_multimodal_route_and_data_url(
        self,
    ) -> None:
        gateway = FakeTutorGateway()
        agent = self._agent(gateway)

        with tempfile.TemporaryDirectory() as directory:
            image_path = (
                Path(directory)
                / "figure.png"
            )

            # The TutorAgent only encodes the local evidence file here.
            # Provider/image decoding is tested separately at gateway level.
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\nTEST"
            )

            answer = agent.answer(
                query="Explain this diagram.",
                intent=self._intent(
                    RequestIntent.DIAGRAM_QUESTION,
                    prefer_visual=True,
                ),
                scope=self._scope(),
                context=self._context(
                    image_path=str(image_path)
                ),
            )

        self.assertEqual(
            len(gateway.calls),
            1,
        )
        self.assertEqual(
            gateway.calls[0]["route"].task,
            ModelTask.TUTOR_MULTIMODAL,
        )

        sent_images = (
            gateway.calls[0]["image_urls"]
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
            answer.answer_type,
            AnswerType.DIAGRAM_EXPLANATION,
        )

    def test_diagram_without_image_does_not_guess(
        self,
    ) -> None:
        gateway = FakeTutorGateway()
        agent = self._agent(gateway)

        answer = agent.answer(
            query="What does this arrow mean?",
            intent=self._intent(
                RequestIntent.DIAGRAM_QUESTION,
                prefer_visual=True,
            ),
            scope=self._scope(),
            context=self._context(
                image_path=None
            ),
        )

        self.assertEqual(
            answer.answer_type,
            AnswerType.INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(
            len(gateway.calls),
            0,
        )

    def test_out_of_scope_never_reaches_tutor_model(
        self,
    ) -> None:
        gateway = FakeTutorGateway()
        agent = self._agent(gateway)

        out_of_scope = QueryScopeDecision(
            is_physics=False,
            school_level=False,
            supported=False,
            status=ScopeStatus.OUT_OF_SCOPE,
            estimated_grade_range=None,
            topics=[],
            confidence=0.99,
            reason="Not a Physics question.",
        )

        with self.assertRaises(
            TutorAgentError
        ):
            agent.answer(
                query="Explain photosynthesis.",
                intent=IntentDecision(
                    intent=(
                        RequestIntent.OUT_OF_SCOPE
                    ),
                    confidence=0.99,
                    language=LanguageCode.ENGLISH,
                    estimated_grade=9,
                    has_physics_request=False,
                    is_follow_up=False,
                    prefer_visual=False,
                ),
                scope=out_of_scope,
                context=self._context(),
            )

        self.assertEqual(
            len(gateway.calls),
            0,
        )


if __name__ == "__main__":
    unittest.main()