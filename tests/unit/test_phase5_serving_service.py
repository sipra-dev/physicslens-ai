from __future__ import annotations

import unittest
from typing import Any

from src.models.contracts import (
    AnswerType,
    ConversationMessage,
    IntentDecision,
    LanguageCode,
    MemorySnapshot,
    QueryRewriteResult,
    QueryScopeDecision,
    QueryUnderstandingResult,
    RequestIntent,
    ScopeStatus,
    SourceCitation,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)
from src.retrieval.models import (
    ContextBundle,
    ContextItem,
    HybridRetrievalResult,
)
from src.serving.service import ServingService


class FakeQueryService:
    def __init__(
        self,
        understanding: QueryUnderstandingResult,
    ) -> None:
        self.understanding = understanding
        self.calls: list[dict[str, Any]] = []

    def understand(
        self,
        *,
        query: str,
        memory: MemorySnapshot | None = None,
        upload_present: bool = False,
    ) -> QueryUnderstandingResult:
        self.calls.append(
            {
                "query": query,
                "memory": memory,
                "upload_present": upload_present,
            }
        )
        return self.understanding


class FakeRetrievalService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs) -> HybridRetrievalResult:
        self.calls.append(dict(kwargs))

        query = kwargs["query"]
        user_id = kwargs["user_id"]
        document_id = kwargs["document_id"]

        text = (
            f"Evidence for {query}. "
            "Acceleration is the rate of change of velocity."
        )

        context = ContextBundle(
            query=query,
            user_id=user_id,
            document_id=document_id,
            items=[
                ContextItem(
                    context_id=f"ctx-{len(self.calls)}",
                    user_id=user_id,
                    document_id=document_id,
                    page_number=2,
                    source_chunk_ids=[
                        f"chunk-{len(self.calls)}"
                    ],
                    parent_id="parent-2",
                    text=text,
                    content_type="text",
                    linked_figure_ids=[],
                    equations=[],
                    image_path=None,
                    caption=None,
                    rerank_score=0.95,
                )
            ],
            total_characters=len(text),
            truncated=False,
        )

        return HybridRetrievalResult(
            query=query,
            context=context,
            evidence_found=True,
        )


class FakeTutorAgent:
    def __init__(
        self,
        answers: list[TutorAnswer],
    ) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def answer(self, **kwargs) -> TutorAnswer:
        self.calls.append(dict(kwargs))

        index = min(
            len(self.calls) - 1,
            len(self.answers) - 1,
        )
        return self.answers[index]


class FakeVerifierAgent:
    def __init__(
        self,
        results: list[VerificationResult],
    ) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def verify(self, **kwargs) -> VerificationResult:
        self.calls.append(dict(kwargs))

        index = min(
            len(self.calls) - 1,
            len(self.results) - 1,
        )
        return self.results[index]


class Phase5ServingServiceTests(unittest.TestCase):
    def _intent(
        self,
        request_intent: RequestIntent,
        *,
        language: LanguageCode = LanguageCode.ENGLISH,
        prefer_visual: bool = False,
        requires_document: bool = False,
    ) -> IntentDecision:
        return IntentDecision(
            intent=request_intent,
            confidence=0.99,
            language=language,
            estimated_grade=9,
            has_physics_request=(
                request_intent
                not in {
                    RequestIntent.GREETING,
                    RequestIntent.OUT_OF_SCOPE,
                    RequestIntent.UNSUPPORTED,
                    RequestIntent.UPLOAD_DOCUMENT,
                }
            ),
            is_follow_up=(
                request_intent
                == RequestIntent.FOLLOW_UP
            ),
            prefer_visual=prefer_visual,
            requires_document=requires_document,
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
                topics=["motion_and_kinematics"],
                confidence=0.99,
                reason="Supported school-level Physics.",
            )

        return QueryScopeDecision(
            is_physics=False,
            school_level=False,
            supported=False,
            status=ScopeStatus.OUT_OF_SCOPE,
            estimated_grade_range=None,
            topics=[],
            confidence=0.99,
            reason="Outside supported Physics scope.",
        )

    def _understanding(
        self,
        *,
        intent: RequestIntent = RequestIntent.PHYSICS_QUESTION,
        active_document_id: str | None = "doc-test",
        rewrite: QueryRewriteResult | None = None,
        in_scope: bool = True,
        language: LanguageCode = LanguageCode.ENGLISH,
        requires_document: bool = False,
    ) -> QueryUnderstandingResult:
        return QueryUnderstandingResult(
            normalized_query="What is acceleration?",
            intent=self._intent(
                intent,
                language=language,
                prefer_visual=(
                    intent == RequestIntent.DIAGRAM_QUESTION
                ),
                requires_document=requires_document,
            ),
            scope=(
                None
                if intent in {
                    RequestIntent.GREETING,
                    RequestIntent.UPLOAD_DOCUMENT,
                }
                else self._scope(in_scope=in_scope)
            ),
            rewrite=rewrite,
            active_document_id=active_document_id,
        )

    def _answer(
        self,
        text: str = "Acceleration is the rate of change of velocity.",
    ) -> TutorAnswer:
        return TutorAnswer(
            answer_type=AnswerType.CONCEPT_EXPLANATION,
            direct_answer=text,
            steps=[],
            formulae=[],
            diagram_explanation=None,
            common_mistake=None,
            final_result=None,
            source_pages=[2],
            citations=[
                SourceCitation(
                    page_number=2,
                    source_chunk_ids=["chunk-1"],
                    figure_id=None,
                )
            ],
        )

    def _verification(
        self,
        action: VerificationAction,
        *,
        issues: list[str] | None = None,
        grounded: bool = True,
    ) -> VerificationResult:
        is_pass = action == VerificationAction.PASS

        return VerificationResult(
            grounded=grounded if not is_pass else True,
            physics_correct=True,
            calculation_correct=True,
            units_correct=True,
            diagram_claims_supported=True,
            within_school_scope=True,
            citation_valid=True,
            issues=issues or [],
            action=action,
            confidence=0.99,
        )

    def _service(
        self,
        *,
        understanding: QueryUnderstandingResult,
        tutor_answers: list[TutorAnswer] | None = None,
        verification_results: list[VerificationResult] | None = None,
    ):
        query = FakeQueryService(understanding)
        retrieval = FakeRetrievalService()
        tutor = FakeTutorAgent(
            tutor_answers or [self._answer()]
        )
        verifier = FakeVerifierAgent(
            verification_results
            or [
                self._verification(
                    VerificationAction.PASS
                )
            ]
        )

        service = ServingService(
            query_service=query,
            retrieval_service=retrieval,
            tutor_agent=tutor,
            verifier_agent=verifier,
        )

        return (
            service,
            query,
            retrieval,
            tutor,
            verifier,
        )

    def test_greeting_fast_path_skips_rag_and_agents(self) -> None:
        understanding = self._understanding(
            intent=RequestIntent.GREETING,
            active_document_id=None,
            language=LanguageCode.BENGALI,
        )

        (
            service,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._service(
            understanding=understanding
        )

        memory = MemorySnapshot(
            language=LanguageCode.BENGALI
        )

        result = service.serve(
            query="হ্যালো",
            user_id="local-user",
            memory=memory,
        )

        self.assertEqual(len(retrieval.calls), 0)
        self.assertEqual(len(tutor.calls), 0)
        self.assertEqual(len(verifier.calls), 0)
        self.assertEqual(result.generation_attempts, 0)
        self.assertEqual(result.retrieval_rounds, 0)
        self.assertEqual(result.next_memory, memory)
        self.assertEqual(
            result.answer.answer_type,
            AnswerType.DIRECT_ANSWER,
        )

    def test_out_of_scope_fast_path_skips_rag_and_agents(self) -> None:
        understanding = self._understanding(
            intent=RequestIntent.OUT_OF_SCOPE,
            active_document_id=None,
            in_scope=False,
        )

        (
            service,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._service(
            understanding=understanding
        )

        result = service.serve(
            query="Explain photosynthesis.",
            user_id="local-user",
        )

        self.assertEqual(len(retrieval.calls), 0)
        self.assertEqual(len(tutor.calls), 0)
        self.assertEqual(len(verifier.calls), 0)
        self.assertEqual(
            result.terminal_action,
            VerificationAction.REJECT_OUT_OF_SCOPE,
        )

    def test_document_rag_passes_in_one_generation(self) -> None:
        understanding = self._understanding(
            requires_document=True
        )

        (
            service,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._service(
            understanding=understanding
        )

        result = service.serve(
            query="What is acceleration?",
            user_id="local-user",
            session_id="session-1",
            document_id="doc-test",
        )

        self.assertEqual(len(retrieval.calls), 1)
        self.assertEqual(len(tutor.calls), 1)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(result.generation_attempts, 1)
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertEqual(
            result.terminal_action,
            VerificationAction.PASS,
        )
        self.assertEqual(
            result.next_memory.active_document_id,
            "doc-test",
        )
        self.assertEqual(
            len(result.next_memory.recent_messages),
            2,
        )

    def test_regenerate_reuses_context_and_stops_after_second_attempt(self) -> None:
        understanding = self._understanding(
            requires_document=True
        )

        (
            service,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._service(
            understanding=understanding,
            tutor_answers=[
                self._answer("First draft."),
                self._answer("Corrected draft."),
            ],
            verification_results=[
                self._verification(
                    VerificationAction.REGENERATE,
                    issues=["Wrong wording."],
                ),
                self._verification(
                    VerificationAction.PASS
                ),
            ],
        )

        result = service.serve(
            query="What is acceleration?",
            user_id="local-user",
            document_id="doc-test",
        )

        self.assertEqual(len(retrieval.calls), 1)
        self.assertEqual(len(tutor.calls), 2)
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(result.generation_attempts, 2)
        self.assertEqual(result.retrieval_rounds, 1)
        self.assertIn(
            "Wrong wording.",
            tutor.calls[1]["verifier_feedback"],
        )
        self.assertEqual(
            result.terminal_action,
            VerificationAction.PASS,
        )

    def test_retry_retrieval_runs_one_broader_round_then_second_generation(self) -> None:
        understanding = self._understanding(
            requires_document=True
        )

        (
            service,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._service(
            understanding=understanding,
            tutor_answers=[
                self._answer("Draft from first retrieval."),
                self._answer("Draft from broader retrieval."),
            ],
            verification_results=[
                self._verification(
                    VerificationAction.RETRY_RETRIEVAL,
                    issues=["Need better evidence."],
                    grounded=False,
                ),
                self._verification(
                    VerificationAction.PASS
                ),
            ],
        )

        result = service.serve(
            query="What is acceleration?",
            user_id="local-user",
            document_id="doc-test",
        )

        self.assertEqual(len(retrieval.calls), 2)
        self.assertEqual(len(tutor.calls), 2)
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(result.retrieval_rounds, 2)
        self.assertEqual(result.generation_attempts, 2)

        self.assertEqual(
            retrieval.calls[0]["dense_top_k"],
            20,
        )
        self.assertEqual(
            retrieval.calls[1]["dense_top_k"],
            30,
        )
        self.assertEqual(
            retrieval.calls[1]["bm25_top_k"],
            30,
        )
        self.assertEqual(
            retrieval.calls[1]["fused_top_k"],
            40,
        )
        self.assertEqual(
            retrieval.calls[1]["max_contexts"],
            8,
        )
        self.assertEqual(
            result.terminal_action,
            VerificationAction.PASS,
        )

    def test_two_failed_generations_fail_closed(self) -> None:
        understanding = self._understanding(
            requires_document=True
        )

        (
            service,
            _query,
            retrieval,
            tutor,
            verifier,
        ) = self._service(
            understanding=understanding,
            tutor_answers=[
                self._answer("Bad first draft."),
                self._answer("Bad second draft."),
            ],
            verification_results=[
                self._verification(
                    VerificationAction.REGENERATE,
                    issues=["First failure."],
                ),
                self._verification(
                    VerificationAction.REGENERATE,
                    issues=["Second failure."],
                ),
            ],
        )

        result = service.serve(
            query="What is acceleration?",
            user_id="local-user",
            document_id="doc-test",
        )

        self.assertEqual(len(tutor.calls), 2)
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(result.generation_attempts, 2)
        self.assertEqual(
            result.answer.answer_type,
            AnswerType.INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(
            result.terminal_action,
            VerificationAction.INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(
            result.verification.action,
            VerificationAction.INSUFFICIENT_EVIDENCE,
        )

    def test_multi_query_plus_hyde_is_bounded_to_three_retrieval_calls(self) -> None:
        rewrite = QueryRewriteResult(
            original_query="Explain acceleration.",
            rewritten_query="Explain acceleration in school Physics.",
            retrieval_queries=[
                "acceleration definition",
                "velocity change per unit time",
            ],
            was_rewritten=True,
            prefer_visual=False,
            preferred_page_numbers=[2],
            referenced_figure_id=None,
            use_hyde=True,
            hyde_text=(
                "A textbook passage explaining acceleration "
                "as change of velocity with time."
            ),
        )

        understanding = self._understanding(
            rewrite=rewrite,
            requires_document=True,
        )

        (
            service,
            _query,
            retrieval,
            _tutor,
            _verifier,
        ) = self._service(
            understanding=understanding
        )

        result = service.serve(
            query="Explain acceleration.",
            user_id="local-user",
            document_id="doc-test",
        )

        self.assertEqual(len(retrieval.calls), 3)
        self.assertEqual(
            result.retrieval_rounds,
            1,
        )

        self.assertEqual(
            [call["query"] for call in retrieval.calls],
            [
                "acceleration definition",
                "velocity change per unit time",
                (
                    "A textbook passage explaining acceleration "
                    "as change of velocity with time."
                ),
            ],
        )

    def test_memory_keeps_only_last_ten_messages_and_updates_page(self) -> None:
        old_messages = []

        for index in range(9):
            old_messages.append(
                ConversationMessage(
                    role=(
                        "user"
                        if index % 2 == 0
                        else "assistant"
                    ),
                    content=f"old-{index}",
                )
            )

        memory = MemorySnapshot(
            active_document_id="doc-old",
            active_page=1,
            recent_messages=old_messages,
            language=LanguageCode.ENGLISH,
            estimated_grade=9,
        )

        understanding = self._understanding(
            active_document_id="doc-test",
            requires_document=True,
        )

        (
            service,
            _query,
            _retrieval,
            _tutor,
            _verifier,
        ) = self._service(
            understanding=understanding
        )

        result = service.serve(
            query="What is acceleration?",
            user_id="local-user",
            document_id="doc-test",
            memory=memory,
        )

        self.assertEqual(
            len(result.next_memory.recent_messages),
            10,
        )
        self.assertEqual(
            result.next_memory.active_document_id,
            "doc-test",
        )
        self.assertEqual(
            result.next_memory.active_page,
            2,
        )
        self.assertEqual(
            result.next_memory.recent_messages[-2].role,
            "user",
        )
        self.assertEqual(
            result.next_memory.recent_messages[-1].role,
            "assistant",
        )


if __name__ == "__main__":
    unittest.main()