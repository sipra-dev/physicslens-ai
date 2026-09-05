from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.graph.nodes.serving_nodes import ServingNodes
from src.models.contracts import (
    AnswerType,
    IntentDecision,
    LanguageCode,
    MemorySnapshot,
    QueryScopeDecision,
    RequestIntent,
    ScopeStatus,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)
from src.retrieval.models import (
    ContextBundle,
    HybridRetrievalResult,
)
from src.retrieval.structural_resolver import (
    AnswerScopeContract,
    StructuralResolutionAction,
    StructuralResolutionStatus,
)


class _RecordingRetrievalService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        **kwargs: Any,
    ) -> HybridRetrievalResult:
        self.calls.append(
            dict(kwargs)
        )

        return HybridRetrievalResult(
            query=kwargs["query"],
            context=ContextBundle(
                query=kwargs["query"],
                user_id=kwargs["user_id"],
                document_id=kwargs["document_id"],
            ),
            evidence_found=True,
        )


class _RecordingTutorAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def answer(
        self,
        **kwargs: Any,
    ) -> TutorAnswer:
        self.calls.append(
            dict(kwargs)
        )

        return TutorAnswer(
            answer_type=(
                AnswerType.CONCEPT_EXPLANATION
            ),
            direct_answer=(
                "An ideal spring follows the resolved source point."
            ),
            steps=[],
            formulae=[],
            source_pages=[],
            citations=[],
        )


class _RecordingVerifierAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def verify(
        self,
        **kwargs: Any,
    ) -> VerificationResult:
        self.calls.append(
            dict(kwargs)
        )

        return VerificationResult(
            grounded=True,
            physics_correct=True,
            calculation_correct=True,
            units_correct=True,
            diagram_claims_supported=True,
            within_school_scope=True,
            citation_valid=True,
            issues=[],
            action=VerificationAction.PASS,
            confidence=1.0,
        )


class _LegacyTutorAgent:
    """
    Old Tutor signature, deliberately without structural_answer_scope.

    If ServingNodes sends the new keyword during a normal semantic turn,
    Python will raise TypeError and this compatibility test will fail.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def answer(
        self,
        *,
        query: str,
        intent: IntentDecision,
        scope: QueryScopeDecision | None,
        context: ContextBundle | None,
        memory: MemorySnapshot,
        semantic_memory_context: str | None,
        strict_document_mode: bool,
        verifier_feedback: list[str] | None,
    ) -> TutorAnswer:
        self.call_count += 1

        return TutorAnswer(
            answer_type=AnswerType.DIRECT_ANSWER,
            direct_answer="Legacy Tutor call remained compatible.",
            steps=[],
            formulae=[],
            source_pages=[],
            citations=[],
        )


class _LegacyVerifierAgent:
    """
    Old Verifier signature, deliberately without structural_answer_scope.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def verify(
        self,
        *,
        query: str,
        intent: IntentDecision,
        scope: QueryScopeDecision | None,
        tutor_answer: TutorAnswer,
        context: ContextBundle | None,
        strict_document_mode: bool,
    ) -> VerificationResult:
        self.call_count += 1

        return VerificationResult(
            grounded=True,
            physics_correct=True,
            calculation_correct=True,
            units_correct=True,
            diagram_claims_supported=True,
            within_school_scope=True,
            citation_valid=True,
            issues=[],
            action=VerificationAction.PASS,
            confidence=1.0,
        )


class StructuralServingWiringTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.intent = IntentDecision(
            intent=RequestIntent.PHYSICS_QUESTION,
            confidence=1.0,
            language=LanguageCode.ENGLISH,
            estimated_grade=10,
            has_physics_request=True,
            is_follow_up=False,
            prefer_visual=False,
        )

        self.query_scope = QueryScopeDecision(
            is_physics=True,
            school_level=True,
            supported=True,
            status=ScopeStatus.IN_SCOPE,
            estimated_grade_range=[9, 10],
            topics=["mechanics"],
            confidence=1.0,
            reason=None,
        )

        self.answer_scope = AnswerScopeContract(
            requested_action=(
                StructuralResolutionAction.EXPLAIN
            ),
            allowed_target_node_ids=[
                "node-4",
            ],
            scope_rules=[
                "Explain only the resolved source item.",
                "Do not include neighbouring points.",
            ],
        )

    @staticmethod
    def _nodes(
        *,
        retrieval_service: Any,
        tutor_agent: Any,
        verifier_agent: Any,
    ) -> ServingNodes:
        return ServingNodes(
            query_service=object(),
            retrieval_service=retrieval_service,
            tutor_agent=tutor_agent,
            verifier_agent=verifier_agent,
        )

    def _base_state(self) -> dict[str, Any]:
        return {
            "request_id": "request-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "raw_query": (
                "Explain the fourth point under ideal spring."
            ),
            "normalized_query": (
                "Explain the fourth point under ideal spring."
            ),
            "intent": self.intent,
            "scope": self.query_scope,
            "memory": MemorySnapshot(),
            "active_document_id": "document-1",
            "strict_document_mode": True,
            "generation_attempts": 0,
        }

    def test_verified_structural_ids_use_one_exact_lookup(
        self,
    ) -> None:
        retrieval = _RecordingRetrievalService()
        nodes = self._nodes(
            retrieval_service=retrieval,
            tutor_agent=_RecordingTutorAgent(),
            verifier_agent=_RecordingVerifierAgent(),
        )

        state = self._base_state()
        state.update(
            {
                "structural_resolution_status": (
                    StructuralResolutionStatus.RESOLVED
                ),
                "structural_fallback_to_semantic": False,
                "structural_linked_retrieval_chunk_ids": [
                    "chunk-4",
                    "chunk-4",
                    "chunk-4b",
                ],
                "structural_linked_parent_chunk_ids": [
                    "parent-2",
                    "parent-2",
                ],
            }
        )

        results = nodes._run_retrieval_round(
            queries=[
                "rewritten query one",
                "rewritten query two",
            ],
            state=state,
            broader=False,
        )

        self.assertEqual(
            len(results),
            1,
        )
        self.assertEqual(
            len(retrieval.calls),
            1,
        )
        self.assertEqual(
            retrieval.calls[0]["required_chunk_ids"],
            (
                "chunk-4",
                "chunk-4b",
            ),
        )
        self.assertEqual(
            retrieval.calls[0]["required_parent_ids"],
            (
                "parent-2",
            ),
        )
        self.assertEqual(
            retrieval.calls[0]["query"],
            "rewritten query one",
        )

    def test_resolved_target_without_linked_evidence_fails_closed(
        self,
    ) -> None:
        retrieval = _RecordingRetrievalService()
        nodes = self._nodes(
            retrieval_service=retrieval,
            tutor_agent=_RecordingTutorAgent(),
            verifier_agent=_RecordingVerifierAgent(),
        )

        state = self._base_state()
        state.update(
            {
                "structural_resolution_status": (
                    StructuralResolutionStatus.RESOLVED
                ),
                "structural_fallback_to_semantic": False,
                "structural_linked_retrieval_chunk_ids": [],
                "structural_linked_parent_chunk_ids": [],
            }
        )

        results = nodes._run_retrieval_round(
            queries=["fourth point"],
            state=state,
            broader=False,
        )

        self.assertEqual(
            retrieval.calls,
            [],
        )
        self.assertEqual(
            len(results),
            1,
        )
        self.assertFalse(
            results[0].evidence_found
        )
        self.assertEqual(
            results[0].failure_reason,
            (
                "RESOLVED_STRUCTURAL_TARGET_HAS_NO_"
                "LINKED_EVIDENCE"
            ),
        )

    def test_same_structural_scope_reaches_tutor_and_verifier(
        self,
    ) -> None:
        tutor = _RecordingTutorAgent()
        verifier = _RecordingVerifierAgent()
        nodes = self._nodes(
            retrieval_service=_RecordingRetrievalService(),
            tutor_agent=tutor,
            verifier_agent=verifier,
        )

        state = self._base_state()
        state["structural_answer_scope"] = (
            self.answer_scope
        )

        tutor_update = nodes.tutor_agent_node(
            state
        )

        verifier_state = {
            **state,
            **tutor_update,
        }

        verifier_update = nodes.verifier_agent_node(
            verifier_state
        )

        self.assertEqual(
            len(tutor.calls),
            1,
        )
        self.assertIs(
            tutor.calls[0]["structural_answer_scope"],
            self.answer_scope,
        )
        self.assertEqual(
            len(verifier.calls),
            1,
        )
        self.assertIs(
            verifier.calls[0]["structural_answer_scope"],
            self.answer_scope,
        )
        self.assertEqual(
            verifier_update["terminal_action"],
            VerificationAction.PASS,
        )

    def test_normal_turn_keeps_legacy_agent_calls_compatible(
        self,
    ) -> None:
        tutor = _LegacyTutorAgent()
        verifier = _LegacyVerifierAgent()
        nodes = self._nodes(
            retrieval_service=_RecordingRetrievalService(),
            tutor_agent=tutor,
            verifier_agent=verifier,
        )

        state = self._base_state()

        self.assertNotIn(
            "structural_answer_scope",
            state,
        )

        tutor_update = nodes.tutor_agent_node(
            state
        )

        verifier_update = nodes.verifier_agent_node(
            {
                **state,
                **tutor_update,
            }
        )

        self.assertEqual(
            tutor.call_count,
            1,
        )
        self.assertEqual(
            verifier.call_count,
            1,
        )
        self.assertEqual(
            verifier_update["terminal_action"],
            VerificationAction.PASS,
        )


if __name__ == "__main__":
    unittest.main()
