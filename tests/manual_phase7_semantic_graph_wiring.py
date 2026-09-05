from __future__ import annotations

from typing import Any

from src.graph.nodes.serving_nodes import ServingNodes
from src.models.contracts import (
    AnswerType,
    IntentDecision,
    LanguageCode,
    MemorySnapshot,
    RequestIntent,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)


class DummyQueryService:
    pass


class DummyRetrievalService:
    pass


class DummyVerifierAgent:
    pass


class FakeTutorAgent:
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
                "Acceleration is the rate of "
                "change of velocity."
            ),
            steps=[],
            formulae=[],
            diagram_explanation=None,
            common_mistake=None,
            final_result=None,
            source_pages=[],
            citations=[],
        )


class FakeSemanticLearningMemory:
    def __init__(
        self,
        *,
        recall_text: str | None = None,
    ) -> None:
        self.recall_text = recall_text

        self.recall_calls: list[
            dict[str, Any]
        ] = []

        self.learn_calls: list[
            dict[str, Any]
        ] = []

    def recall_for_tutor(
        self,
        *,
        user_id: str,
        query_text: str,
    ) -> str | None:
        self.recall_calls.append(
            {
                "user_id": user_id,
                "query_text": query_text,
            }
        )

        return self.recall_text

    def learn_from_turn(
        self,
        **kwargs: Any,
    ) -> None:
        self.learn_calls.append(
            dict(kwargs)
        )


def make_intent() -> IntentDecision:
    return IntentDecision(
        intent=(
            RequestIntent.PHYSICS_QUESTION
        ),
        confidence=0.99,
        language=LanguageCode.ENGLISH,
        estimated_grade=9,
        has_physics_request=True,
        is_follow_up=False,
        prefer_visual=False,
    )


def make_answer() -> TutorAnswer:
    return TutorAnswer(
        answer_type=(
            AnswerType.CONCEPT_EXPLANATION
        ),
        direct_answer=(
            "Acceleration describes how "
            "velocity changes with time."
        ),
        steps=[],
        formulae=[],
        diagram_explanation=None,
        common_mistake=None,
        final_result=None,
        source_pages=[],
        citations=[],
    )


def make_pass_verification(
) -> VerificationResult:
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
        confidence=0.99,
    )


def build_nodes(
    semantic_memory: (
        FakeSemanticLearningMemory
    ),
    tutor: FakeTutorAgent,
) -> ServingNodes:
    return ServingNodes(
        query_service=DummyQueryService(),
        retrieval_service=(
            DummyRetrievalService()
        ),
        tutor_agent=tutor,
        verifier_agent=(
            DummyVerifierAgent()
        ),
        session_store=None,
        long_term_memory=None,
        semantic_learning_memory=(
            semantic_memory
        ),
        query_cache=None,
        output_guard=None,
    )


def test_semantic_recall_reaches_tutor(
) -> None:
    semantic_memory = (
        FakeSemanticLearningMemory(
            recall_text=(
                "Student previously confused "
                "speed with acceleration."
            )
        )
    )

    tutor = FakeTutorAgent()

    nodes = build_nodes(
        semantic_memory,
        tutor,
    )

    state = {
        "user_id": "student-1",
        "session_id": "session-1",
        "normalized_query": (
            "Can acceleration exist "
            "while slowing down?"
        ),
        "rewritten_query": (
            "relationship between "
            "acceleration and slowing down"
        ),
        "intent": make_intent(),
        "scope": None,
        "memory": MemorySnapshot(),
        "reranked_context": None,
        "strict_document_mode": False,
        "generation_attempts": 0,
        "cache_hit": False,
    }

    result = nodes.tutor_agent_node(
        state
    )

    assert (
        len(
            semantic_memory.recall_calls
        )
        == 1
    )

    assert (
        semantic_memory
        .recall_calls[0][
            "user_id"
        ]
        == "student-1"
    )

    assert (
        semantic_memory
        .recall_calls[0][
            "query_text"
        ]
        == (
            "relationship between "
            "acceleration and slowing down"
        )
    )

    assert len(tutor.calls) == 1

    assert (
        tutor.calls[0][
            "semantic_memory_context"
        ]
        == (
            "Student previously confused "
            "speed with acceleration."
        )
    )

    assert (
        result[
            "generation_attempts"
        ]
        == 1
    )

    print(
        "SEMANTIC_RECALL_REACHES_TUTOR=True"
    )


def test_pass_turn_writes_semantic_memory(
) -> None:
    semantic_memory = (
        FakeSemanticLearningMemory()
    )

    tutor = FakeTutorAgent()

    nodes = build_nodes(
        semantic_memory,
        tutor,
    )

    state = {
        "user_id": "student-1",
        "session_id": "session-1",
        "normalized_query": (
            "I thought acceleration "
            "means moving fast."
        ),
        "intent": make_intent(),
        "memory": MemorySnapshot(),
        "final_answer": make_answer(),
        "verification_result": (
            make_pass_verification()
        ),
        "cache_hit": False,
        "active_document_id": None,
        "language": (
            LanguageCode.ENGLISH
        ),
        "estimated_grade": 9,
    }

    result = (
        nodes.memory_write_decision(
            state
        )
    )

    assert (
        result[
            "should_write_memory"
        ]
        is True
    )

    assert (
        len(
            semantic_memory.learn_calls
        )
        == 1
    )

    call = (
        semantic_memory.learn_calls[0]
    )

    assert (
        call["user_id"]
        == "student-1"
    )

    assert (
        call["student_text"]
        == (
            "I thought acceleration "
            "means moving fast."
        )
    )

    assert (
        call[
            "verification_passed"
        ]
        is True
    )

    assert (
        call["session_id"]
        == "session-1"
    )

    assert (
        "Verifier action: PASS."
        in call["verifier_text"]
    )

    print(
        "PASS_TURN_WRITES_SEMANTIC_MEMORY=True"
    )


def test_cache_hit_does_not_relearn(
) -> None:
    semantic_memory = (
        FakeSemanticLearningMemory()
    )

    tutor = FakeTutorAgent()

    nodes = build_nodes(
        semantic_memory,
        tutor,
    )

    state = {
        "user_id": "student-1",
        "session_id": "session-1",
        "normalized_query": (
            "Explain acceleration."
        ),
        "intent": make_intent(),
        "memory": MemorySnapshot(),
        "final_answer": make_answer(),
        "verification_result": (
            make_pass_verification()
        ),
        "cache_hit": True,
        "active_document_id": None,
        "language": (
            LanguageCode.ENGLISH
        ),
        "estimated_grade": 9,
    }

    result = (
        nodes.memory_write_decision(
            state
        )
    )

    assert (
        result[
            "should_write_memory"
        ]
        is True
    )

    assert (
        len(
            semantic_memory.learn_calls
        )
        == 0
    )

    print(
        "CACHE_HIT_DOES_NOT_RELEARN=True"
    )


def main() -> None:
    test_semantic_recall_reaches_tutor()

    test_pass_turn_writes_semantic_memory()

    test_cache_hit_does_not_relearn()

    print()
    print(
        "PHASE7_SEMANTIC_GRAPH_WIRING=PASS"
    )


if __name__ == "__main__":
    main()