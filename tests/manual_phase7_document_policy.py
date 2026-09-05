from __future__ import annotations

from types import SimpleNamespace
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
)


class DummyService:
    pass


def make_nodes() -> ServingNodes:
    return ServingNodes(
        query_service=DummyService(),
        retrieval_service=DummyService(),
        tutor_agent=DummyService(),
        verifier_agent=DummyService(),
        session_store=None,
        long_term_memory=None,
        semantic_learning_memory=None,
        query_cache=None,
        output_guard=None,
    )


def make_intent() -> IntentDecision:
    return IntentDecision(
        intent=RequestIntent.PHYSICS_QUESTION,
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
            "Newton's laws describe motion "
            "and forces."
        ),
        steps=[],
        formulae=[],
        diagram_explanation=None,
        common_mistake=None,
        final_result=None,
        source_pages=[],
        citations=[],
    )


def make_understanding(
    document_id: str | None,
) -> Any:
    return SimpleNamespace(
        active_document_id=document_id
    )


def test_general_question_bypasses_old_pdf(
) -> None:
    nodes = make_nodes()

    memory = MemorySnapshot(
        active_document_id="thermo.pdf",
        active_page=8,
        last_selected_figure_id="fig-8",
        language=LanguageCode.ENGLISH,
        estimated_grade=9,
    )

    state = {
        "normalized_query": (
            "State Newton's three laws."
        ),
        "memory": memory,
        "query_understanding": (
            make_understanding(
                "thermo.pdf"
            )
        ),
        "strict_document_mode": None,
        "intent": make_intent(),
    }

    result = (
        nodes.resolve_active_document(
            state
        )
    )

    assert (
        result["active_document_id"]
        is None
    )

    assert (
        result["strict_document_mode"]
        is False
    )

    print(
        "GENERAL_QUESTION_BYPASSES_PDF=True"
    )


def test_general_turn_does_not_leak_page(
) -> None:
    nodes = make_nodes()

    memory = MemorySnapshot(
        active_document_id="thermo.pdf",
        active_page=8,
        last_selected_figure_id="fig-8",
        language=LanguageCode.ENGLISH,
        estimated_grade=9,
    )

    state = {
        "active_document_id": None,
        "memory": memory,
        "requested_language": None,
        "language": LanguageCode.ENGLISH,
        "estimated_grade": 9,
    }

    result = (
        nodes.load_short_term_memory(
            state
        )
    )

    # Current general turn must NOT expose
    # old page/figure context.
    assert (
        result["active_page"]
        is None
    )

    assert (
        result["referenced_figure_id"]
        is None
    )

    # But persistent memory still remembers
    # the selected PDF context.
    assert (
        result["memory"]
        .active_document_id
        == "thermo.pdf"
    )

    assert (
        result["memory"]
        .active_page
        == 8
    )

    assert (
        result["memory"]
        .last_selected_figure_id
        == "fig-8"
    )

    print(
        "GENERAL_TURN_NO_PAGE_LEAK=True"
    )


def test_general_answer_keeps_pdf_in_memory(
) -> None:
    nodes = make_nodes()

    memory = MemorySnapshot(
        active_document_id="thermo.pdf",
        active_page=8,
        last_selected_figure_id="fig-8",
        language=LanguageCode.ENGLISH,
        estimated_grade=9,
    )

    state = {
        "user_id": "student-1",
        "session_id": "session-1",
        "normalized_query": (
            "State Newton's three laws."
        ),
        "intent": make_intent(),
        "memory": memory,
        "final_answer": make_answer(),
        "active_document_id": None,
        "active_page": None,
        "referenced_figure_id": None,
        "language": LanguageCode.ENGLISH,
        "estimated_grade": 9,
        "cache_hit": False,
    }

    result = (
        nodes.memory_write_decision(
            state
        )
    )

    next_memory = (
        result["next_memory"]
    )

    assert (
        next_memory.active_document_id
        == "thermo.pdf"
    )

    assert (
        next_memory.active_page
        == 8
    )

    assert (
        next_memory
        .last_selected_figure_id
        == "fig-8"
    )

    print(
        "PDF_RETAINED_AFTER_GENERAL_TURN=True"
    )


def test_later_page_question_reuses_pdf(
) -> None:
    nodes = make_nodes()

    memory = MemorySnapshot(
        active_document_id="thermo.pdf",
        active_page=8,
        last_selected_figure_id="fig-8",
        language=LanguageCode.ENGLISH,
        estimated_grade=9,
    )

    state = {
        "normalized_query": (
            "Explain page 8."
        ),
        "memory": memory,
        "query_understanding": (
            make_understanding(
                "thermo.pdf"
            )
        ),
        "strict_document_mode": None,
        "intent": make_intent(),
    }

    result = (
        nodes.resolve_active_document(
            state
        )
    )

    assert (
        result["active_document_id"]
        == "thermo.pdf"
    )

    assert (
        result["strict_document_mode"]
        is True
    )

    state.update(result)

    loaded = (
        nodes.load_short_term_memory(
            state
        )
    )

    assert (
        loaded["active_page"]
        == 8
    )

    assert (
        loaded["referenced_figure_id"]
        == "fig-8"
    )

    print(
        "LATER_PDF_QUESTION_REUSES_PDF=True"
    )


def test_explicit_document_always_wins(
) -> None:
    nodes = make_nodes()

    memory = MemorySnapshot(
        active_document_id="old.pdf",
    )

    state = {
        "normalized_query": (
            "Explain energy."
        ),
        "memory": memory,
        "explicit_document_id": (
            "new.pdf"
        ),
        "query_understanding": (
            make_understanding(
                "old.pdf"
            )
        ),
        "strict_document_mode": None,
        "intent": make_intent(),
    }

    result = (
        nodes.resolve_active_document(
            state
        )
    )

    assert (
        result["active_document_id"]
        == "new.pdf"
    )

    assert (
        result["strict_document_mode"]
        is True
    )

    print(
        "EXPLICIT_DOCUMENT_WINS=True"
    )


def test_document_request_without_pdf_fails_closed(
) -> None:
    nodes = make_nodes()

    state = {
        "normalized_query": (
            "Explain page 5."
        ),
        "memory": MemorySnapshot(),
        "query_understanding": (
            make_understanding(None)
        ),
        "strict_document_mode": None,
        "intent": make_intent(),
    }

    result = (
        nodes.resolve_active_document(
            state
        )
    )

    assert (
        result["active_document_id"]
        is None
    )

    assert (
        result["strict_document_mode"]
        is True
    )

    assert (
        result["terminal_action"]
        == VerificationAction
        .INSUFFICIENT_EVIDENCE
    )

    print(
        "DOCUMENT_REQUEST_WITHOUT_PDF_FAILS_CLOSED=True"
    )


def main() -> None:
    test_general_question_bypasses_old_pdf()

    test_general_turn_does_not_leak_page()

    test_general_answer_keeps_pdf_in_memory()

    test_later_page_question_reuses_pdf()

    test_explicit_document_always_wins()

    test_document_request_without_pdf_fails_closed()

    print()
    print(
        "PHASE7_DOCUMENT_POLICY_TEST=PASS"
    )


if __name__ == "__main__":
    main()