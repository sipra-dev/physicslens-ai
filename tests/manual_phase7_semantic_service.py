from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.memory.semantic_models import (
    SemanticLearningMemoryRecord,
    SemanticMemoryKind,
)
from src.memory.semantic_service import (
    SemanticLearningMemoryService,
)


@dataclass
class FakeMatch:
    record: SemanticLearningMemoryRecord
    similarity_score: float


class FakeExtractor:
    def __init__(
        self,
        result: SemanticLearningMemoryRecord | None,
        *,
        should_raise: bool = False,
    ) -> None:
        self.result = result
        self.should_raise = should_raise
        self.calls: list[dict[str, Any]] = []

    def extract(
        self,
        **kwargs: Any,
    ) -> SemanticLearningMemoryRecord | None:
        self.calls.append(
            dict(kwargs)
        )

        if self.should_raise:
            raise RuntimeError(
                "fake extractor failure"
            )

        return self.result


class FakeStore:
    def __init__(
        self,
        *,
        search_results: list[Any] | None = None,
        upsert_result: bool = True,
        search_should_raise: bool = False,
        upsert_should_raise: bool = False,
    ) -> None:
        self.search_results = (
            list(search_results or [])
        )

        self.upsert_result = (
            upsert_result
        )

        self.search_should_raise = (
            search_should_raise
        )

        self.upsert_should_raise = (
            upsert_should_raise
        )

        self.search_calls: list[
            dict[str, Any]
        ] = []

        self.upsert_calls: list[
            SemanticLearningMemoryRecord
        ] = []

    def search(
        self,
        **kwargs: Any,
    ) -> list[Any]:
        self.search_calls.append(
            dict(kwargs)
        )

        if self.search_should_raise:
            raise RuntimeError(
                "fake search failure"
            )

        return list(
            self.search_results
        )

    def upsert(
        self,
        record: SemanticLearningMemoryRecord,
    ) -> bool:
        self.upsert_calls.append(
            record
        )

        if self.upsert_should_raise:
            raise RuntimeError(
                "fake upsert failure"
            )

        return self.upsert_result


def make_memory(
    *,
    user_id: str = "student-1",
) -> SemanticLearningMemoryRecord:
    return SemanticLearningMemoryRecord(
        user_id=user_id,
        kind=(
            SemanticMemoryKind.MISCONCEPTION
        ),
        topic="Mechanics",
        concept="Acceleration",
        text=(
            "Student confuses speed "
            "with acceleration."
        ),
        confidence=0.95,
        source_session_id="session-1",
        source_document_id="doc-1",
    )


def test_recall_for_tutor() -> None:
    memory = make_memory()

    store = FakeStore(
        search_results=[
            FakeMatch(
                record=memory,
                similarity_score=0.92,
            )
        ]
    )

    extractor = FakeExtractor(
        None
    )

    service = (
        SemanticLearningMemoryService(
            extractor=extractor,
            store=store,
            top_k=5,
            minimum_score=0.45,
            max_context_characters=2000,
        )
    )

    context = (
        service.recall_for_tutor(
            user_id="student-1",
            query_text=(
                "Can acceleration happen "
                "while slowing down?"
            ),
        )
    )

    assert context is not None

    assert (
        "Student confuses speed "
        "with acceleration."
        in context
    )

    assert len(
        store.search_calls
    ) == 1

    search_call = (
        store.search_calls[0]
    )

    assert (
        search_call["user_id"]
        == "student-1"
    )

    assert (
        search_call["query_text"]
        == (
            "Can acceleration happen "
            "while slowing down?"
        )
    )

    print(
        "RECALL_FOR_TUTOR=True"
    )


def test_verifier_fail_blocks_write() -> None:
    memory = make_memory()

    extractor = FakeExtractor(
        memory
    )

    store = FakeStore()

    service = (
        SemanticLearningMemoryService(
            extractor=extractor,
            store=store,
        )
    )

    result = (
        service.learn_from_turn(
            user_id="student-1",
            student_text=(
                "Acceleration just means "
                "moving fast."
            ),
            tutor_text=(
                "Acceleration is the rate "
                "of change of velocity."
            ),
            verification_passed=False,
            verifier_text=(
                "Verifier action: REGENERATE."
            ),
            session_id="session-1",
            document_id="doc-1",
        )
    )

    assert result is None

    assert (
        len(extractor.calls) == 0
    )

    assert (
        len(store.upsert_calls) == 0
    )

    print(
        "VERIFIER_FAIL_BLOCKS_WRITE=True"
    )


def test_extractor_none_blocks_write() -> None:
    extractor = FakeExtractor(
        None
    )

    store = FakeStore()

    service = (
        SemanticLearningMemoryService(
            extractor=extractor,
            store=store,
        )
    )

    result = (
        service.learn_from_turn(
            user_id="student-1",
            student_text=(
                "What is Newton's "
                "second law?"
            ),
            tutor_text=(
                "Newton's second law "
                "relates force, mass, "
                "and acceleration."
            ),
            verification_passed=True,
            verifier_text=(
                "Verifier action: PASS."
            ),
            session_id="session-1",
            document_id=None,
        )
    )

    assert result is None

    assert (
        len(extractor.calls) == 1
    )

    assert (
        len(store.upsert_calls) == 0
    )

    print(
        "EXTRACTOR_NO_MEMORY_BLOCKS_WRITE=True"
    )


def test_verified_memory_is_stored() -> None:
    memory = make_memory()

    extractor = FakeExtractor(
        memory
    )

    store = FakeStore(
        upsert_result=True
    )

    service = (
        SemanticLearningMemoryService(
            extractor=extractor,
            store=store,
        )
    )

    result = (
        service.learn_from_turn(
            user_id="student-1",
            student_text=(
                "I thought acceleration "
                "just means moving fast."
            ),
            tutor_text=(
                "Speed tells how fast. "
                "Acceleration tells how "
                "velocity changes."
            ),
            verification_passed=True,
            verifier_text=(
                "Verifier action: PASS."
            ),
            session_id="session-1",
            document_id="doc-1",
        )
    )

    assert result is not None

    assert (
        result.memory_id
        == memory.memory_id
    )

    assert (
        len(extractor.calls) == 1
    )

    assert (
        len(store.upsert_calls) == 1
    )

    assert (
        store.upsert_calls[0].memory_id
        == memory.memory_id
    )

    print(
        "VERIFIED_MEMORY_STORED=True"
    )


def test_upsert_failure_is_non_blocking() -> None:
    memory = make_memory()

    extractor = FakeExtractor(
        memory
    )

    store = FakeStore(
        upsert_result=False
    )

    service = (
        SemanticLearningMemoryService(
            extractor=extractor,
            store=store,
        )
    )

    result = (
        service.learn_from_turn(
            user_id="student-1",
            student_text=(
                "I confuse speed "
                "and acceleration."
            ),
            tutor_text=(
                "They are different "
                "physical quantities."
            ),
            verification_passed=True,
            verifier_text=(
                "Verifier action: PASS."
            ),
        )
    )

    assert result is None

    assert (
        len(store.upsert_calls) == 1
    )

    print(
        "UPSERT_FAILURE_NON_BLOCKING=True"
    )


def test_search_failure_is_non_blocking() -> None:
    extractor = FakeExtractor(
        None
    )

    store = FakeStore(
        search_should_raise=True
    )

    service = (
        SemanticLearningMemoryService(
            extractor=extractor,
            store=store,
        )
    )

    context = (
        service.recall_for_tutor(
            user_id="student-1",
            query_text=(
                "Explain acceleration."
            ),
        )
    )

    assert context is None

    print(
        "SEARCH_FAILURE_NON_BLOCKING=True"
    )


def main() -> None:
    test_recall_for_tutor()

    test_verifier_fail_blocks_write()

    test_extractor_none_blocks_write()

    test_verified_memory_is_stored()

    test_upsert_failure_is_non_blocking()

    test_search_failure_is_non_blocking()

    print()
    print(
        "PHASE7_SEMANTIC_SERVICE_TEST=PASS"
    )


if __name__ == "__main__":
    main()