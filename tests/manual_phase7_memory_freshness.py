from __future__ import annotations

from typing import Any

from src.memory.semantic_models import (
    SemanticLearningMemoryRecord,
    SemanticMemoryKind,
    SemanticMemoryMatch,
    SemanticMemoryStatus,
)
from src.memory.semantic_service import (
    SemanticLearningMemoryService,
)


class FakeExtractor:
    def __init__(
        self,
        record: SemanticLearningMemoryRecord,
    ) -> None:
        self.record = record

    def extract(
        self,
        **kwargs: Any,
    ) -> SemanticLearningMemoryRecord:
        return self.record


class FakeStore:
    def __init__(
        self,
        related_memories: list[
            SemanticMemoryMatch
        ] | None = None,
    ) -> None:
        self.related_memories = list(
            related_memories or []
        )

        self.upsert_calls: list[
            SemanticLearningMemoryRecord
        ] = []

        self.search_calls: list[
            dict[str, Any]
        ] = []

    def search(
        self,
        **kwargs: Any,
    ) -> list[SemanticMemoryMatch]:
        self.search_calls.append(
            dict(kwargs)
        )

        return list(
            self.related_memories
        )

    def upsert(
        self,
        record: SemanticLearningMemoryRecord,
    ) -> bool:
        self.upsert_calls.append(
            record
        )

        return True


def make_record(
    *,
    memory_id: str,
    kind: SemanticMemoryKind,
    text: str,
    topic: str = "Mechanics",
    concept: str = "Acceleration",
    status: SemanticMemoryStatus = (
        SemanticMemoryStatus.ACTIVE
    ),
) -> SemanticLearningMemoryRecord:
    return SemanticLearningMemoryRecord(
        memory_id=memory_id,
        user_id="student-1",
        kind=kind,
        topic=topic,
        concept=concept,
        text=text,
        confidence=0.95,
        status=status,
        source_session_id="session-1",
    )


def learn(
    service: SemanticLearningMemoryService,
) -> SemanticLearningMemoryRecord | None:
    return service.learn_from_turn(
        user_id="student-1",
        student_text=(
            "Student learning evidence."
        ),
        tutor_text=(
            "Verified tutor explanation."
        ),
        verification_passed=True,
        verifier_text=(
            "Verifier action: PASS."
        ),
        session_id="session-2",
    )


def test_same_signal_reinforces_old_memory(
) -> None:
    old_memory = make_record(
        memory_id="old-misconception",
        kind=(
            SemanticMemoryKind.MISCONCEPTION
        ),
        text=(
            "Student confuses speed "
            "with acceleration."
        ),
    )

    new_memory = make_record(
        memory_id="new-duplicate",
        kind=(
            SemanticMemoryKind.MISCONCEPTION
        ),
        text=(
            "Student confuses speed "
            "with acceleration."
        ),
    )

    match = SemanticMemoryMatch(
        record=old_memory,
        similarity_score=0.96,
    )

    store = FakeStore(
        [match]
    )

    service = (
        SemanticLearningMemoryService(
            extractor=FakeExtractor(
                new_memory
            ),
            store=store,
        )
    )

    result = learn(service)

    assert result is not None

    # Same old ID must be reused.
    assert (
        result.memory_id
        == "old-misconception"
    )

    # Only one upsert:
    # reinforce old, don't create duplicate.
    assert len(
        store.upsert_calls
    ) == 1

    assert (
        store.upsert_calls[0]
        .memory_id
        == "old-misconception"
    )

    assert (
        store.upsert_calls[0]
        .status
        == SemanticMemoryStatus.ACTIVE
    )

    print(
        "SAME_SIGNAL_REINFORCED=True"
    )


def test_mastery_resolves_old_problem(
) -> None:
    old_problem = make_record(
        memory_id="old-problem",
        kind=(
            SemanticMemoryKind.MISCONCEPTION
        ),
        text=(
            "Student thinks acceleration "
            "simply means moving fast."
        ),
    )

    new_mastery = make_record(
        memory_id="new-mastery",
        kind=SemanticMemoryKind.MASTERY,
        text=(
            "Student correctly distinguishes "
            "speed from acceleration."
        ),
    )

    match = SemanticMemoryMatch(
        record=old_problem,
        similarity_score=0.91,
    )

    store = FakeStore(
        [match]
    )

    service = (
        SemanticLearningMemoryService(
            extractor=FakeExtractor(
                new_mastery
            ),
            store=store,
        )
    )

    result = learn(service)

    assert result is not None

    # First store the new evidence.
    assert (
        store.upsert_calls[0]
        .memory_id
        == "new-mastery"
    )

    # Then retire the old problem.
    assert len(
        store.upsert_calls
    ) == 2

    updated_old = (
        store.upsert_calls[1]
    )

    assert (
        updated_old.memory_id
        == "old-problem"
    )

    assert (
        updated_old.status
        == SemanticMemoryStatus.RESOLVED
    )

    print(
        "MASTERY_RESOLVES_OLD_PROBLEM=True"
    )


def test_new_problem_supersedes_old_mastery(
) -> None:
    old_mastery = make_record(
        memory_id="old-mastery",
        kind=SemanticMemoryKind.MASTERY,
        text=(
            "Student understands "
            "acceleration correctly."
        ),
    )

    new_problem = make_record(
        memory_id="new-problem",
        kind=(
            SemanticMemoryKind.KNOWLEDGE_GAP
        ),
        text=(
            "Student no longer distinguishes "
            "velocity from acceleration."
        ),
    )

    match = SemanticMemoryMatch(
        record=old_mastery,
        similarity_score=0.90,
    )

    store = FakeStore(
        [match]
    )

    service = (
        SemanticLearningMemoryService(
            extractor=FakeExtractor(
                new_problem
            ),
            store=store,
        )
    )

    result = learn(service)

    assert result is not None

    assert len(
        store.upsert_calls
    ) == 2

    assert (
        store.upsert_calls[0]
        .memory_id
        == "new-problem"
    )

    old_updated = (
        store.upsert_calls[1]
    )

    assert (
        old_updated.memory_id
        == "old-mastery"
    )

    assert (
        old_updated.status
        == SemanticMemoryStatus.SUPERSEDED
    )

    print(
        "NEW_PROBLEM_SUPERSEDES_MASTERY=True"
    )


def test_unrelated_memory_stays_separate(
) -> None:
    old_memory = make_record(
        memory_id="old-force-memory",
        kind=(
            SemanticMemoryKind.MISCONCEPTION
        ),
        topic="Forces",
        concept="Newton Second Law",
        text=(
            "Student confuses mass "
            "and force."
        ),
    )

    new_memory = make_record(
        memory_id="new-optics-memory",
        kind=SemanticMemoryKind.MASTERY,
        topic="Optics",
        concept="Reflection",
        text=(
            "Student understands the "
            "law of reflection."
        ),
    )

    match = SemanticMemoryMatch(
        record=old_memory,
        similarity_score=0.92,
    )

    store = FakeStore(
        [match]
    )

    service = (
        SemanticLearningMemoryService(
            extractor=FakeExtractor(
                new_memory
            ),
            store=store,
        )
    )

    result = learn(service)

    assert result is not None

    # New unrelated memory is stored.
    assert len(
        store.upsert_calls
    ) == 1

    assert (
        store.upsert_calls[0]
        .memory_id
        == "new-optics-memory"
    )

    print(
        "UNRELATED_MEMORY_KEPT_SEPARATE=True"
    )


def test_stale_memory_not_sent_to_tutor(
) -> None:
    resolved = make_record(
        memory_id="resolved-memory",
        kind=(
            SemanticMemoryKind.MISCONCEPTION
        ),
        text=(
            "Student once confused speed "
            "with acceleration."
        ),
        status=(
            SemanticMemoryStatus.RESOLVED
        ),
    )

    match = SemanticMemoryMatch(
        record=resolved,
        similarity_score=0.99,
    )

    service = (
        SemanticLearningMemoryService(
            extractor=FakeExtractor(
                resolved
            ),
            store=FakeStore(),
        )
    )

    context = (
        service.build_tutor_context(
            memories=[match]
        )
    )

    assert context is None

    print(
        "STALE_MEMORY_HIDDEN_FROM_TUTOR=True"
    )


def main() -> None:
    test_same_signal_reinforces_old_memory()

    test_mastery_resolves_old_problem()

    test_new_problem_supersedes_old_mastery()

    test_unrelated_memory_stays_separate()

    test_stale_memory_not_sent_to_tutor()

    print()
    print(
        "PHASE7_MEMORY_FRESHNESS_TEST=PASS"
    )


if __name__ == "__main__":
    main()