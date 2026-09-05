from __future__ import annotations

from datetime import datetime, timezone

from src.memory.pinecone_store import (
    PineconeSemanticMemoryStore,
)
from src.memory.semantic_extractor import (
    SemanticMemoryExtractor,
)
from src.memory.semantic_models import (
    SemanticLearningMemoryRecord,
    SemanticMemoryKind,
    SemanticMemoryMatch,
    SemanticMemoryStatus,
)


class SemanticLearningMemoryService:
    """
    High-level semantic learning-memory service.

    Responsibilities:
    1. Recall relevant old learning memories before
       the Tutor generates a new answer.
    2. Convert recalled memories into bounded context
       that can later be injected into the Tutor.
    3. Inspect a completed verified tutoring turn.
    4. Store only useful durable learning signals.
    5. Keep learning memory fresh:
       - reinforce repeated evidence;
       - resolve old misconceptions / gaps after mastery;
       - supersede stale mastery when later verified
         evidence shows a new gap or misconception.
    """

    DEFAULT_TOP_K = 5
    DEFAULT_MINIMUM_SCORE = 0.45
    DEFAULT_MAX_CONTEXT_CHARACTERS = 2000

    # Search a slightly wider candidate set when
    # deciding whether a newly extracted memory
    # duplicates or contradicts an old one.
    DEFAULT_FRESHNESS_TOP_K = 8

    # Same kind + same topic/concept must also be
    # highly similar before we treat it as repeated
    # evidence instead of a separate memory.
    DEFAULT_REINFORCEMENT_SCORE = 0.88

    # Contradiction handling is deliberately more
    # conservative than ordinary recall.
    DEFAULT_CONFLICT_SCORE = 0.60

    def __init__(
        self,
        *,
        extractor: SemanticMemoryExtractor,
        store: PineconeSemanticMemoryStore,
        top_k: int = DEFAULT_TOP_K,
        minimum_score: float = (
            DEFAULT_MINIMUM_SCORE
        ),
        max_context_characters: int = (
            DEFAULT_MAX_CONTEXT_CHARACTERS
        ),
        freshness_top_k: int = (
            DEFAULT_FRESHNESS_TOP_K
        ),
        reinforcement_score: float = (
            DEFAULT_REINFORCEMENT_SCORE
        ),
        conflict_score: float = (
            DEFAULT_CONFLICT_SCORE
        ),
    ) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        if freshness_top_k <= 0:
            raise ValueError(
                "freshness_top_k must be positive."
            )

        if not (
            -1.0
            <= minimum_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_score must be between "
                "-1.0 and 1.0."
            )

        if not (
            -1.0
            <= reinforcement_score
            <= 1.0
        ):
            raise ValueError(
                "reinforcement_score must be "
                "between -1.0 and 1.0."
            )

        if not (
            -1.0
            <= conflict_score
            <= 1.0
        ):
            raise ValueError(
                "conflict_score must be between "
                "-1.0 and 1.0."
            )

        if max_context_characters <= 0:
            raise ValueError(
                "max_context_characters "
                "must be positive."
            )

        self.extractor = extractor
        self.store = store

        self.top_k = top_k

        self.minimum_score = (
            minimum_score
        )

        self.max_context_characters = (
            max_context_characters
        )

        self.freshness_top_k = (
            freshness_top_k
        )

        self.reinforcement_score = (
            reinforcement_score
        )

        self.conflict_score = (
            conflict_score
        )

    # =========================================================
    # READ PATH
    # =========================================================

    def recall(
        self,
        *,
        user_id: str,
        query_text: str,
    ) -> list[SemanticMemoryMatch]:
        """
        Find ACTIVE old learning memories relevant
        to the student's current question.

        Pinecone failure is treated as no memory,
        not as a failure of the Tutor workflow.
        """

        normalized_user_id = (
            user_id.strip()
        )

        normalized_query = (
            query_text.strip()
        )

        if not normalized_user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        if not normalized_query:
            return []

        try:
            return self.store.search(
                user_id=normalized_user_id,
                query_text=normalized_query,
                top_k=self.top_k,
                minimum_score=(
                    self.minimum_score
                ),
            )

        except Exception:
            return []

    def build_tutor_context(
        self,
        *,
        memories: list[
            SemanticMemoryMatch
        ],
    ) -> str | None:
        """
        Convert retrieved ACTIVE memories into
        a short bounded Tutor-personalization block.

        These memories are learning information,
        not document evidence.
        """

        if not memories:
            return None

        lines: list[str] = [
            (
                "Relevant prior learning "
                "memory about this student:"
            )
        ]

        for memory_match in memories:
            record = (
                memory_match.record
            )

            # Defence in depth:
            # Pinecone search normally already filters
            # to ACTIVE records, but never expose stale
            # lifecycle records to the Tutor.
            if (
                record.status
                != SemanticMemoryStatus.ACTIVE
            ):
                continue

            score = (
                memory_match
                .similarity_score
            )

            parts = [
                (
                    f"- Type: "
                    f"{record.kind.value}"
                ),
                (
                    f"  Topic: "
                    f"{record.topic}"
                ),
            ]

            if record.concept:
                parts.append(
                    (
                        f"  Concept: "
                        f"{record.concept}"
                    )
                )

            parts.extend(
                [
                    (
                        f"  Memory: "
                        f"{record.text}"
                    ),
                    (
                        f"  Relevance: "
                        f"{score:.3f}"
                    ),
                ]
            )

            candidate = "\n".join(
                parts
            )

            current_text = "\n".join(
                lines
            )

            projected_length = (
                len(current_text)
                + len(candidate)
                + 1
            )

            if (
                projected_length
                > self.max_context_characters
            ):
                break

            lines.append(
                candidate
            )

        if len(lines) == 1:
            return None

        lines.extend(
            [
                "",
                (
                    "Use this only to adapt "
                    "the teaching approach."
                ),
                (
                    "Do not assume the memory "
                    "is infallible."
                ),
                (
                    "Current user evidence "
                    "takes priority."
                ),
            ]
        )

        result = "\n".join(
            lines
        )

        return result[
            : self.max_context_characters
        ]

    def recall_for_tutor(
        self,
        *,
        user_id: str,
        query_text: str,
    ) -> str | None:
        """
        Convenience method:

        current question
        -> Pinecone ACTIVE-memory recall
        -> Tutor-ready context
        """

        memories = self.recall(
            user_id=user_id,
            query_text=query_text,
        )

        return self.build_tutor_context(
            memories=memories,
        )

    # =========================================================
    # WRITE PATH
    # =========================================================

    def learn_from_turn(
        self,
        *,
        user_id: str,
        student_text: str,
        tutor_text: str,
        verification_passed: bool,
        verifier_text: str | None = None,
        session_id: str | None = None,
        document_id: str | None = None,
    ) -> SemanticLearningMemoryRecord | None:
        """
        Analyse one completed tutoring interaction
        and persist a useful semantic memory.

        Only verified successful Tutor interactions
        are eligible.

        Freshness policy:

        A. Same signal again:
           reinforce existing memory instead of
           creating endless duplicates.

        B. New mastery:
           resolve an old misconception / gap
           about the same concept.

        C. New misconception / knowledge gap:
           supersede an older mastery memory about
           the same concept.

        D. Unrelated learning:
           keep both memories ACTIVE.
        """

        if not verification_passed:
            return None

        try:
            record = (
                self.extractor.extract(
                    user_id=user_id,
                    student_text=student_text,
                    tutor_text=tutor_text,
                    verifier_text=(
                        verifier_text
                    ),
                    session_id=session_id,
                    document_id=document_id,
                )
            )

        except Exception:
            return None

        if record is None:
            return None

        # -----------------------------------------
        # Find ACTIVE old memories that may refer
        # to the same learning concept.
        #
        # Failure here must not prevent storing
        # the newly verified signal.
        # -----------------------------------------

        related_memories = (
            self._find_related_memories(
                record
            )
        )

        # -----------------------------------------
        # CASE 1:
        # Same learning signal appears again.
        #
        # Example:
        # old mastery of Newton's 2nd law
        # +
        # new mastery of Newton's 2nd law
        #
        # Reinforce existing record instead of
        # creating duplicate Pinecone memories.
        # -----------------------------------------

        reinforcement_candidate = (
            self._select_reinforcement_candidate(
                new_record=record,
                memories=related_memories,
            )
        )

        if (
            reinforcement_candidate
            is not None
        ):
            reinforced_record = (
                self._reinforce_record(
                    old_record=(
                        reinforcement_candidate
                        .record
                    ),
                    new_record=record,
                )
            )

            try:
                stored = self.store.upsert(
                    reinforced_record
                )

            except Exception:
                return None

            if not stored:
                return None

            return reinforced_record

        # -----------------------------------------
        # CASE 2:
        # Genuine new signal.
        #
        # Store NEW evidence first.
        #
        # Important:
        # do not retire old memory first and then
        # risk losing both if the new upsert fails.
        # -----------------------------------------

        try:
            stored = self.store.upsert(
                record
            )

        except Exception:
            return None

        if not stored:
            return None

        # -----------------------------------------
        # CASE 3:
        # New evidence contradicts old evidence.
        #
        # We keep history but remove stale records
        # from ACTIVE retrieval by changing status.
        # -----------------------------------------

        conflicts = (
            self._find_conflicts(
                new_record=record,
                memories=related_memories,
            )
        )

        for memory_match in conflicts:
            old_record = (
                memory_match.record
            )

            new_status = (
                self._replacement_status(
                    old_record=old_record,
                    new_record=record,
                )
            )

            updated_old = (
                old_record.model_copy(
                    update={
                        "status": (
                            new_status
                        ),
                        "updated_at": (
                            self._utc_now()
                        ),
                    }
                )
            )

            try:
                # Best-effort lifecycle update.
                #
                # The Tutor response must never
                # fail because Pinecone freshness
                # maintenance failed.
                self.store.upsert(
                    updated_old
                )

            except Exception:
                continue

        return record

    # =========================================================
    # FRESHNESS POLICY
    # =========================================================

    def _find_related_memories(
        self,
        record: SemanticLearningMemoryRecord,
    ) -> list[SemanticMemoryMatch]:
        """
        Retrieve possible ACTIVE duplicates or
        contradictions for one newly extracted signal.

        Failure is intentionally fail-open:
        the new verified memory can still be stored.
        """

        try:
            return self.store.search(
                user_id=record.user_id,
                query_text=(
                    record.embedding_text
                ),
                top_k=(
                    self.freshness_top_k
                ),
                include_resolved=False,
                minimum_score=(
                    min(
                        self.conflict_score,
                        self.reinforcement_score,
                    )
                ),
            )

        except Exception:
            return []

    def _select_reinforcement_candidate(
        self,
        *,
        new_record: SemanticLearningMemoryRecord,
        memories: list[
            SemanticMemoryMatch
        ],
    ) -> SemanticMemoryMatch | None:
        """
        Find the strongest old memory that represents
        essentially the same learning signal.
        """

        candidates: list[
            SemanticMemoryMatch
        ] = []

        for memory_match in memories:
            old_record = (
                memory_match.record
            )

            if (
                old_record.status
                != SemanticMemoryStatus.ACTIVE
            ):
                continue

            if (
                old_record.kind
                != new_record.kind
            ):
                continue

            if not self._same_subject(
                old_record,
                new_record,
            ):
                continue

            if (
                memory_match
                .similarity_score
                < self.reinforcement_score
            ):
                continue

            candidates.append(
                memory_match
            )

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: (
                item.similarity_score
            ),
        )

    def _find_conflicts(
        self,
        *,
        new_record: SemanticLearningMemoryRecord,
        memories: list[
            SemanticMemoryMatch
        ],
    ) -> list[SemanticMemoryMatch]:
        """
        Return ACTIVE old memories contradicted by
        the newly verified learning evidence.
        """

        conflicts: list[
            SemanticMemoryMatch
        ] = []

        for memory_match in memories:
            old_record = (
                memory_match.record
            )

            if (
                old_record.status
                != SemanticMemoryStatus.ACTIVE
            ):
                continue

            if (
                memory_match
                .similarity_score
                < self.conflict_score
            ):
                continue

            if not self._same_subject(
                old_record,
                new_record,
            ):
                continue

            if not self._kinds_conflict(
                old_record.kind,
                new_record.kind,
            ):
                continue

            conflicts.append(
                memory_match
            )

        return conflicts

    def _reinforce_record(
        self,
        *,
        old_record: SemanticLearningMemoryRecord,
        new_record: SemanticLearningMemoryRecord,
    ) -> SemanticLearningMemoryRecord:
        """
        Refresh an existing memory after repeated
        verified evidence.

        Preserve the old memory_id so Pinecone
        overwrites the same record instead of
        creating a duplicate.
        """

        now = self._utc_now()

        return old_record.model_copy(
            update={
                "confidence": max(
                    old_record.confidence,
                    new_record.confidence,
                ),
                "status": (
                    SemanticMemoryStatus.ACTIVE
                ),
                "source_session_id": (
                    new_record.source_session_id
                    or old_record
                    .source_session_id
                ),
                "source_document_id": (
                    new_record
                    .source_document_id
                    or old_record
                    .source_document_id
                ),
                "updated_at": now,
                "last_reinforced_at": now,
            }
        )

    @classmethod
    def _same_subject(
        cls,
        left: SemanticLearningMemoryRecord,
        right: SemanticLearningMemoryRecord,
    ) -> bool:
        """
        Conservative concept matching.

        Exact normalized topic is required.

        If both records contain a concept, the
        normalized concept must also match.

        If only one has a concept, do NOT assume
        they describe the same subject.
        """

        if (
            cls._normalize_key(
                left.topic
            )
            != cls._normalize_key(
                right.topic
            )
        ):
            return False

        left_concept = (
            cls._normalize_optional_key(
                left.concept
            )
        )

        right_concept = (
            cls._normalize_optional_key(
                right.concept
            )
        )

        if (
            left_concept is None
            and right_concept is None
        ):
            return True

        if (
            left_concept is None
            or right_concept is None
        ):
            return False

        return (
            left_concept
            == right_concept
        )

    @staticmethod
    def _kinds_conflict(
        old_kind: SemanticMemoryKind,
        new_kind: SemanticMemoryKind,
    ) -> bool:
        """
        Only clear pedagogical contradictions are
        automatically retired.

        misconception / knowledge_gap
            <-> mastery

        Support preference is independent and
        should not be retired by subject mastery.
        """

        difficulty_kinds = {
            SemanticMemoryKind.MISCONCEPTION,
            SemanticMemoryKind.KNOWLEDGE_GAP,
        }

        if (
            old_kind
            in difficulty_kinds
            and new_kind
            == SemanticMemoryKind.MASTERY
        ):
            return True

        if (
            old_kind
            == SemanticMemoryKind.MASTERY
            and new_kind
            in difficulty_kinds
        ):
            return True

        return False

    @staticmethod
    def _replacement_status(
        *,
        old_record: SemanticLearningMemoryRecord,
        new_record: SemanticLearningMemoryRecord,
    ) -> SemanticMemoryStatus:
        """
        Decide how the old contradictory record
        should be labelled.

        Old difficulty + new mastery:
            RESOLVED

        Old mastery + new difficulty:
            SUPERSEDED
        """

        difficulty_kinds = {
            SemanticMemoryKind.MISCONCEPTION,
            SemanticMemoryKind.KNOWLEDGE_GAP,
        }

        if (
            old_record.kind
            in difficulty_kinds
            and new_record.kind
            == SemanticMemoryKind.MASTERY
        ):
            return (
                SemanticMemoryStatus.RESOLVED
            )

        return (
            SemanticMemoryStatus.SUPERSEDED
        )

    @staticmethod
    def _normalize_key(
        value: str,
    ) -> str:
        return " ".join(
            value.strip().lower().split()
        )

    @classmethod
    def _normalize_optional_key(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = cls._normalize_key(
            value
        )

        return normalized or None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(
            timezone.utc
        )