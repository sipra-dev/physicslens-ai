from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.models import (
    RetrievalChunk,
)

from src.retrieval.models import (
    RetrievalHit,
)


@dataclass(frozen=True)
class RetrievalFilter:
    """
    Retrieval isolation and metadata filtering.

    user_id and document_id are intentionally mandatory.
    """

    user_id: str
    document_id: str

    page_numbers: tuple[int, ...] | None = None

    content_types: tuple[str, ...] | None = None

    topics: tuple[str, ...] | None = None

    grade: int | None = None

    include_visual: bool = True

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError(
                "RetrievalFilter.user_id cannot be empty."
            )

        if not self.document_id.strip():
            raise ValueError(
                "RetrievalFilter.document_id "
                "cannot be empty."
            )

        if (
            self.grade is not None
            and not 1 <= self.grade <= 12
        ):
            raise ValueError(
                "grade must be between 1 and 12."
            )


def chunk_matches_filter(
    *,
    chunk: RetrievalChunk,
    retrieval_filter: RetrievalFilter,
) -> bool:
    """
    Filter RetrievalChunk objects.

    Used during Phase 3 dense/BM25 retrieval.
    """

    if (
        chunk.user_id
        != retrieval_filter.user_id
    ):
        return False

    if (
        chunk.document_id
        != retrieval_filter.document_id
    ):
        return False

    if (
        not retrieval_filter.include_visual
        and chunk.chunk_kind == "visual"
    ):
        return False

    if retrieval_filter.page_numbers:
        if (
            chunk.page_number
            not in retrieval_filter.page_numbers
        ):
            return False

    if retrieval_filter.content_types:
        if (
            chunk.content_type
            not in retrieval_filter.content_types
        ):
            return False

    if retrieval_filter.topics:
        chunk_topics = set(
            chunk.topics
        )

        required_topics = set(
            retrieval_filter.topics
        )

        if not (
            chunk_topics
            & required_topics
        ):
            return False

    if retrieval_filter.grade is not None:
        grade = retrieval_filter.grade

        if (
            chunk.grade_min is not None
            and grade < chunk.grade_min
        ):
            return False

        if (
            chunk.grade_max is not None
            and grade > chunk.grade_max
        ):
            return False

    return True


def hit_matches_filter(
    *,
    hit: RetrievalHit,
    retrieval_filter: RetrievalFilter,
) -> bool:
    """
    Filter RetrievalHit objects.

    Dense/BM25 already filter at chunk level,
    but Phase 4 applies metadata filtering again
    after fusion as defence in depth.

    This helps prevent accidental cross-user
    or cross-document retrieval.
    """

    if (
        hit.user_id
        != retrieval_filter.user_id
    ):
        return False

    if (
        hit.document_id
        != retrieval_filter.document_id
    ):
        return False

    if (
        not retrieval_filter.include_visual
        and hit.chunk_kind == "visual"
    ):
        return False

    if retrieval_filter.page_numbers:
        if (
            hit.page_number
            not in retrieval_filter.page_numbers
        ):
            return False

    if retrieval_filter.content_types:
        if (
            hit.content_type
            not in retrieval_filter.content_types
        ):
            return False

    if retrieval_filter.topics:
        hit_topics = set(
            hit.topics
        )

        required_topics = set(
            retrieval_filter.topics
        )

        if not (
            hit_topics
            & required_topics
        ):
            return False

    if retrieval_filter.grade is not None:
        grade = retrieval_filter.grade

        if (
            hit.grade_min is not None
            and grade < hit.grade_min
        ):
            return False

        if (
            hit.grade_max is not None
            and grade > hit.grade_max
        ):
            return False

    return True