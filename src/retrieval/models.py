from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    chunk_id: str

    user_id: str
    document_id: str

    page_number: int = Field(
        ge=1
    )

    text: str

    content_type: str
    chunk_kind: str

    parent_id: str | None = None

    topics: list[str] = Field(
        default_factory=list
    )

    grade_min: int | None = None
    grade_max: int | None = None

    linked_figure_ids: list[str] = Field(
        default_factory=list
    )

    image_path: str | None = None

    caption: str | None = None

    score: float

    retrieval_source: str


class FusedRetrievalHit(BaseModel):
    hit: RetrievalHit

    rrf_score: float = Field(
        ge=0.0
    )

    source_ranks: dict[str, int] = Field(
        default_factory=dict
    )

    source_scores: dict[str, float] = Field(
        default_factory=dict
    )


class RerankedRetrievalHit(BaseModel):
    hit: RetrievalHit

    rrf_score: float = Field(
        ge=0.0
    )

    rerank_score: float

    source_ranks: dict[str, int] = Field(
        default_factory=dict
    )

    source_scores: dict[str, float] = Field(
        default_factory=dict
    )


class ContextItem(BaseModel):
    context_id: str

    user_id: str
    document_id: str

    page_number: int = Field(
        ge=1
    )

    source_chunk_ids: list[str] = Field(
        default_factory=list
    )

    parent_id: str | None = None

    text: str

    content_type: str

    linked_figure_ids: list[str] = Field(
        default_factory=list
    )

    equations: list[str] = Field(
        default_factory=list
    )

    image_path: str | None = None
    caption: str | None = None

    rerank_score: float


class ContextBundle(BaseModel):
    query: str

    user_id: str
    document_id: str

    items: list[ContextItem] = Field(
        default_factory=list
    )

    total_characters: int = 0

    truncated: bool = False


class HybridRetrievalResult(BaseModel):
    query: str

    dense_hits: list[RetrievalHit] = Field(
        default_factory=list
    )

    bm25_hits: list[RetrievalHit] = Field(
        default_factory=list
    )

    fused_hits: list[FusedRetrievalHit] = Field(
        default_factory=list
    )

    reranked_hits: list[
        RerankedRetrievalHit
    ] = Field(
        default_factory=list
    )

    context: ContextBundle

    evidence_found: bool = False

    failure_reason: str | None = None