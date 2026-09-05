from __future__ import annotations

from pydantic import BaseModel, Field

from src.retrieval.models import (
    ContextBundle,
    FusedRetrievalHit,
    RetrievalHit,
    RerankedRetrievalHit,
)


class RetrievalSearchRequest(BaseModel):
    user_id: str = Field(
        default="local-user",
        min_length=1,
    )

    document_id: str = Field(
        min_length=1,
    )

    query: str = Field(
        min_length=1,
        max_length=2000,
    )


class RetrievalSearchResponse(BaseModel):
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