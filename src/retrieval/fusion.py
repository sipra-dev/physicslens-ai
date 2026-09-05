from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.filters import (
    RetrievalFilter,
    hit_matches_filter,
)
from src.retrieval.models import (
    FusedRetrievalHit,
    RetrievalHit,
)


@dataclass
class _FusionAccumulator:
    hit: RetrievalHit
    rrf_score: float
    source_ranks: dict[str, int]
    source_scores: dict[str, float]


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF).

    Dense and BM25 scores live on different scales,
    so we combine their ranks rather than directly
    adding their raw scores.
    """

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        preferred_page_boost: float = 1.15,
        visual_boost: float = 1.10,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        if dense_weight <= 0:
            raise ValueError(
                "dense_weight must be positive."
            )

        if bm25_weight <= 0:
            raise ValueError(
                "bm25_weight must be positive."
            )

        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.preferred_page_boost = (
            preferred_page_boost
        )

        self.visual_boost = visual_boost

    def fuse(
        self,
        *,
        dense_hits: list[RetrievalHit],
        bm25_hits: list[RetrievalHit],
        retrieval_filter: RetrievalFilter,
        top_k: int = 30,
        preferred_page_numbers: (
            tuple[int, ...] | None
        ) = None,
        prefer_visual: bool = False,
    ) -> list[FusedRetrievalHit]:
        if top_k <= 0:
            return []

        accumulators: dict[
            str,
            _FusionAccumulator,
        ] = {}

        self._add_source(
            source_name="dense",
            hits=dense_hits,
            source_weight=self.dense_weight,
            retrieval_filter=retrieval_filter,
            accumulators=accumulators,
        )

        self._add_source(
            source_name="bm25",
            hits=bm25_hits,
            source_weight=self.bm25_weight,
            retrieval_filter=retrieval_filter,
            accumulators=accumulators,
        )

        fused: list[
            FusedRetrievalHit
        ] = []

        preferred_pages = set(
            preferred_page_numbers or ()
        )

        for accumulator in (
            accumulators.values()
        ):
            final_score = (
                accumulator.rrf_score
            )

            hit = accumulator.hit

            # Later the router / active-page resolver can
            # use this hook for queries such as:
            # "Explain this diagram."
            if (
                preferred_pages
                and hit.page_number
                in preferred_pages
            ):
                final_score *= (
                    self.preferred_page_boost
                )

            if (
                prefer_visual
                and hit.chunk_kind == "visual"
            ):
                final_score *= (
                    self.visual_boost
                )

            fused.append(
                FusedRetrievalHit(
                    hit=hit,
                    rrf_score=final_score,
                    source_ranks=(
                        accumulator.source_ranks
                    ),
                    source_scores=(
                        accumulator.source_scores
                    ),
                )
            )

        fused.sort(
            key=lambda item: item.rrf_score,
            reverse=True,
        )

        return fused[:top_k]

    def _add_source(
        self,
        *,
        source_name: str,
        hits: list[RetrievalHit],
        source_weight: float,
        retrieval_filter: RetrievalFilter,
        accumulators: dict[
            str,
            _FusionAccumulator,
        ],
    ) -> None:
        for rank, hit in enumerate(
            hits,
            start=1,
        ):
            if not hit_matches_filter(
                hit=hit,
                retrieval_filter=(
                    retrieval_filter
                ),
            ):
                continue

            contribution = (
                source_weight
                / (self.rrf_k + rank)
            )

            existing = accumulators.get(
                hit.chunk_id
            )

            if existing is None:
                accumulators[
                    hit.chunk_id
                ] = _FusionAccumulator(
                    hit=hit,
                    rrf_score=contribution,
                    source_ranks={
                        source_name: rank
                    },
                    source_scores={
                        source_name: hit.score
                    },
                )

                continue

            existing.rrf_score += (
                contribution
            )

            existing.source_ranks[
                source_name
            ] = rank

            existing.source_scores[
                source_name
            ] = hit.score