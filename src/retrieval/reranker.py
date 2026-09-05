from __future__ import annotations

from sentence_transformers import (
    CrossEncoder,
)

from src.retrieval.models import (
    FusedRetrievalHit,
    RerankedRetrievalHit,
)


class RerankerError(Exception):
    """Raised when cross-encoder reranking fails."""


class CrossEncoderReranker:
    """
    Second-stage reranker.

    Stage 1:
        FAISS + BM25 + RRF

    Stage 2:
        Cross-Encoder query/document scoring
    """

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int = 8,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        self.model_name = model_name
        self.batch_size = batch_size

        self._model: CrossEncoder | None = (
            None
        )

    @property
    def model(
        self,
    ) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name
            )

        return self._model

    def rerank(
        self,
        *,
        query: str,
        candidates: list[
            FusedRetrievalHit
        ],
        top_k: int = 8,
    ) -> list[RerankedRetrievalHit]:
        normalized_query = query.strip()

        if not normalized_query:
            return []

        if not candidates:
            return []

        if top_k <= 0:
            return []

        documents = [
            candidate.hit.text
            for candidate in candidates
        ]

        try:
            ranked = self.model.rank(
                normalized_query,
                documents,
                top_k=min(
                    top_k,
                    len(documents),
                ),
                return_documents=False,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

        except Exception as exc:
            raise RerankerError(
                "Cross-encoder reranking failed."
            ) from exc

        results: list[
            RerankedRetrievalHit
        ] = []

        for item in ranked:
            corpus_id = int(
                item["corpus_id"]
            )

            score = float(
                item["score"]
            )

            if (
                corpus_id < 0
                or corpus_id >= len(candidates)
            ):
                continue

            candidate = candidates[
                corpus_id
            ]

            results.append(
                RerankedRetrievalHit(
                    hit=candidate.hit,
                    rrf_score=(
                        candidate.rrf_score
                    ),
                    rerank_score=score,
                    source_ranks=(
                        candidate.source_ranks
                    ),
                    source_scores=(
                        candidate.source_scores
                    ),
                )
            )

        return results