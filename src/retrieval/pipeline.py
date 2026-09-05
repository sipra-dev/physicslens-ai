from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.bm25 import (
    BM25Retriever,
)
from src.retrieval.compression import (
    ContextCompressor,
)
from src.retrieval.dense import (
    DenseRetriever,
)
from src.retrieval.filters import (
    RetrievalFilter,
)
from src.retrieval.fusion import (
    ReciprocalRankFusion,
)
from src.retrieval.models import (
    ContextBundle,
    HybridRetrievalResult,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
)


class HybridRetrievalPipeline:
    """
    Complete retrieval-serving pipeline.

    Dense + BM25
        ↓
    RRF
        ↓
    Cross-Encoder reranking
        ↓
    Parent expansion
        ↓
    Context compression
    """

    def __init__(
        self,
        *,
        embedding_model_name: str,
        reranker_model_name: str,
        rrf_k: int = 60,
        candidate_pool_size: int = 30,
        per_source_top_k: int = 30,
        reranker_top_k: int = 8,
        final_context_count: int = 6,
        reranker_batch_size: int = 8,
        max_context_characters: int = 12000,
        max_context_item_characters: int = 3000,
    ) -> None:
        self.candidate_pool_size = (
            candidate_pool_size
        )

        self.per_source_top_k = (
            per_source_top_k
        )

        self.reranker_top_k = (
            reranker_top_k
        )

        self.final_context_count = (
            final_context_count
        )

        self.dense = DenseRetriever(
            model_name=embedding_model_name
        )

        self.bm25 = BM25Retriever()

        self.fusion = ReciprocalRankFusion(
            rrf_k=rrf_k
        )

        self.reranker = (
            CrossEncoderReranker(
                model_name=(
                    reranker_model_name
                ),
                batch_size=(
                    reranker_batch_size
                ),
            )
        )

        self.compressor = (
            ContextCompressor(
                max_context_characters=(
                    max_context_characters
                ),
                max_item_characters=(
                    max_context_item_characters
                ),
            )
        )

    def retrieve(
        self,
        *,
        query: str,
        index_manifest: dict[str, Any],
        retrieval_filter: RetrievalFilter,
        preferred_page_numbers: (
            tuple[int, ...] | None
        ) = None,
        prefer_visual: bool = False,
    ) -> HybridRetrievalResult:
        normalized_query = query.strip()

        if not normalized_query:
            return self._empty_result(
                query=query,
                retrieval_filter=(
                    retrieval_filter
                ),
                reason="Query is empty.",
            )

        dense_directory = self._path_from_manifest(
            manifest=index_manifest,
            key="dense_index_directory",
        )

        bm25_directory = self._path_from_manifest(
            manifest=index_manifest,
            key="bm25_index_directory",
        )

        parent_chunks_path = (
            dense_directory
            / "parent_chunks.json"
        )

        dense_hits = self.dense.search(
            query=normalized_query,
            index_directory=(
                dense_directory
            ),
            retrieval_filter=(
                retrieval_filter
            ),
            top_k=self.per_source_top_k,
        )

        bm25_hits = self.bm25.search(
            query=normalized_query,
            index_directory=(
                bm25_directory
            ),
            retrieval_filter=(
                retrieval_filter
            ),
            top_k=self.per_source_top_k,
        )

        fused_hits = self.fusion.fuse(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            retrieval_filter=(
                retrieval_filter
            ),
            top_k=self.candidate_pool_size,
            preferred_page_numbers=(
                preferred_page_numbers
            ),
            prefer_visual=prefer_visual,
        )

        if not fused_hits:
            return HybridRetrievalResult(
                query=normalized_query,
                dense_hits=dense_hits,
                bm25_hits=bm25_hits,
                fused_hits=[],
                reranked_hits=[],
                context=ContextBundle(
                    query=normalized_query,
                    user_id=(
                        retrieval_filter.user_id
                    ),
                    document_id=(
                        retrieval_filter.document_id
                    ),
                    items=[],
                    total_characters=0,
                    truncated=False,
                ),
                evidence_found=False,
                failure_reason=(
                    "No relevant retrieval "
                    "candidates were found."
                ),
            )

        reranked_hits = (
            self.reranker.rerank(
                query=normalized_query,
                candidates=fused_hits,
                top_k=self.reranker_top_k,
            )
        )

        if not reranked_hits:
            return HybridRetrievalResult(
                query=normalized_query,
                dense_hits=dense_hits,
                bm25_hits=bm25_hits,
                fused_hits=fused_hits,
                reranked_hits=[],
                context=ContextBundle(
                    query=normalized_query,
                    user_id=(
                        retrieval_filter.user_id
                    ),
                    document_id=(
                        retrieval_filter.document_id
                    ),
                    items=[],
                    total_characters=0,
                    truncated=False,
                ),
                evidence_found=False,
                failure_reason=(
                    "Candidates were retrieved "
                    "but reranking produced no "
                    "usable evidence."
                ),
            )

        context = (
            self.compressor.compress(
                query=normalized_query,
                reranked_hits=(
                    reranked_hits
                ),
                parent_chunks_path=(
                    parent_chunks_path
                ),
                user_id=(
                    retrieval_filter.user_id
                ),
                document_id=(
                    retrieval_filter.document_id
                ),
                max_contexts=(
                    self.final_context_count
                ),
            )
        )

        return HybridRetrievalResult(
            query=normalized_query,
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            fused_hits=fused_hits,
            reranked_hits=(
                reranked_hits
            ),
            context=context,
            evidence_found=bool(
                context.items
            ),
            failure_reason=(
                None
                if context.items
                else (
                    "Retrieved evidence could "
                    "not be converted into "
                    "grounded context."
                )
            ),
        )

    def _path_from_manifest(
        self,
        *,
        manifest: dict[str, Any],
        key: str,
    ) -> Path:
        value = manifest.get(key)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"Index manifest is missing {key}."
            )

        path = Path(value).resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"Index directory does not exist: "
                f"{path}"
            )

        return path

    def _empty_result(
        self,
        *,
        query: str,
        retrieval_filter: RetrievalFilter,
        reason: str,
    ) -> HybridRetrievalResult:
        return HybridRetrievalResult(
            query=query,
            dense_hits=[],
            bm25_hits=[],
            fused_hits=[],
            reranked_hits=[],
            context=ContextBundle(
                query=query,
                user_id=(
                    retrieval_filter.user_id
                ),
                document_id=(
                    retrieval_filter.document_id
                ),
                items=[],
                total_characters=0,
                truncated=False,
            ),
            evidence_found=False,
            failure_reason=reason,
        )