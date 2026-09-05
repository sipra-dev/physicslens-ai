from __future__ import annotations

import json
import re
from pathlib import Path

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.compression import ContextCompressor
from src.retrieval.dense import DenseRetriever
from src.retrieval.filters import (
    RetrievalFilter,
    hit_matches_filter,
)
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.models import (
    ContextBundle,
    FusedRetrievalHit,
    HybridRetrievalResult,
    RerankedRetrievalHit,
    RetrievalHit,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
)


_SAFE_COMPONENT = re.compile(
    r"[^A-Za-z0-9._-]+"
)


class RetrievalServiceError(Exception):
    """Raised when the hybrid retrieval pipeline fails."""


class RetrievalService:
    """
    Phase-4 deterministic retrieval pipeline.

    Flow:

        Query
          ↓
        Dense + BM25
          ↓
        Reciprocal Rank Fusion
          ↓
        Top 30
          ↓
        Metadata filtering
          ↓
        CrossEncoder reranking
          ↓
        Relevance score gate
          ↓
        Top 5-8 relevant hits
          ↓
        Parent expansion
          ↓
        Context compression
          ↓
        HybridRetrievalResult

    This service does NOT generate an answer.
    It only prepares grounded evidence for the
    future Tutor Agent.
    """

    def __init__(
        self,
        *,
        vector_store_directory: Path,
        bm25_store_directory: Path,
        embedding_model_name: str,
        reranker_model_name: str,
        rrf_k: int = 60,
        reranker_batch_size: int = 8,
        max_context_characters: int = 12000,
        max_item_characters: int = 3000,
        minimum_rerank_score: float = 0.0,
    ) -> None:
        if not embedding_model_name.strip():
            raise ValueError(
                "embedding_model_name cannot be empty."
            )

        if not reranker_model_name.strip():
            raise ValueError(
                "reranker_model_name cannot be empty."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be positive."
            )

        self.vector_store_directory = (
            vector_store_directory.resolve()
        )

        self.bm25_store_directory = (
            bm25_store_directory.resolve()
        )

        self.minimum_rerank_score = (
            float(minimum_rerank_score)
        )

        self.dense_retriever = DenseRetriever(
            model_name=embedding_model_name
        )

        self.bm25_retriever = BM25Retriever()

        self.fusion = ReciprocalRankFusion(
            rrf_k=rrf_k,
        )

        self.reranker = CrossEncoderReranker(
            model_name=reranker_model_name,
            batch_size=reranker_batch_size,
        )

        self.compressor = ContextCompressor(
            max_context_characters=(
                max_context_characters
            ),
            max_item_characters=(
                max_item_characters
            ),
        )

    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        fused_top_k: int = 30,
        rerank_top_k: int = 8,
        max_contexts: int = 6,
        page_numbers: (
            tuple[int, ...] | None
        ) = None,
        content_types: (
            tuple[str, ...] | None
        ) = None,
        topics: (
            tuple[str, ...] | None
        ) = None,
        grade: int | None = None,
        include_visual: bool = True,
        preferred_page_numbers: (
            tuple[int, ...] | None
        ) = None,
        prefer_visual: bool = False,
        required_chunk_ids: (
            tuple[str, ...] | None
        ) = None,
        required_parent_ids: (
            tuple[str, ...] | None
        ) = None,
    ) -> HybridRetrievalResult:
        """
        Run the complete Phase-4 retrieval pipeline.
        """

        normalized_query = query.strip()
        normalized_user_id = user_id.strip()
        normalized_document_id = (
            document_id.strip()
        )

        if not normalized_query:
            return self._empty_result(
                query="",
                user_id=normalized_user_id,
                document_id=(
                    normalized_document_id
                ),
                reason="EMPTY_QUERY",
            )

        if not normalized_user_id:
            raise RetrievalServiceError(
                "user_id cannot be empty."
            )

        if not normalized_document_id:
            raise RetrievalServiceError(
                "document_id cannot be empty."
            )

        if dense_top_k <= 0:
            raise RetrievalServiceError(
                "dense_top_k must be positive."
            )

        if bm25_top_k <= 0:
            raise RetrievalServiceError(
                "bm25_top_k must be positive."
            )

        if fused_top_k <= 0:
            raise RetrievalServiceError(
                "fused_top_k must be positive."
            )

        if rerank_top_k <= 0:
            raise RetrievalServiceError(
                "rerank_top_k must be positive."
            )

        if max_contexts <= 0:
            raise RetrievalServiceError(
                "max_contexts must be positive."
            )

        normalized_required_chunk_ids = (
            self._normalized_identifier_tuple(
                required_chunk_ids
            )
        )

        normalized_required_parent_ids = (
            self._normalized_identifier_tuple(
                required_parent_ids
            )
        )

        retrieval_filter = RetrievalFilter(
            user_id=normalized_user_id,
            document_id=(
                normalized_document_id
            ),
            page_numbers=page_numbers,
            content_types=content_types,
            topics=topics,
            grade=grade,
            include_visual=include_visual,
        )

        (
            dense_directory,
            bm25_directory,
            parent_chunks_path,
        ) = self._document_paths(
            user_id=normalized_user_id,
            document_id=(
                normalized_document_id
            ),
        )

        self._validate_index_files(
            dense_directory=dense_directory,
            bm25_directory=bm25_directory,
        )

        try:
            # ---------------------------------
            # 0. Verified structural evidence
            # ---------------------------------
            #
            # A structure-aware resolver may already have selected exact
            # retrieval chunk IDs (or, for older artifacts, parent IDs).
            # In that case we load those real indexed records directly.
            # We do not ask semantic retrieval to rediscover an already
            # verified source identity, and we never substitute a nearby
            # semantic hit when a required ID is missing.
            if (
                normalized_required_chunk_ids
                or normalized_required_parent_ids
            ):
                return self._retrieve_required_structural_evidence(
                    query=normalized_query,
                    user_id=normalized_user_id,
                    document_id=normalized_document_id,
                    dense_metadata_path=(
                        dense_directory
                        / "dense_metadata.json"
                    ),
                    parent_chunks_path=(
                        parent_chunks_path
                    ),
                    retrieval_filter=(
                        retrieval_filter
                    ),
                    required_chunk_ids=(
                        normalized_required_chunk_ids
                    ),
                    required_parent_ids=(
                        normalized_required_parent_ids
                    ),
                    rerank_top_k=rerank_top_k,
                    max_contexts=max_contexts,
                    prefer_visual=prefer_visual,
                )

            # ---------------------------------
            # 1. Dense retrieval
            # ---------------------------------

            dense_hits = (
                self.dense_retriever.search(
                    query=normalized_query,
                    index_directory=(
                        dense_directory
                    ),
                    retrieval_filter=(
                        retrieval_filter
                    ),
                    top_k=dense_top_k,
                )
            )

            # ---------------------------------
            # 2. BM25 retrieval
            # ---------------------------------

            bm25_hits = (
                self.bm25_retriever.search(
                    query=normalized_query,
                    index_directory=(
                        bm25_directory
                    ),
                    retrieval_filter=(
                        retrieval_filter
                    ),
                    top_k=bm25_top_k,
                )
            )

            if (
                not dense_hits
                and not bm25_hits
            ):
                return self._result_without_evidence(
                    query=normalized_query,
                    user_id=normalized_user_id,
                    document_id=(
                        normalized_document_id
                    ),
                    dense_hits=dense_hits,
                    bm25_hits=bm25_hits,
                    reason="NO_RETRIEVAL_HITS",
                )

            # ---------------------------------
            # 3. Reciprocal Rank Fusion
            # ---------------------------------

            fused_hits = self.fusion.fuse(
                dense_hits=dense_hits,
                bm25_hits=bm25_hits,
                retrieval_filter=(
                    retrieval_filter
                ),
                top_k=fused_top_k,
                preferred_page_numbers=(
                    preferred_page_numbers
                ),
                prefer_visual=prefer_visual,
            )

            # ---------------------------------
            # 4. Metadata filtering AGAIN
            #
            # Dense and BM25 already filter.
            # Fusion also applies the filter.
            #
            # We deliberately verify once more
            # after fusion as defence in depth.
            # ---------------------------------

            filtered_fused_hits = [
                fused_hit
                for fused_hit in fused_hits
                if hit_matches_filter(
                    hit=fused_hit.hit,
                    retrieval_filter=(
                        retrieval_filter
                    ),
                )
            ]

            # Respect the design's candidate
            # pool boundary.
            filtered_fused_hits = (
                filtered_fused_hits[
                    :fused_top_k
                ]
            )

            if not filtered_fused_hits:
                return HybridRetrievalResult(
                    query=normalized_query,
                    dense_hits=dense_hits,
                    bm25_hits=bm25_hits,
                    fused_hits=[],
                    reranked_hits=[],
                    context=self._empty_context(
                        query=normalized_query,
                        user_id=(
                            normalized_user_id
                        ),
                        document_id=(
                            normalized_document_id
                        ),
                    ),
                    evidence_found=False,
                    failure_reason=(
                        "NO_FILTERED_CANDIDATES"
                    ),
                )

            # ---------------------------------
            # 5. CrossEncoder reranking
            # ---------------------------------

            reranked_hits = (
                self.reranker.rerank(
                    query=normalized_query,
                    candidates=(
                        filtered_fused_hits
                    ),
                    top_k=min(
                        rerank_top_k,
                        8,
                    ),
                )
            )

            # A CrossEncoder is text-only. For an explicitly visual
            # question, the ordinary text top-k can accidentally omit
            # the image-bearing chunk even though RRF retrieved it.
            #
            # We do NOT bypass relevance checking and we do NOT invent
            # a score. We separately score the strongest fused visual
            # candidate with the same CrossEncoder and add it to the
            # reranked pool only when it was absent from the text top-k.
            if prefer_visual and include_visual:
                reranked_hits = (
                    self._retain_best_visual_candidate(
                        query=normalized_query,
                        fused_hits=(
                            filtered_fused_hits
                        ),
                        reranked_hits=reranked_hits,
                    )
                )

            if not reranked_hits:
                return HybridRetrievalResult(
                    query=normalized_query,
                    dense_hits=dense_hits,
                    bm25_hits=bm25_hits,
                    fused_hits=(
                        filtered_fused_hits
                    ),
                    reranked_hits=[],
                    context=self._empty_context(
                        query=normalized_query,
                        user_id=(
                            normalized_user_id
                        ),
                        document_id=(
                            normalized_document_id
                        ),
                    ),
                    evidence_found=False,
                    failure_reason=(
                        "NO_RERANKED_HITS"
                    ),
                )

            # ---------------------------------
            # 6. Relevance score gate
            #
            # CrossEncoder returns a relevance
            # score for every reranked candidate.
            #
            # Strict document mode should not
            # pass clearly negative evidence to
            # the future Tutor Agent.
            # ---------------------------------

            relevant_reranked_hits = [
                item
                for item in reranked_hits
                if (
                    item.rerank_score
                    >= self.minimum_rerank_score
                )
            ]

            # CrossEncoder scores are ranking scores,
            # not calibrated probabilities. A hard
            # zero cutoff can occasionally remove useful
            # evidence even when Dense and BM25 agree.
            #
            # If only one positive-scoring hit survives,
            # treat that as weak reranker confidence and
            # recover candidates supported by BOTH Dense
            # and BM25. This preserves hybrid-consensus
            # evidence without reopening candidates that
            # came from only one retrieval source.
            if len(relevant_reranked_hits) == 1:
                surviving_chunk_ids = {
                    item.hit.chunk_id
                    for item in relevant_reranked_hits
                }

                surviving_parent_ids = {
                    item.hit.parent_id
                    or item.hit.chunk_id
                    for item in relevant_reranked_hits
                }

                max_supporting_parents = 2
                added_supporting_parents = 0

                for item in reranked_hits:
                    if (
                        item.hit.chunk_id
                        in surviving_chunk_ids
                    ):
                        continue

                    source_names = set(
                        item.source_ranks.keys()
                    )

                    if not {
                        "dense",
                        "bm25",
                    }.issubset(source_names):
                        continue

                    candidate_parent_id = (
                        item.hit.parent_id
                        or item.hit.chunk_id
                    )

                    if (
                        candidate_parent_id
                        in surviving_parent_ids
                    ):
                        continue

                    relevant_reranked_hits.append(
                        item
                    )
                    surviving_chunk_ids.add(
                        item.hit.chunk_id
                    )
                    surviving_parent_ids.add(
                        candidate_parent_id
                    )

                    added_supporting_parents += 1

                    if (
                        added_supporting_parents
                        >= max_supporting_parents
                    ):
                        break

            # For source-image turns, the CrossEncoder's absolute score is
            # not a reliable gate for the image-bearing visual chunk.
            #
            # This matters not only for diagram questions but also for
            # numericals whose givens live inside an uploaded image:
            # OCR/text evidence can survive the score gate while the real
            # visual chunk is removed by the absolute CrossEncoder cutoff.
            #
            # Keep the ordinary text relevance gate intact. When visual
            # evidence is requested, ensure that one trustworthy image-bearing
            # chunk is also preserved when the already-selected evidence does
            # not contain one.
            #
            # Recovery remains conservative:
            #   - visual mode must be explicitly requested;
            #   - the candidate must be a real visual chunk;
            #   - it must carry a local image_path that exists;
            #   - if a preferred page is known, recover only from that page;
            #   - without a page hint, recover only when there is exactly
            #     one valid visual candidate, avoiding multi-figure guessing.
            if prefer_visual and include_visual:
                selected_has_visual = any(
                    self._has_existing_visual_evidence(
                        item.hit
                    )
                    for item in relevant_reranked_hits
                )

                if not selected_has_visual:
                    recovered_visual = (
                        self._recover_structural_visual_evidence(
                            reranked_hits=reranked_hits,
                            preferred_page_numbers=(
                                preferred_page_numbers
                            ),
                        )
                    )

                    if recovered_visual is not None:
                        recovered_chunk_id = (
                            recovered_visual.hit.chunk_id
                        )

                        if all(
                            item.hit.chunk_id
                            != recovered_chunk_id
                            for item
                            in relevant_reranked_hits
                        ):
                            relevant_reranked_hits.append(
                                recovered_visual
                            )

                relevant_reranked_hits = (
                    self._prioritize_relevant_visual_hit(
                        relevant_reranked_hits
                    )
                )

            if not relevant_reranked_hits:
                return HybridRetrievalResult(
                    query=normalized_query,
                    dense_hits=dense_hits,
                    bm25_hits=bm25_hits,
                    fused_hits=(
                        filtered_fused_hits
                    ),
                    reranked_hits=[],
                    context=self._empty_context(
                        query=normalized_query,
                        user_id=(
                            normalized_user_id
                        ),
                        document_id=(
                            normalized_document_id
                        ),
                    ),
                    evidence_found=False,
                    failure_reason=(
                        "NO_RELEVANT_RERANKED_HITS"
                    ),
                )

            # ---------------------------------
            # 7. Parent expansion +
            #    deterministic compression
            # ---------------------------------

            context = self.compressor.compress(
                query=normalized_query,
                reranked_hits=(
                    relevant_reranked_hits
                ),
                parent_chunks_path=(
                    parent_chunks_path
                ),
                user_id=normalized_user_id,
                document_id=(
                    normalized_document_id
                ),
                max_contexts=max_contexts,
            )

            evidence_found = bool(
                context.items
            )

            failure_reason = (
                None
                if evidence_found
                else "NO_USABLE_CONTEXT"
            )

            return HybridRetrievalResult(
                query=normalized_query,
                dense_hits=dense_hits,
                bm25_hits=bm25_hits,
                fused_hits=(
                    filtered_fused_hits
                ),
                reranked_hits=(
                    relevant_reranked_hits
                ),
                context=context,
                evidence_found=(
                    evidence_found
                ),
                failure_reason=(
                    failure_reason
                ),
            )

        except RetrievalServiceError:
            raise

        except Exception as exc:
            raise RetrievalServiceError(
                "Hybrid document retrieval failed."
            ) from exc

    def _retrieve_required_structural_evidence(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
        dense_metadata_path: Path,
        parent_chunks_path: Path,
        retrieval_filter: RetrievalFilter,
        required_chunk_ids: tuple[str, ...],
        required_parent_ids: tuple[str, ...],
        rerank_top_k: int,
        max_contexts: int,
        prefer_visual: bool,
    ) -> HybridRetrievalResult:
        """
        Load resolver-verified evidence by canonical indexed identity.

        Exact chunk IDs are authoritative when present. Parent IDs are used
        only for older structure artifacts that do not contain linked child
        chunk IDs. Missing required identities fail closed: a semantically
        nearby chunk is never substituted for the requested source item.
        """

        try:
            metadata_payload = json.loads(
                dense_metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise RetrievalServiceError(
                "Structural evidence metadata could not be read."
            ) from exc

        required_chunk_set = set(
            required_chunk_ids
        )
        required_parent_set = set(
            required_parent_ids
        )

        use_exact_chunk_ids = bool(
            required_chunk_set
        )

        selected_hits: list[RetrievalHit] = []
        seen_chunk_ids: set[str] = set()

        for raw_record in (
            self._iter_indexed_chunk_records(
                metadata_payload
            )
        ):
            record = self._flatten_chunk_record(
                raw_record
            )

            chunk_id = str(
                record.get("chunk_id", "")
            ).strip()

            parent_id_value = record.get(
                "parent_id"
            )
            parent_id = (
                str(parent_id_value).strip()
                if parent_id_value is not None
                else ""
            )

            if use_exact_chunk_ids:
                if chunk_id not in required_chunk_set:
                    continue
            elif parent_id not in required_parent_set:
                continue

            if (
                not chunk_id
                or chunk_id in seen_chunk_ids
            ):
                continue

            hit_payload = {
                **record,
                # This is an exact canonical-ID identity match, not a
                # semantic similarity probability.
                "score": 1.0,
                "retrieval_source": "structural",
            }

            try:
                hit = RetrievalHit.model_validate(
                    hit_payload
                )
            except Exception:
                # An incomplete/corrupt metadata record is not turned into
                # invented evidence. The missing-ID check below fails closed.
                continue

            if (
                hit.user_id != user_id
                or hit.document_id != document_id
            ):
                continue

            if not hit_matches_filter(
                hit=hit,
                retrieval_filter=retrieval_filter,
            ):
                continue

            selected_hits.append(hit)
            seen_chunk_ids.add(chunk_id)

        if use_exact_chunk_ids:
            missing_chunk_ids = (
                required_chunk_set
                - seen_chunk_ids
            )

            if missing_chunk_ids:
                return self._empty_result(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    reason=(
                        "REQUIRED_STRUCTURAL_CHUNKS_NOT_FOUND"
                    ),
                )

            chunk_order = {
                chunk_id: index
                for index, chunk_id in enumerate(
                    required_chunk_ids
                )
            }

            selected_hits.sort(
                key=lambda hit: (
                    chunk_order.get(
                        hit.chunk_id,
                        len(chunk_order),
                    ),
                    hit.chunk_id,
                )
            )

        else:
            found_parent_ids = {
                hit.parent_id
                for hit in selected_hits
                if hit.parent_id is not None
            }

            if (
                required_parent_set
                - found_parent_ids
            ):
                return self._empty_result(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    reason=(
                        "REQUIRED_STRUCTURAL_PARENTS_NOT_FOUND"
                    ),
                )

            parent_order = {
                parent_id: index
                for index, parent_id in enumerate(
                    required_parent_ids
                )
            }

            selected_hits.sort(
                key=lambda hit: (
                    parent_order.get(
                        hit.parent_id or "",
                        len(parent_order),
                    ),
                    hit.page_number,
                    hit.chunk_id,
                )
            )

        if not selected_hits:
            return self._empty_result(
                query=query,
                user_id=user_id,
                document_id=document_id,
                reason="NO_STRUCTURAL_EVIDENCE",
            )

        fused_hits = [
            FusedRetrievalHit(
                hit=hit,
                # Structural identity is a separate deterministic source,
                # so no fake dense/BM25 RRF score is assigned.
                rrf_score=0.0,
                source_ranks={
                    "structural": rank
                },
                source_scores={
                    "structural": 1.0
                },
            )
            for rank, hit in enumerate(
                selected_hits,
                start=1,
            )
        ]

        reranked_hits = self.reranker.rerank(
            query=query,
            candidates=fused_hits,
            # Every item is resolver-linked evidence. Reranking orders the
            # evidence; it does not silently discard a required identity.
            top_k=max(
                rerank_top_k,
                len(fused_hits),
            ),
        )

        if not reranked_hits:
            return HybridRetrievalResult(
                query=query,
                dense_hits=[],
                bm25_hits=[],
                fused_hits=fused_hits,
                reranked_hits=[],
                context=self._empty_context(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                ),
                evidence_found=False,
                failure_reason=(
                    "NO_STRUCTURAL_RERANKED_HITS"
                ),
            )

        if prefer_visual:
            reranked_hits = (
                self._prioritize_relevant_visual_hit(
                    reranked_hits
                )
            )

        context = self.compressor.compress(
            query=query,
            reranked_hits=reranked_hits,
            parent_chunks_path=(
                parent_chunks_path
            ),
            user_id=user_id,
            document_id=document_id,
            max_contexts=max(
                max_contexts,
                len(reranked_hits),
            ),
        )

        evidence_found = bool(
            context.items
        )

        return HybridRetrievalResult(
            query=query,
            dense_hits=[],
            bm25_hits=[],
            fused_hits=fused_hits,
            reranked_hits=reranked_hits,
            context=context,
            evidence_found=evidence_found,
            failure_reason=(
                None
                if evidence_found
                else "NO_USABLE_STRUCTURAL_CONTEXT"
            ),
        )

    @staticmethod
    def _normalized_identifier_tuple(
        values: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        normalized: list[str] = []

        for value in values or ():
            item = str(value).strip()

            if item and item not in normalized:
                normalized.append(item)

        return tuple(normalized)

    @classmethod
    def _iter_indexed_chunk_records(
        cls,
        payload,
    ):
        """Yield chunk-shaped dictionaries from supported JSON envelopes."""

        if isinstance(payload, dict):
            flattened = cls._flatten_chunk_record(
                payload
            )

            if isinstance(
                flattened.get("chunk_id"),
                str,
            ):
                yield flattened

            for value in payload.values():
                if isinstance(
                    value,
                    (dict, list),
                ):
                    yield from (
                        cls._iter_indexed_chunk_records(
                            value
                        )
                    )

        elif isinstance(payload, list):
            for value in payload:
                yield from (
                    cls._iter_indexed_chunk_records(
                        value
                    )
                )

    @staticmethod
    def _flatten_chunk_record(
        record: dict,
    ) -> dict:
        """Merge common metadata wrappers without changing source values."""

        flattened: dict = {}

        for key in (
            "metadata",
            "hit",
            "chunk",
            "record",
        ):
            nested = record.get(key)

            if isinstance(nested, dict):
                flattened.update(nested)

        flattened.update(
            {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "metadata",
                    "hit",
                    "chunk",
                    "record",
                }
            }
        )

        return flattened

    def _retain_best_visual_candidate(
        self,
        *,
        query: str,
        fused_hits: list[FusedRetrievalHit],
        reranked_hits: list[RerankedRetrievalHit],
    ) -> list[RerankedRetrievalHit]:
        """
        Preserve one image-bearing candidate for visual questions.

        The normal CrossEncoder top-k is text-only and may omit a
        useful visual hit. We therefore take the highest-RRF visual
        candidate that actually carries an image_path, score it with
        the SAME CrossEncoder, and append it only if it is not already
        present. No synthetic relevance score is created here.
        """
        if any(
            self._has_visual_evidence(item.hit)
            for item in reranked_hits
        ):
            return reranked_hits

        best_visual_fused = next(
            (
                item
                for item in fused_hits
                if self._has_visual_evidence(
                    item.hit
                )
            ),
            None,
        )

        if best_visual_fused is None:
            return reranked_hits

        visual_reranked = self.reranker.rerank(
            query=query,
            candidates=[best_visual_fused],
            top_k=1,
        )

        if not visual_reranked:
            return reranked_hits

        visual_item = visual_reranked[0]

        if any(
            item.hit.chunk_id
            == visual_item.hit.chunk_id
            for item in reranked_hits
        ):
            return reranked_hits

        return [
            *reranked_hits,
            visual_item,
        ]

    @classmethod
    def _recover_structural_visual_evidence(
        cls,
        *,
        reranked_hits: list[RerankedRetrievalHit],
        preferred_page_numbers: (
            tuple[int, ...] | None
        ),
    ) -> RerankedRetrievalHit | None:
        """
        Recover trustworthy visual evidence when a generic deictic
        visual query receives only negative CrossEncoder scores.

        This is intentionally NOT a global score-threshold bypass.
        It is a separate structural gate for explicitly visual turns.

        Recovery is allowed only when the candidate:
          - is a visual chunk,
          - has a non-empty image_path,
          - points to an existing local image file.

        Ambiguity is handled conservatively:
          - if preferred pages are known, only candidates on those
            pages are considered;
          - without a preferred page, recovery occurs only when there
            is exactly one valid visual candidate.

        When multiple valid visuals exist on one preferred page, the
        CrossEncoder and RRF scores are used only for relative ranking,
        never as an absolute calibrated probability.
        """
        valid_visual_hits = [
            item
            for item in reranked_hits
            if cls._has_existing_visual_evidence(
                item.hit
            )
        ]

        if not valid_visual_hits:
            return None

        preferred_pages = {
            int(page_number)
            for page_number in (
                preferred_page_numbers or ()
            )
            if int(page_number) > 0
        }

        if preferred_pages:
            page_visual_hits = [
                item
                for item in valid_visual_hits
                if item.hit.page_number
                in preferred_pages
            ]

            if not page_visual_hits:
                return None

            return max(
                page_visual_hits,
                key=lambda item: (
                    item.rerank_score,
                    item.rrf_score,
                    item.hit.chunk_id,
                ),
            )

        if len(valid_visual_hits) != 1:
            return None

        return valid_visual_hits[0]

    @staticmethod
    def _has_existing_visual_evidence(
        hit,
    ) -> bool:
        """
        Return True only for a visual chunk whose referenced local
        image file currently exists.
        """
        image_path = (
            hit.image_path.strip()
            if isinstance(
                hit.image_path,
                str,
            )
            else ""
        )

        if (
            hit.chunk_kind != "visual"
            or not image_path
        ):
            return False

        try:
            return Path(image_path).is_file()
        except OSError:
            return False

    @classmethod
    def _prioritize_relevant_visual_hit(
        cls,
        hits: list[RerankedRetrievalHit],
    ) -> list[RerankedRetrievalHit]:
        """
        Put the best already-relevant image-bearing hit first.

        This only affects visual turns and only after the ordinary
        minimum rerank-score gate has accepted the item. It prevents
        ContextCompressor's bounded max_contexts window from dropping
        the required image behind several text-only contexts.
        """
        visual_hits = [
            item
            for item in hits
            if cls._has_visual_evidence(
                item.hit
            )
        ]

        if not visual_hits:
            return hits

        best_visual = max(
            visual_hits,
            key=lambda item: (
                item.rerank_score,
                item.rrf_score,
                item.hit.chunk_id,
            ),
        )

        return [
            best_visual,
            *(
                item
                for item in hits
                if item.hit.chunk_id
                != best_visual.hit.chunk_id
            ),
        ]

    @staticmethod
    def _has_visual_evidence(
        hit,
    ) -> bool:
        image_path = (
            hit.image_path.strip()
            if isinstance(
                hit.image_path,
                str,
            )
            else ""
        )

        return bool(
            hit.chunk_kind == "visual"
            and image_path
        )

    def _document_paths(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> tuple[
        Path,
        Path,
        Path,
    ]:
        """
        Reconstruct exactly the same document-specific
        index paths used by LocalDocumentIndexer.
        """

        safe_user_id = self._safe_component(
            user_id,
            fallback="local-user",
        )

        safe_document_id = (
            self._safe_component(
                document_id,
                fallback="document",
            )
        )

        dense_directory = (
            self.vector_store_directory
            / "users"
            / safe_user_id
            / "documents"
            / safe_document_id
        )

        bm25_directory = (
            self.bm25_store_directory
            / "users"
            / safe_user_id
            / "documents"
            / safe_document_id
        )

        parent_chunks_path = (
            dense_directory
            / "parent_chunks.json"
        )

        return (
            dense_directory,
            bm25_directory,
            parent_chunks_path,
        )

    def _validate_index_files(
        self,
        *,
        dense_directory: Path,
        bm25_directory: Path,
    ) -> None:
        """
        Fail clearly if the requested document has
        not been indexed or its index is incomplete.
        """

        dense_index_path = (
            dense_directory
            / "dense.faiss"
        )

        dense_metadata_path = (
            dense_directory
            / "dense_metadata.json"
        )

        bm25_path = (
            bm25_directory
            / "bm25_corpus.json"
        )

        missing: list[str] = []

        if not dense_index_path.is_file():
            missing.append(
                str(dense_index_path)
            )

        if not dense_metadata_path.is_file():
            missing.append(
                str(dense_metadata_path)
            )

        if not bm25_path.is_file():
            missing.append(
                str(bm25_path)
            )

        if missing:
            raise RetrievalServiceError(
                "Document retrieval indexes "
                "are missing or incomplete. "
                "The document may not be READY. "
                f"Missing files: {missing}"
            )

    def _safe_component(
        self,
        value: str,
        *,
        fallback: str,
    ) -> str:
        """
        Keep path construction consistent with
        LocalDocumentIndexer.
        """

        cleaned = _SAFE_COMPONENT.sub(
            "_",
            value.strip(),
        )

        cleaned = cleaned.strip(
            "._"
        )

        return cleaned or fallback

    def _empty_context(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
    ) -> ContextBundle:
        return ContextBundle(
            query=query,
            user_id=user_id,
            document_id=document_id,
            items=[],
            total_characters=0,
            truncated=False,
        )

    def _empty_result(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
        reason: str,
    ) -> HybridRetrievalResult:
        return HybridRetrievalResult(
            query=query,
            dense_hits=[],
            bm25_hits=[],
            fused_hits=[],
            reranked_hits=[],
            context=self._empty_context(
                query=query,
                user_id=user_id,
                document_id=document_id,
            ),
            evidence_found=False,
            failure_reason=reason,
        )

    def _result_without_evidence(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
        dense_hits,
        bm25_hits,
        reason: str,
    ) -> HybridRetrievalResult:
        return HybridRetrievalResult(
            query=query,
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            fused_hits=[],
            reranked_hits=[],
            context=self._empty_context(
                query=query,
                user_id=user_id,
                document_id=document_id,
            ),
            evidence_found=False,
            failure_reason=reason,
        )
