from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import settings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.filters import RetrievalFilter
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import CrossEncoderReranker


# -----------------------------------------
# Use our already validated SHM document.
# -----------------------------------------

USER_ID = "local-user"

DOCUMENT_ID = (
    "7e814389ea2142bcb7f9b6bfc5f9234b"
)


# -----------------------------------------
# Exact index directories created by
# LocalDocumentIndexer.
# -----------------------------------------

dense_directory = (
    settings.vector_store_dir
    / "users"
    / USER_ID
    / "documents"
    / DOCUMENT_ID
)

bm25_directory = (
    settings.bm25_store_dir
    / "users"
    / USER_ID
    / "documents"
    / DOCUMENT_ID
)


# -----------------------------------------
# Create retrieval components.
# -----------------------------------------

bm25 = BM25Retriever()

dense = DenseRetriever(
    model_name=settings.embedding_model_name
)

fusion = ReciprocalRankFusion(
    rrf_k=settings.rrf_k
)

reranker = CrossEncoderReranker(
    model_name=settings.reranker_model_name,
    batch_size=settings.reranker_batch_size,
)


# =========================================
# 5A — BM25 TEST
# =========================================

print("\n")
print("=" * 60)
print("5A — BM25")
print("=" * 60)

bm25_query = (
    "spring force constant 120 N/m "
    "frequency 6.00 Hz mass"
)

normal_filter = RetrievalFilter(
    user_id=USER_ID,
    document_id=DOCUMENT_ID,
)

bm25_hits = bm25.search(
    query=bm25_query,
    index_directory=bm25_directory,
    retrieval_filter=normal_filter,
    top_k=30,
)

print(
    "BM25 hit count:",
    len(bm25_hits),
)

for rank, hit in enumerate(
    bm25_hits[:5],
    start=1,
):
    print(
        f"\nBM25 rank {rank}"
    )
    print(
        "chunk_id:",
        hit.chunk_id,
    )
    print(
        "page:",
        hit.page_number,
    )
    print(
        "score:",
        hit.score,
    )
    print(
        "source:",
        hit.retrieval_source,
    )
    print(
        "text:",
        hit.text[:300],
    )

assert bm25_hits, (
    "5A FAILED: BM25 returned no hits."
)

assert all(
    hit.retrieval_source == "bm25"
    for hit in bm25_hits
), (
    "5A FAILED: BM25 source label is wrong."
)

print(
    "\n5A BM25 PASS ✅"
)


# =========================================
# 5B — DENSE RETRIEVAL TEST
# =========================================

print("\n")
print("=" * 60)
print("5B — DENSE")
print("=" * 60)

dense_query = (
    "What force brings an oscillating "
    "object back toward equilibrium?"
)

dense_hits = dense.search(
    query=dense_query,
    index_directory=dense_directory,
    retrieval_filter=normal_filter,
    top_k=30,
)

print(
    "Dense hit count:",
    len(dense_hits),
)

for rank, hit in enumerate(
    dense_hits[:5],
    start=1,
):
    print(
        f"\nDense rank {rank}"
    )
    print(
        "chunk_id:",
        hit.chunk_id,
    )
    print(
        "page:",
        hit.page_number,
    )
    print(
        "score:",
        hit.score,
    )
    print(
        "source:",
        hit.retrieval_source,
    )
    print(
        "text:",
        hit.text[:300],
    )

assert dense_hits, (
    "5B FAILED: Dense returned no hits."
)

assert all(
    hit.retrieval_source == "dense"
    for hit in dense_hits
), (
    "5B FAILED: Dense source label is wrong."
)

print(
    "\n5B Dense PASS ✅"
)


# =========================================
# 5C — RRF FUSION TEST
# =========================================

print("\n")
print("=" * 60)
print("5C — RRF FUSION")
print("=" * 60)

hybrid_query = (
    "spring force constant 120 N/m "
    "frequency 6.00 Hz mass"
)

dense_for_rrf = dense.search(
    query=hybrid_query,
    index_directory=dense_directory,
    retrieval_filter=normal_filter,
    top_k=30,
)

bm25_for_rrf = bm25.search(
    query=hybrid_query,
    index_directory=bm25_directory,
    retrieval_filter=normal_filter,
    top_k=30,
)

fused_hits = fusion.fuse(
    dense_hits=dense_for_rrf,
    bm25_hits=bm25_for_rrf,
    retrieval_filter=normal_filter,
    top_k=30,
)

print(
    "Fused hit count:",
    len(fused_hits),
)

for rank, item in enumerate(
    fused_hits[:5],
    start=1,
):
    print(
        f"\nRRF rank {rank}"
    )
    print(
        "chunk_id:",
        item.hit.chunk_id,
    )
    print(
        "page:",
        item.hit.page_number,
    )
    print(
        "RRF score:",
        item.rrf_score,
    )
    print(
        "source ranks:",
        item.source_ranks,
    )
    print(
        "source scores:",
        item.source_scores,
    )

assert fused_hits, (
    "5C FAILED: RRF returned no hits."
)

has_hybrid_consensus = any(
    (
        "dense" in item.source_ranks
        and "bm25" in item.source_ranks
    )
    for item in fused_hits
)

assert has_hybrid_consensus, (
    "5C FAILED: No candidate appeared "
    "in both Dense and BM25."
)

print(
    "\n5C RRF PASS ✅"
)


# =========================================
# 5D — METADATA FILTER TEST
# =========================================

print("\n")
print("=" * 60)
print("5D — METADATA FILTER")
print("=" * 60)

page_two_filter = RetrievalFilter(
    user_id=USER_ID,
    document_id=DOCUMENT_ID,
    page_numbers=(2,),
)

filtered_dense_hits = dense.search(
    query=hybrid_query,
    index_directory=dense_directory,
    retrieval_filter=page_two_filter,
    top_k=30,
)

filtered_bm25_hits = bm25.search(
    query=hybrid_query,
    index_directory=bm25_directory,
    retrieval_filter=page_two_filter,
    top_k=30,
)

print(
    "Filtered Dense count:",
    len(filtered_dense_hits),
)

print(
    "Filtered BM25 count:",
    len(filtered_bm25_hits),
)

for hit in filtered_dense_hits:
    print(
        "Dense filtered page:",
        hit.page_number,
        "|",
        hit.chunk_id,
    )

for hit in filtered_bm25_hits:
    print(
        "BM25 filtered page:",
        hit.page_number,
        "|",
        hit.chunk_id,
    )

assert (
    filtered_dense_hits
    or filtered_bm25_hits
), (
    "5D FAILED: Page-2 filter "
    "returned no evidence."
)

assert all(
    hit.page_number == 2
    for hit in filtered_dense_hits
), (
    "5D FAILED: Dense leaked a hit "
    "outside page 2."
)

assert all(
    hit.page_number == 2
    for hit in filtered_bm25_hits
), (
    "5D FAILED: BM25 leaked a hit "
    "outside page 2."
)

print(
    "\n5D Metadata Filter PASS ✅"
)


# =========================================
# 5E — CROSSENCODER RERANKER TEST
# =========================================

print("\n")
print("=" * 60)
print("5E — CROSSENCODER")
print("=" * 60)

reranked_hits = reranker.rerank(
    query=hybrid_query,
    candidates=fused_hits,
    top_k=min(
        settings.reranker_top_k,
        8,
    ),
)

print(
    "Reranked hit count:",
    len(reranked_hits),
)

for rank, item in enumerate(
    reranked_hits,
    start=1,
):
    print(
        f"\nRerank position {rank}"
    )
    print(
        "chunk_id:",
        item.hit.chunk_id,
    )
    print(
        "page:",
        item.hit.page_number,
    )
    print(
        "rerank score:",
        item.rerank_score,
    )
    print(
        "RRF score:",
        item.rrf_score,
    )
    print(
        "text:",
        item.hit.text[:300],
    )

assert reranked_hits, (
    "5E FAILED: CrossEncoder "
    "returned no hits."
)

print(
    "\n5E CrossEncoder PASS ✅"
)


# =========================================
# 5F — TOP-K LIMIT TEST
# =========================================

print("\n")
print("=" * 60)
print("5F — TOP-K LIMITS")
print("=" * 60)

print(
    "Dense:",
    len(dense_for_rrf),
    "/ max 30",
)

print(
    "BM25:",
    len(bm25_for_rrf),
    "/ max 30",
)

print(
    "Fused:",
    len(fused_hits),
    "/ max 30",
)

print(
    "Reranked:",
    len(reranked_hits),
    "/ max 8",
)

assert len(
    dense_for_rrf
) <= 30

assert len(
    bm25_for_rrf
) <= 30

assert len(
    fused_hits
) <= 30

assert len(
    reranked_hits
) <= 8

print(
    "\n5F Top-k PASS ✅"
)


# =========================================
# FINAL
# =========================================

print("\n")
print("=" * 60)
print("POINT 5 COMPLETE ✅")
print("=" * 60)

print(
    "BM25 ✅\n"
    "Dense ✅\n"
    "RRF ✅\n"
    "Metadata Filter ✅\n"
    "CrossEncoder ✅\n"
    "Top-k ✅"
)