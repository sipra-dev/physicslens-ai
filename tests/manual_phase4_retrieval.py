from src.config import settings
from src.retrieval.service import RetrievalService


DOCUMENT_ID = (
    "7e814389ea2142bcb7f9b6bfc5f9234b"
)

USER_ID = "local-user"

QUERY = "What is simple harmonic motion?"


service = RetrievalService(
    vector_store_directory=(
        settings.vector_store_dir
    ),
    bm25_store_directory=(
        settings.bm25_store_dir
    ),
    embedding_model_name=(
        settings.embedding_model_name
    ),
    reranker_model_name=(
        settings.reranker_model_name
    ),
    reranker_batch_size=(
        settings.reranker_batch_size
    ),
    max_context_characters=(
        settings.max_context_characters
    ),
    max_item_characters=(
        settings.max_context_item_characters
    ),
)


result = service.retrieve(
    query=QUERY,
    user_id=USER_ID,
    document_id=DOCUMENT_ID,
    dense_top_k=(
        settings.retrieval_per_source_top_k
    ),
    bm25_top_k=(
        settings.retrieval_per_source_top_k
    ),
    fused_top_k=(
        settings.hybrid_candidate_pool_size
    ),
    rerank_top_k=(
        settings.reranker_top_k
    ),
    max_contexts=(
        settings.final_context_count
    ),
)


print("\n==============================")
print("PHASE 4 RETRIEVAL TEST")
print("==============================")

print(
    "\nDense hits:",
    len(result.dense_hits),
)

print(
    "BM25 hits:",
    len(result.bm25_hits),
)

print(
    "Fused hits:",
    len(result.fused_hits),
)

print(
    "Reranked hits:",
    len(result.reranked_hits),
)

print(
    "Evidence found:",
    result.evidence_found,
)

print(
    "Failure reason:",
    result.failure_reason,
)


print("\n==============================")
print("TOP RERANKED HITS")
print("==============================")

for index, item in enumerate(
    result.reranked_hits,
    start=1,
):
    print(
        f"\n--- Hit {index} ---"
    )

    print(
        "Chunk ID:",
        item.hit.chunk_id,
    )

    print(
        "Parent ID:",
        item.hit.parent_id,
    )

    print(
        "Page:",
        item.hit.page_number,
    )

    print(
        "Rerank score:",
        item.rerank_score,
    )

    print(
        "Text:",
        item.hit.text[:500],
    )


print("\n==============================")
print("FINAL PARENT CONTEXTS")
print("==============================")

for index, item in enumerate(
    result.context.items,
    start=1,
):
    print(
        f"\n--- Context {index} ---"
    )

    print(
        "Context ID:",
        item.context_id,
    )

    print(
        "Parent ID:",
        item.parent_id,
    )

    print(
        "Page:",
        item.page_number,
    )

    print(
        "Source chunk IDs:",
        item.source_chunk_ids,
    )

    print(
        "Equations:",
        item.equations,
    )

    print(
        "Text:",
        item.text[:1200],
    )


print("\n==============================")
print("CONTEXT SUMMARY")
print("==============================")

print(
    "Context count:",
    len(result.context.items),
)

print(
    "Total characters:",
    result.context.total_characters,
)

print(
    "Truncated:",
    result.context.truncated,
)