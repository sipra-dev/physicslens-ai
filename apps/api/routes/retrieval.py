from __future__ import annotations

from functools import partial

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from apps.api.schemas.retrieval import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)

from src.config import settings

from src.runtime_services import (
    get_retrieval_service,
)


router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
)


# =========================================================
# SHARED RETRIEVAL SERVICE
# =========================================================
#
# RetrievalService is NOT created inside this route.
#
# It comes from the shared runtime layer so:
#
# - retrieval route
# - chat route
# - semantic cache
#
# can reuse the same RetrievalService / DenseRetriever.
# =========================================================

retrieval_service = (
    get_retrieval_service()
)


@router.post(
    "/search",
    response_model=RetrievalSearchResponse,
    summary="Search an indexed Physics document",
)
async def search_retrieval(
    request: RetrievalSearchRequest,
) -> RetrievalSearchResponse:

    retrieval_call = partial(
        retrieval_service.retrieve,

        query=request.query,

        user_id=request.user_id,

        document_id=request.document_id,

        dense_top_k=(
            settings
            .retrieval_per_source_top_k
        ),

        bm25_top_k=(
            settings
            .retrieval_per_source_top_k
        ),

        fused_top_k=(
            settings
            .hybrid_candidate_pool_size
        ),

        rerank_top_k=(
            settings
            .reranker_top_k
        ),

        max_contexts=(
            settings
            .final_context_count
        ),
    )

    result = await run_in_threadpool(
        retrieval_call
    )

    return RetrievalSearchResponse(
        **result.model_dump()
    )