from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from src.config import settings


if TYPE_CHECKING:
    from src.cache.semantic_cache import SemanticCache
    from src.ingestion.service import IngestionService
    from src.memory.long_term_service import (
        LongTermMemoryService,
    )
    from src.memory.pinecone_store import (
        PineconeSemanticMemoryStore,
    )
    from src.memory.semantic_extractor import (
        SemanticMemoryExtractor,
    )
    from src.memory.semantic_llm_adapter import (
        SemanticMemoryLLMAdapter,
    )
    from src.memory.semantic_service import (
        SemanticLearningMemoryService,
    )
    from src.memory.session_store import (
        RedisSessionStore,
    )
    from src.models.gateway import LLMGateway
    from src.models.routing import ModelRouter
    from src.retrieval.service import (
        RetrievalService,
    )
    from src.retrieval.structural_resolver import (
        StructuralResolver,
    )
    from src.storage import LocalStorage


@lru_cache(maxsize=1)
def get_local_storage() -> "LocalStorage":
    from src.storage import LocalStorage

    return LocalStorage(
        root_directory=settings.upload_dir
    )


@lru_cache(maxsize=1)
def get_ingestion_service() -> "IngestionService":
    """
    Build the shared ingestion service.

    The shared gateway and router enable structural
    Vision indexing without creating another independent
    OpenAI client or duplicating model-routing policy.
    """

    from src.ingestion.service import IngestionService

    return IngestionService(
        storage=get_local_storage(),

        application_settings=settings,

        llm_gateway=get_llm_gateway(),

        model_router=get_model_router(),
    )


@lru_cache(maxsize=1)
def get_model_router() -> "ModelRouter":
    """
    Build the shared model router using central settings.

    Policy:
    - query understanding / scope / rewrite
      -> configured classifier model
    - general Physics answering
      -> configured general model
    - general numerical reasoning
      -> configured reasoning model
    - document structural indexing
      -> configured document-capable model
    - structural document reference resolution
      -> configured document-capable model
    - document-dependent answering
      -> configured document-capable model
    - general verification
      -> configured general verifier
    - document verification
      -> configured document verifier

    Exact model names come from settings/environment.
    """

    from src.models.routing import ModelRouter

    return ModelRouter(
        classifier_model=(
            settings.query_understanding_model
        ),

        text_model=(
            settings.general_reasoning_model
        ),

        multimodal_model=(
            settings.document_reasoning_model
        ),

        reasoning_model=(
            settings.general_reasoning_model
        ),

        verifier_model=(
            settings.general_verifier_model
        ),

        fallback_model=(
            settings.model_fallback_model
        ),

        document_model=(
            settings.document_reasoning_model
        ),

        document_verifier_model=(
            settings.document_verifier_model
        ),
    )


@lru_cache(maxsize=1)
def get_llm_gateway() -> "LLMGateway":
    from src.models.gateway import LLMGateway

    return LLMGateway(
        model_router=get_model_router(),

        api_key=settings.openai_api_key,

        timeout_seconds=(
            settings.model_gateway_timeout_seconds
        ),
    )


@lru_cache(maxsize=1)
def get_structural_resolver(
) -> "StructuralResolver":
    """
    Build the one shared structure-aware resolver.

    It reads the canonical structure artifact from the same
    LocalStorage used by ingestion and reuses the same model
    gateway/router. It does not replace semantic retrieval.
    """

    from src.retrieval.structural_resolver import (
        StructuralResolver,
    )

    return StructuralResolver(
        storage=get_local_storage(),

        gateway=get_llm_gateway(),

        model_router=get_model_router(),
    )


@lru_cache(maxsize=1)
def get_semantic_memory_llm_adapter(
) -> "SemanticMemoryLLMAdapter":
    from src.memory.semantic_llm_adapter import (
        SemanticMemoryLLMAdapter,
    )

    return SemanticMemoryLLMAdapter(
        gateway=get_llm_gateway(),

        model_router=get_model_router(),
    )


@lru_cache(maxsize=1)
def get_semantic_memory_extractor(
) -> "SemanticMemoryExtractor":
    from src.memory.semantic_extractor import (
        SemanticMemoryExtractor,
    )

    return SemanticMemoryExtractor(
        llm=get_semantic_memory_llm_adapter(),

        confidence_threshold=0.75,
    )


@lru_cache(maxsize=1)
def get_semantic_learning_memory_service(
) -> "SemanticLearningMemoryService":
    from src.memory.semantic_service import (
        SemanticLearningMemoryService,
    )

    return SemanticLearningMemoryService(
        extractor=get_semantic_memory_extractor(),

        store=get_semantic_memory_store(),

        top_k=5,

        minimum_score=0.45,

        max_context_characters=2000,
    )


@lru_cache(maxsize=1)
def get_retrieval_service() -> "RetrievalService":
    """
    Build the one shared retrieval service used by:

    - retrieval API route
    - chat flow
    - semantic cache
    - semantic memory store

    RRF configuration comes from central settings.
    """

    from src.retrieval.service import RetrievalService

    return RetrievalService(
        vector_store_directory=(
            settings.vector_store_dir
        ),

        bm25_store_directory=(
            settings.bm25_store_dir
        ),

        embedding_model_name=(
            settings.embedding_model_name
        ),

        # Reciprocal Rank Fusion constant.
        # This was previously missing from the shared
        # runtime construction.
        rrf_k=(
            settings.rrf_k
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

        minimum_rerank_score=0.0,
    )


@lru_cache(maxsize=1)
def get_semantic_memory_store(
) -> "PineconeSemanticMemoryStore":
    from src.memory.pinecone_store import (
        PineconeSemanticMemoryStore,
    )

    retrieval_service = get_retrieval_service()

    return PineconeSemanticMemoryStore(
        embedder=(
            retrieval_service.dense_retriever
        ),

        api_key=settings.pinecone_api_key,

        index_name=(
            settings.pinecone_learning_memory_index
        ),

        expected_dimension=384,
    )


@lru_cache(maxsize=1)
def get_semantic_cache() -> "SemanticCache":
    from src.cache.semantic_cache import (
        SemanticCache,
    )

    retrieval_service = get_retrieval_service()

    return SemanticCache(
        dense_retriever=(
            retrieval_service.dense_retriever
        ),

        similarity_threshold=0.85,

        default_ttl_seconds=3600,

        max_semantic_candidates=100,
    )


@lru_cache(maxsize=1)
def get_session_store() -> "RedisSessionStore":
    from src.memory.session_store import (
        RedisSessionStore,
    )

    return RedisSessionStore()


@lru_cache(maxsize=1)
def get_long_term_memory_service(
) -> "LongTermMemoryService":
    from src.memory.long_term_service import (
        LongTermMemoryService,
    )

    return LongTermMemoryService()


__all__ = [
    "get_local_storage",
    "get_ingestion_service",
    "get_model_router",
    "get_llm_gateway",
    "get_structural_resolver",
    "get_semantic_memory_llm_adapter",
    "get_retrieval_service",
    "get_semantic_cache",
    "get_session_store",
    "get_long_term_memory_service",
    "get_semantic_memory_store",
    "get_semantic_memory_extractor",
    "get_semantic_learning_memory_service",
]
