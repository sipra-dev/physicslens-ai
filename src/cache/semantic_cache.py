from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import redis
from redis.exceptions import RedisError

from src.config.settings import settings


if TYPE_CHECKING:
    from src.retrieval.dense import DenseRetriever


# =========================================================
# ATOMIC DOCUMENT INVALIDATION
# =========================================================
#
# The registry contains every semantic-cache key belonging
# to one user + one document.
#
# Redis executes Lua scripts atomically, so there is no
# SMEMBERS -> DELETE gap inside this operation.
# =========================================================

_INVALIDATE_DOCUMENT_SCRIPT = """
local members = redis.call(
    'SMEMBERS',
    KEYS[1]
)

for _, member in ipairs(members) do
    redis.call(
        'DEL',
        member
    )
end

redis.call(
    'DEL',
    KEYS[1]
)

return #members
"""


class SemanticCache:
    """
    Redis-backed semantic answer cache.

    Important:
    - Uses the application's central Redis configuration.
    - Does NOT create its own embedding model.
    - Reuses the existing DenseRetriever.
    - Exact lookup first.
    - Semantic lookup second.
    - Only verifier-PASS answers may be stored.
    - Cache is isolated by:
        user
        document
        page
        figure
        language
        grade
    - Redis failure behaves like a cache miss.
    - Document cache entries are registered so they can
      be invalidated on delete/re-upload/re-index.
    """

    def __init__(
        self,
        *,
        dense_retriever: "DenseRetriever",
        similarity_threshold: float = 0.85,
        default_ttl_seconds: int = 3600,
        max_semantic_candidates: int = 100,
    ) -> None:
        if not (
            0.0
            < similarity_threshold
            <= 1.0
        ):
            raise ValueError(
                "similarity_threshold must be "
                "between 0 and 1."
            )

        if default_ttl_seconds <= 0:
            raise ValueError(
                "default_ttl_seconds must be positive."
            )

        if max_semantic_candidates <= 0:
            raise ValueError(
                "max_semantic_candidates must be positive."
            )

        self.dense_retriever = (
            dense_retriever
        )

        self.similarity_threshold = float(
            similarity_threshold
        )

        self.default_ttl_seconds = int(
            default_ttl_seconds
        )

        self.max_semantic_candidates = int(
            max_semantic_candidates
        )

        # -------------------------------------------------
        # REDIS CONNECTION
        # -------------------------------------------------
        #
        # Redis Cloud comes from REDIS_URL through settings.
        # Localhost remains a development fallback.
        # -------------------------------------------------

        redis_url = (
            settings.redis_url
            or "redis://localhost:6379/0"
        )

        self.pool = (
            redis.ConnectionPool.from_url(
                redis_url,
                decode_responses=True,
                max_connections=50,
                socket_connect_timeout=2,
                socket_timeout=3,
                health_check_interval=30,
            )
        )

        self.client = redis.Redis(
            connection_pool=self.pool
        )

    # =====================================================
    # GET
    # =====================================================

    def get(
        self,
        *,
        user_id: str,
        document_id: str,
        query: str,
        active_page: int | None = None,
        figure_id: str | None = None,
        language: str | None = None,
        grade: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Try exact cache lookup first.

        If there is no exact match, perform
        semantic lookup inside the same
        user/document/context scope.

        Redis failure behaves like a cache miss.
        """

        normalized_query = (
            self._normalize_query(
                query
            )
        )

        if not normalized_query:
            return None

        scope_prefix = (
            self._scope_prefix(
                user_id=user_id,
                document_id=document_id,
                active_page=active_page,
                figure_id=figure_id,
                language=language,
                grade=grade,
            )
        )

        exact_key = (
            self._answer_key(
                scope_prefix=scope_prefix,
                query=normalized_query,
            )
        )

        # -------------------------------------------------
        # 1. EXACT LOOKUP
        # -------------------------------------------------

        try:
            exact_raw = (
                self.client.get(
                    exact_key
                )
            )

            if exact_raw is not None:
                exact_payload = (
                    self._decode_payload(
                        exact_raw
                    )
                )

                if (
                    exact_payload
                    is not None
                ):
                    answer = (
                        exact_payload.get(
                            "answer"
                        )
                    )

                    if isinstance(
                        answer,
                        dict,
                    ):
                        return answer

            # ---------------------------------------------
            # 2. LOAD SEMANTIC CANDIDATES
            # ---------------------------------------------

            index_key = (
                self._index_key(
                    scope_prefix
                )
            )

            candidate_keys = (
                self.client.zrevrange(
                    index_key,
                    0,
                    (
                        self
                        .max_semantic_candidates
                        - 1
                    ),
                )
            )

            if not candidate_keys:
                return None

            cached_values = (
                self.client.mget(
                    candidate_keys
                )
            )

        except RedisError:
            # Redis outage must not break
            # the Tutor workflow.
            return None

        # -------------------------------------------------
        # 3. EMBED CURRENT QUERY
        # -------------------------------------------------
        #
        # IMPORTANT:
        # DenseRetriever belongs to RetrievalService.
        # SemanticCache does NOT instantiate another model.
        # -------------------------------------------------

        try:
            query_embedding = (
                self.dense_retriever
                .embed_text(
                    normalized_query
                )
            )

        except Exception:
            return None

        # -------------------------------------------------
        # 4. FIND BEST SEMANTIC MATCH
        # -------------------------------------------------

        best_answer: (
            dict[str, Any]
            | None
        ) = None

        best_similarity = -1.0

        for cached_raw in cached_values:
            if cached_raw is None:
                continue

            cached_payload = (
                self._decode_payload(
                    cached_raw
                )
            )

            if cached_payload is None:
                continue

            cached_embedding = (
                cached_payload.get(
                    "query_embedding"
                )
            )

            if not isinstance(
                cached_embedding,
                list,
            ):
                continue

            try:
                cached_vector = (
                    np.asarray(
                        cached_embedding,
                        dtype=np.float32,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                cached_vector.ndim != 1
                or cached_vector.shape
                != query_embedding.shape
            ):
                continue

            # DenseRetriever embeddings are already
            # normalized, so dot product gives cosine
            # similarity.
            similarity = float(
                np.dot(
                    query_embedding,
                    cached_vector,
                )
            )

            if (
                similarity
                < self.similarity_threshold
            ):
                continue

            if (
                similarity
                <= best_similarity
            ):
                continue

            answer = (
                cached_payload.get(
                    "answer"
                )
            )

            if not isinstance(
                answer,
                dict,
            ):
                continue

            best_similarity = (
                similarity
            )

            best_answer = (
                answer
            )

        return best_answer

    # =====================================================
    # SET
    # =====================================================

    def set(
        self,
        *,
        user_id: str,
        document_id: str,
        query: str,
        answer: dict[str, Any],
        verification_status: str,
        active_page: int | None = None,
        figure_id: str | None = None,
        language: str | None = None,
        grade: int | None = None,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Store only verifier-approved answers.

        Every cached answer is also registered under
        the owning user + document so the full document
        cache can later be invalidated safely.
        """

        # -------------------------------------------------
        # PASS-ONLY CACHE WRITE
        # -------------------------------------------------

        if (
            verification_status
            .strip()
            .upper()
            != "PASS"
        ):
            return False

        normalized_query = (
            self._normalize_query(
                query
            )
        )

        if not normalized_query:
            return False

        if not isinstance(
            answer,
            dict,
        ):
            return False

        ttl = (
            self.default_ttl_seconds
            if ttl_seconds is None
            else int(
                ttl_seconds
            )
        )

        if ttl <= 0:
            return False

        # -------------------------------------------------
        # EMBED QUERY
        # -------------------------------------------------

        try:
            query_embedding = (
                self.dense_retriever
                .embed_text(
                    normalized_query
                )
            )

        except Exception:
            return False

        # -------------------------------------------------
        # BUILD ISOLATED KEYS
        # -------------------------------------------------

        scope_prefix = (
            self._scope_prefix(
                user_id=user_id,
                document_id=document_id,
                active_page=active_page,
                figure_id=figure_id,
                language=language,
                grade=grade,
            )
        )

        key = (
            self._answer_key(
                scope_prefix=scope_prefix,
                query=normalized_query,
            )
        )

        index_key = (
            self._index_key(
                scope_prefix
            )
        )

        registry_key = (
            self._document_registry_key(
                user_id=user_id,
                document_id=document_id,
            )
        )

        # -------------------------------------------------
        # CACHE PAYLOAD
        # -------------------------------------------------

        payload = {
            "query": (
                normalized_query
            ),
            "query_embedding": (
                query_embedding.tolist()
            ),
            "answer": (
                answer
            ),
            "verification_status": (
                "PASS"
            ),
        }

        # -------------------------------------------------
        # WRITE TO REDIS
        # -------------------------------------------------

        try:
            pipeline = (
                self.client.pipeline(
                    transaction=False
                )
            )

            # ---------------------------------------------
            # Actual cached answer.
            # ---------------------------------------------

            pipeline.set(
                key,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
                ex=ttl,
            )

            # ---------------------------------------------
            # Semantic candidate index.
            #
            # score = write timestamp
            # member = answer-cache key
            # ---------------------------------------------

            pipeline.zadd(
                index_key,
                {
                    key: (
                        time.time()
                    )
                },
            )

            # ---------------------------------------------
            # INDEX TTL HARDENING
            #
            # NX:
            # set expiration only when the index currently
            # has no expiration.
            #
            # GT:
            # after that, only replace the existing TTL
            # when the new expiration is LONGER.
            #
            # Therefore a later short-lived cache entry
            # can never accidentally shorten the lifetime
            # needed by an older long-lived entry.
            # ---------------------------------------------

            pipeline.expire(
                index_key,
                ttl,
                nx=True,
            )

            pipeline.expire(
                index_key,
                ttl,
                gt=True,
            )

            # ---------------------------------------------
            # DOCUMENT REGISTRY
            #
            # Keep both:
            # - answer key
            # - semantic index key
            #
            # under one user + document registry.
            # ---------------------------------------------

            pipeline.sadd(
                registry_key,
                key,
                index_key,
            )

            # ---------------------------------------------
            # REGISTRY TTL HARDENING
            #
            # Same rule:
            # NEVER allow a later shorter cache TTL to
            # shorten this registry.
            #
            # Otherwise the registry could disappear while
            # an older cached answer still exists, making
            # document invalidation incomplete.
            # ---------------------------------------------

            pipeline.expire(
                registry_key,
                ttl,
                nx=True,
            )

            pipeline.expire(
                registry_key,
                ttl,
                gt=True,
            )

            pipeline.execute()

            return True

        except RedisError:
            # Cache is an optimization.
            # Redis failure must not break answering.
            return False

    # =====================================================
    # DOCUMENT INVALIDATION
    # =====================================================

    def invalidate_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        """
        Delete all semantic-answer cache data belonging
        to one user's document.

        Intended for:
        - document deletion
        - document re-upload
        - document re-index

        Redis Lua execution is atomic.

        Redis failure is fail-open and does not crash
        the document lifecycle operation.
        """

        normalized_user_id = (
            user_id.strip()
        )

        normalized_document_id = (
            document_id.strip()
        )

        if not normalized_user_id:
            return False

        if not normalized_document_id:
            return False

        registry_key = (
            self._document_registry_key(
                user_id=(
                    normalized_user_id
                ),
                document_id=(
                    normalized_document_id
                ),
            )
        )

        try:
            self.client.eval(
                _INVALIDATE_DOCUMENT_SCRIPT,
                1,
                registry_key,
            )

            return True

        except RedisError:
            return False

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    def health_check(
        self,
    ) -> bool:
        """
        Check whether Redis is reachable.
        """

        try:
            return bool(
                self.client.ping()
            )

        except RedisError:
            return False

    # =====================================================
    # CLOSE
    # =====================================================

    def close(
        self,
    ) -> None:
        """
        Close Redis client and connection pool.
        """

        try:
            self.client.close()

        finally:
            self.pool.disconnect()

    # =====================================================
    # CACHE SCOPE
    # =====================================================

    def _scope_prefix(
        self,
        *,
        user_id: str,
        document_id: str,
        active_page: int | None,
        figure_id: str | None,
        language: str | None,
        grade: int | None,
    ) -> str:
        """
        Build an isolated semantic-cache namespace.

        Different combinations of:
        - user
        - document
        - page
        - figure
        - language
        - grade

        cannot accidentally share the same answer cache.
        """

        user_hash = (
            self._short_hash(
                user_id.strip()
            )
        )

        document_hash = (
            self._short_hash(
                document_id.strip()
            )
        )

        context_raw = "\x1f".join(
            [
                str(
                    active_page
                    or 0
                ),
                (
                    figure_id.strip()
                    if figure_id
                    else "no-figure"
                ),
                (
                    str(
                        language
                    ).strip()
                    if language
                    else "unknown"
                ),
                str(
                    grade
                    or 0
                ),
            ]
        )

        context_hash = (
            self._short_hash(
                context_raw
            )
        )

        return (
            "phymentor:answer:"
            f"{user_hash}:"
            f"{document_hash}:"
            f"{context_hash}"
        )

    # =====================================================
    # ANSWER KEY
    # =====================================================

    def _answer_key(
        self,
        *,
        scope_prefix: str,
        query: str,
    ) -> str:
        """
        Build the exact answer-cache key.

        Query hashing prevents very long raw queries from
        becoming Redis key names.
        """

        query_hash = hashlib.sha256(
            query.casefold().encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            f"{scope_prefix}:"
            f"{query_hash}"
        )

    # =====================================================
    # SEMANTIC INDEX KEY
    # =====================================================

    @staticmethod
    def _index_key(
        scope_prefix: str,
    ) -> str:
        """
        Redis sorted-set key containing semantic-cache
        candidate answer keys for one isolated context.
        """

        return (
            f"{scope_prefix}:index"
        )

    # =====================================================
    # DOCUMENT REGISTRY KEY
    # =====================================================

    @classmethod
    def _document_registry_key(
        cls,
        *,
        user_id: str,
        document_id: str,
    ) -> str:
        """
        One registry for one user + one document.

        Every answer/index key created for the document is
        registered here so deletion/re-upload/re-index can
        invalidate the document without a global SCAN.
        """

        user_hash = (
            cls._short_hash(
                user_id.strip()
            )
        )

        document_hash = (
            cls._short_hash(
                document_id.strip()
            )
        )

        return (
            "phymentor:answer-registry:"
            f"{user_hash}:"
            f"{document_hash}"
        )

    # =====================================================
    # QUERY NORMALIZATION
    # =====================================================

    @staticmethod
    def _normalize_query(
        query: str,
    ) -> str:
        """
        Collapse repeated whitespace while preserving
        the query's meaningful content.
        """

        return " ".join(
            query.strip().split()
        )

    # =====================================================
    # SHORT HASH
    # =====================================================

    @staticmethod
    def _short_hash(
        value: str,
    ) -> str:
        """
        Compact deterministic hash used for Redis
        namespace components.
        """

        return hashlib.sha256(
            value.encode(
                "utf-8"
            )
        ).hexdigest()[:24]

    # =====================================================
    # PAYLOAD DECODING
    # =====================================================

    @staticmethod
    def _decode_payload(
        value: str,
    ) -> dict[str, Any] | None:
        """
        Safely decode a Redis JSON cache payload.
        """

        try:
            payload = json.loads(
                value
            )

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

        if not isinstance(
            payload,
            dict,
        ):
            return None

        return payload


__all__ = [
    "SemanticCache",
]