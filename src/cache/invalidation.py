from __future__ import annotations

from redis.exceptions import RedisError

from src.cache.redis_client import (
    redis_client,
)
from src.cache.semantic_cache import (
    SemanticCache,
)


# =========================================================
# DOCUMENT CACHE INVALIDATION
# =========================================================
#
# This module is deliberately lightweight.
#
# Document upload/delete/re-index should be able to clear
# Redis answer-cache entries WITHOUT loading:
#
# - DenseRetriever
# - SentenceTransformer
# - reranker
# - chat graph
#
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


async def invalidate_document_cache(
    *,
    user_id: str,
    document_id: str,
) -> bool:
    """
    Remove all registered semantic-answer cache
    entries belonging to one user's document.

    Redis failure is fail-open:
    document lifecycle operations may continue.
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
        SemanticCache
        ._document_registry_key(
            user_id=normalized_user_id,
            document_id=(
                normalized_document_id
            ),
        )
    )

    try:
        await redis_client.client.eval(
            _INVALIDATE_DOCUMENT_SCRIPT,
            1,
            registry_key,
        )

        return True

    except RedisError:
        return False


__all__ = [
    "invalidate_document_cache",
]