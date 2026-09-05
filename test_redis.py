from __future__ import annotations

from src.runtime_services import (
    get_semantic_cache,
)


def main() -> None:
    """
    Manual Redis semantic-cache smoke test.

    This file is intentionally safe for pytest collection:
    importing it does not execute Redis/cache operations.

    Run manually with:

        python test_redis.py
    """

    semantic_cache = get_semantic_cache()

    if not semantic_cache.health_check():
        print(
            "Redis is unavailable. "
            "Semantic-cache smoke test skipped."
        )
        return

    user_id = "user1"
    document_id = "document1"

    source_query = (
        "Explain simple harmonic motion."
    )

    cached_answer = {
        "direct_answer": (
            "This answer belongs only to "
            "user1 and document1."
        ),
    }

    stored = semantic_cache.set(
        user_id=user_id,
        document_id=document_id,
        query=source_query,
        answer=cached_answer,
        verification_status="PASS",
        ttl_seconds=300,
    )

    # Same user + same document + exact query -> HIT.
    exact_hit = semantic_cache.get(
        user_id=user_id,
        document_id=document_id,
        query=source_query,
    )

    # Same user + same document + similar query.
    # This exercises semantic lookup. Whether it hits depends
    # on the configured embedding model and similarity threshold.
    semantic_hit = semantic_cache.get(
        user_id=user_id,
        document_id=document_id,
        query="What is simple harmonic motion?",
    )

    # Different user -> must MISS.
    wrong_user = semantic_cache.get(
        user_id="user2",
        document_id=document_id,
        query=source_query,
    )

    # Same user, different document -> must MISS.
    wrong_document = semantic_cache.get(
        user_id=user_id,
        document_id="document2",
        query=source_query,
    )

    print("Stored:", stored)
    print("Exact same scope:", exact_hit is not None)
    print(
        "Semantic same scope:",
        semantic_hit is not None,
    )
    print("Different user:", wrong_user)
    print("Different document:", wrong_document)

    # These checks are deterministic and verify isolation.
    assert stored is True
    assert exact_hit is not None
    assert wrong_user is None
    assert wrong_document is None


if __name__ == "__main__":
    main()
