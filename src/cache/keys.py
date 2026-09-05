import hashlib


def session_key(
    user_id: str,
    session_id: str,
) -> str:
    return f"session:{user_id}:{session_id}"


def cache_key(
    user_id: str,
    document_id: str,
    query: str,
) -> str:
    normalized_query = query.strip().lower()

    query_hash = hashlib.sha256(
        normalized_query.encode("utf-8")
    ).hexdigest()

    return (
        f"cache:{user_id}:"
        f"{document_id}:"
        f"{query_hash}"
    )


def rate_limit_key(
    user_id: str,
) -> str:
    return f"rate:{user_id}"


def short_memory_key(
    user_id: str,
    session_id: str,
) -> str:
    return (
        f"memory:short:"
        f"{user_id}:"
        f"{session_id}"
    )


def long_memory_key(
    user_id: str,
) -> str:
    return f"memory:long:{user_id}"


def checkpoint_key(
    user_id: str,
    session_id: str,
) -> str:
    return (
        f"checkpoint:"
        f"{user_id}:"
        f"{session_id}"
    )