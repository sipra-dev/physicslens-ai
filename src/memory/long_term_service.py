from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.memory.long_term_models import (
    LongTermMemoryProfile,
)
from src.memory.postgres_store import (
    PostgresLongTermMemoryStore,
)
from src.models.contracts import (
    LanguageCode,
    MemorySnapshot,
)


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


class LongTermMemoryService:
    """
    Bridge between LangGraph memory candidates
    and durable PostgreSQL memory.

    Redis:
        temporary conversation/session memory

    PostgreSQL:
        durable student learning profile
    """

    def __init__(
        self,
        *,
        store: (
            PostgresLongTermMemoryStore
            | None
        ) = None,
    ) -> None:
        self.store = (
            store
            or PostgresLongTermMemoryStore()
        )

    # =========================================================
    # LOAD DURABLE MEMORY INTO SESSION CONTEXT
    # =========================================================

    def hydrate_memory(
        self,
        *,
        user_id: str,
        memory: MemorySnapshot,
    ) -> MemorySnapshot:
        """
        Load durable preferences from PostgreSQL.

        Existing session-specific values win.

        Example:
        Redis already knows current language ->
        keep Redis value.

        New session has no language ->
        recover durable language from PostgreSQL.
        """

        normalized_user_id = (
            user_id.strip()
        )

        if not normalized_user_id:
            return memory

        profile = self.store.load_profile(
            user_id=normalized_user_id
        )

        if profile is None:
            return memory

        updates: dict[str, Any] = {}

        if (
            (
                memory.language is None
                or memory.language
                == LanguageCode.UNKNOWN
            )
            and profile.preferred_language
            is not None
        ):
            updates["language"] = (
                profile.preferred_language
            )

        if (
            memory.estimated_grade is None
            and profile.grade is not None
        ):
            updates["estimated_grade"] = (
                profile.grade
            )

        if not updates:
            return memory

        return memory.model_copy(
            update=updates
        )

    # =========================================================
    # WRITE LANGGRAPH MEMORY CANDIDATES
    # =========================================================

    def write_candidates(
        self,
        *,
        user_id: str,
        candidates: list[
            dict[str, Any]
        ],
    ) -> bool:
        """
        Persist durable candidates produced
        by the graph.

        Currently the graph emits:
        - language_preference
        - grade

        Other durable profile fields remain
        preserved.
        """

        normalized_user_id = (
            user_id.strip()
        )

        if (
            not normalized_user_id
            or not candidates
        ):
            return False

        existing = self.store.load_profile(
            user_id=normalized_user_id
        )

        if existing is None:
            profile = LongTermMemoryProfile(
                user_id=normalized_user_id
            )
        else:
            profile = existing

        preferred_language = (
            profile.preferred_language
        )

        grade = profile.grade

        changed = False

        for candidate in candidates:
            kind = str(
                candidate.get(
                    "kind",
                    "",
                )
            ).strip()

            value = candidate.get(
                "value"
            )

            if (
                kind
                == "language_preference"
            ):
                try:
                    language = LanguageCode(
                        str(value)
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    language
                    != LanguageCode.UNKNOWN
                    and language
                    != preferred_language
                ):
                    preferred_language = (
                        language
                    )
                    changed = True

            elif kind == "grade":
                try:
                    candidate_grade = int(
                        value
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if not (
                    1
                    <= candidate_grade
                    <= 12
                ):
                    continue

                if (
                    candidate_grade
                    != grade
                ):
                    grade = candidate_grade
                    changed = True

        if not changed:
            return False

        updated_profile = (
            profile.model_copy(
                update={
                    "preferred_language": (
                        preferred_language
                    ),
                    "grade": grade,
                    "updated_at": _utc_now(),
                }
            )
        )

        self.store.save_profile(
            updated_profile
        )

        return True


__all__ = [
    "LongTermMemoryService",
]