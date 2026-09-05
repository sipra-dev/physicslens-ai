from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4

from src.memory.pinecone_store import (
    PineconeSemanticMemoryStore,
)
from src.memory.semantic_models import (
    SemanticLearningMemoryRecord,
    SemanticMemoryKind,
    SemanticMemoryStatus,
)


class Fake384Embedder:
    """
    Lightweight deterministic embedding.

    No OpenAI.
    No SentenceTransformer.
    No Torch.
    """

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        _ = text

        return [
            1.0,
            *([0.0] * 383),
        ]


def main() -> None:
    user_id = (
        "phase7-pinecone-test-"
        f"{uuid4().hex}"
    )

    memory_id = (
        "phase7-memory-"
        f"{uuid4().hex}"
    )

    now = datetime.now(
        timezone.utc
    )

    store = PineconeSemanticMemoryStore(
        embedder=Fake384Embedder(),
        expected_dimension=384,
    )

    record = SemanticLearningMemoryRecord(
        memory_id=memory_id,
        user_id=user_id,
        kind=(
            SemanticMemoryKind.MISCONCEPTION
        ),
        topic="mechanics",
        concept="force_and_motion",
        text=(
            "The learner appears to believe "
            "that a continuous force is required "
            "to keep an object moving."
        ),
        confidence=0.95,
        status=(
            SemanticMemoryStatus.ACTIVE
        ),
        source_session_id=(
            "phase7-test-session"
        ),
        source_document_id=None,
        created_at=now,
        updated_at=now,
        last_reinforced_at=now,
        schema_version=1,
    )

    try:
        # -----------------------------------------
        # 1. UPSERT
        # -----------------------------------------

        stored = store.upsert(
            record
        )

        print(
            "UPSERT_OK =",
            stored,
        )

        if not stored:
            raise RuntimeError(
                "Pinecone upsert failed."
            )

        # Pinecone may need a short moment
        # before the new vector is searchable.
        found = False

        # -----------------------------------------
        # 2. SEARCH
        # -----------------------------------------

        for _ in range(8):
            matches = store.search(
                user_id=user_id,
                query_text=(
                    "The student thinks force "
                    "must always keep motion going."
                ),
                top_k=5,
                minimum_score=0.90,
            )

            found = any(
                match.record.memory_id
                == memory_id
                for match in matches
            )

            if found:
                break

            time.sleep(1)

        print(
            "SEARCH_FOUND =",
            found,
        )

        if not found:
            raise RuntimeError(
                "Stored memory was not found "
                "by semantic search."
            )

        # -----------------------------------------
        # 3. DELETE
        # -----------------------------------------

        deleted = store.delete(
            user_id=user_id,
            memory_id=memory_id,
        )

        print(
            "DELETE_OK =",
            deleted,
        )

        if not deleted:
            raise RuntimeError(
                "Pinecone delete failed."
            )

        # -----------------------------------------
        # 4. CONFIRM DELETE
        # -----------------------------------------

        still_present = True

        for _ in range(8):
            matches = store.search(
                user_id=user_id,
                query_text=(
                    "The student thinks force "
                    "must always keep motion going."
                ),
                top_k=5,
                minimum_score=0.90,
            )

            still_present = any(
                match.record.memory_id
                == memory_id
                for match in matches
            )

            if not still_present:
                break

            time.sleep(1)

        print(
            "DELETE_CONFIRMED =",
            not still_present,
        )

        if still_present:
            raise RuntimeError(
                "Deleted memory is still "
                "searchable."
            )

        print(
            "PHASE7_PINECONE_STORE_TEST = PASS"
        )

    finally:
        # Best-effort cleanup in case the test
        # failed halfway through.
        store.delete(
            user_id=user_id,
            memory_id=memory_id,
        )


if __name__ == "__main__":
    main()