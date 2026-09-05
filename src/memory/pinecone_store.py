from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol

from pinecone import Pinecone

from src.config import settings
from src.memory.semantic_models import (
    SemanticLearningMemoryRecord,
    SemanticMemoryKind,
    SemanticMemoryMatch,
    SemanticMemoryStatus,
)


class SemanticEmbeddingProtocol(Protocol):
    """
    Minimal embedding interface needed by semantic memory.

    Runtime will reuse the existing dense retriever /
    embedding pipeline instead of loading another model.
    """

    def embed_text(
        self,
        text: str,
    ) -> Any:
        ...


class PineconeSemanticMemoryStore:
    """
    Pinecone-backed semantic learning-memory store.

    Every user gets a separate deterministic namespace.
    Raw user IDs are not exposed as namespace names.
    """

    def __init__(
        self,
        *,
        embedder: SemanticEmbeddingProtocol,
        api_key: str | None = None,
        index_name: str | None = None,
        expected_dimension: int = 384,
    ) -> None:
        if expected_dimension <= 0:
            raise ValueError(
                "expected_dimension must be positive."
            )

        resolved_api_key = (
            api_key
            or settings.pinecone_api_key
        )

        resolved_index_name = (
            index_name
            or settings.pinecone_learning_memory_index
        )

        if not resolved_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not configured."
            )

        if not resolved_index_name:
            raise RuntimeError(
                "Pinecone learning-memory index "
                "name is not configured."
            )

        self.embedder = embedder

        self.expected_dimension = (
            expected_dimension
        )

        self._pc = Pinecone(
            api_key=resolved_api_key
        )

        self._index = self._pc.index(
            resolved_index_name
        )

    def upsert(
        self,
        record: SemanticLearningMemoryRecord,
    ) -> bool:
        """
        Embed and store one validated learning memory.

        Failure must not break the main Tutor workflow.
        """

        try:
            vector = self._embed(
                record.embedding_text
            )

            metadata = {
                "kind": record.kind.value,
                "topic": record.topic,
                "text": record.text,
                "confidence": (
                    float(record.confidence)
                ),
                "status": record.status.value,
                "created_at": (
                    record.created_at.isoformat()
                ),
                "updated_at": (
                    record.updated_at.isoformat()
                ),
                "last_reinforced_at": (
                    record.last_reinforced_at
                    .isoformat()
                ),
                "schema_version": (
                    int(record.schema_version)
                ),
            }

            if record.concept is not None:
                metadata["concept"] = (
                    record.concept
                )

            if (
                record.source_session_id
                is not None
            ):
                metadata[
                    "source_session_id"
                ] = record.source_session_id

            if (
                record.source_document_id
                is not None
            ):
                metadata[
                    "source_document_id"
                ] = record.source_document_id

            response = self._index.upsert(
                vectors=[
                    (
                        record.memory_id,
                        vector,
                        metadata,
                    )
                ],
                namespace=self._namespace(
                    record.user_id
                ),
            )

            upserted_count = getattr(
                response,
                "upserted_count",
                None,
            )

            return (
                upserted_count == 1
                if upserted_count
                is not None
                else True
            )

        except Exception:
            return False

    def search(
        self,
        *,
        user_id: str,
        query_text: str,
        top_k: int = 5,
        kinds: Sequence[
            SemanticMemoryKind
        ] | None = None,
        include_resolved: bool = False,
        minimum_score: float = 0.0,
    ) -> list[SemanticMemoryMatch]:
        """
        Search only this user's semantic memories.

        By default only ACTIVE memories are returned.
        """

        normalized_user_id = (
            user_id.strip()
        )

        normalized_query = (
            query_text.strip()
        )

        if not normalized_user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        if not normalized_query:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        if not (
            -1.0
            <= minimum_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_score must be between "
                "-1.0 and 1.0."
            )

        metadata_filter = (
            self._build_filter(
                kinds=kinds,
                include_resolved=(
                    include_resolved
                ),
            )
        )

        try:
            query_vector = self._embed(
                normalized_query
            )

            response = self._index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=self._namespace(
                    normalized_user_id
                ),
                filter=metadata_filter,
                include_metadata=True,
                include_values=False,
            )

        except Exception:
            return []

        matches = getattr(
            response,
            "matches",
            None,
        )

        if not matches:
            return []

        results: list[
            SemanticMemoryMatch
        ] = []

        for match in matches:
            score = getattr(
                match,
                "score",
                None,
            )

            metadata = getattr(
                match,
                "metadata",
                None,
            )

            memory_id = getattr(
                match,
                "id",
                None,
            )

            if (
                score is None
                or not isinstance(
                    metadata,
                    dict,
                )
                or not memory_id
            ):
                continue

            numeric_score = float(
                score
            )

            if (
                numeric_score
                < minimum_score
            ):
                continue

            try:
                record = (
                    SemanticLearningMemoryRecord(
                        memory_id=str(
                            memory_id
                        ),
                        user_id=(
                            normalized_user_id
                        ),
                        kind=metadata[
                            "kind"
                        ],
                        topic=metadata[
                            "topic"
                        ],
                        concept=metadata.get(
                            "concept"
                        ),
                        text=metadata[
                            "text"
                        ],
                        confidence=float(
                            metadata.get(
                                "confidence",
                                1.0,
                            )
                        ),
                        status=metadata.get(
                            "status",
                            (
                                SemanticMemoryStatus
                                .ACTIVE
                                .value
                            ),
                        ),
                        source_session_id=(
                            metadata.get(
                                "source_session_id"
                            )
                        ),
                        source_document_id=(
                            metadata.get(
                                "source_document_id"
                            )
                        ),
                        created_at=metadata[
                            "created_at"
                        ],
                        updated_at=metadata[
                            "updated_at"
                        ],
                        last_reinforced_at=(
                            metadata[
                                "last_reinforced_at"
                            ]
                        ),
                        schema_version=int(
                            metadata.get(
                                "schema_version",
                                1,
                            )
                        ),
                    )
                )

                results.append(
                    SemanticMemoryMatch(
                        record=record,
                        similarity_score=(
                            numeric_score
                        ),
                    )
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

        return results

    def delete(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> bool:
        """
        Delete one memory from one user's namespace.
        """

        normalized_user_id = (
            user_id.strip()
        )

        normalized_memory_id = (
            memory_id.strip()
        )

        if not normalized_user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        if not normalized_memory_id:
            raise ValueError(
                "memory_id cannot be empty."
            )

        try:
            self._index.delete(
                ids=[
                    normalized_memory_id
                ],
                namespace=self._namespace(
                    normalized_user_id
                ),
            )

            return True

        except Exception:
            return False

    def _embed(
        self,
        text: str,
    ) -> list[float]:
        raw_vector = (
            self.embedder.embed_text(
                text
            )
        )

        if hasattr(
            raw_vector,
            "tolist",
        ):
            raw_vector = (
                raw_vector.tolist()
            )

        try:
            vector = [
                float(value)
                for value
                in raw_vector
            ]

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Embedding must be a "
                "one-dimensional numeric vector."
            ) from exc

        if (
            len(vector)
            != self.expected_dimension
        ):
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected "
                f"{self.expected_dimension}, "
                f"got {len(vector)}."
            )

        return vector

    @staticmethod
    def _namespace(
        user_id: str,
    ) -> str:
        normalized_user_id = (
            user_id.strip()
        )

        if not normalized_user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        digest = hashlib.sha256(
            normalized_user_id.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            "phymentor-user-"
            f"{digest[:32]}"
        )

    @staticmethod
    def _build_filter(
        *,
        kinds: Sequence[
            SemanticMemoryKind
        ] | None,
        include_resolved: bool,
    ) -> dict[str, Any] | None:
        clauses: list[
            dict[str, Any]
        ] = []

        if not include_resolved:
            clauses.append(
                {
                    "status": {
                        "$eq": (
                            SemanticMemoryStatus
                            .ACTIVE
                            .value
                        )
                    }
                }
            )

        if kinds:
            clauses.append(
                {
                    "kind": {
                        "$in": [
                            kind.value
                            for kind
                            in kinds
                        ]
                    }
                }
            )

        if not clauses:
            return None

        if len(clauses) == 1:
            return clauses[0]

        return {
            "$and": clauses
        }