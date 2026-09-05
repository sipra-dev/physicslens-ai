from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


def _utc_now() -> datetime:
    """
    Return a timezone-aware UTC timestamp.
    """
    return datetime.now(timezone.utc)


class SemanticMemoryKind(str, Enum):
    """
    Types of learning information that are useful
    to retrieve later by semantic similarity.
    """

    MISCONCEPTION = "misconception"
    KNOWLEDGE_GAP = "knowledge_gap"
    MASTERY = "mastery"
    SUPPORT_PREFERENCE = "support_preference"


class SemanticMemoryStatus(str, Enum):
    """
    Lifecycle state of a semantic learning memory.

    Freshness / decay policy will be implemented
    separately in Phase 7 #7.
    """

    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPERSEDED = "superseded"


class SemanticLearningMemoryRecord(BaseModel):
    """
    One durable semantic learning-memory record.

    Example:
        User repeatedly confuses velocity with acceleration.

    The text is embedded and stored in Pinecone.
    Metadata is used for filtering and isolation.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )

    memory_id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=128,
    )

    user_id: str = Field(
        min_length=1,
        max_length=256,
    )

    kind: SemanticMemoryKind

    topic: str = Field(
        min_length=1,
        max_length=256,
    )

    concept: str | None = Field(
        default=None,
        max_length=256,
    )

    text: str = Field(
        min_length=1,
        max_length=4000,
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    status: SemanticMemoryStatus = (
        SemanticMemoryStatus.ACTIVE
    )

    source_session_id: str | None = Field(
        default=None,
        max_length=256,
    )

    source_document_id: str | None = Field(
        default=None,
        max_length=256,
    )

    created_at: datetime = Field(
        default_factory=_utc_now,
    )

    updated_at: datetime = Field(
        default_factory=_utc_now,
    )

    last_reinforced_at: datetime = Field(
        default_factory=_utc_now,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
    )

    @field_validator(
        "memory_id",
        "user_id",
        "topic",
        "text",
    )
    @classmethod
    def _strip_required_text(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Value cannot be empty."
            )

        return cleaned

    @field_validator(
        "concept",
        "source_session_id",
        "source_document_id",
    )
    @classmethod
    def _strip_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @property
    def embedding_text(self) -> str:
        """
        Stable text representation that will later
        be converted into a 384-dimensional vector.
        """

        parts = [
            f"type: {self.kind.value}",
            f"topic: {self.topic}",
        ]

        if self.concept:
            parts.append(
                f"concept: {self.concept}"
            )

        parts.append(
            f"memory: {self.text}"
        )

        return "\n".join(parts)


class SemanticMemoryMatch(BaseModel):
    """
    One semantic-memory search result returned
    after querying Pinecone.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    record: SemanticLearningMemoryRecord

    similarity_score: float = Field(
        ge=-1.0,
        le=1.0,
    )