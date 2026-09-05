from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from src.models.contracts import LanguageCode


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


class TopicProgress(BaseModel):
    """
    Durable learning progress for one Physics topic.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    topic: str = Field(
        min_length=1,
        max_length=200,
    )

    mastery_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    attempts: int = Field(
        default=0,
        ge=0,
    )

    correct_attempts: int = Field(
        default=0,
        ge=0,
    )

    last_seen_at: datetime = Field(
        default_factory=_utc_now
    )

    updated_at: datetime = Field(
        default_factory=_utc_now
    )

    @field_validator(
        "topic"
    )
    @classmethod
    def _normalize_topic(
        cls,
        value: str,
    ) -> str:
        normalized = " ".join(
            value.strip().split()
        )

        if not normalized:
            raise ValueError(
                "topic cannot be empty."
            )

        return normalized


class MisconceptionRecord(BaseModel):
    """
    Stable misconception learned about a student.

    Example:
    Student repeatedly thinks velocity and
    acceleration are the same thing.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    concept: str = Field(
        min_length=1,
        max_length=200,
    )

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    status: Literal[
        "active",
        "improving",
        "resolved",
    ] = "active"

    source: str | None = Field(
        default=None,
        max_length=500,
    )

    first_seen_at: datetime = Field(
        default_factory=_utc_now
    )

    last_seen_at: datetime = Field(
        default_factory=_utc_now
    )

    @field_validator(
        "concept",
        "description",
    )
    @classmethod
    def _normalize_required_text(
        cls,
        value: str,
    ) -> str:
        normalized = " ".join(
            value.strip().split()
        )

        if not normalized:
            raise ValueError(
                "value cannot be empty."
            )

        return normalized


class LongTermMemoryProfile(BaseModel):
    """
    Durable student learning profile.

    Unlike Redis session memory, this data is
    intended to survive session expiry and
    application restarts.

    PostgreSQL will persist this model.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    user_id: str = Field(
        min_length=1,
        max_length=255,
    )

    preferred_language: (
        LanguageCode | None
    ) = None

    grade: int | None = Field(
        default=None,
        ge=1,
        le=12,
    )

    learning_style: str | None = Field(
        default=None,
        max_length=200,
    )

    progress: list[
        TopicProgress
    ] = Field(
        default_factory=list
    )

    misconceptions: list[
        MisconceptionRecord
    ] = Field(
        default_factory=list
    )

    created_at: datetime = Field(
        default_factory=_utc_now
    )

    updated_at: datetime = Field(
        default_factory=_utc_now
    )

    @field_validator(
        "user_id"
    )
    @classmethod
    def _normalize_user_id(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "user_id cannot be empty."
            )

        return normalized

    @field_validator(
        "learning_style"
    )
    @classmethod
    def _normalize_learning_style(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = " ".join(
            value.strip().split()
        )

        return (
            normalized
            or None
        )


__all__ = [
    "LongTermMemoryProfile",
    "MisconceptionRecord",
    "TopicProgress",
]