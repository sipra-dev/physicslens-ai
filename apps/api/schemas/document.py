from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    PROCESSING = "PROCESSING"
    INDEXING = "INDEXING"
    READY = "READY"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class DocumentUploadResponse(BaseModel):
    document_id: str
    user_id: str
    status: DocumentStatus
    processing_stage: str

    original_filename: str
    stored_filename: str
    content_type: str
    file_extension: str

    size_bytes: int = Field(ge=1)

    sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    page_count: int | None = Field(
        default=None,
        ge=1,
    )

    image_width: int | None = Field(
        default=None,
        ge=1,
    )

    image_height: int | None = Field(
        default=None,
        ge=1,
    )

    storage_path: str
    uploaded_at: datetime

    scope_classification: (
        dict[str, Any] | None
    ) = None

    artifacts: dict[str, str] = Field(
        default_factory=dict
    )

    processing_error: str | None = None
    message: str

    index_manifest: (
        dict[str, Any] | None
    ) = None


class DocumentStatusResponse(
    DocumentUploadResponse
):
    pass


class DocumentDeleteResponse(BaseModel):
    document_id: str
    user_id: str
    deleted: bool
    message: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None