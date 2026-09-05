from __future__ import annotations

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from apps.api.schemas import (
    DocumentDeleteResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)

from src.cache.invalidation import (
    invalidate_document_cache,
)

from src.config import settings
from src.runtime_services import (
    get_ingestion_service,
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a school-level Physics PDF or image",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(
        default=settings.default_local_user_id
    ),
) -> DocumentUploadResponse:

    normalized_user_id = (
        user_id.strip()
    )

    if not normalized_user_id:
        raise ValueError(
            "user_id cannot be empty."
        )

    ingestion_service = (
        get_ingestion_service()
    )

    try:
        file_bytes = await file.read(
            settings.max_upload_size_bytes
            + 1
        )

        uploaded_metadata = (
            ingestion_service
            .upload_document(
                user_id=normalized_user_id,
                filename=file.filename,
                content_type=file.content_type,
                file_bytes=file_bytes,
            )
        )

        document_id = str(
            uploaded_metadata[
                "document_id"
            ]
        ).strip()

        if not document_id:
            raise ValueError(
                "Uploaded document did not "
                "produce a valid document_id."
            )

        # -------------------------------------------------
        # PHASE 7 — DUPLICATE DOCUMENT FAST PATH
        # -------------------------------------------------
        #
        # The ingestion layer has already compared
        # SHA-256 fingerprints.
        #
        # If this file already exists:
        #
        #   - reuse existing document_id
        #   - do NOT save another copy
        #   - do NOT invalidate its valid cache
        #   - do NOT run OCR/chunking/indexing again
        # -------------------------------------------------

        deduplicated = bool(
            uploaded_metadata.get(
                "_deduplicated",
                False,
            )
        )

        if deduplicated:
            response_metadata = dict(
                uploaded_metadata
            )

            response_metadata.pop(
                "_deduplicated",
                None,
            )

            return DocumentUploadResponse(
                **response_metadata
            )

        # -------------------------------------------------
        # NEW DOCUMENT — CACHE INVALIDATION
        # -------------------------------------------------

        await invalidate_document_cache(
            user_id=normalized_user_id,
            document_id=document_id,
        )

        # -------------------------------------------------
        # NEW DOCUMENT — NORMAL PROCESSING
        # -------------------------------------------------

        if (
            settings
            .process_uploads_in_background
        ):
            response_metadata = (
                ingestion_service
                .queue_document_processing(
                    user_id=(
                        normalized_user_id
                    ),
                    document_id=document_id,
                )
            )

            background_tasks.add_task(
                ingestion_service
                .process_document_background,
                user_id=(
                    normalized_user_id
                ),
                document_id=document_id,
            )

        else:
            response_metadata = (
                ingestion_service
                .process_document(
                    user_id=(
                        normalized_user_id
                    ),
                    document_id=document_id,
                )
            )

        # Internal dedupe marker should never be
        # exposed as part of the public API model.
        response_metadata = dict(
            response_metadata
        )

        response_metadata.pop(
            "_deduplicated",
            None,
        )

        return DocumentUploadResponse(
            **response_metadata
        )

    finally:
        await file.close()


# =========================================================
# DOCUMENT STATUS
# =========================================================

@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document processing status",
)
async def get_document_status(
    document_id: str,
    user_id: str = Query(
        default=settings.default_local_user_id
    ),
) -> DocumentStatusResponse:

    normalized_user_id = (
        user_id.strip()
    )

    normalized_document_id = (
        document_id.strip()
    )

    if not normalized_user_id:
        raise ValueError(
            "user_id cannot be empty."
        )

    if not normalized_document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    ingestion_service = (
        get_ingestion_service()
    )

    result = (
        ingestion_service
        .get_document_status(
            user_id=normalized_user_id,
            document_id=(
                normalized_document_id
            ),
        )
    )

    return DocumentStatusResponse(
        **result
    )


# =========================================================
# RE-INDEX DOCUMENT
# =========================================================

@router.post(
    "/{document_id}/reindex",
    response_model=DocumentStatusResponse,
    summary="Re-process and re-index an existing document",
)
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Query(
        default=settings.default_local_user_id
    ),
    x_user_id: str | None = Header(
        default=None,
        alias="X-User-ID",
    ),
) -> DocumentStatusResponse:
    """
    Re-run the ingestion/indexing pipeline for an
    existing document.

    Important order:

        verify document exists
              ↓
        invalidate old answer cache
              ↓
        queue/process existing document
              ↓
        build fresh FAISS/BM25 indexes

    This prevents answers generated from an old index
    from being returned after re-indexing.
    """

    resolved_user_id = (
        x_user_id.strip()
        if (
            x_user_id
            and x_user_id.strip()
        )
        else user_id.strip()
    )

    normalized_document_id = (
        document_id.strip()
    )

    if not resolved_user_id:
        raise ValueError(
            "user_id cannot be empty."
        )

    if not normalized_document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    ingestion_service = (
        get_ingestion_service()
    )

    # -----------------------------------------------------
    # VERIFY THAT THE DOCUMENT EXISTS
    # -----------------------------------------------------

    existing_metadata = (
        ingestion_service
        .get_document_status(
            user_id=resolved_user_id,
            document_id=(
                normalized_document_id
            ),
        )
    )

    current_status = str(
        existing_metadata.get(
            "status",
            "",
        )
    ).strip().upper()

    current_stage = str(
        existing_metadata.get(
            "processing_stage",
            "",
        )
    ).strip().upper()

    # -----------------------------------------------------
    # DO NOT START TWO INDEXING JOBS FOR THE SAME DOCUMENT
    # -----------------------------------------------------

    if (
        current_status
        in {
            "PROCESSING",
            "INDEXING",
        }
        or current_stage
        in {
            "QUEUED",
            "PARSING",
            "LAYOUT_ANALYSIS",
            "OCR",
            "EQUATION_EXTRACTION",
            "FIGURE_EXTRACTION",
            "SCOPE_CLASSIFICATION",
            "CHUNKING",
            "FAISS_BM25_INDEXING",
        }
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=(
                "This document is already "
                "being processed or indexed."
            ),
        )

    # -----------------------------------------------------
    # PHASE 7 — CACHE INVALIDATION
    # -----------------------------------------------------
    #
    # This is the critical Phase-7 rule:
    #
    # old document evidence
    #       ↓
    # old cached answers
    #
    # must disappear BEFORE the document's
    # retrieval index changes.
    # -----------------------------------------------------

    await invalidate_document_cache(
        user_id=resolved_user_id,
        document_id=(
            normalized_document_id
        ),
    )

    # -----------------------------------------------------
    # RE-PROCESS EXISTING DOCUMENT
    # -----------------------------------------------------

    if (
        settings
        .process_uploads_in_background
    ):
        response_metadata = (
            ingestion_service
            .queue_document_processing(
                user_id=resolved_user_id,
                document_id=(
                    normalized_document_id
                ),
            )
        )

        background_tasks.add_task(
            ingestion_service
            .process_document_background,
            user_id=resolved_user_id,
            document_id=(
                normalized_document_id
            ),
        )

    else:
        response_metadata = (
            ingestion_service
            .process_document(
                user_id=resolved_user_id,
                document_id=(
                    normalized_document_id
                ),
            )
        )

    return DocumentStatusResponse(
        **response_metadata
    )


# =========================================================
# DELETE DOCUMENT
# =========================================================

@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a local document",
)
async def delete_document(
    document_id: str,
    user_id: str = Query(
        default=settings.default_local_user_id
    ),
    x_user_id: str | None = Header(
        default=None,
        alias="X-User-ID",
    ),
) -> DocumentDeleteResponse:

    resolved_user_id = (
        x_user_id.strip()
        if (
            x_user_id
            and x_user_id.strip()
        )
        else user_id.strip()
    )

    normalized_document_id = (
        document_id.strip()
    )

    if not resolved_user_id:
        raise ValueError(
            "user_id cannot be empty."
        )

    if not normalized_document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    # -----------------------------------------------------
    # PHASE 7 — CACHE INVALIDATION
    # -----------------------------------------------------

    await invalidate_document_cache(
        user_id=resolved_user_id,
        document_id=(
            normalized_document_id
        ),
    )

    ingestion_service = (
        get_ingestion_service()
    )

    ingestion_service.delete_document(
        user_id=resolved_user_id,
        document_id=(
            normalized_document_id
        ),
    )

    return DocumentDeleteResponse(
        document_id=(
            normalized_document_id
        ),
        user_id=resolved_user_id,
        deleted=True,
        message=(
            "The document, all local "
            "artifacts, and related answer "
            "cache were deleted successfully."
        ),
    )