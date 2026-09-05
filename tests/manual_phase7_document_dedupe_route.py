from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks

import apps.api.routes.documents as documents_route


class FakeUploadFile:
    def __init__(self) -> None:
        self.filename = "renamed-physics.pdf"
        self.content_type = "application/pdf"
        self.closed = False

    async def read(
        self,
        size: int = -1,
    ) -> bytes:
        return b"same-existing-pdf"

    async def close(
        self,
    ) -> None:
        self.closed = True


class FakeIngestionService:
    def __init__(self) -> None:
        self.upload_calls = 0
        self.processing_called = False

    def upload_document(
        self,
        *,
        user_id: str,
        filename: str | None,
        content_type: str | None,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        self.upload_calls += 1

        return {
            "document_id": "existing-doc-123",
            "user_id": user_id,
            "status": "READY",
            "processing_stage": "READY",
            "original_filename": "physics.pdf",
            "stored_filename": "physics.pdf",
            "content_type": "application/pdf",
            "file_extension": ".pdf",
            "size_bytes": 123,
            "sha256": "a" * 64,
            "page_count": 3,
            "image_width": None,
            "image_height": None,
            "storage_path": (
                "users/test-user/documents/"
                "existing-doc-123/original/"
                "physics.pdf"
            ),
            "uploaded_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "scope_classification": None,
            "artifacts": {},
            "processing_error": None,
            "index_manifest": {},
            "_deduplicated": True,
            "message": (
                "This file was already uploaded. "
                "The existing document is being reused."
            ),
        }

    def queue_document_processing(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.processing_called = True

        raise RuntimeError(
            "Duplicate document must not be queued."
        )

    def process_document(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.processing_called = True

        raise RuntimeError(
            "Duplicate document must not be processed."
        )

    def process_document_background(
        self,
        **kwargs: Any,
    ) -> None:
        self.processing_called = True

        raise RuntimeError(
            "Duplicate document must not run "
            "background processing."
        )


async def main() -> None:
    original_ingestion_service = (
        documents_route.ingestion_service
    )

    original_invalidator = (
        documents_route.invalidate_document_cache
    )

    fake_service = FakeIngestionService()
    fake_file = FakeUploadFile()

    cache_invalidation_called = False

    async def forbidden_cache_invalidation(
        *,
        user_id: str,
        document_id: str,
    ) -> Any:
        nonlocal cache_invalidation_called

        cache_invalidation_called = True

        raise RuntimeError(
            "Duplicate document must not "
            "invalidate its existing cache."
        )

    try:
        documents_route.ingestion_service = (
            fake_service
        )

        documents_route.invalidate_document_cache = (
            forbidden_cache_invalidation
        )

        response = (
            await documents_route.upload_document(
                background_tasks=(
                    BackgroundTasks()
                ),
                file=fake_file,  # type: ignore[arg-type]
                user_id="test-user",
            )
        )

        existing_document_reused = (
            response.document_id
            == "existing-doc-123"
        )

        print(
            "EXISTING_DOCUMENT_REUSED="
            f"{existing_document_reused}"
        )

        no_processing = (
            not fake_service.processing_called
        )

        print(
            "PROCESSING_SKIPPED="
            f"{no_processing}"
        )

        no_cache_invalidation = (
            not cache_invalidation_called
        )

        print(
            "CACHE_INVALIDATION_SKIPPED="
            f"{no_cache_invalidation}"
        )

        upload_called_once = (
            fake_service.upload_calls == 1
        )

        print(
            "UPLOAD_CHECK_CALLED_ONCE="
            f"{upload_called_once}"
        )

        file_closed = fake_file.closed

        print(
            "UPLOAD_FILE_CLOSED="
            f"{file_closed}"
        )

        all_ok = all(
            (
                existing_document_reused,
                no_processing,
                no_cache_invalidation,
                upload_called_once,
                file_closed,
            )
        )

        print()

        if all_ok:
            print(
                "PHASE7_DOCUMENT_DEDUPE_ROUTE_TEST=PASS"
            )
        else:
            print(
                "PHASE7_DOCUMENT_DEDUPE_ROUTE_TEST=FAIL"
            )

    finally:
        documents_route.ingestion_service = (
            original_ingestion_service
        )

        documents_route.invalidate_document_cache = (
            original_invalidator
        )


if __name__ == "__main__":
    asyncio.run(
        main()
    )