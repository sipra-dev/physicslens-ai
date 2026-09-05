from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.ingestion.service import (
    DocumentProcessingError,
    IngestionService,
)


class FakeStorage:
    def __init__(
        self,
        *,
        root: Path,
    ) -> None:
        self.root = root

        self.metadata = {
            "document_id": "test-document",
            "user_id": "test-user",
            "status": "UPLOADED",
            "processing_stage": "UPLOADED",
            "storage_path": "source.pdf",
            "artifacts": {},
            "processing_error": None,
        }

        self.delete_called = False
        self.writes: list[dict] = []

    def read_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ):
        return dict(self.metadata)

    def get_document_directory(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> Path:
        directory = (
            self.root
            / "users"
            / user_id
            / "documents"
            / document_id
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    def write_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
        metadata: dict,
    ) -> None:
        self.metadata = dict(metadata)
        self.writes.append(
            dict(metadata)
        )

    def delete_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        self.delete_called = True
        return True


class FailingParser:
    def parse(
        self,
        **kwargs,
    ):
        raise RuntimeError(
            "Simulated parser crash"
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        storage = FakeStorage(
            root=root
        )

        service = object.__new__(
            IngestionService
        )

        service.storage = storage

        service.settings = SimpleNamespace(
            upload_dir=root
        )

        service.parser = FailingParser()

        failure_caught = False

        try:
            service.process_document(
                user_id="test-user",
                document_id="test-document",
            )

        except DocumentProcessingError:
            failure_caught = True

        final_metadata = (
            storage.metadata
        )

        status_failed = (
            final_metadata.get("status")
            == "FAILED"
        )

        stage_failed = (
            final_metadata.get(
                "processing_stage"
            )
            == "FAILED"
        )

        error_recorded = (
            "RuntimeError"
            in str(
                final_metadata.get(
                    "processing_error"
                )
            )
        )

        original_preserved = (
            not storage.delete_called
        )

        not_stuck_processing = (
            final_metadata.get("status")
            not in {
                "PROCESSING",
                "INDEXING",
            }
        )

        print(
            f"FAILURE_CAUGHT="
            f"{failure_caught}"
        )

        print(
            f"STATUS_FAILED="
            f"{status_failed}"
        )

        print(
            f"STAGE_FAILED="
            f"{stage_failed}"
        )

        print(
            f"PROCESSING_ERROR_RECORDED="
            f"{error_recorded}"
        )

        print(
            f"ORIGINAL_DOCUMENT_PRESERVED="
            f"{original_preserved}"
        )

        print(
            f"NOT_STUCK_PROCESSING="
            f"{not_stuck_processing}"
        )

        all_passed = all(
            [
                failure_caught,
                status_failed,
                stage_failed,
                error_recorded,
                original_preserved,
                not_stuck_processing,
            ]
        )

        print()

        if all_passed:
            print(
                "PHASE7_PROCESSING_FAILURE_TEST=PASS"
            )
            return

        raise SystemExit(
            "PHASE7_PROCESSING_FAILURE_TEST=FAIL"
        )


if __name__ == "__main__":
    main()