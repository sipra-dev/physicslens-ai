from __future__ import annotations

from src.ingestion.service import (
    IngestionService,
)


class FakeStorage:
    def __init__(self) -> None:
        self.metadata_exists = True
        self.delete_called = False
        self.calls: list[str] = []

    def read_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ):
        if not self.metadata_exists:
            return None

        return {
            "user_id": user_id,
            "document_id": document_id,
        }

    def delete_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        self.delete_called = True
        self.calls.append(
            "storage_delete"
        )

        return True


class FakeIndexer:
    def __init__(
        self,
        *,
        fail: bool = False,
    ) -> None:
        self.fail = fail
        self.delete_called = False
        self.calls: list[str] = []

    def delete_document_indexes(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        self.delete_called = True
        self.calls.append(
            "index_delete"
        )

        if self.fail:
            raise RuntimeError(
                "Simulated index cleanup failure"
            )

        return True


def build_service(
    *,
    storage: FakeStorage,
    indexer: FakeIndexer,
) -> IngestionService:
    service = object.__new__(
        IngestionService
    )

    service.storage = storage
    service.indexer = indexer

    return service


def main() -> None:
    # =====================================================
    # TEST 1 — NORMAL COMPLETE DELETE
    # =====================================================

    storage = FakeStorage()
    indexer = FakeIndexer()

    service = build_service(
        storage=storage,
        indexer=indexer,
    )

    deleted = service.delete_document(
        user_id="test-user",
        document_id="test-document",
    )

    indexes_deleted = (
        indexer.delete_called
    )

    document_deleted = (
        storage.delete_called
    )

    correct_order = (
        indexer.calls
        + storage.calls
        == [
            "index_delete",
            "storage_delete",
        ]
    )

    print(
        f"DELETE_RETURNED_TRUE="
        f"{deleted is True}"
    )

    print(
        f"INDEXES_DELETED="
        f"{indexes_deleted}"
    )

    print(
        f"DOCUMENT_DELETED="
        f"{document_deleted}"
    )

    print(
        f"INDEXES_DELETED_FIRST="
        f"{correct_order}"
    )

    # =====================================================
    # TEST 2 — INDEX CLEANUP FAILURE
    # =====================================================

    failing_storage = FakeStorage()

    failing_indexer = FakeIndexer(
        fail=True
    )

    failing_service = build_service(
        storage=failing_storage,
        indexer=failing_indexer,
    )

    failure_caught = False

    try:
        failing_service.delete_document(
            user_id="test-user",
            document_id="test-document",
        )

    except RuntimeError:
        failure_caught = True

    document_preserved = (
        not failing_storage.delete_called
    )

    print(
        f"INDEX_FAILURE_CAUGHT="
        f"{failure_caught}"
    )

    print(
        f"DOCUMENT_PRESERVED_ON_INDEX_FAILURE="
        f"{document_preserved}"
    )

    # =====================================================
    # FINAL
    # =====================================================

    all_passed = all(
        [
            deleted is True,
            indexes_deleted,
            document_deleted,
            correct_order,
            failure_caught,
            document_preserved,
        ]
    )

    print()

    if all_passed:
        print(
            "PHASE7_DOCUMENT_DELETE_CLEANUP_TEST=PASS"
        )

        return

    raise SystemExit(
        "PHASE7_DOCUMENT_DELETE_CLEANUP_TEST=FAIL"
    )


if __name__ == "__main__":
    main()