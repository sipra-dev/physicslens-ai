from __future__ import annotations

import tempfile
from pathlib import Path

from src.ingestion.indexer import (
    DocumentIndexingError,
    LocalDocumentIndexer,
)


class FakeChunk:
    def __init__(
        self,
        *,
        chunk_kind: str = "child",
    ) -> None:
        self.chunk_kind = chunk_kind


class FakeParent:
    def model_dump(
        self,
        *,
        mode: str,
    ) -> dict:
        return {
            "parent_id": "parent-1",
            "text": "test parent",
        }


class FakeChunkingResult:
    user_id = "test-user"
    document_id = "test-document"

    retrieval_chunks = [
        FakeChunk(),
    ]

    parent_chunks = [
        FakeParent(),
    ]


class FakeDenseRetriever:
    def build(
        self,
        *,
        chunks,
        index_directory: Path,
    ) -> None:
        index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            index_directory
            / "dense-built.txt"
        ).write_text(
            "dense built",
            encoding="utf-8",
        )


class FailingBM25Retriever:
    def build(
        self,
        *,
        chunks,
        index_directory: Path,
    ) -> None:
        index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            index_directory
            / "partial-bm25.txt"
        ).write_text(
            "partial bm25",
            encoding="utf-8",
        )

        raise RuntimeError(
            "Simulated BM25 crash"
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        vector_root = (
            root / "vector"
        )

        bm25_root = (
            root / "bm25"
        )

        # Avoid loading real embedding models.
        indexer = object.__new__(
            LocalDocumentIndexer
        )

        indexer.vector_store_directory = (
            vector_root.resolve()
        )

        indexer.bm25_store_directory = (
            bm25_root.resolve()
        )

        indexer.dense_retriever = (
            FakeDenseRetriever()
        )

        indexer.bm25_retriever = (
            FailingBM25Retriever()
        )

        dense_final = (
            vector_root
            / "users"
            / "test-user"
            / "documents"
            / "test-document"
        )

        bm25_final = (
            bm25_root
            / "users"
            / "test-user"
            / "documents"
            / "test-document"
        )

        # ---------------------------------------------
        # Existing good indexes
        # ---------------------------------------------

        dense_final.mkdir(
            parents=True,
            exist_ok=True,
        )

        bm25_final.mkdir(
            parents=True,
            exist_ok=True,
        )

        dense_marker = (
            dense_final / "OLD_GOOD_DENSE.txt"
        )

        bm25_marker = (
            bm25_final / "OLD_GOOD_BM25.txt"
        )

        dense_marker.write_text(
            "old dense",
            encoding="utf-8",
        )

        bm25_marker.write_text(
            "old bm25",
            encoding="utf-8",
        )

        # ---------------------------------------------
        # Simulate failed re-index
        # ---------------------------------------------

        failed_as_expected = False

        try:
            indexer.index_document(
                chunking_result=(
                    FakeChunkingResult()
                )
            )

        except DocumentIndexingError:
            failed_as_expected = True

        # Old valid indexes must still exist.
        old_dense_preserved = (
            dense_marker.is_file()
        )

        old_bm25_preserved = (
            bm25_marker.is_file()
        )

        # No staging directories should remain.
        dense_parent = (
            dense_final.parent
        )

        bm25_parent = (
            bm25_final.parent
        )

        dense_staging_left = any(
            path.name.startswith(
                ".test-document.building-"
            )
            for path
            in dense_parent.iterdir()
        )

        bm25_staging_left = any(
            path.name.startswith(
                ".test-document.building-"
            )
            for path
            in bm25_parent.iterdir()
        )

        cleanup_ok = (
            not dense_staging_left
            and not bm25_staging_left
        )

        print(
            f"FAILURE_CAUGHT="
            f"{failed_as_expected}"
        )

        print(
            f"OLD_DENSE_PRESERVED="
            f"{old_dense_preserved}"
        )

        print(
            f"OLD_BM25_PRESERVED="
            f"{old_bm25_preserved}"
        )

        print(
            f"PARTIAL_BUILD_CLEANED="
            f"{cleanup_ok}"
        )

        # ---------------------------------------------
        # Test explicit index deletion
        # ---------------------------------------------

        deleted = (
            indexer.delete_document_indexes(
                user_id="test-user",
                document_id="test-document",
            )
        )

        delete_cleanup_ok = (
            deleted
            and not dense_final.exists()
            and not bm25_final.exists()
        )

        print(
            f"DOCUMENT_INDEX_DELETE_OK="
            f"{delete_cleanup_ok}"
        )

        all_passed = all(
            [
                failed_as_expected,
                old_dense_preserved,
                old_bm25_preserved,
                cleanup_ok,
                delete_cleanup_ok,
            ]
        )

        print()

        if all_passed:
            print(
                "PHASE7_INDEXER_CLEANUP_TEST=PASS"
            )
            return

        raise SystemExit(
            "PHASE7_INDEXER_CLEANUP_TEST=FAIL"
        )


if __name__ == "__main__":
    main()