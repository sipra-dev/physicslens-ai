from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.ingestion.models import (
    ChunkingResult,
)
from src.retrieval.bm25 import (
    BM25Retriever,
)
from src.retrieval.dense import (
    DenseRetriever,
)


_SAFE_COMPONENT = re.compile(
    r"[^A-Za-z0-9._-]+"
)


class DocumentIndexingError(Exception):
    pass


class LocalDocumentIndexer:
    """
    Build and persist local retrieval indexes.

    Dense:
        SentenceTransformer -> FAISS

    Sparse:
        BM25 corpus

    Parent chunks:
        JSON store for later parent-context expansion.

    Phase 7 hardening:
    - build in temporary staging directories
    - publish only after full success
    - clean partial builds on failure
    - preserve old valid indexes during re-index
    """

    def __init__(
        self,
        *,
        vector_store_directory: Path,
        bm25_store_directory: Path,
        embedding_model_name: str,
    ) -> None:
        self.vector_store_directory = (
            vector_store_directory.resolve()
        )

        self.bm25_store_directory = (
            bm25_store_directory.resolve()
        )

        self.dense_retriever = DenseRetriever(
            model_name=embedding_model_name
        )

        self.bm25_retriever = (
            BM25Retriever()
        )

    def index_document(
        self,
        *,
        chunking_result: ChunkingResult,
    ) -> dict[str, Any]:

        user_id = (
            chunking_result.user_id
        )

        document_id = (
            chunking_result.document_id
        )

        retrieval_chunks = (
            chunking_result.retrieval_chunks
        )

        if not retrieval_chunks:
            raise DocumentIndexingError(
                "No retrieval chunks are "
                "available for indexing."
            )

        (
            dense_directory,
            bm25_directory,
        ) = self._document_index_directories(
            user_id=user_id,
            document_id=document_id,
        )

        build_id = uuid4().hex

        dense_staging_directory = (
            dense_directory.parent
            / (
                f".{dense_directory.name}"
                f".building-{build_id}"
            )
        )

        bm25_staging_directory = (
            bm25_directory.parent
            / (
                f".{bm25_directory.name}"
                f".building-{build_id}"
            )
        )

        dense_backup_directory = (
            dense_directory.parent
            / (
                f".{dense_directory.name}"
                f".backup-{build_id}"
            )
        )

        bm25_backup_directory = (
            bm25_directory.parent
            / (
                f".{bm25_directory.name}"
                f".backup-{build_id}"
            )
        )

        try:
            dense_staging_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            bm25_staging_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            self.dense_retriever.build(
                chunks=retrieval_chunks,
                index_directory=(
                    dense_staging_directory
                ),
            )

            self.bm25_retriever.build(
                chunks=retrieval_chunks,
                index_directory=(
                    bm25_staging_directory
                ),
            )

            parent_path = (
                dense_staging_directory
                / "parent_chunks.json"
            )

            self._write_json_atomic(
                path=parent_path,
                payload={
                    "user_id": user_id,
                    "document_id": (
                        document_id
                    ),
                    "parents": [
                        parent.model_dump(
                            mode="json"
                        )
                        for parent
                        in chunking_result
                        .parent_chunks
                    ],
                },
            )

            manifest = {
                "user_id": user_id,
                "document_id": (
                    document_id
                ),
                "parent_chunk_count": len(
                    chunking_result
                    .parent_chunks
                ),
                "retrieval_chunk_count": (
                    len(
                        retrieval_chunks
                    )
                ),
                "child_chunk_count": sum(
                    1
                    for chunk
                    in retrieval_chunks
                    if (
                        chunk.chunk_kind
                        == "child"
                    )
                ),
                "visual_chunk_count": sum(
                    1
                    for chunk
                    in retrieval_chunks
                    if (
                        chunk.chunk_kind
                        == "visual"
                    )
                ),
                "dense_index_directory": str(
                    dense_directory
                ),
                "bm25_index_directory": str(
                    bm25_directory
                ),
            }

            manifest_path = (
                dense_staging_directory
                / "index_manifest.json"
            )

            self._write_json_atomic(
                path=manifest_path,
                payload=manifest,
            )

            self._publish_staged_indexes(
                dense_staging_directory=(
                    dense_staging_directory
                ),
                bm25_staging_directory=(
                    bm25_staging_directory
                ),
                dense_directory=(
                    dense_directory
                ),
                bm25_directory=(
                    bm25_directory
                ),
                dense_backup_directory=(
                    dense_backup_directory
                ),
                bm25_backup_directory=(
                    bm25_backup_directory
                ),
            )

            return manifest

        except DocumentIndexingError:
            raise

        except Exception as exc:
            raise DocumentIndexingError(
                "Local document indexing failed."
            ) from exc

        finally:
            self._remove_directory_if_exists(
                dense_staging_directory
            )

            self._remove_directory_if_exists(
                bm25_staging_directory
            )

    def delete_document_indexes(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        """
        Remove FAISS + BM25 indexes for
        one user's document.
        """

        (
            dense_directory,
            bm25_directory,
        ) = self._document_index_directories(
            user_id=user_id,
            document_id=document_id,
        )

        existed = (
            dense_directory.exists()
            or bm25_directory.exists()
        )

        try:
            self._remove_directory_if_exists(
                dense_directory
            )

            self._remove_directory_if_exists(
                bm25_directory
            )

        except OSError as exc:
            raise DocumentIndexingError(
                "Document indexes could not "
                "be deleted."
            ) from exc

        return existed

    def _publish_staged_indexes(
        self,
        *,
        dense_staging_directory: Path,
        bm25_staging_directory: Path,
        dense_directory: Path,
        bm25_directory: Path,
        dense_backup_directory: Path,
        bm25_backup_directory: Path,
    ) -> None:

        dense_backup_created = False
        bm25_backup_created = False

        dense_new_published = False
        bm25_new_published = False

        try:
            if dense_directory.exists():
                os.replace(
                    dense_directory,
                    dense_backup_directory,
                )

                dense_backup_created = True

            if bm25_directory.exists():
                os.replace(
                    bm25_directory,
                    bm25_backup_directory,
                )

                bm25_backup_created = True

            os.replace(
                dense_staging_directory,
                dense_directory,
            )

            dense_new_published = True

            os.replace(
                bm25_staging_directory,
                bm25_directory,
            )

            bm25_new_published = True

        except Exception as exc:
            rollback_errors: list[
                Exception
            ] = []

            if dense_new_published:
                try:
                    self._remove_directory_if_exists(
                        dense_directory
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(
                        rollback_exc
                    )

            if bm25_new_published:
                try:
                    self._remove_directory_if_exists(
                        bm25_directory
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(
                        rollback_exc
                    )

            if (
                dense_backup_created
                and dense_backup_directory.exists()
            ):
                try:
                    os.replace(
                        dense_backup_directory,
                        dense_directory,
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(
                        rollback_exc
                    )

            if (
                bm25_backup_created
                and bm25_backup_directory.exists()
            ):
                try:
                    os.replace(
                        bm25_backup_directory,
                        bm25_directory,
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(
                        rollback_exc
                    )

            if rollback_errors:
                raise DocumentIndexingError(
                    "Publishing the new indexes "
                    "failed and rollback was "
                    "incomplete."
                ) from exc

            raise DocumentIndexingError(
                "Publishing the new indexes failed. "
                "The previous indexes were restored."
            ) from exc

        else:
            self._remove_directory_if_exists(
                dense_backup_directory
            )

            self._remove_directory_if_exists(
                bm25_backup_directory
            )

    def _document_index_directories(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> tuple[Path, Path]:

        safe_user_id = self._safe_component(
            user_id,
            fallback="local-user",
        )

        safe_document_id = (
            self._safe_component(
                document_id,
                fallback="document",
            )
        )

        dense_directory = (
            self.vector_store_directory
            / "users"
            / safe_user_id
            / "documents"
            / safe_document_id
        )

        bm25_directory = (
            self.bm25_store_directory
            / "users"
            / safe_user_id
            / "documents"
            / safe_document_id
        )

        return (
            dense_directory,
            bm25_directory,
        )

    def _safe_component(
        self,
        value: str,
        *,
        fallback: str,
    ) -> str:
        cleaned = _SAFE_COMPONENT.sub(
            "_",
            value.strip(),
        )

        cleaned = cleaned.strip(
            "._"
        )

        return cleaned or fallback

    def _remove_directory_if_exists(
        self,
        path: Path,
    ) -> None:
        if not path.exists():
            return

        shutil.rmtree(
            path
        )

    def _write_json_atomic(
        self,
        *,
        path: Path,
        payload: Any,
    ) -> None:
        temporary_path = (
            path.with_suffix(
                path.suffix + ".tmp"
            )
        )

        try:
            with temporary_path.open(
                mode="w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

                file.flush()

                os.fsync(
                    file.fileno()
                )

            os.replace(
                temporary_path,
                path,
            )

        except Exception:
            temporary_path.unlink(
                missing_ok=True
            )

            raise