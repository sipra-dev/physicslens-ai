from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.storage.base import (
    StorageBackend,
    StorageError,
    StoredFile,
)


_SAFE_COMPONENT_PATTERN = re.compile(
    r"[^a-zA-Z0-9._-]+"
)

_SHA256_PATTERN = re.compile(
    r"^[0-9a-fA-F]{64}$"
)


def _sanitize_path_component(
    value: str,
    *,
    fallback: str,
) -> str:
    cleaned_value = Path(
        value
    ).name.strip()

    cleaned_value = (
        _SAFE_COMPONENT_PATTERN.sub(
            "_",
            cleaned_value,
        )
    )

    cleaned_value = cleaned_value.strip(
        "._"
    )

    return cleaned_value or fallback


def _sanitize_filename(
    filename: str,
) -> str:
    original_name = Path(
        filename
    ).name

    stem = Path(
        original_name
    ).stem

    suffix = Path(
        original_name
    ).suffix.lower()

    safe_stem = _sanitize_path_component(
        stem,
        fallback="uploaded_file",
    )

    safe_suffix = re.sub(
        r"[^a-zA-Z0-9.]",
        "",
        suffix,
    )

    return (
        f"{safe_stem}"
        f"{safe_suffix}"
    )


class LocalStorage(StorageBackend):
    """
    Local filesystem implementation of the
    storage interface.
    """

    def __init__(
        self,
        root_directory: Path,
    ) -> None:
        self.root_directory = (
            root_directory.resolve()
        )

        self.root_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def get_document_directory(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> Path:
        safe_user_id = (
            _sanitize_path_component(
                user_id,
                fallback="local-user",
            )
        )

        safe_document_id = (
            _sanitize_path_component(
                document_id,
                fallback="document",
            )
        )

        return (
            self.root_directory
            / "users"
            / safe_user_id
            / "documents"
            / safe_document_id
        )

    def save_original_file(
        self,
        *,
        user_id: str,
        document_id: str,
        original_filename: str,
        file_bytes: bytes,
    ) -> StoredFile:
        if not file_bytes:
            raise StorageError(
                "Cannot save an empty file."
            )

        document_directory = (
            self.get_document_directory(
                user_id=user_id,
                document_id=document_id,
            )
        )

        original_directory = (
            document_directory
            / "original"
        )

        original_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        stored_filename = (
            _sanitize_filename(
                original_filename
            )
        )

        destination_path = (
            original_directory
            / stored_filename
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=original_directory,
                prefix=".upload_",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(
                    file_bytes
                )

                temporary_file.flush()

                os.fsync(
                    temporary_file.fileno()
                )

                temporary_path = Path(
                    temporary_file.name
                )

            os.replace(
                temporary_path,
                destination_path,
            )

        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True
                )

            raise StorageError(
                "The uploaded file could "
                "not be saved."
            ) from exc

        relative_path = (
            destination_path
            .relative_to(
                self.root_directory
            )
            .as_posix()
        )

        return StoredFile(
            user_id=user_id,
            document_id=document_id,
            original_filename=(
                original_filename
            ),
            stored_filename=(
                stored_filename
            ),
            relative_path=(
                relative_path
            ),
            absolute_path=(
                destination_path
            ),
            size_bytes=len(
                file_bytes
            ),
        )

    def write_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
        metadata: dict[str, Any],
    ) -> None:
        document_directory = (
            self.get_document_directory(
                user_id=user_id,
                document_id=document_id,
            )
        )

        document_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path = (
            document_directory
            / "metadata.json"
        )

        temporary_path = (
            document_directory
            / ".metadata.tmp"
        )

        try:
            with temporary_path.open(
                mode="w",
                encoding="utf-8",
            ) as metadata_file:
                json.dump(
                    metadata,
                    metadata_file,
                    ensure_ascii=False,
                    indent=2,
                )

                metadata_file.flush()

                os.fsync(
                    metadata_file.fileno()
                )

            os.replace(
                temporary_path,
                metadata_path,
            )

        except (
            OSError,
            TypeError,
        ) as exc:
            temporary_path.unlink(
                missing_ok=True
            )

            raise StorageError(
                "Document metadata could "
                "not be saved."
            ) from exc

    def read_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any] | None:
        metadata_path = (
            self.get_document_directory(
                user_id=user_id,
                document_id=document_id,
            )
            / "metadata.json"
        )

        if not metadata_path.is_file():
            return None

        try:
            with metadata_path.open(
                mode="r",
                encoding="utf-8",
            ) as metadata_file:
                data = json.load(
                    metadata_file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise StorageError(
                "Document metadata could "
                "not be read."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise StorageError(
                "Document metadata has "
                "an invalid format."
            )

        return data

    def read_document_json_artifact(
        self,
        *,
        user_id: str,
        document_id: str,
        artifact_path: str,
    ) -> Any | None:
        """
        Read a JSON artifact that belongs to one user's one document.

        The supplied path is storage-root-relative and normally comes from
        document metadata. The resolved path must remain inside the requested
        document directory. This keeps artifact access isolated without
        hard-coding any particular artifact name or type.
        """

        normalized_artifact_path = (
            artifact_path.strip()
        )

        if not normalized_artifact_path:
            raise ValueError(
                "artifact_path cannot be empty."
            )

        relative_path = Path(
            normalized_artifact_path
        )

        if relative_path.is_absolute():
            raise StorageError(
                "Document artifact path must be relative."
            )

        document_directory = (
            self.get_document_directory(
                user_id=user_id,
                document_id=document_id,
            ).resolve()
        )

        artifact_file = (
            self.root_directory
            / relative_path
        ).resolve()

        try:
            artifact_file.relative_to(
                document_directory
            )
        except ValueError as exc:
            raise StorageError(
                "Document artifact path is outside "
                "the requested document directory."
            ) from exc

        if not artifact_file.is_file():
            return None

        try:
            with artifact_file.open(
                mode="r",
                encoding="utf-8",
            ) as input_file:
                return json.load(
                    input_file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise StorageError(
                "Document JSON artifact could "
                "not be read."
            ) from exc

    def find_document_by_sha256(
        self,
        *,
        user_id: str,
        sha256: str,
    ) -> dict[str, Any] | None:
        """
        Search this user's existing documents
        for an identical uploaded file.

        Matching is based on SHA-256 file
        fingerprint, not filename.
        """

        normalized_user_id = (
            user_id.strip()
        )

        normalized_sha256 = (
            sha256.strip().lower()
        )

        if not normalized_user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        if not _SHA256_PATTERN.fullmatch(
            normalized_sha256
        ):
            raise ValueError(
                "sha256 must be a valid "
                "64-character hexadecimal "
                "SHA-256 digest."
            )

        safe_user_id = (
            _sanitize_path_component(
                normalized_user_id,
                fallback="local-user",
            )
        )

        documents_directory = (
            self.root_directory
            / "users"
            / safe_user_id
            / "documents"
        )

        if not documents_directory.is_dir():
            return None

        try:
            document_directories = sorted(
                (
                    path
                    for path
                    in documents_directory.iterdir()
                    if path.is_dir()
                ),
                key=lambda path: path.name,
            )

        except OSError as exc:
            raise StorageError(
                "Stored documents could "
                "not be inspected."
            ) from exc

        for document_directory in (
            document_directories
        ):
            metadata = (
                self.read_document_metadata(
                    user_id=(
                        normalized_user_id
                    ),
                    document_id=(
                        document_directory.name
                    ),
                )
            )

            if metadata is None:
                continue

            stored_sha256 = metadata.get(
                "sha256"
            )

            if not isinstance(
                stored_sha256,
                str,
            ):
                continue

            if (
                stored_sha256
                .strip()
                .lower()
                == normalized_sha256
            ):
                return metadata

        return None

    def document_exists(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        return (
            self.get_document_directory(
                user_id=user_id,
                document_id=document_id,
            )
            .is_dir()
        )

    def delete_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        document_directory = (
            self.get_document_directory(
                user_id=user_id,
                document_id=document_id,
            )
        )

        if not document_directory.exists():
            return False

        try:
            shutil.rmtree(
                document_directory
            )

        except OSError as exc:
            raise StorageError(
                "The document could "
                "not be deleted."
            ) from exc

        return True