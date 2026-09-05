from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class StorageError(Exception):
    """
    Raised when the configured storage backend
    cannot complete an operation.
    """


@dataclass(frozen=True)
class StoredFile:
    user_id: str
    document_id: str
    original_filename: str
    stored_filename: str
    relative_path: str
    absolute_path: Path
    size_bytes: int


class StorageBackend(ABC):
    """
    Storage abstraction.

    LocalStorage implements this interface now.

    A future S3Storage class can implement the
    same methods without changing the ingestion
    service.
    """

    @abstractmethod
    def save_original_file(
        self,
        *,
        user_id: str,
        document_id: str,
        original_filename: str,
        file_bytes: bytes,
    ) -> StoredFile:
        raise NotImplementedError

    @abstractmethod
    def write_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
        metadata: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_document_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def read_document_json_artifact(
        self,
        *,
        user_id: str,
        document_id: str,
        artifact_path: str,
    ) -> Any | None:
        """
        Read one JSON artifact that belongs to the specified document.

        artifact_path is the storage-root-relative path already recorded in
        document metadata, for example an analysis artifact path.

        Implementations must verify that the resolved artifact still belongs
        to the requested user's requested document before reading it.

        This method is intentionally generic. It is not figure-, equation-,
        chunk-, topic-, or document-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def find_document_by_sha256(
        self,
        *,
        user_id: str,
        sha256: str,
    ) -> dict[str, Any] | None:
        """
        Find an existing document belonging to
        the same user with identical file content.

        Returns matching document metadata,
        otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def document_exists(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_document_directory(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> Path:
        raise NotImplementedError