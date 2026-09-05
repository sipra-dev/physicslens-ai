from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import Settings


_EXTENSION_TO_MIME_TYPES: dict[str, set[str]] = {
    "pdf": {
        "application/pdf",
        "application/octet-stream",
    },
    "png": {
        "image/png",
        "application/octet-stream",
    },
    "jpg": {
        "image/jpeg",
        "application/octet-stream",
    },
    "jpeg": {
        "image/jpeg",
        "application/octet-stream",
    },
    "webp": {
        "image/webp",
        "application/octet-stream",
    },
}


@dataclass(frozen=True)
class ValidationResult:
    original_filename: str
    extension: str
    content_type: str
    size_bytes: int
    sha256: str
    page_count: int | None = None
    image_width: int | None = None
    image_height: int | None = None


class FileValidationError(Exception):
    """
    Controlled validation error safe to expose through the API.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.status_code = status_code


def _get_extension(filename: str) -> str:
    extension = (
        Path(filename)
        .suffix
        .lower()
        .lstrip(".")
    )

    if not extension:
        raise FileValidationError(
            "The uploaded file does not have a file extension.",
            status_code=415,
        )

    return extension


def _validate_binary_signature(
    *,
    file_bytes: bytes,
    extension: str,
) -> None:
    if extension == "pdf":
        if not file_bytes.startswith(b"%PDF-"):
            raise FileValidationError(
                "The file content is not a valid PDF.",
                status_code=415,
            )

        return

    if extension == "png":
        if not file_bytes.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise FileValidationError(
                "The file content is not a valid PNG image.",
                status_code=415,
            )

        return

    if extension in {"jpg", "jpeg"}:
        if not file_bytes.startswith(b"\xff\xd8\xff"):
            raise FileValidationError(
                "The file content is not a valid JPEG image.",
                status_code=415,
            )

        return

    if extension == "webp":
        valid_webp_signature = (
            len(file_bytes) >= 12
            and file_bytes[0:4] == b"RIFF"
            and file_bytes[8:12] == b"WEBP"
        )

        if not valid_webp_signature:
            raise FileValidationError(
                "The file content is not a valid WEBP image.",
                status_code=415,
            )


def _validate_pdf(
    *,
    file_bytes: bytes,
    maximum_pages: int,
) -> int:
    try:
        reader = PdfReader(
            BytesIO(file_bytes),
            strict=False,
        )

    except (
        PdfReadError,
        OSError,
        ValueError,
    ) as exc:
        raise FileValidationError(
            "The PDF is corrupted or cannot be read.",
            status_code=400,
        ) from exc

    if reader.is_encrypted:
        raise FileValidationError(
            "Password-protected PDFs are not supported.",
            status_code=400,
        )

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise FileValidationError(
            "The PDF page structure could not be read.",
            status_code=400,
        ) from exc

    if page_count < 1:
        raise FileValidationError(
            "The PDF does not contain any pages.",
            status_code=400,
        )

    if page_count > maximum_pages:
        raise FileValidationError(
            (
                f"The PDF contains {page_count} pages. "
                f"The maximum allowed number is {maximum_pages}."
            ),
            status_code=413,
        )

    return page_count


def _validate_image(
    *,
    file_bytes: bytes,
    maximum_pixels: int,
) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(
                BytesIO(file_bytes)
            ) as image:
                width, height = image.size

                if width < 1 or height < 1:
                    raise FileValidationError(
                        "The image has invalid dimensions.",
                        status_code=400,
                    )

                if width * height > maximum_pixels:
                    raise FileValidationError(
                        (
                            "The image resolution is too large. "
                            f"The current limit is "
                            f"{maximum_pixels:,} pixels."
                        ),
                        status_code=413,
                    )

                image.verify()

    except FileValidationError:
        raise

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
    ) as exc:
        raise FileValidationError(
            "The image is corrupted or cannot be read.",
            status_code=400,
        ) from exc

    return width, height


def validate_upload(
    *,
    filename: str | None,
    content_type: str | None,
    file_bytes: bytes,
    application_settings: Settings,
) -> ValidationResult:
    original_filename = (
        Path(filename).name.strip()
        if filename
        else ""
    )

    if not original_filename:
        raise FileValidationError(
            "The uploaded file must have a filename.",
            status_code=400,
        )

    if not file_bytes:
        raise FileValidationError(
            "The uploaded file is empty.",
            status_code=400,
        )

    size_bytes = len(file_bytes)

    if (
        size_bytes
        > application_settings.max_upload_size_bytes
    ):
        raise FileValidationError(
            (
                "The uploaded file is too large. "
                f"The maximum allowed size is "
                f"{application_settings.max_upload_size_mb} MB."
            ),
            status_code=413,
        )

    extension = _get_extension(
        original_filename
    )

    if (
        extension
        not in application_settings.allowed_extensions
    ):
        allowed = ", ".join(
            f".{value}"
            for value
            in application_settings.allowed_extensions
        )

        raise FileValidationError(
            (
                f"Unsupported file extension '.{extension}'. "
                f"Allowed extensions: {allowed}."
            ),
            status_code=415,
        )

    normalized_content_type = (
        content_type.strip().lower()
        if content_type
        else "application/octet-stream"
    )

    if (
        normalized_content_type
        not in application_settings.allowed_mime_types
    ):
        raise FileValidationError(
            (
                f"Unsupported content type "
                f"'{normalized_content_type}'."
            ),
            status_code=415,
        )

    expected_mime_types = (
        _EXTENSION_TO_MIME_TYPES.get(
            extension,
            set(),
        )
    )

    if (
        expected_mime_types
        and normalized_content_type
        not in expected_mime_types
    ):
        raise FileValidationError(
            (
                "The file extension and content type "
                "do not match."
            ),
            status_code=415,
        )

    _validate_binary_signature(
        file_bytes=file_bytes,
        extension=extension,
    )

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    page_count: int | None = None
    image_width: int | None = None
    image_height: int | None = None

    if extension == "pdf":
        page_count = _validate_pdf(
            file_bytes=file_bytes,
            maximum_pages=(
                application_settings.max_pdf_pages
            ),
        )

    else:
        image_width, image_height = (
            _validate_image(
                file_bytes=file_bytes,
                maximum_pixels=(
                    application_settings.max_image_pixels
                ),
            )
        )

    return ValidationResult(
        original_filename=original_filename,
        extension=extension,
        content_type=normalized_content_type,
        size_bytes=size_bytes,
        sha256=file_hash,
        page_count=page_count,
        image_width=image_width,
        image_height=image_height,
    )