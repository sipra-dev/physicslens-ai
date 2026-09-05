from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image

from src.ingestion.models import (
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
)


class DocumentParsingError(Exception):
    pass


class DocumentParser:
    """
    Fast document parser.

    Responsibilities:
    - extract native PDF text blocks;
    - preserve native embedded image blocks when PyMuPDF exposes them;
    - render every PDF page to PNG for downstream visual-region detection;
    - parse standalone image uploads as one full-page visual block.

    This parser deliberately does NOT scan or cluster PDF vector drawings.
    Vector/raster/mixed visuals will be handled later from the rendered page
    image, so parsing remains bounded and fast.
    """

    def __init__(
        self,
        *,
        render_dpi: int = 180,
        minimum_native_text_characters: int = 40,
    ) -> None:
        if render_dpi <= 0:
            raise ValueError(
                "render_dpi must be greater than zero."
            )

        self.render_dpi = render_dpi
        self.minimum_native_text_characters = (
            minimum_native_text_characters
        )

    def parse(
        self,
        *,
        document_id: str,
        source_path: Path,
        output_directory: Path,
    ) -> ParsedDocument:
        source_path = source_path.resolve()

        if not source_path.is_file():
            raise DocumentParsingError(
                "The source document does not exist."
            )

        extension = (
            source_path.suffix.lower().lstrip(".")
        )

        rendered_directory = (
            output_directory / "rendered_pages"
        )

        rendered_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if extension == "pdf":
            return self._parse_pdf(
                document_id=document_id,
                source_path=source_path,
                rendered_directory=rendered_directory,
            )

        if extension in {
            "png",
            "jpg",
            "jpeg",
            "webp",
        }:
            return self._parse_image(
                document_id=document_id,
                source_path=source_path,
                rendered_directory=rendered_directory,
                extension=extension,
            )

        raise DocumentParsingError(
            f"Unsupported file extension: {extension}"
        )

    def _parse_pdf(
        self,
        *,
        document_id: str,
        source_path: Path,
        rendered_directory: Path,
    ) -> ParsedDocument:
        try:
            pdf_document = fitz.open(source_path)
        except Exception as exc:
            raise DocumentParsingError(
                "The PDF could not be opened by PyMuPDF."
            ) from exc

        parsed_pages: list[ParsedPage] = []

        try:
            for page_index in range(
                pdf_document.page_count
            ):
                page = pdf_document.load_page(
                    page_index
                )

                page_number = page_index + 1

                rendered_path = (
                    rendered_directory
                    / f"page_{page_number:04d}.png"
                )

                pixmap = page.get_pixmap(
                    dpi=self.render_dpi,
                    alpha=False,
                )

                pixmap.save(rendered_path)

                native_text = page.get_text(
                    "text",
                    sort=True,
                ).strip()

                raw_blocks = page.get_text(
                    "blocks",
                    sort=True,
                )

                parsed_blocks: list[ParsedBlock] = []

                for fallback_index, raw_block in enumerate(
                    raw_blocks
                ):
                    if len(raw_block) < 5:
                        continue

                    x0 = float(raw_block[0])
                    y0 = float(raw_block[1])
                    x1 = float(raw_block[2])
                    y1 = float(raw_block[3])

                    if (
                        x1 <= x0
                        or y1 <= y0
                    ):
                        continue

                    block_text = str(
                        raw_block[4] or ""
                    ).strip()

                    block_number = (
                        int(raw_block[5])
                        if (
                            len(raw_block) > 5
                            and isinstance(
                                raw_block[5],
                                int,
                            )
                        )
                        else fallback_index
                    )

                    block_type_number = (
                        int(raw_block[6])
                        if (
                            len(raw_block) > 6
                            and isinstance(
                                raw_block[6],
                                int,
                            )
                        )
                        else 0
                    )

                    block_type = (
                        "image"
                        if block_type_number == 1
                        else "text"
                    )

                    parsed_blocks.append(
                        ParsedBlock(
                            block_id=(
                                f"p{page_number}_"
                                f"b{block_number}"
                            ),
                            page_number=page_number,
                            block_number=block_number,
                            block_type=block_type,
                            bbox=BoundingBox(
                                x0=x0,
                                y0=y0,
                                x1=x1,
                                y1=y1,
                            ),
                            text=block_text,
                            source=(
                                "image"
                                if block_type == "image"
                                else "native"
                            ),
                        )
                    )

                requires_ocr = (
                    len(native_text)
                    < self.minimum_native_text_characters
                )

                parsed_pages.append(
                    ParsedPage(
                        page_number=page_number,
                        width=float(
                            page.rect.width
                        ),
                        height=float(
                            page.rect.height
                        ),
                        rendered_width=pixmap.width,
                        rendered_height=pixmap.height,
                        rendered_image_path=str(
                            rendered_path.resolve()
                        ),
                        native_text=native_text,
                        native_text_length=len(
                            native_text
                        ),
                        blocks=parsed_blocks,
                        requires_ocr=requires_ocr,
                    )
                )

        except Exception as exc:
            raise DocumentParsingError(
                "One or more PDF pages could not be parsed."
            ) from exc

        finally:
            pdf_document.close()

        return ParsedDocument(
            document_id=document_id,
            source_path=str(source_path),
            file_extension="pdf",
            page_count=len(parsed_pages),
            pages=parsed_pages,
        )

    def _parse_image(
        self,
        *,
        document_id: str,
        source_path: Path,
        rendered_directory: Path,
        extension: str,
    ) -> ParsedDocument:
        rendered_path = (
            rendered_directory / "page_0001.png"
        )

        try:
            with Image.open(source_path) as image:
                normalized_image = image.convert("RGB")

                normalized_image.save(
                    rendered_path,
                    format="PNG",
                )

                width, height = (
                    normalized_image.size
                )

        except OSError as exc:
            raise DocumentParsingError(
                "The uploaded image could not be opened."
            ) from exc

        full_page_block = ParsedBlock(
            block_id="p1_b0",
            page_number=1,
            block_number=0,
            block_type="image",
            bbox=BoundingBox(
                x0=0.0,
                y0=0.0,
                x1=float(width),
                y1=float(height),
            ),
            text="",
            source="image",
        )

        page = ParsedPage(
            page_number=1,
            width=float(width),
            height=float(height),
            rendered_width=width,
            rendered_height=height,
            rendered_image_path=str(
                rendered_path.resolve()
            ),
            native_text="",
            native_text_length=0,
            blocks=[
                full_page_block
            ],
            requires_ocr=True,
        )

        return ParsedDocument(
            document_id=document_id,
            source_path=str(source_path),
            file_extension=extension,
            page_count=1,
            pages=[
                page
            ],
        )