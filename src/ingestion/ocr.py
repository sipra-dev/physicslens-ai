from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image
from pytesseract import Output

from src.ingestion.models import (
    BoundingBox,
    OCRDocumentResult,
    OCRPageResult,
    OCRWord,
    ParsedDocument,
)


class OCRService:
    def __init__(
        self,
        *,
        languages: str = "eng",
        minimum_confidence: float = 25.0,
        tesseract_command: str | None = None,
    ) -> None:
        self.languages = languages
        self.minimum_confidence = (
            minimum_confidence
        )

        if tesseract_command:
            pytesseract.pytesseract.tesseract_cmd = (
                tesseract_command
            )

    def is_available(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def process(
        self,
        parsed_document: ParsedDocument,
    ) -> OCRDocumentResult:
        available = self.is_available()

        page_results: list[OCRPageResult] = []

        for page in parsed_document.pages:
            if not page.requires_ocr:
                page_results.append(
                    OCRPageResult(
                        page_number=page.page_number,
                        attempted=False,
                        available=available,
                        used=False,
                        text="",
                        error=None,
                    )
                )
                continue

            if not available:
                page_results.append(
                    OCRPageResult(
                        page_number=page.page_number,
                        attempted=False,
                        available=False,
                        used=False,
                        text="",
                        error=(
                            "Tesseract OCR is not installed "
                            "or is not available in PATH."
                        ),
                    )
                )
                continue

            page_results.append(
                self._ocr_page(
                    page_number=page.page_number,
                    image_path=Path(
                        page.rendered_image_path
                    ),
                )
            )

        return OCRDocumentResult(
            document_id=parsed_document.document_id,
            engine="tesseract",
            pages=page_results,
        )

    def _ocr_page(
        self,
        *,
        page_number: int,
        image_path: Path,
    ) -> OCRPageResult:
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                data = pytesseract.image_to_data(
                    image,
                    lang=self.languages,
                    output_type=Output.DICT,
                    config="--psm 6",
                )

        except Exception as exc:
            return OCRPageResult(
                page_number=page_number,
                attempted=True,
                available=True,
                used=False,
                text="",
                error=str(exc),
            )

        accepted_words: list[OCRWord] = []
        text_parts: list[str] = []
        confidences: list[float] = []

        total_items = len(
            data.get("text", [])
        )

        for index in range(total_items):
            text = str(
                data["text"][index]
            ).strip()

            if not text:
                continue

            try:
                confidence = float(
                    data["conf"][index]
                )
            except (TypeError, ValueError):
                confidence = -1.0

            if confidence < self.minimum_confidence:
                continue

            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])

            accepted_words.append(
                OCRWord(
                    text=text,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x0=float(left),
                        y0=float(top),
                        x1=float(left + width),
                        y1=float(top + height),
                    ),
                )
            )

            text_parts.append(text)
            confidences.append(confidence)

        combined_text = " ".join(
            text_parts
        ).strip()

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )

        return OCRPageResult(
            page_number=page_number,
            attempted=True,
            available=True,
            used=bool(combined_text),
            text=combined_text,
            average_confidence=average_confidence,
            words=accepted_words,
            error=None,
        )