from __future__ import annotations

import re
import unicodedata

from src.ingestion.models import (
    DocumentLayout,
    LayoutBlock,
    LayoutBlockType,
    PageLayout,
    ParsedBlock,
    ParsedDocument,
)


# A list marker may be:
#
# 1.
# 1)
# (1)
# a.
# (a)
# i.
# (iv)
_ITEM_TOKEN_PATTERN = (
    r"(?:"
    r"\d+(?:\.\d+)*"
    r"|[A-Za-z]"
    r"|[IVXLCDMivxlcdm]+"
    r")"
)

_ITEM_PREFIX_PATTERN = (
    rf"(?:"
    rf"\({_ITEM_TOKEN_PATTERN}\)"
    rf"|{_ITEM_TOKEN_PATTERN}[.)]"
    rf")"
)


_QUESTION_PATTERN = re.compile(
    rf"^\s*(?:"
    rf"(?:"
    rf"q(?:uestion)?\.?\s*\d*(?:\.\d+)*"
    rf"|exercise(?:\s+\d+(?:\.\d+)*)?"
    rf"|problem(?:\s+\d+(?:\.\d+)*)?"
    rf")\b"
    rf"|"
    rf"(?:{_ITEM_PREFIX_PATTERN}\s*)?"
    rf"(?:calculate|find|what|why|how|when|which)\b"
    rf")",
    re.IGNORECASE,
)


_EXAMPLE_PATTERN = re.compile(
    r"^\s*(?:worked\s+example|example)\b",
    re.IGNORECASE,
)


_ANSWER_PATTERN = re.compile(
    r"^\s*(?:answer|solution)\b",
    re.IGNORECASE,
)


# Common Unicode and plain-text bullet markers.
_BULLET_ITEM_PATTERN = re.compile(
    r"^\s*"
    r"(?:"
    r"[•◦▪▫‣⁃●○■□◆◇]"
    r"|[-*–—]"
    r")"
    r"\s+\S",
)


# Ordered-list markers such as:
#
# 1. Text
# 2) Text
# (3) Text
# a. Text
# (b) Text
# iv. Text
_NUMBERED_ITEM_PATTERN = re.compile(
    rf"^\s*{_ITEM_PREFIX_PATTERN}\s+\S",
)


# Caption text is only an optional naming/linking signal.
# Figure detection itself does NOT depend on this pattern.
_CAPTION_PATTERN = re.compile(
    r"^\s*[\(\[\{]?\s*"
    r"(?:fig(?:ure)?|diagram|graph)"
    r"\s*\.?\s*"
    r"(?:"
    r"\d+(?:[.\-]\d+)*"
    r"|[A-Za-z]\d+(?:[.\-]\d+)*"
    r"|[IVXLCDM]+"
    r")"
    r"\s*[\)\]\}]?"
    r"(?:\s*[:.\-]\s*.*)?$",
    re.IGNORECASE,
)


class LayoutAnalyzer:
    """
    Classify parser-produced blocks into document layout types.

    Visual detection is parser-driven:

    - if DocumentParser emits block_type="image", this analyzer
      marks it FIGURE;

    - figure-caption text is only an optional metadata/linking
      clue;

    - absence of words such as "Fig", "Figure", "Diagram", or
      "Graph" does not stop visual detection.

    List detection is deterministic:

    - bullet markers become BULLET_ITEM;

    - ordered markers become NUMBERED_ITEM;

    - explicit questions, problems, examples, and solutions are
      classified before generic list detection.
    """

    def analyze(
        self,
        parsed_document: ParsedDocument,
    ) -> DocumentLayout:
        pages: list[PageLayout] = []

        for page in parsed_document.pages:
            classified_blocks = [
                self._classify_block(
                    block=block,
                    page_width=page.width,
                    page_height=page.height,
                )
                for block in page.blocks
            ]

            pages.append(
                PageLayout(
                    page_number=page.page_number,
                    blocks=classified_blocks,
                )
            )

        return DocumentLayout(
            document_id=parsed_document.document_id,
            pages=pages,
        )

    def _classify_block(
        self,
        *,
        block: ParsedBlock,
        page_width: float,
        page_height: float,
    ) -> LayoutBlock:
        text = block.text.strip()

        # Parser-detected images always remain visual figures.
        if block.block_type == "image":
            return LayoutBlock(
                **block.model_dump(
                    exclude={"block_type"}
                ),
                block_type=LayoutBlockType.FIGURE,
                confidence=0.95,
            )

        if not text:
            detected_type = LayoutBlockType.UNKNOWN
            confidence = 0.20

        elif _CAPTION_PATTERN.match(text):
            detected_type = (
                LayoutBlockType.FIGURE_CAPTION
            )
            confidence = 0.85

        elif _EXAMPLE_PATTERN.match(text):
            detected_type = (
                LayoutBlockType.WORKED_EXAMPLE
            )
            confidence = 0.85

        elif _ANSWER_PATTERN.match(text):
            detected_type = LayoutBlockType.ANSWER
            confidence = 0.80

        # Question detection comes before numbered-item
        # detection.
        #
        # This ensures that:
        #
        # 1. Calculate the force...
        #
        # is treated as a question rather than an ordinary
        # numbered point.
        elif (
            _QUESTION_PATTERN.match(text)
            or text.endswith("?")
        ):
            detected_type = LayoutBlockType.QUESTION
            confidence = 0.80

        elif _BULLET_ITEM_PATTERN.match(text):
            detected_type = (
                LayoutBlockType.BULLET_ITEM
            )
            confidence = 0.90

        elif _NUMBERED_ITEM_PATTERN.match(text):
            detected_type = (
                LayoutBlockType.NUMBERED_ITEM
            )
            confidence = 0.90

        elif self._looks_like_equation(text):
            detected_type = LayoutBlockType.EQUATION
            confidence = 0.80

        elif self._looks_like_title(
            text=text,
            y0=block.bbox.y0,
            page_height=page_height,
        ):
            detected_type = LayoutBlockType.TITLE
            confidence = 0.75

        elif self._looks_like_heading(text):
            detected_type = LayoutBlockType.HEADING
            confidence = 0.70

        else:
            detected_type = LayoutBlockType.PARAGRAPH
            confidence = 0.65

        return LayoutBlock(
            block_id=block.block_id,
            page_number=block.page_number,
            block_number=block.block_number,
            block_type=detected_type,
            bbox=block.bbox,
            text=text,
            source=block.source,
            confidence=confidence,
        )

    def _looks_like_equation(
        self,
        text: str,
    ) -> bool:
        """
        Generic Unicode-aware equation detection.

        This intentionally avoids a Physics-specific symbol
        whitelist.
        """

        normalized = " ".join(
            text.split()
        )

        if (
            not normalized
            or len(normalized) > 160
        ):
            return False

        has_math_symbol = any(
            unicodedata.category(character) == "Sm"
            for character in normalized
        )

        if has_math_symbol:
            return True

        if "=" in normalized:
            left, _, right = normalized.partition("=")

            return bool(
                left.strip()
                and right.strip()
                and len(left.strip()) <= 24
            )

        return False

    def _looks_like_title(
        self,
        *,
        text: str,
        y0: float,
        page_height: float,
    ) -> bool:
        normalized = " ".join(
            text.split()
        )

        if not 2 <= len(normalized) <= 100:
            return False

        near_top = (
            y0 <= page_height * 0.20
        )

        title_like_case = (
            normalized.isupper()
            or normalized.istitle()
        )

        return (
            near_top
            and title_like_case
        )

    def _looks_like_heading(
        self,
        text: str,
    ) -> bool:
        normalized = " ".join(
            text.split()
        )

        if not 2 <= len(normalized) <= 120:
            return False

        if normalized.endswith(
            (".", "?", "!", ";")
        ):
            return False

        word_count = len(
            normalized.split()
        )

        return (
            word_count <= 12
            and (
                normalized.istitle()
                or normalized.isupper()
            )
        )