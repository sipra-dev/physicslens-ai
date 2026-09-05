from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Literal

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from src.ingestion.models import (
    DocumentLayout,
    LayoutBlock,
    LayoutBlockType,
    ParsedDocument,
    ParsedPage,
)


_CORRUPTED_MATH_PATTERN = re.compile(
    r"(?:!{2,}|�|[\"#$%&]{3,})"
)

_STANDALONE_BANG_PATTERN = re.compile(
    r"(?<!\w)!+(?!\w)"
)

_MATH_MARKER_PATTERN = re.compile(
    r"[=±×÷∑√∆ΔθλμρσπωΦφ]|"
    r"\b(?:sin|cos|tan|log|sqrt)\b",
    re.IGNORECASE,
)

_LEADING_OR_TRAILING_OPERATOR_PATTERN = re.compile(
    r"(?:^[=+\-×÷/]|[=+\-×÷/]$)"
)


_PLACEHOLDER_FRAGMENT_PATTERN = re.compile(
    r"[!�]"
)

_ORPHAN_MATH_TOKEN_PATTERN = re.compile(
    r"^[\s()\[\]{}.,;:+\-×÷/=]*"
    r"(?:[A-Za-zΑ-Ωα-ω𝑨-𝒛0-9]+)"
    r"[\s()\[\]{}.,;:+\-×÷/=]*$"
)


_VISION_META_COMMENTARY_PATTERN = re.compile(
    r"""
    (?:
        \\{1,2}text\{
            \s*
            \(?
            \s*
            (?:and\s+)?
            (?:
                continuation\s+not\s+shown
                (?:\s+in\s+(?:the\s+)?(?:image|crop))?
                |
                not\s+(?:fully\s+)?visible
                (?:\s+in\s+(?:the\s+)?(?:image|crop))?
                |
                not\s+shown\s+in\s+(?:the\s+)?(?:image|crop)
                |
                outside\s+(?:the\s+)?(?:image|crop)
                |
                cropped\s+(?:out|off)
            )
            \s*
            \)?
            \s*
        \}
        |
        \(
            \s*
            (?:and\s+)?
            (?:
                continuation\s+not\s+shown
                (?:\s+in\s+(?:the\s+)?(?:image|crop))?
                |
                not\s+(?:fully\s+)?visible
                (?:\s+in\s+(?:the\s+)?(?:image|crop))?
                |
                not\s+shown\s+in\s+(?:the\s+)?(?:image|crop)
                |
                outside\s+(?:the\s+)?(?:image|crop)
                |
                cropped\s+(?:out|off)
            )
            \s*
        \)
    )
    """,
    flags=(
        re.IGNORECASE
        | re.VERBOSE
    ),
)


class EquationArtifact(BaseModel):
    artifact_id: str
    page_number: int = Field(ge=1)

    # Anchor block retained for backward compatibility.
    source_block_id: str

    # Full provenance for a recovered multi-block region.
    source_block_ids: list[str] = Field(
        default_factory=list
    )

    original_text: str = ""
    transcribed_text: str = ""

    equations: list[str] = Field(
        default_factory=list
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    extraction_method: Literal[
        "native",
        "openai_vision",
        "openai_vision_region",
    ]

    region_kind: Literal[
        "single_block",
        "multi_block",
    ] = "single_block"

    crop_image_path: str | None = None

    # True only when visual source transcription is
    # trustworthy enough to replace broken native text.
    replacement_safe: bool = False

    # If a formula-only region is visibly corrupted and
    # recovery is not trustworthy, suppress the broken
    # original evidence instead of indexing garbage.
    suppress_original_on_failure: bool = False

    # Helps the chunker decide whether a successfully
    # recovered region should be treated as an equation.
    formula_dominant: bool = True


class EquationExtractionResult(BaseModel):
    document_id: str
    enabled: bool
    model: str | None = None

    artifacts: list[EquationArtifact] = Field(
        default_factory=list
    )

    errors: list[str] = Field(
        default_factory=list
    )


class EquationExtractor:
    """
    Source-faithful Physics equation recovery.

    Clean native equations are kept as native evidence.
    Corrupted or structurally fragmented equations are
    recovered from rendered source-image regions.

    Important:
    - Never "repair" a formula by guessing from broken text.
    - Multi-block formula fragments are grouped spatially.
    - Vision receives the rendered source region itself.
    - If visual transcription is uncertain, formula-only
      corrupted evidence is suppressed rather than trusted.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "gpt-4.1-mini",
        timeout_seconds: float = 45.0,
        minimum_replacement_confidence: float = 0.70,
    ) -> None:
        self.api_key = (
            api_key.strip()
            if api_key
            else None
        )

        self.model = model
        self.timeout_seconds = (
            timeout_seconds
        )

        self.minimum_replacement_confidence = (
            minimum_replacement_confidence
        )

        self.client = (
            OpenAI(
                api_key=self.api_key,
                timeout=timeout_seconds,
            )
            if self.api_key
            else None
        )

    def process(
        self,
        *,
        parsed_document: ParsedDocument,
        document_layout: DocumentLayout,
        output_directory: Path,
    ) -> EquationExtractionResult:
        """
        Recover equation evidence page by page.

        Strategy:
        1. Detect corrupted/fragmented equation seeds.
        2. Group nearby equation fragments into regions.
        3. Crop each region from the rendered source image.
        4. Transcribe the full visible formula region.
        5. Keep remaining clean equation blocks natively.
        """

        layout_by_page = {
            page.page_number: page
            for page in document_layout.pages
        }

        parsed_by_page = {
            page.page_number: page
            for page in parsed_document.pages
        }

        artifacts: list[
            EquationArtifact
        ] = []

        errors: list[str] = []

        crop_directory = (
            output_directory / "equations"
        )

        crop_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for page_number, layout_page in (
            layout_by_page.items()
        ):
            parsed_page = (
                parsed_by_page.get(
                    page_number
                )
            )

            if parsed_page is None:
                continue

            ordered_blocks = [
                block
                for block in sorted(
                    layout_page.blocks,
                    key=lambda item: (
                        item.bbox.y0,
                        item.bbox.x0,
                        item.block_number,
                    ),
                )
                if (
                    block.block_type
                    not in {
                        LayoutBlockType.FIGURE,
                        LayoutBlockType.FIGURE_CAPTION,
                    }
                    and block.text.strip()
                )
            ]

            recovery_regions = (
                self._build_recovery_regions(
                    blocks=ordered_blocks,
                    page_width=parsed_page.width,
                    page_height=parsed_page.height,
                )
            )

            covered_block_ids: set[str] = (
                set()
            )

            for region_number, region_blocks in (
                enumerate(
                    recovery_regions,
                    start=1,
                )
            ):
                region_ids = [
                    block.block_id
                    for block in region_blocks
                ]

                covered_block_ids.update(
                    region_ids
                )

                formula_dominant = (
                    self._is_formula_dominant_region(
                        region_blocks
                    )
                )

                if self.client is None:
                    errors.append(
                        (
                            f"page {page_number} "
                            f"region {region_number}: "
                            "visual equation recovery "
                            "skipped because "
                            "OPENAI_API_KEY is unavailable."
                        )
                    )

                    artifacts.append(
                        self._failed_region_artifact(
                            page_number=page_number,
                            region_number=region_number,
                            blocks=region_blocks,
                            formula_dominant=(
                                formula_dominant
                            ),
                        )
                    )

                    continue

                try:
                    crop_path = (
                        self._crop_region(
                            parsed_page=(
                                parsed_page
                            ),
                            blocks=region_blocks,
                            crop_directory=(
                                crop_directory
                            ),
                            region_number=(
                                region_number
                            ),
                        )
                    )

                    artifact = (
                        self._transcribe_region(
                            page_number=(
                                page_number
                            ),
                            region_number=(
                                region_number
                            ),
                            blocks=region_blocks,
                            crop_path=crop_path,
                            formula_dominant=(
                                formula_dominant
                            ),
                        )
                    )

                    artifacts.append(
                        artifact
                    )

                except Exception as exc:
                    errors.append(
                        (
                            f"page {page_number} "
                            f"region {region_number}: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    )

                    artifacts.append(
                        self._failed_region_artifact(
                            page_number=page_number,
                            region_number=region_number,
                            blocks=region_blocks,
                            formula_dominant=(
                                formula_dominant
                            ),
                        )
                    )

            # Preserve clean equation evidence that was not
            # already absorbed into a visual recovery region.
            for block in ordered_blocks:
                if (
                    block.block_id
                    in covered_block_ids
                ):
                    continue

                if not self.is_equation_candidate(
                    block
                ):
                    continue

                text = block.text.strip()

                # Any remaining block that still looks
                # corrupted should never be certified as
                # clean native evidence.
                if self.needs_visual_recovery(
                    text
                ):
                    continue

                equations = (
                    self._native_equations(
                        text
                    )
                )

                if not equations:
                    continue

                artifacts.append(
                    EquationArtifact(
                        artifact_id=(
                            f"eq_{page_number}_"
                            f"{block.block_id}"
                        ),
                        page_number=page_number,
                        source_block_id=(
                            block.block_id
                        ),
                        source_block_ids=[
                            block.block_id
                        ],
                        original_text=text,
                        transcribed_text=text,
                        equations=equations,
                        confidence=max(
                            block.confidence,
                            0.75,
                        ),
                        extraction_method=(
                            "native"
                        ),
                        region_kind=(
                            "single_block"
                        ),
                        crop_image_path=None,
                        replacement_safe=False,
                        suppress_original_on_failure=False,
                        formula_dominant=(
                            block.block_type
                            == LayoutBlockType.EQUATION
                        ),
                    )
                )

        return EquationExtractionResult(
            document_id=(
                parsed_document.document_id
            ),
            enabled=(
                self.client is not None
            ),
            model=(
                self.model
                if self.client is not None
                else None
            ),
            artifacts=artifacts,
            errors=errors,
        )

    def is_equation_candidate(
        self,
        block: LayoutBlock,
    ) -> bool:
        text = block.text.strip()

        if not text:
            return False

        if (
            block.block_type
            == LayoutBlockType.EQUATION
        ):
            return True

        # A paragraph/worked example may contain a damaged
        # formula even when layout classification kept the
        # whole region as prose.
        return (
            self.needs_visual_recovery(
                text
            )
            and bool(
                _MATH_MARKER_PATTERN.search(
                    text
                )
            )
        )

    def needs_visual_recovery(
        self,
        text: str,
    ) -> bool:
        normalized = " ".join(
            text.split()
        )

        if not normalized:
            return False

        if _CORRUPTED_MATH_PATTERN.search(
            normalized
        ):
            return True

        # In this PDF family, isolated "!" inside a
        # formula-like string is an extraction placeholder.
        # Ordinary prose punctuation such as "Great!" does
        # not satisfy the math-context requirement.
        if (
            "!" in normalized
            and bool(
                _MATH_MARKER_PATTERN.search(
                    normalized
                )
            )
        ):
            return True

        standalone_bangs = len(
            _STANDALONE_BANG_PATTERN.findall(
                normalized
            )
        )

        if standalone_bangs >= 3:
            return True

        return self._looks_structurally_fragmented(
            text
        )

    def _looks_structurally_fragmented(
        self,
        text: str,
    ) -> bool:
        normalized = " ".join(
            text.split()
        )

        if not normalized:
            return False

        if (
            _LEADING_OR_TRAILING_OPERATOR_PATTERN
            .search(normalized)
        ):
            return True

        raw_lines = [
            " ".join(
                line.split()
            )
            for line in text.splitlines()
            if line.strip()
        ]

        if len(raw_lines) >= 3:
            short_math_lines = sum(
                1
                for line in raw_lines
                if (
                    len(line) <= 16
                    and (
                        _MATH_MARKER_PATTERN
                        .search(line)
                        or any(
                            character.isdigit()
                            for character in line
                        )
                    )
                )
            )

            if short_math_lines >= 2:
                return True

        # A very short equation ending in an operator or
        # consisting mostly of disconnected formula pieces
        # is not safe native evidence.
        if (
            len(normalized) <= 40
            and "=" in normalized
            and normalized.count("=") >= 2
            and len(normalized.split()) <= 6
        ):
            return True

        return False

    def _is_recovery_seed(
        self,
        block: LayoutBlock,
    ) -> bool:
        text = block.text.strip()

        if not text:
            return False

        # Strong corruption signals are always recovery seeds.
        if (
            _CORRUPTED_MATH_PATTERN.search(
                text
            )
            or (
                "!" in text
                and (
                    block.block_type
                    == LayoutBlockType.EQUATION
                    or bool(
                        _MATH_MARKER_PATTERN.search(
                            text
                        )
                    )
                )
            )
        ):
            return True

        # Structurally fragmented equation blocks (fractions,
        # stacked lines, detached operators) should also seed
        # a visual region even when no "!" placeholder exists.
        return (
            block.block_type
            == LayoutBlockType.EQUATION
            and self._looks_structurally_fragmented(
                text
            )
        )

    def _is_placeholder_fragment(
        self,
        text: str,
    ) -> bool:
        normalized = " ".join(
            text.split()
        )

        if not normalized:
            return False

        if not _PLACEHOLDER_FRAGMENT_PATTERN.search(
            normalized
        ):
            return False

        # Keep this intentionally conservative: only short
        # extraction debris is treated as a formula fragment.
        # Longer prose containing an exclamation mark is not.
        return len(normalized) <= 80

    def _is_orphan_math_fragment(
        self,
        block: LayoutBlock,
    ) -> bool:
        text = " ".join(
            block.text.split()
        )

        if not text:
            return False

        if len(text) > 18:
            return False

        if (
            block.block_type
            != LayoutBlockType.EQUATION
        ):
            return False

        if "=" in text:
            return True

        return bool(
            _ORPHAN_MATH_TOKEN_PATTERN.fullmatch(
                text
            )
        )

    def _supports_equation_region(
        self,
        block: LayoutBlock,
    ) -> bool:
        text = " ".join(
            block.text.split()
        )

        if not text:
            return False

        if self._is_placeholder_fragment(
            text
        ):
            return True

        if self._is_orphan_math_fragment(
            block
        ):
            return True

        if (
            block.block_type
            == LayoutBlockType.EQUATION
        ):
            return True

        if self.needs_visual_recovery(
            text
        ):
            return True

        # Small labels/units that contain mathematical
        # notation can be part of the same rendered formula.
        return (
            len(text) <= 180
            and bool(
                _MATH_MARKER_PATTERN.search(
                    text
                )
            )
        )

    def _build_recovery_regions(
        self,
        *,
        blocks: list[LayoutBlock],
        page_width: float,
        page_height: float,
    ) -> list[list[LayoutBlock]]:
        """
        Build connected components of nearby equation-like
        blocks. Only components containing at least one
        corrupted/fragmented seed are sent to Vision.
        """

        candidates = [
            block
            for block in blocks
            if self._supports_equation_region(
                block
            )
        ]

        if not candidates:
            return []

        adjacency: dict[
            str,
            set[str],
        ] = {
            block.block_id: set()
            for block in candidates
        }

        by_id = {
            block.block_id: block
            for block in candidates
        }

        for first_index, first in enumerate(
            candidates
        ):
            for second in candidates[
                first_index + 1:
            ]:
                if self._blocks_are_neighbors(
                    first=first,
                    second=second,
                    page_width=page_width,
                    page_height=page_height,
                ):
                    adjacency[
                        first.block_id
                    ].add(
                        second.block_id
                    )

                    adjacency[
                        second.block_id
                    ].add(
                        first.block_id
                    )

        regions: list[
            list[LayoutBlock]
        ] = []

        visited: set[str] = set()

        for candidate in candidates:
            if (
                candidate.block_id
                in visited
            ):
                continue

            stack = [
                candidate.block_id
            ]

            component_ids: list[str] = []

            while stack:
                current_id = stack.pop()

                if current_id in visited:
                    continue

                visited.add(
                    current_id
                )

                component_ids.append(
                    current_id
                )

                stack.extend(
                    adjacency[
                        current_id
                    ]
                    - visited
                )

            component = [
                by_id[block_id]
                for block_id in component_ids
            ]

            if not any(
                self._is_recovery_seed(
                    block
                )
                for block in component
            ):
                continue

            component = sorted(
                component,
                key=lambda item: (
                    item.bbox.y0,
                    item.bbox.x0,
                    item.block_number,
                ),
            )

            regions.append(
                component
            )

        return regions

    def _blocks_are_neighbors(
        self,
        *,
        first: LayoutBlock,
        second: LayoutBlock,
        page_width: float,
        page_height: float,
    ) -> bool:
        if (
            first.page_number
            != second.page_number
        ):
            return False

        first_width = max(
            first.bbox.x1
            - first.bbox.x0,
            1.0,
        )

        second_width = max(
            second.bbox.x1
            - second.bbox.x0,
            1.0,
        )

        first_height = max(
            first.bbox.y1
            - first.bbox.y0,
            1.0,
        )

        second_height = max(
            second.bbox.y1
            - second.bbox.y0,
            1.0,
        )

        horizontal_gap = max(
            0.0,
            max(
                first.bbox.x0,
                second.bbox.x0,
            )
            - min(
                first.bbox.x1,
                second.bbox.x1,
            ),
        )

        vertical_gap = max(
            0.0,
            max(
                first.bbox.y0,
                second.bbox.y0,
            )
            - min(
                first.bbox.y1,
                second.bbox.y1,
            ),
        )

        horizontal_overlap = max(
            0.0,
            min(
                first.bbox.x1,
                second.bbox.x1,
            )
            - max(
                first.bbox.x0,
                second.bbox.x0,
            ),
        )

        vertical_overlap = max(
            0.0,
            min(
                first.bbox.y1,
                second.bbox.y1,
            )
            - max(
                first.bbox.y0,
                second.bbox.y0,
            ),
        )

        first_center_y = (
            first.bbox.y0
            + first.bbox.y1
        ) / 2.0

        second_center_y = (
            second.bbox.y0
            + second.bbox.y1
        ) / 2.0

        same_row = (
            vertical_overlap
            >= min(
                first_height,
                second_height,
            ) * 0.20
            or abs(
                first_center_y
                - second_center_y
            )
            <= (
                max(
                    first_height,
                    second_height,
                )
                * 0.85
                + 4.0
            )
        )

        same_row_gap_limit = max(
            90.0,
            page_width * 0.32,
        )

        if (
            same_row
            and horizontal_gap
            <= same_row_gap_limit
        ):
            return True

        horizontal_overlap_ratio = (
            horizontal_overlap
            / max(
                min(
                    first_width,
                    second_width,
                ),
                1.0,
            )
        )

        stacked_gap_limit = max(
            24.0,
            page_height * 0.035,
        )

        if (
            horizontal_overlap_ratio >= 0.10
            and vertical_gap
            <= stacked_gap_limit
        ):
            return True

        # PyMuPDF block numbering often keeps nearby pieces
        # of the same rendered formula close even when a
        # fraction/subscript shifts their bounding boxes.
        if (
            abs(
                first.block_number
                - second.block_number
            )
            <= 3
            and vertical_gap
            <= max(
                36.0,
                page_height * 0.05,
            )
            and horizontal_gap
            <= max(
                120.0,
                page_width * 0.25,
            )
        ):
            return True

        return False

    def _is_formula_dominant_region(
        self,
        blocks: list[LayoutBlock],
    ) -> bool:
        if not blocks:
            return False

        formula_like = 0

        for block in blocks:
            text = " ".join(
                block.text.split()
            )

            if (
                block.block_type
                == LayoutBlockType.EQUATION
                or (
                    len(text) <= 180
                    and (
                        _MATH_MARKER_PATTERN
                        .search(text)
                        or self.needs_visual_recovery(
                            text
                        )
                    )
                )
            ):
                formula_like += 1

        return (
            formula_like
            / len(blocks)
            >= 0.75
        )

    def _crop_region(
        self,
        *,
        parsed_page: ParsedPage,
        blocks: list[LayoutBlock],
        crop_directory: Path,
        region_number: int,
    ) -> Path:
        if not blocks:
            raise ValueError(
                "Equation region cannot be empty."
            )

        image_path = Path(
            parsed_page.rendered_image_path
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                (
                    "Rendered page image does "
                    f"not exist: {image_path}"
                )
            )

        min_x = min(
            block.bbox.x0
            for block in blocks
        )

        min_y = min(
            block.bbox.y0
            for block in blocks
        )

        max_x = max(
            block.bbox.x1
            for block in blocks
        )

        max_y = max(
            block.bbox.y1
            for block in blocks
        )

        # Single fragments need more horizontal context,
        # because the rest of a fraction/formula may have
        # been emitted as neighboring PyMuPDF blocks.
        horizontal_padding_points = (
            parsed_page.width * (
                0.12
                if len(blocks) == 1
                else 0.06
            )
        )

        vertical_padding_points = max(
            16.0,
            parsed_page.height * 0.02,
        )

        min_x = max(
            0.0,
            min_x
            - horizontal_padding_points,
        )

        max_x = min(
            parsed_page.width,
            max_x
            + horizontal_padding_points,
        )

        min_y = max(
            0.0,
            min_y
            - vertical_padding_points,
        )

        max_y = min(
            parsed_page.height,
            max_y
            + vertical_padding_points,
        )

        with Image.open(
            image_path
        ) as image:
            image = image.convert(
                "RGB"
            )

            scale_x = (
                parsed_page.rendered_width
                / max(
                    parsed_page.width,
                    1.0,
                )
            )

            scale_y = (
                parsed_page.rendered_height
                / max(
                    parsed_page.height,
                    1.0,
                )
            )

            left = max(
                0,
                int(
                    min_x
                    * scale_x
                ),
            )

            top = max(
                0,
                int(
                    min_y
                    * scale_y
                ),
            )

            right = min(
                image.width,
                int(
                    max_x
                    * scale_x
                ),
            )

            bottom = min(
                image.height,
                int(
                    max_y
                    * scale_y
                ),
            )

            if (
                right <= left
                or bottom <= top
            ):
                left = 0
                top = 0
                right = image.width
                bottom = image.height

            crop = image.crop(
                (
                    left,
                    top,
                    right,
                    bottom,
                )
            )

            crop_path = (
                crop_directory
                / (
                    f"page_"
                    f"{parsed_page.page_number:04d}_"
                    f"region_"
                    f"{region_number:03d}.png"
                )
            )

            crop.save(
                crop_path,
                format="PNG",
            )

        return crop_path.resolve()

    def _transcribe_region(
        self,
        *,
        page_number: int,
        region_number: int,
        blocks: list[LayoutBlock],
        crop_path: Path,
        formula_dominant: bool,
    ) -> EquationArtifact:
        if self.client is None:
            raise RuntimeError(
                "OpenAI client is unavailable."
            )

        ordered_blocks = sorted(
            blocks,
            key=lambda item: (
                item.bbox.y0,
                item.bbox.x0,
                item.block_number,
            ),
        )

        anchor = ordered_blocks[0]

        source_block_ids = [
            block.block_id
            for block in ordered_blocks
        ]

        original_text = "\n".join(
            block.text.strip()
            for block in ordered_blocks
            if block.text.strip()
        )

        image_data_url = (
            self._image_data_url(
                crop_path
            )
        )

        schema = {
            "type": "object",
            "properties": {
                "transcribed_text": {
                    "type": "string",
                },
                "equations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "transcribed_text",
                "equations",
                "confidence",
            ],
            "additionalProperties": False,
        }

        response = (
            self.client.responses.create(
                model=self.model,
                instructions=(
                    "You are a source-faithful "
                    "mathematical transcription engine. "
                    "Read ONLY what is visibly present in "
                    "the supplied Physics document crop. "
                    "The crop may contain one equation "
                    "fragmented across several PDF text "
                    "blocks, multiple lines, a fraction, "
                    "superscripts, or subscripts. Reconstruct "
                    "the visual reading order from the IMAGE, "
                    "not from guessed Physics knowledge. "
                    "Preserve variables, operators, fractions, "
                    "roots, superscripts, subscripts, Greek "
                    "letters, units, labels, and nearby "
                    "explanatory text. Do not solve, simplify, "
                    "correct using domain knowledge, or infer "
                    "a symbol that is not visually legible. "
                    "If a symbol is genuinely unreadable, "
                    "write [unclear] instead of guessing. "
                    "In equations, return complete visible "
                    "equations whenever the crop shows them; "
                    "do not return isolated '=' tokens merely "
                    "because PDF extraction was fragmented. "
                    "Do not add editorial or crop-status commentary "
                    "such as 'continuation not shown', 'not visible', "
                    "'outside the image', or similar notes. If the "
                    "visible source ends, simply stop transcribing."
                ),
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": (
                                    "input_text"
                                ),
                                "text": (
                                    "Transcribe this exact "
                                    "Physics equation region "
                                    "in visual reading order."
                                ),
                            },
                            {
                                "type": (
                                    "input_image"
                                ),
                                "image_url": (
                                    image_data_url
                                ),
                                "detail": "high",
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": (
                            "json_schema"
                        ),
                        "name": (
                            "physics_equation_"
                            "transcription"
                        ),
                        "strict": True,
                        "schema": schema,
                    }
                },
                max_output_tokens=2500,
            )
        )

        payload = json.loads(
            response.output_text
        )

        transcribed_text = (
            self._strip_vision_meta_commentary(
                str(
                    payload.get(
                        "transcribed_text",
                        "",
                    )
                )
            )
        )

        equations = list(
            dict.fromkeys(
                cleaned
                for item in payload.get(
                    "equations",
                    [],
                )
                if (
                    cleaned
                    := self._strip_vision_meta_commentary(
                        str(item)
                    )
                )
            )
        )

        try:
            confidence = float(
                payload.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        replacement_safe = (
            bool(transcribed_text)
            and bool(equations)
            and confidence
            >= self.minimum_replacement_confidence
            and self._unclear_ratio(
                transcribed_text
            )
            <= 0.08
            and not self.needs_visual_recovery(
                transcribed_text
            )
        )

        region_kind = (
            "multi_block"
            if len(
                source_block_ids
            ) > 1
            else "single_block"
        )

        extraction_method = (
            "openai_vision_region"
            if region_kind
            == "multi_block"
            else "openai_vision"
        )

        return EquationArtifact(
            artifact_id=(
                f"eq_{page_number}_"
                f"region_{region_number}"
            ),
            page_number=page_number,
            source_block_id=(
                anchor.block_id
            ),
            source_block_ids=(
                source_block_ids
            ),
            original_text=(
                original_text
            ),
            transcribed_text=(
                transcribed_text
            ),
            equations=equations,
            confidence=confidence,
            extraction_method=(
                extraction_method
            ),
            region_kind=region_kind,
            crop_image_path=str(
                crop_path
            ),
            replacement_safe=(
                replacement_safe
            ),
            suppress_original_on_failure=(
                formula_dominant
            ),
            formula_dominant=(
                formula_dominant
            ),
        )

    def _failed_region_artifact(
        self,
        *,
        page_number: int,
        region_number: int,
        blocks: list[LayoutBlock],
        formula_dominant: bool,
    ) -> EquationArtifact:
        ordered_blocks = sorted(
            blocks,
            key=lambda item: (
                item.bbox.y0,
                item.bbox.x0,
                item.block_number,
            ),
        )

        anchor = ordered_blocks[0]

        source_block_ids = [
            block.block_id
            for block in ordered_blocks
        ]

        return EquationArtifact(
            artifact_id=(
                f"eq_{page_number}_"
                f"region_{region_number}"
            ),
            page_number=page_number,
            source_block_id=(
                anchor.block_id
            ),
            source_block_ids=(
                source_block_ids
            ),
            original_text="\n".join(
                block.text.strip()
                for block in ordered_blocks
                if block.text.strip()
            ),
            transcribed_text="",
            equations=[],
            confidence=0.0,
            extraction_method=(
                "openai_vision_region"
                if len(
                    source_block_ids
                ) > 1
                else "openai_vision"
            ),
            region_kind=(
                "multi_block"
                if len(
                    source_block_ids
                ) > 1
                else "single_block"
            ),
            crop_image_path=None,
            replacement_safe=False,
            suppress_original_on_failure=(
                formula_dominant
            ),
            formula_dominant=(
                formula_dominant
            ),
        )

    def _native_equations(
        self,
        text: str,
    ) -> list[str]:
        normalized = " ".join(
            text.split()
        )

        if not normalized:
            return []

        if self.needs_visual_recovery(
            text
        ):
            return []

        if not _MATH_MARKER_PATTERN.search(
            normalized
        ):
            return []

        # Preserve the complete native expression instead
        # of splitting PDF line fragments into meaningless
        # entries such as ["=", "="].
        return [
            normalized
        ]

    def _strip_vision_meta_commentary(
        self,
        text: str,
    ) -> str:
        """
        Keep source transcription only.

        Vision models must not contribute editorial comments
        about the crop itself. This removes generic crop-status
        commentary without inventing or repairing source text.
        """

        if not text:
            return ""

        cleaned = (
            _VISION_META_COMMENTARY_PATTERN.sub(
                " ",
                text,
            )
        )

        # Remove empty TeX wrappers left behind by filtering.
        cleaned = re.sub(
            r"\\{1,2}text\{\s*\}",
            " ",
            cleaned,
        )

        cleaned = re.sub(
            r"[ \t]+\n",
            "\n",
            cleaned,
        )

        cleaned = re.sub(
            r"\n[ \t]+",
            "\n",
            cleaned,
        )

        cleaned = re.sub(
            r"\n{3,}",
            "\n\n",
            cleaned,
        )

        cleaned = re.sub(
            r"[ \t]{2,}",
            " ",
            cleaned,
        )

        return cleaned.strip()

    def _image_data_url(
        self,
        image_path: Path,
    ) -> str:
        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode(
            "ascii"
        )

        return (
            "data:image/png;base64,"
            + encoded
        )

    def _unclear_ratio(
        self,
        text: str,
    ) -> float:
        if not text:
            return 1.0

        unclear_count = (
            text.lower().count(
                "[unclear]"
            )
        )

        token_count = max(
            len(
                text.split()
            ),
            1,
        )

        return (
            unclear_count
            / token_count
        )