from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from src.ingestion.equations import (
    EquationArtifact,
    EquationExtractionResult,
)
from src.ingestion.models import (
    ChunkingResult,
    DocumentLayout,
    DocumentStructure,
    FigureArtifact,
    FigureExtractionResult,
    LayoutBlock,
    LayoutBlockType,
    OCRDocumentResult,
    ParentChunk,
    ParsedDocument,
    RetrievalChunk,
    ScopeClassification,
    StructuralNode,
    StructuralNodeType,
)


class DocumentChunkingError(Exception):
    """Raised when document chunk creation fails."""


class HierarchicalChunker:
    """
    Create hierarchical multimodal chunks.

    Parent chunks preserve section-level context.
    Child chunks are used for precise retrieval.
    Visual chunks represent figures and diagrams.
    """

    def __init__(
        self,
        *,
        child_max_characters: int = 900,
        child_overlap_characters: int = 120,
        minimum_child_characters: int = 80,
        structural_text_match_minimum: float = 0.55,
        max_structural_fallback_links: int = 3,
    ) -> None:
        if child_max_characters < 200:
            raise ValueError(
                "child_max_characters must be at least 200."
            )

        if child_overlap_characters < 0:
            raise ValueError(
                "child_overlap_characters cannot be negative."
            )

        if (
            child_overlap_characters
            >= child_max_characters
        ):
            raise ValueError(
                "child_overlap_characters must be "
                "smaller than child_max_characters."
            )

        if not 0.0 <= structural_text_match_minimum <= 1.0:
            raise ValueError(
                "structural_text_match_minimum must be between 0 and 1."
            )

        if max_structural_fallback_links < 1:
            raise ValueError(
                "max_structural_fallback_links must be at least 1."
            )

        self.child_max_characters = (
            child_max_characters
        )

        self.child_overlap_characters = (
            child_overlap_characters
        )

        self.minimum_child_characters = (
            minimum_child_characters
        )

        self.structural_text_match_minimum = (
            structural_text_match_minimum
        )

        self.max_structural_fallback_links = (
            max_structural_fallback_links
        )

    def chunk(
        self,
        *,
        user_id: str,
        parsed_document: ParsedDocument,
        document_layout: DocumentLayout,
        ocr_result: OCRDocumentResult,
        figure_result: FigureExtractionResult,
        equation_result: EquationExtractionResult,
        scope_result: ScopeClassification,
        document_structure: DocumentStructure | None = None,
    ) -> ChunkingResult:
        if not user_id.strip():
            raise DocumentChunkingError(
                "user_id cannot be empty."
            )

        document_id = parsed_document.document_id

        if (
            document_structure is not None
            and document_structure.document_id
            != document_id
        ):
            raise DocumentChunkingError(
                "DocumentStructure belongs to a different document."
            )

        layout_by_page = {
            page.page_number: page
            for page in document_layout.pages
        }

        ocr_by_page = {
            page.page_number: page
            for page in ocr_result.pages
        }

        figures_by_page = defaultdict(list)

        for figure in figure_result.figures:
            figures_by_page[
                figure.page_number
            ].append(figure)

        for page_number in list(
            figures_by_page.keys()
        ):
            figures_by_page[
                page_number
            ] = sorted(
                figures_by_page[
                    page_number
                ],
                key=self._figure_sort_key,
            )

        equations_by_block: dict[
            str,
            EquationArtifact,
        ] = {}

        for artifact in equation_result.artifacts:
            source_ids = (
                artifact.source_block_ids
                or [
                    artifact.source_block_id
                ]
            )

            for block_id in source_ids:
                equations_by_block[
                    block_id
                ] = artifact

        parent_chunks: list[ParentChunk] = []
        retrieval_chunks: list[
            RetrievalChunk
        ] = []

        for parsed_page in parsed_document.pages:
            page_number = parsed_page.page_number

            layout_page = layout_by_page.get(
                page_number
            )

            if layout_page is None:
                continue

            (
                page_blocks,
                source_ids_by_block,
            ) = self._apply_equation_transcriptions(
                blocks=list(
                    layout_page.blocks
                ),
                equations_by_block=(
                    equations_by_block
                ),
            )

            # OCR-only/scanned page fallback.
            if not self._has_meaningful_text(
                page_blocks
            ):
                ocr_page = ocr_by_page.get(
                    page_number
                )

                if (
                    ocr_page is not None
                    and ocr_page.text.strip()
                ):
                    fallback_block = LayoutBlock(
                        block_id=(
                            f"p{page_number}_ocr"
                        ),
                        page_number=page_number,
                        block_number=999999,
                        block_type=(
                            LayoutBlockType.PARAGRAPH
                        ),
                        bbox=(
                            parsed_page.blocks[0].bbox
                            if parsed_page.blocks
                            else self._full_page_bbox(
                                parsed_page
                            )
                        ),
                        text=ocr_page.text.strip(),
                        source="ocr",
                        confidence=(
                            (
                                ocr_page.average_confidence
                                or 50.0
                            )
                            / 100.0
                        ),
                    )

                    page_blocks.append(
                        fallback_block
                    )

                    source_ids_by_block[
                        fallback_block.block_id
                    ] = [
                        fallback_block.block_id
                    ]

            sections = self._build_sections(
                page_blocks
            )

            page_figures = figures_by_page.get(
                page_number,
                [],
            )

            for section_index, section in enumerate(
                sections,
                start=1,
            ):
                parent_id = (
                    f"{document_id}_"
                    f"p{page_number}_"
                    f"section{section_index}"
                )

                heading = section["heading"]

                blocks: list[LayoutBlock] = (
                    section["blocks"]
                )

                section_text = (
                    self._section_text(
                        heading=heading,
                        blocks=blocks,
                    )
                )

                if not section_text.strip():
                    continue

                equations = (
                    self._section_equations(
                        blocks=blocks,
                        equations_by_block=(
                            equations_by_block
                        ),
                    )
                )

                linked_figures = (
                    self._section_figure_ids(
                        section_text=section_text,
                        section_blocks=blocks,
                        page_figures=page_figures,
                        single_section=(
                            len(sections) == 1
                        ),
                    )
                )

                child_chunks = (
                    self._create_child_chunks(
                        user_id=user_id,
                        document_id=document_id,
                        page_number=page_number,
                        parent_id=parent_id,
                        heading=heading,
                        blocks=blocks,
                        topics=scope_result.topics,
                        grade_min=(
                            scope_result
                            .estimated_grade_min
                        ),
                        grade_max=(
                            scope_result
                            .estimated_grade_max
                        ),
                        linked_figure_ids=(
                            linked_figures
                        ),
                        page_figures=page_figures,
                        source_ids_by_block=(
                            source_ids_by_block
                        ),
                    )
                )

                parent = ParentChunk(
                    parent_id=parent_id,
                    user_id=user_id,
                    document_id=document_id,
                    page_number=page_number,
                    heading=heading,
                    text=section_text,
                    topics=scope_result.topics,
                    grade_min=(
                        scope_result
                        .estimated_grade_min
                    ),
                    grade_max=(
                        scope_result
                        .estimated_grade_max
                    ),
                    figures=linked_figures,
                    equations=equations,
                    child_ids=[
                        child.chunk_id
                        for child in child_chunks
                    ],
                )

                parent_chunks.append(parent)

                retrieval_chunks.extend(
                    child_chunks
                )

            retrieval_chunks.extend(
                self._create_visual_chunks(
                    user_id=user_id,
                    document_id=document_id,
                    page_number=page_number,
                    page_figures=page_figures,
                    parent_chunks=[
                        parent
                        for parent in parent_chunks
                        if (
                            parent.page_number
                            == page_number
                        )
                    ],
                    topics=scope_result.topics,
                    grade_min=(
                        scope_result
                        .estimated_grade_min
                    ),
                    grade_max=(
                        scope_result
                        .estimated_grade_max
                    ),
                )
            )

        if not retrieval_chunks:
            raise DocumentChunkingError(
                "No retrieval-ready chunks could "
                "be created from the document."
            )

        if document_structure is not None:
            self._link_structural_evidence(
                document_structure=document_structure,
                parent_chunks=parent_chunks,
                retrieval_chunks=retrieval_chunks,
            )

        return ChunkingResult(
            document_id=document_id,
            user_id=user_id,
            parent_chunks=parent_chunks,
            retrieval_chunks=retrieval_chunks,
        )

    def _apply_equation_transcriptions(
        self,
        *,
        blocks: list[LayoutBlock],
        equations_by_block: dict[
            str,
            EquationArtifact,
        ],
    ) -> tuple[
        list[LayoutBlock],
        dict[
            str,
            list[str],
        ],
    ]:
        """
        Apply source-image equation recovery.

        For a successful multi-block recovery:
        - write the recovered region once at its anchor,
        - suppress the duplicate companion fragments,
        - retain every original block id as provenance.

        For an unsuccessful formula-only recovery:
        - suppress visibly corrupted formula fragments rather
          than indexing them as trustworthy evidence.

        For prose-dominant regions:
        - keep the original prose if recovery is uncertain.
        """

        corrected: list[
            LayoutBlock
        ] = []

        source_ids_by_block: dict[
            str,
            list[str],
        ] = {}

        emitted_artifact_ids: set[
            str
        ] = set()

        for block in blocks:
            artifact = (
                equations_by_block.get(
                    block.block_id
                )
            )

            if artifact is None:
                sanitized_block = (
                    self._sanitize_unrecovered_block(
                        block
                    )
                )

                if sanitized_block is None:
                    continue

                corrected.append(
                    sanitized_block
                )

                source_ids_by_block[
                    sanitized_block.block_id
                ] = [
                    block.block_id
                ]

                continue

            source_ids = (
                artifact.source_block_ids
                or [
                    artifact.source_block_id
                ]
            )

            # Successful visual recovery replaces the entire
            # fragmented region exactly once at its anchor.
            if (
                artifact.replacement_safe
                and artifact.transcribed_text.strip()
            ):
                if (
                    artifact.artifact_id
                    in emitted_artifact_ids
                ):
                    continue

                if (
                    block.block_id
                    != artifact.source_block_id
                ):
                    continue

                corrected_type = (
                    LayoutBlockType.EQUATION
                    if (
                        artifact.formula_dominant
                        and artifact.equations
                    )
                    else block.block_type
                )

                corrected_block = (
                    block.model_copy(
                        update={
                            "text": (
                                artifact
                                .transcribed_text
                                .strip()
                            ),
                            "source": "vision",
                            "confidence": max(
                                block.confidence,
                                artifact.confidence,
                            ),
                            "block_type": (
                                corrected_type
                            ),
                        }
                    )
                )

                corrected.append(
                    corrected_block
                )

                source_ids_by_block[
                    corrected_block.block_id
                ] = list(
                    dict.fromkeys(
                        source_ids
                    )
                )

                emitted_artifact_ids.add(
                    artifact.artifact_id
                )

                continue

            # Failed recovery of a formula-dominant corrupted
            # region: do not leak broken equation placeholders
            # such as "!!", "! =", or detached fractions into
            # retrieval.
            if (
                artifact.suppress_original_on_failure
            ):
                continue

            # Prose-dominant uncertain region: preserve the
            # source text because dropping the full paragraph
            # could lose legitimate explanatory content.
            sanitized_block = (
                self._sanitize_unrecovered_block(
                    block
                )
            )

            if sanitized_block is None:
                continue

            corrected.append(
                sanitized_block
            )

            source_ids_by_block[
                sanitized_block.block_id
            ] = [
                block.block_id
            ]

        corrected = (
            self._deduplicate_overlapping_blocks(
                corrected
            )
        )

        retained_ids = {
            block.block_id
            for block in corrected
        }

        source_ids_by_block = {
            block_id: source_ids
            for block_id, source_ids
            in source_ids_by_block.items()
            if block_id in retained_ids
        }

        return (
            corrected,
            source_ids_by_block,
        )

    def _deduplicate_overlapping_blocks(
        self,
        blocks: list[LayoutBlock],
    ) -> list[LayoutBlock]:
        """
        Remove duplicate short labels/prose without deleting
        unique source evidence.

        Two safe cases are handled:
        1. exact adjacent duplicates from native/OCR,
        2. a short native label repeated inside a nearby
           source-faithful vision recovery block.

        The second rule is intentionally conservative:
        equation-like text is never removed by this helper.
        """

        if len(blocks) < 2:
            return blocks

        ordered = sorted(
            blocks,
            key=lambda block: (
                block.bbox.y0,
                block.bbox.x0,
                block.block_number,
            ),
        )

        keep = [
            True
            for _ in ordered
        ]

        normalized = [
            self._normalize_for_dedup(
                block.text
            )
            for block in ordered
        ]

        # Pass 1: collapse exact adjacent duplicates.
        for index in range(
            len(ordered) - 1
        ):
            if (
                normalized[index]
                and normalized[index]
                == normalized[index + 1]
            ):
                first = ordered[index]
                second = ordered[index + 1]

                if (
                    second.source.startswith(
                        "vision"
                    )
                    and not first.source.startswith(
                        "vision"
                    )
                ):
                    keep[index] = False
                else:
                    keep[
                        index + 1
                    ] = False

        # Pass 2: remove a short native label if the same
        # information is already present inside a nearby
        # vision-recovered block.
        vision_entries = [
            (
                index,
                block,
                self._plain_text_for_dedup(
                    block.text
                ),
            )
            for index, block in enumerate(
                ordered
            )
            if block.source.startswith(
                "vision"
            )
        ]

        for index, block in enumerate(
            ordered
        ):
            if not keep[index]:
                continue

            if block.source.startswith(
                "vision"
            ):
                continue

            if not self._is_short_label_like(
                block.text
            ):
                continue

            candidate = (
                self._plain_text_for_dedup(
                    block.text
                )
            )

            if not candidate:
                continue

            for (
                vision_index,
                vision_block,
                vision_text,
            ) in vision_entries:
                if not keep[
                    vision_index
                ]:
                    continue

                # Only compare nearby extraction blocks.
                # This prevents a repeated label in a distant
                # section of the same page from being removed.
                if (
                    abs(
                        block.block_number
                        - vision_block.block_number
                    )
                    > 24
                ):
                    continue

                if (
                    candidate == vision_text
                    or candidate in vision_text
                ):
                    keep[index] = False
                    break

        return [
            block
            for index, block in enumerate(
                ordered
            )
            if keep[index]
        ]

    def _normalize_for_dedup(
        self,
        text: str,
    ) -> str:
        normalized = re.sub(
            r"\s+",
            " ",
            text,
        ).strip().casefold()

        normalized = re.sub(
            r"\s*([:;,.])\s*",
            r"\1",
            normalized,
        )

        return normalized

    def _plain_text_for_dedup(
        self,
        text: str,
    ) -> str:
        """
        Normalize lightweight LaTeX wrappers only for duplicate
        comparison. This does not change stored evidence.
        """

        normalized = text

        # Unwrap common text-only LaTeX containers produced by
        # vision transcription.
        for command in (
            "textbf",
            "textit",
            "text",
            "mathrm",
        ):
            pattern = (
                r"\\"
                + command
                + r"\{([^{}]*)\}"
            )

            previous = None

            while (
                previous
                != normalized
            ):
                previous = normalized
                normalized = re.sub(
                    pattern,
                    r"\1",
                    normalized,
                )

        normalized = normalized.replace(
            "{",
            " ",
        ).replace(
            "}",
            " ",
        )

        normalized = normalized.replace(
            "\\",
            " ",
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip().casefold()

        normalized = re.sub(
            r"\s*([:;,.])\s*",
            r"\1",
            normalized,
        )

        return normalized

    def _is_short_label_like(
        self,
        text: str,
    ) -> bool:
        normalized = (
            self._plain_text_for_dedup(
                text
            )
        )

        if (
            not normalized
            or len(normalized) > 90
        ):
            return False

        # Never deduplicate equation-like evidence here.
        if (
            "=" in normalized
            or "\\frac" in text
            or "\\sqrt" in text
        ):
            return False

        words = re.findall(
            r"\w+",
            normalized,
            flags=re.UNICODE,
        )

        if len(words) > 10:
            return False

        return (
            normalized.endswith(":")
            or "/" in normalized
            or len(normalized) <= 55
        )

    def _is_decorative_separator_text(
        self,
        text: str,
    ) -> bool:
        """
        Detect separator-only extraction debris across Unicode
        dash/soft-hyphen variants.

        No source text containing letters or digits is removed.
        """

        compact = "".join(
            character
            for character in text
            if (
                not character.isspace()
                and unicodedata.category(
                    character
                )
                != "Cf"
            )
        )

        if (
            not compact
            or len(compact) > 24
        ):
            return False

        if any(
            character.isalnum()
            for character in compact
        ):
            return False

        return all(
            unicodedata.category(
                character
            ).startswith(
                (
                    "P",
                    "S",
                )
            )
            for character in compact
        )

    def _normalize_block_text(
        self,
        block: LayoutBlock,
    ) -> str:
        """
        Normalize extraction whitespace without flattening
        intentional multi-line vision equations.

        Native/OCR PDF text often contains CR/tab separators
        between individual words; those become normal spaces.
        Vision recovery may intentionally use line breaks, so
        its line structure is retained.
        """

        text = block.text.strip()

        if not text:
            return ""

        if block.source.startswith(
            "vision"
        ):
            lines: list[str] = []

            for raw_line in text.splitlines():
                line = re.sub(
                    r"[ \t\r\f\v]+",
                    " ",
                    raw_line,
                ).strip()

                if line:
                    lines.append(
                        line
                    )
                elif (
                    lines
                    and lines[-1] != ""
                ):
                    lines.append(
                        ""
                    )

            while (
                lines
                and lines[-1] == ""
            ):
                lines.pop()

            return "\n".join(
                lines
            ).strip()

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

    def _clean_assembled_text(
        self,
        text: str,
    ) -> str:
        """
        Final retrieval-text hygiene.

        This only removes adjacent duplicate lines and empty
        spacing. It does not infer or rewrite source content.
        """

        if not text:
            return ""

        cleaned_lines: list[str] = []
        previous_key: str | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                if (
                    cleaned_lines
                    and cleaned_lines[-1] != ""
                ):
                    cleaned_lines.append(
                        ""
                    )
                continue

            key = self._normalize_for_dedup(
                line
            )

            if (
                key
                and key == previous_key
            ):
                continue

            cleaned_lines.append(
                line
            )

            previous_key = key

        while (
            cleaned_lines
            and cleaned_lines[-1] == ""
        ):
            cleaned_lines.pop()

        return "\n".join(
            cleaned_lines
        ).strip()

    def _sanitize_unrecovered_block(
        self,
        block: LayoutBlock,
    ) -> LayoutBlock | None:
        """
        Remove extraction debris that was not covered by a
        trustworthy equation artifact.

        We never guess a missing symbol. We either:
        - drop a tiny formula-only debris block, or
        - remove only the visibly broken token and retain the
          readable prose around it.
        """

        original = block.text.strip()

        if not original:
            return None

        normalized = re.sub(
            r"\s+",
            " ",
            original,
        ).strip()

        # Decorative separator-only blocks are layout noise,
        # including Unicode dash + soft-hyphen variants.
        if self._is_decorative_separator_text(
            normalized
        ):
            return None

        # A single mathematical glyph can be emitted as a
        # separate paragraph block by PDF extractors. Drop the
        # orphan glyph regardless of paragraph/equation label;
        # a trustworthy equation-region artifact carries the
        # real formula when recovery succeeds.
        compact = normalized.replace(
            " ",
            "",
        )

        math_symbol_only = bool(
            re.fullmatch(
                r"[\u0370-\u03ff"
                r"\U0001d400-\U0001d7ff"
                r"∑√∆Δ±×÷π]+",
                compact,
            )
        )

        if (
            len(compact) <= 3
            and math_symbol_only
            and "=" not in compact
        ):
            return None

        # Orphan one-token equation fragments such as "T",
        # "ω", or "-" are not useful retrieval evidence.
        if (
            block.block_type
            == LayoutBlockType.EQUATION
            and len(normalized) <= 3
            and "=" not in normalized
        ):
            return None

        has_placeholder = (
            "!" in normalized
            or "�" in normalized
            or bool(
                re.search(
                    r'["#$%&]{3,}',
                    normalized,
                )
            )
        )

        if not has_placeholder:
            return block

        # Pure/near-pure corrupted formula debris should not
        # be indexed at all.
        readable_without_debris = re.sub(
            r"\S*[!�]\S*",
            " ",
            normalized,
        )

        readable_without_debris = re.sub(
            r'["#$%&]{3,}',
            " ",
            readable_without_debris,
        )

        readable_without_debris = re.sub(
            r"\s+",
            " ",
            readable_without_debris,
        ).strip()

        alphanumeric_count = sum(
            character.isalnum()
            for character
            in readable_without_debris
        )

        if (
            not readable_without_debris
            or alphanumeric_count < 4
            or (
                len(normalized) <= 24
                and block.block_type
                == LayoutBlockType.EQUATION
            )
        ):
            return None

        # Preserve readable prose while omitting the unknown
        # corrupted symbol/token. This is omission, not
        # reconstruction.
        return block.model_copy(
            update={
                "text": readable_without_debris,
                "source": (
                    f"{block.source}_debris_filtered"
                ),
                "confidence": min(
                    block.confidence,
                    0.70,
                ),
            }
        )

    def _section_equations(
        self,
        *,
        blocks: list[LayoutBlock],
        equations_by_block: dict[
            str,
            EquationArtifact,
        ],
    ) -> list[str]:
        equations: list[str] = []
        used_artifacts: set[str] = set()

        for block in blocks:
            artifact = (
                equations_by_block.get(
                    block.block_id
                )
            )

            if artifact is not None:
                if (
                    artifact.artifact_id
                    in used_artifacts
                ):
                    continue

                trustworthy = (
                    artifact.extraction_method
                    == "native"
                    or artifact.replacement_safe
                )

                if (
                    trustworthy
                    and artifact.equations
                ):
                    equations.extend(
                        artifact.equations
                    )

                    used_artifacts.add(
                        artifact.artifact_id
                    )

                continue

            if (
                block.block_type
                == LayoutBlockType.EQUATION
                and block.text.strip()
            ):
                equations.append(
                    block.text.strip()
                )

        return list(
            dict.fromkeys(
                equation.strip()
                for equation in equations
                if equation.strip()
            )
        )

    def _has_meaningful_text(
        self,
        blocks: list[LayoutBlock],
    ) -> bool:
        total_text = " ".join(
            block.text.strip()
            for block in blocks
            if block.text.strip()
        )

        return len(total_text) >= 40

    def _build_sections(
        self,
        blocks: list[LayoutBlock],
    ) -> list[dict]:
        """
        Preserve document structure instead of blindly
        splitting every N characters.
        """

        sections: list[dict] = []

        current_heading: str | None = None
        current_blocks: list[
            LayoutBlock
        ] = []

        ordered_blocks = sorted(
            blocks,
            key=lambda block: (
                block.bbox.y0,
                block.bbox.x0,
                block.block_number,
            ),
        )

        for block in ordered_blocks:
            if (
                block.block_type
                in {
                    LayoutBlockType.TITLE,
                    LayoutBlockType.HEADING,
                }
                and block.text.strip()
            ):
                if current_blocks:
                    sections.append(
                        {
                            "heading": (
                                current_heading
                            ),
                            "blocks": (
                                current_blocks
                            ),
                        }
                    )

                    current_blocks = []

                current_heading = (
                    block.text.strip()
                )

                continue

            if block.block_type == (
                LayoutBlockType.FIGURE
            ):
                continue

            if block.text.strip():
                current_blocks.append(block)

        if current_blocks or current_heading:
            sections.append(
                {
                    "heading": current_heading,
                    "blocks": current_blocks,
                }
            )

        if not sections:
            text_blocks = [
                block
                for block in ordered_blocks
                if block.text.strip()
            ]

            if text_blocks:
                sections.append(
                    {
                        "heading": None,
                        "blocks": text_blocks,
                    }
                )

        return sections

    def _section_text(
        self,
        *,
        heading: str | None,
        blocks: list[LayoutBlock],
    ) -> str:
        parts: list[str] = []

        if heading:
            normalized_heading = re.sub(
                r"\s+",
                " ",
                heading,
            ).strip()

            if normalized_heading:
                parts.append(
                    normalized_heading
                )

        for block in blocks:
            text = (
                self._normalize_block_text(
                    block
                )
            )

            if not text:
                continue

            parts.append(text)

        return self._clean_assembled_text(
            "\n".join(parts)
        )

    def _create_child_chunks(
        self,
        *,
        user_id: str,
        document_id: str,
        page_number: int,
        parent_id: str,
        heading: str | None,
        blocks: list[LayoutBlock],
        topics: list[str],
        grade_min: int | None,
        grade_max: int | None,
        linked_figure_ids: list[str],
        page_figures: list[
            FigureArtifact
        ] | None = None,
        source_ids_by_block: dict[
            str,
            list[str],
        ],
    ) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []

        current_parts: list[str] = []
        current_block_ids: list[str] = []
        current_types: list[str] = []
        current_blocks: list[
            LayoutBlock
        ] = []

        child_number = 1

        def flush() -> None:
            nonlocal child_number
            nonlocal current_parts
            nonlocal current_block_ids
            nonlocal current_types
            nonlocal current_blocks

            combined = (
                self._clean_assembled_text(
                    "\n".join(
                        current_parts
                    )
                )
            )

            if not combined:
                current_parts = []
                current_block_ids = []
                current_types = []
                current_blocks = []
                return

            content_type = (
                self._dominant_content_type(
                    current_types
                )
            )

            chunk_linked_figure_ids = (
                self._resolve_child_figure_ids(
                    chunk_text=combined,
                    chunk_blocks=current_blocks,
                    content_type=content_type,
                    section_figure_ids=(
                        linked_figure_ids
                    ),
                    page_figures=(
                        page_figures
                        or []
                    ),
                )
            )

            chunk = RetrievalChunk(
                chunk_id=(
                    f"{parent_id}_"
                    f"c{child_number}"
                ),
                user_id=user_id,
                document_id=document_id,
                page_number=page_number,
                chunk_kind="child",
                text=combined,
                content_type=content_type,
                parent_id=parent_id,
                topics=topics,
                grade_min=grade_min,
                grade_max=grade_max,
                source_block_ids=(
                    current_block_ids.copy()
                ),
                linked_figure_ids=(
                    chunk_linked_figure_ids
                ),
            )

            chunks.append(chunk)

            child_number += 1

            overlap_text = (
                self._tail_overlap(
                    combined
                )
            )

            current_parts = (
                [overlap_text]
                if overlap_text
                else []
            )

            current_block_ids = []
            current_types = []
            current_blocks = []

        if heading:
            current_parts.append(
                f"Heading: {heading}"
            )

        for block in blocks:
            text = (
                self._normalize_block_text(
                    block
                )
            )

            if not text:
                continue

            # Equations/questions/examples are semantic
            # units: avoid cutting them in the middle.
            atomic_block = (
                block.block_type
                in {
                    LayoutBlockType.EQUATION,
                    LayoutBlockType.QUESTION,
                    LayoutBlockType.ANSWER,
                    LayoutBlockType.WORKED_EXAMPLE,
                    LayoutBlockType.FIGURE_CAPTION,
                }
            )

            projected_length = len(
                "\n".join(
                    current_parts + [text]
                )
            )

            if (
                current_parts
                and projected_length
                > self.child_max_characters
            ):
                flush()

            if (
                len(text)
                > self.child_max_characters
                and not atomic_block
            ):
                pieces = (
                    self._split_long_text(
                        text
                    )
                )

                for piece in pieces:
                    if current_parts:
                        flush()

                    current_parts = [piece]
                    current_block_ids = (
                        source_ids_by_block.get(
                            block.block_id,
                            [
                                block.block_id
                            ],
                        ).copy()
                    )

                    current_types = [
                        block.block_type.value
                    ]

                    current_blocks = [
                        block
                    ]

                    flush()

                continue

            current_parts.append(text)

            block_source_ids = (
                source_ids_by_block.get(
                    block.block_id,
                    [
                        block.block_id
                    ],
                )
            )

            current_block_ids = list(
                dict.fromkeys(
                    current_block_ids
                    + block_source_ids
                )
            )

            current_types.append(
                block.block_type.value
            )

            current_blocks.append(
                block
            )

        if current_parts:
            flush()

        # Very tiny final chunks are merged into
        # their predecessor where possible.
        if (
            len(chunks) >= 2
            and len(chunks[-1].text)
            < self.minimum_child_characters
        ):
            last = chunks.pop()
            previous = chunks[-1]

            merged_text = (
                previous.text.rstrip()
                + "\n"
                + last.text.lstrip()
            )

            previous.text = merged_text

            previous.source_block_ids = list(
                dict.fromkeys(
                    previous.source_block_ids
                    + last.source_block_ids
                )
            )

            previous.linked_figure_ids = list(
                dict.fromkeys(
                    previous.linked_figure_ids
                    + last.linked_figure_ids
                )
            )

        return chunks

    def _split_long_text(
        self,
        text: str,
    ) -> list[str]:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        pieces: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            candidate = (
                f"{current} {sentence}".strip()
            )

            if (
                len(candidate)
                <= self.child_max_characters
            ):
                current = candidate
                continue

            if current:
                pieces.append(current)

            if (
                len(sentence)
                <= self.child_max_characters
            ):
                current = sentence
                continue

            start = 0

            while start < len(sentence):
                end = (
                    start
                    + self.child_max_characters
                )

                pieces.append(
                    sentence[start:end].strip()
                )

                start = max(
                    end
                    - self.child_overlap_characters,
                    start + 1,
                )

            current = ""

        if current:
            pieces.append(current)

        return [
            piece
            for piece in pieces
            if piece.strip()
        ]

    def _tail_overlap(
        self,
        text: str,
    ) -> str:
        if self.child_overlap_characters <= 0:
            return ""

        if len(text) <= (
            self.child_overlap_characters
        ):
            return ""

        tail = text[
            -self.child_overlap_characters:
        ]

        first_space = tail.find(" ")

        if first_space >= 0:
            tail = tail[
                first_space + 1:
            ]

        return tail.strip()

    def _dominant_content_type(
        self,
        block_types: list[str],
    ) -> str:
        priority = [
            LayoutBlockType.QUESTION.value,
            LayoutBlockType.WORKED_EXAMPLE.value,
            LayoutBlockType.EQUATION.value,
            LayoutBlockType.ANSWER.value,
            LayoutBlockType.PARAGRAPH.value,
            LayoutBlockType.FIGURE_CAPTION.value,
        ]

        for item in priority:
            if item in block_types:
                return item

        return "concept"

    def _create_visual_chunks(
        self,
        *,
        user_id: str,
        document_id: str,
        page_number: int,
        page_figures: list[FigureArtifact],
        parent_chunks: list[ParentChunk],
        topics: list[str],
        grade_min: int | None,
        grade_max: int | None,
    ) -> list[RetrievalChunk]:
        """
        Create retrieval-ready visual chunks.

        The canonical FigureArtifact keeps the actual image plus provenance-
        separated visual semantics. This method turns those fields into one
        searchable text representation without collapsing source evidence and
        standard Physics explanation into the same provenance category.

        Backward compatibility:
        - older figures that only contain semantic_description/caption remain
          indexable;
        - existing RetrievalChunk contracts are unchanged.
        """

        visual_chunks: list[
            RetrievalChunk
        ] = []

        for index, figure in enumerate(
            page_figures,
            start=1,
        ):
            parent = self._best_parent_for_figure(
                figure_id=figure.figure_id,
                parents=parent_chunks,
            )

            caption = (
                figure.caption.strip()
                if figure.caption
                and figure.caption.strip()
                else None
            )

            exact_source_label = (
                figure.exact_source_label.strip()
                if (
                    figure.exact_source_label
                    and figure.exact_source_label.strip()
                )
                else None
            )

            source_description = (
                figure.source_description.strip()
                if (
                    figure.source_description
                    and figure.source_description.strip()
                )
                else None
            )

            standard_physics_explanation = (
                figure.standard_physics_explanation.strip()
                if (
                    figure.standard_physics_explanation
                    and figure.standard_physics_explanation.strip()
                )
                else None
            )

            derived_interpretation = (
                figure.derived_interpretation.strip()
                if (
                    figure.derived_interpretation
                    and figure.derived_interpretation.strip()
                )
                else None
            )

            semantic_description = (
                figure.semantic_description.strip()
                if (
                    figure.semantic_description
                    and figure.semantic_description.strip()
                )
                else None
            )

            visible_labels = list(
                dict.fromkeys(
                    label.strip()
                    for label in figure.visible_labels
                    if (
                        isinstance(label, str)
                        and label.strip()
                    )
                )
            )

            nearby_text = (
                figure.nearby_text.strip()
                if (
                    figure.nearby_text
                    and figure.nearby_text.strip()
                )
                else None
            )

            text_parts = [
                (
                    "Visual evidence from the uploaded "
                    "Physics document."
                )
            ]

            if (
                figure.document_figure_index
                is not None
            ):
                text_parts.append(
                    "Document figure position: "
                    + str(
                        figure.document_figure_index
                    )
                )

            if (
                figure.page_figure_index
                is not None
            ):
                text_parts.append(
                    "Page figure position: "
                    + str(
                        figure.page_figure_index
                    )
                )

            if exact_source_label:
                text_parts.append(
                    "Exact source label: "
                    + exact_source_label
                )

            # -------------------------------------------------
            # SOURCE-SIDE VISUAL EVIDENCE
            # -------------------------------------------------

            if source_description:
                text_parts.append(
                    "Source visual description: "
                    + source_description
                )

            if visible_labels:
                text_parts.append(
                    "Visible labels and notation: "
                    + " | ".join(
                        visible_labels
                    )
                )

            # -------------------------------------------------
            # STANDARD PHYSICS KNOWLEDGE
            # -------------------------------------------------

            if standard_physics_explanation:
                text_parts.append(
                    "Standard Physics explanation: "
                    + standard_physics_explanation
                )

            # -------------------------------------------------
            # DERIVED INTERPRETATION
            # -------------------------------------------------

            if derived_interpretation:
                text_parts.append(
                    "Derived interpretation: "
                    + derived_interpretation
                )

            # Older stored FigureArtifact records may not have
            # the richer fields above. Keep them searchable.
            has_rich_semantics = any(
                (
                    source_description,
                    standard_physics_explanation,
                    derived_interpretation,
                    visible_labels,
                )
            )

            if (
                not has_rich_semantics
                and semantic_description
            ):
                text_parts.append(
                    "Visual description: "
                    + semantic_description
                )

            elif (
                not has_rich_semantics
                and caption
            ):
                text_parts.append(
                    "Caption: "
                    + caption
                )

            if nearby_text:
                text_parts.append(
                    "Nearby source text: "
                    + nearby_text[:1200]
                )

            if parent and parent.heading:
                text_parts.append(
                    f"Section: {parent.heading}"
                )

            if parent:
                contextual_text = (
                    parent.text[:600].strip()
                )

                if contextual_text:
                    text_parts.append(
                        "Linked document context: "
                        + contextual_text
                    )

            visual_chunks.append(
                RetrievalChunk(
                    chunk_id=(
                        f"{document_id}_"
                        f"p{page_number}_"
                        f"fig{index}"
                    ),
                    user_id=user_id,
                    document_id=document_id,
                    page_number=page_number,
                    chunk_kind="visual",
                    text="\n".join(
                        text_parts
                    ),
                    content_type="figure",
                    parent_id=(
                        parent.parent_id
                        if parent
                        else None
                    ),
                    topics=topics,
                    grade_min=grade_min,
                    grade_max=grade_max,
                    linked_figure_ids=[
                        figure.figure_id
                    ],
                    image_path=(
                        figure.image_path
                    ),
                    caption=caption,
                )
            )

        return visual_chunks

    @staticmethod
    def _figure_sort_key(
        figure: FigureArtifact,
    ) -> tuple[
        int,
        int,
        float,
        float,
        str,
    ]:
        """
        Deterministic figure order used by ingestion/chunking.

        LLM output never determines positional order.
        """

        return (
            (
                figure.document_figure_index
                if figure.document_figure_index
                is not None
                else 10**9
            ),
            (
                figure.page_figure_index
                if figure.page_figure_index
                is not None
                else 10**9
            ),
            figure.bbox.y0,
            figure.bbox.x0,
            figure.figure_id,
        )

    def _section_figure_ids(
        self,
        *,
        section_text: str,
        section_blocks: list[
            LayoutBlock
        ],
        page_figures: list[
            FigureArtifact
        ],
        single_section: bool,
    ) -> list[str]:
        """
        Build bounded section-level figure candidates.

        Exact source-label references are strongest. Spatial membership is
        the generic fallback. A one-section page may retain every page figure
        at the parent level for broad context, while child chunks below are
        linked more precisely.
        """

        explicit = [
            figure.figure_id
            for figure in page_figures
            if (
                figure.exact_source_label
                and self._text_mentions_figure_label(
                    text=section_text,
                    source_label=(
                        figure.exact_source_label
                    ),
                )
            )
        ]

        if explicit:
            return list(
                dict.fromkeys(
                    explicit
                )
            )

        spatial = [
            figure.figure_id
            for figure in page_figures
            if self._figure_belongs_to_section(
                figure_bbox=figure.bbox,
                section_blocks=section_blocks,
            )
        ]

        if spatial:
            return list(
                dict.fromkeys(
                    spatial
                )
            )

        if single_section:
            return [
                figure.figure_id
                for figure in page_figures
            ]

        return []

    def _resolve_child_figure_ids(
        self,
        *,
        chunk_text: str,
        chunk_blocks: list[
            LayoutBlock
        ],
        content_type: str,
        section_figure_ids: list[str],
        page_figures: list[
            FigureArtifact
        ],
    ) -> list[str]:
        """
        Link a child text chunk only to figures that are plausibly needed.

        Resolution order:
        1. explicit source label in the text;
        2. one unambiguous nearby section figure for a question/worked example;
        3. nearest figure when the text contains a generic visual reference.

        No Physics-topic names or semantic tags are hard-coded here.
        """

        if not page_figures:
            return []

        ordered = sorted(
            page_figures,
            key=self._figure_sort_key,
        )

        explicit = [
            figure.figure_id
            for figure in ordered
            if (
                figure.exact_source_label
                and self._text_mentions_figure_label(
                    text=chunk_text,
                    source_label=(
                        figure.exact_source_label
                    ),
                )
            )
        ]

        if explicit:
            return list(
                dict.fromkeys(
                    explicit
                )
            )

        section_candidates = [
            figure
            for figure in ordered
            if figure.figure_id
            in section_figure_ids
        ]

        structural_problem = (
            content_type
            in {
                LayoutBlockType.QUESTION.value,
                LayoutBlockType.WORKED_EXAMPLE.value,
            }
        )

        has_visual_reference = (
            self._contains_generic_visual_reference(
                chunk_text
            )
        )

        if (
            structural_problem
            and len(section_candidates) == 1
        ):
            candidate = section_candidates[0]

            if self._figure_is_near_blocks(
                figure=candidate,
                blocks=chunk_blocks,
            ):
                return [
                    candidate.figure_id
                ]

        if (
            has_visual_reference
            or structural_problem
        ):
            candidates = (
                section_candidates
                if section_candidates
                else ordered
            )

            nearest = self._nearest_figure_to_blocks(
                figures=candidates,
                blocks=chunk_blocks,
            )

            if nearest is not None:
                return [
                    nearest.figure_id
                ]

        return []

    @staticmethod
    def _contains_generic_visual_reference(
        text: str,
    ) -> bool:
        """
        Detect only generic visual-reference wording.

        This is structural language detection, not a Physics-topic mapping.
        """

        return bool(
            re.search(
                (
                    r"(?i)\b("
                    r"fig(?:ure)?\.?|"
                    r"diagram|image|graph|plot|"
                    r"illustration|drawing|"
                    r"shown\s+(?:above|below|here)|"
                    r"as\s+shown|see\s+(?:the\s+)?"
                    r"(?:fig(?:ure)?|diagram|image|graph)"
                    r")\b"
                ),
                text,
            )
        )

    @staticmethod
    def _figure_label_key(
        value: str,
    ) -> str | None:
        """
        Canonicalize only the identifier part of a source figure label.

        Examples:
            "Fig. 3" -> "3"
            "Figure 2.1" -> "2.1"

        The original source label remains stored unchanged elsewhere.
        """

        match = re.search(
            (
                r"(?i)(?<![A-Za-z])"
                r"(?:fig(?:ure)?\.?)\s*"
                r"(?:no\.?\s*)?"
                r"([A-Za-z]?\d+(?:\.\d+)*"
                r"(?:\s*\([A-Za-z0-9]+\))?)"
                r"(?![A-Za-z0-9])"
            ),
            value,
        )

        if match is None:
            return None

        return re.sub(
            r"\s+",
            "",
            match.group(1),
        ).casefold()

    @classmethod
    def _text_mentions_figure_label(
        cls,
        *,
        text: str,
        source_label: str,
    ) -> bool:
        label_key = cls._figure_label_key(
            source_label
        )

        if not label_key:
            return False

        for match in re.finditer(
            (
                r"(?i)(?<![A-Za-z])"
                r"(?:fig(?:ure)?\.?)\s*"
                r"(?:no\.?\s*)?"
                r"([A-Za-z]?\d+(?:\.\d+)*"
                r"(?:\s*\([A-Za-z0-9]+\))?)"
                r"(?![A-Za-z0-9])"
            ),
            text,
        ):
            candidate_key = re.sub(
                r"\s+",
                "",
                match.group(1),
            ).casefold()

            if candidate_key == label_key:
                return True

        return False

    @staticmethod
    def _figure_block_distance(
        *,
        figure: FigureArtifact,
        block: LayoutBlock,
    ) -> float:
        if (
            block.bbox.y0
            >= figure.bbox.y1
        ):
            vertical_gap = (
                block.bbox.y0
                - figure.bbox.y1
            )
        elif (
            block.bbox.y1
            <= figure.bbox.y0
        ):
            vertical_gap = (
                figure.bbox.y0
                - block.bbox.y1
            )
        else:
            vertical_gap = 0.0

        if (
            block.bbox.x0
            >= figure.bbox.x1
        ):
            horizontal_gap = (
                block.bbox.x0
                - figure.bbox.x1
            )
        elif (
            block.bbox.x1
            <= figure.bbox.x0
        ):
            horizontal_gap = (
                figure.bbox.x0
                - block.bbox.x1
            )
        else:
            horizontal_gap = 0.0

        # Vertical proximity is normally more meaningful for textbook flow;
        # horizontal distance is a smaller tie-breaker.
        return (
            vertical_gap
            + 0.25 * horizontal_gap
        )

    def _figure_is_near_blocks(
        self,
        *,
        figure: FigureArtifact,
        blocks: list[
            LayoutBlock
        ],
        maximum_distance: float = 220.0,
    ) -> bool:
        if not blocks:
            return False

        distance = min(
            self._figure_block_distance(
                figure=figure,
                block=block,
            )
            for block in blocks
        )

        return distance <= maximum_distance

    def _nearest_figure_to_blocks(
        self,
        *,
        figures: list[
            FigureArtifact
        ],
        blocks: list[
            LayoutBlock
        ],
        maximum_distance: float = 220.0,
    ) -> FigureArtifact | None:
        if not figures or not blocks:
            return None

        ranked = sorted(
            (
                (
                    min(
                        self._figure_block_distance(
                            figure=figure,
                            block=block,
                        )
                        for block in blocks
                    ),
                    self._figure_sort_key(
                        figure
                    ),
                    figure,
                )
                for figure in figures
            ),
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        best_distance, _, best = (
            ranked[0]
        )

        if best_distance > maximum_distance:
            return None

        # If two figures are essentially equally near, do not guess unless
        # an exact source label already resolved the reference above.
        if len(ranked) >= 2:
            second_distance = ranked[1][0]

            if abs(
                second_distance
                - best_distance
            ) <= 12.0:
                return None

        return best

    def _best_parent_for_figure(
        self,
        *,
        figure_id: str,
        parents: list[ParentChunk],
    ) -> ParentChunk | None:
        for parent in parents:
            if figure_id in parent.figures:
                return parent

        if parents:
            return parents[0]

        return None

    def _figure_belongs_to_section(
        self,
        *,
        figure_bbox,
        section_blocks: list[
            LayoutBlock
        ],
    ) -> bool:
        if not section_blocks:
            return False

        min_y = min(
            block.bbox.y0
            for block in section_blocks
        )

        max_y = max(
            block.bbox.y1
            for block in section_blocks
        )

        figure_center = (
            figure_bbox.y0
            + figure_bbox.y1
        ) / 2.0

        margin = 80.0

        return (
            min_y - margin
            <= figure_center
            <= max_y + margin
        )

    def _link_structural_evidence(
        self,
        *,
        document_structure: DocumentStructure,
        parent_chunks: list[ParentChunk],
        retrieval_chunks: list[RetrievalChunk],
    ) -> None:
        """
        Link structural nodes and existing semantic chunks in both directions.

        Strong source-block provenance wins. Text matching is only a bounded
        fallback on pages already covered by the structural node. A weak match
        is left unresolved instead of being forced.

        `document_structure` is updated in place so the ingestion service can
        persist the same object after chunking without changing ChunkingResult's
        established public contract.
        """

        parents_by_id = {
            parent.parent_id: parent
            for parent in parent_chunks
        }

        child_chunks_by_page: defaultdict[
            int,
            list[RetrievalChunk],
        ] = defaultdict(list)

        visual_chunks_by_page: defaultdict[
            int,
            list[RetrievalChunk],
        ] = defaultdict(list)

        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in retrieval_chunks
        }

        for chunk in retrieval_chunks:
            if chunk.chunk_kind == "visual":
                visual_chunks_by_page[
                    chunk.page_number
                ].append(chunk)
            else:
                child_chunks_by_page[
                    chunk.page_number
                ].append(chunk)

        figure_chunk_by_node_id = (
            self._positional_figure_chunk_links(
                nodes=document_structure.nodes,
                visual_chunks_by_page=visual_chunks_by_page,
            )
        )

        nodes_by_id = {
            node.node_id: node
            for node in document_structure.nodes
        }

        for node in sorted(
            document_structure.nodes,
            key=lambda item: (
                item.document_order,
                item.node_id,
            ),
        ):
            linked_chunks: list[RetrievalChunk]

            if node.node_type == StructuralNodeType.FIGURE:
                linked_chunks = self._figure_chunks_for_node(
                    node=node,
                    visual_chunks_by_page=visual_chunks_by_page,
                    positional_chunk_id=(
                        figure_chunk_by_node_id.get(
                            node.node_id
                        )
                    ),
                    chunks_by_id=chunks_by_id,
                )
            else:
                linked_chunks = self._text_chunks_for_node(
                    node=node,
                    child_chunks_by_page=child_chunks_by_page,
                )

            linked_chunk_ids = [
                chunk.chunk_id
                for chunk in linked_chunks
            ]

            linked_parent_ids = list(
                dict.fromkeys(
                    chunk.parent_id
                    for chunk in linked_chunks
                    if chunk.parent_id
                    and chunk.parent_id in parents_by_id
                )
            )

            linked_parent_ids = list(
                dict.fromkeys(
                    linked_parent_ids
                    + self._heading_parent_ids(
                        node=node,
                        parent_chunks=parent_chunks,
                    )
                )
            )

            linked_figure_ids = list(
                dict.fromkeys(
                    figure_id
                    for chunk in linked_chunks
                    for figure_id in chunk.linked_figure_ids
                )
            )

            node.linked_retrieval_chunk_ids = (
                linked_chunk_ids
            )
            node.linked_parent_chunk_ids = (
                linked_parent_ids
            )
            node.linked_figure_ids = list(
                dict.fromkeys(
                    node.linked_figure_ids
                    + linked_figure_ids
                )
            )

            for chunk in linked_chunks:
                chunk.structural_node_ids = list(
                    dict.fromkeys(
                        chunk.structural_node_ids
                        + [node.node_id]
                    )
                )

            for parent_id in linked_parent_ids:
                parent = parents_by_id[parent_id]
                parent.structural_node_ids = list(
                    dict.fromkeys(
                        parent.structural_node_ids
                        + [node.node_id]
                    )
                )

        # A problem/point may depend on a separate figure node. Propagate the
        # canonical FigureArtifact IDs recovered through that visual node.
        for node in document_structure.nodes:
            related_figure_ids: list[str] = []

            for visual_node_id in node.related_visual_node_ids:
                visual_node = nodes_by_id.get(
                    visual_node_id
                )

                if visual_node is not None:
                    related_figure_ids.extend(
                        visual_node.linked_figure_ids
                    )

            node.linked_figure_ids = list(
                dict.fromkeys(
                    node.linked_figure_ids
                    + related_figure_ids
                )
            )

        # Parent links also inherit every structural ID carried by their
        # existing child chunks. This retains the old parent/child hierarchy.
        for chunk in retrieval_chunks:
            if (
                chunk.parent_id
                and chunk.parent_id in parents_by_id
            ):
                parent = parents_by_id[
                    chunk.parent_id
                ]
                parent.structural_node_ids = list(
                    dict.fromkeys(
                        parent.structural_node_ids
                        + chunk.structural_node_ids
                    )
                )

    def _text_chunks_for_node(
        self,
        *,
        node: StructuralNode,
        child_chunks_by_page: dict[
            int,
            list[RetrievalChunk],
        ],
    ) -> list[RetrievalChunk]:
        candidates = [
            chunk
            for page_number in self._node_page_numbers(
                node
            )
            for chunk in child_chunks_by_page.get(
                page_number,
                [],
            )
        ]

        if not candidates:
            return []

        source_block_ids = {
            block_id
            for span in node.source_spans
            for block_id in span.block_ids
        }

        direct_matches = [
            chunk
            for chunk in candidates
            if source_block_ids.intersection(
                chunk.source_block_ids
            )
        ]

        if direct_matches:
            return self._deduplicate_chunks(
                direct_matches
            )

        ranked: list[
            tuple[float, str, RetrievalChunk]
        ] = []

        for chunk in candidates:
            score = self._structural_text_match_score(
                node_text=node.text,
                chunk_text=chunk.text,
            )

            if score >= self.structural_text_match_minimum:
                ranked.append(
                    (
                        score,
                        chunk.chunk_id,
                        chunk,
                    )
                )

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            item[2]
            for item in ranked[
                : self.max_structural_fallback_links
            ]
        ]

    def _figure_chunks_for_node(
        self,
        *,
        node: StructuralNode,
        visual_chunks_by_page: dict[
            int,
            list[RetrievalChunk],
        ],
        positional_chunk_id: str | None,
        chunks_by_id: dict[str, RetrievalChunk],
    ) -> list[RetrievalChunk]:
        candidates = [
            chunk
            for page_number in self._node_page_numbers(
                node
            )
            for chunk in visual_chunks_by_page.get(
                page_number,
                [],
            )
        ]

        if not candidates:
            return []

        visible_references = [
            value
            for value in (
                node.exact_source_label,
                node.label,
                *node.visual_labels,
            )
            if value and value.strip()
        ]

        label_matches = [
            chunk
            for chunk in candidates
            if any(
                self._normalized_phrase_occurs(
                    phrase=reference,
                    text=chunk.text,
                )
                for reference in visible_references
            )
        ]

        if label_matches:
            return self._deduplicate_chunks(
                label_matches
            )

        if positional_chunk_id:
            positional_chunk = chunks_by_id.get(
                positional_chunk_id
            )

            if positional_chunk is not None:
                return [positional_chunk]

        if len(candidates) == 1:
            return candidates

        return []

    def _positional_figure_chunk_links(
        self,
        *,
        nodes: list[StructuralNode],
        visual_chunks_by_page: dict[
            int,
            list[RetrievalChunk],
        ],
    ) -> dict[str, str]:
        figure_nodes_by_page: defaultdict[
            int,
            list[StructuralNode],
        ] = defaultdict(list)

        for node in nodes:
            if node.node_type != StructuralNodeType.FIGURE:
                continue

            page_numbers = self._node_page_numbers(
                node
            )

            if page_numbers:
                figure_nodes_by_page[
                    page_numbers[0]
                ].append(node)

        result: dict[str, str] = {}

        for page_number, page_nodes in (
            figure_nodes_by_page.items()
        ):
            ordered_nodes = sorted(
                page_nodes,
                key=lambda item: (
                    item.document_order,
                    item.node_id,
                ),
            )

            page_chunks = visual_chunks_by_page.get(
                page_number,
                [],
            )

            if len(ordered_nodes) != len(page_chunks):
                continue

            for node, chunk in zip(
                ordered_nodes,
                page_chunks,
            ):
                result[node.node_id] = chunk.chunk_id

        return result

    def _heading_parent_ids(
        self,
        *,
        node: StructuralNode,
        parent_chunks: list[ParentChunk],
    ) -> list[str]:
        if node.node_type not in {
            StructuralNodeType.TITLE,
            StructuralNodeType.HEADING,
            StructuralNodeType.SUBHEADING,
            StructuralNodeType.SECTION,
        }:
            return []

        node_heading = (
            node.title
            or node.label
            or node.text
        )

        normalized_node_heading = (
            self._normalize_for_dedup(
                node_heading
            )
        )

        if not normalized_node_heading:
            return []

        page_numbers = set(
            self._node_page_numbers(node)
        )

        return [
            parent.parent_id
            for parent in parent_chunks
            if parent.page_number in page_numbers
            and parent.heading
            and self._normalize_for_dedup(
                parent.heading
            )
            == normalized_node_heading
        ]

    def _structural_text_match_score(
        self,
        *,
        node_text: str,
        chunk_text: str,
    ) -> float:
        normalized_node = self._normalize_for_dedup(
            node_text
        )
        normalized_chunk = self._normalize_for_dedup(
            chunk_text
        )

        if not normalized_node or not normalized_chunk:
            return 0.0

        if normalized_node in normalized_chunk:
            return 1.0

        if normalized_chunk in normalized_node:
            return 0.95

        node_tokens = self._structural_tokens(
            normalized_node
        )
        chunk_tokens = self._structural_tokens(
            normalized_chunk
        )

        if not node_tokens or not chunk_tokens:
            return 0.0

        common_tokens = node_tokens.intersection(
            chunk_tokens
        )

        if len(common_tokens) < 3:
            return 0.0

        return len(common_tokens) / min(
            len(node_tokens),
            len(chunk_tokens),
        )

    @staticmethod
    def _structural_tokens(
        normalized_text: str,
    ) -> set[str]:
        return {
            token
            for token in re.findall(
                r"\w+",
                normalized_text,
                flags=re.UNICODE,
            )
            if len(token) >= 2
        }

    @staticmethod
    def _normalized_phrase_occurs(
        *,
        phrase: str,
        text: str,
    ) -> bool:
        normalized_phrase = " ".join(
            phrase.casefold().split()
        )
        normalized_text = " ".join(
            text.casefold().split()
        )

        return bool(
            normalized_phrase
            and normalized_phrase in normalized_text
        )

    @staticmethod
    def _node_page_numbers(
        node: StructuralNode,
    ) -> list[int]:
        from_spans = sorted(
            {
                span.page_number
                for span in node.source_spans
            }
        )

        if from_spans:
            return from_spans

        if (
            node.page_start is not None
            and node.page_end is not None
            and node.page_end >= node.page_start
        ):
            return list(
                range(
                    node.page_start,
                    node.page_end + 1,
                )
            )

        return []

    @staticmethod
    def _deduplicate_chunks(
        chunks: list[RetrievalChunk],
    ) -> list[RetrievalChunk]:
        result: list[RetrievalChunk] = []
        seen: set[str] = set()

        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue

            seen.add(chunk.chunk_id)
            result.append(chunk)

        return result

    def _full_page_bbox(
        self,
        parsed_page,
    ):
        from src.ingestion.models import (
            BoundingBox,
        )

        return BoundingBox(
            x0=0.0,
            y0=0.0,
            x1=parsed_page.width,
            y1=parsed_page.height,
        )
