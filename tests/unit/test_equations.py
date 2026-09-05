from __future__ import annotations

import unittest

from src.ingestion.chunker import (
    HierarchicalChunker,
)
from src.ingestion.equations import (
    EquationArtifact,
    EquationExtractor,
)
from src.ingestion.models import (
    BoundingBox,
    LayoutBlock,
    LayoutBlockType,
)


def make_block(
    *,
    block_id: str,
    block_number: int,
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    block_type: LayoutBlockType = (
        LayoutBlockType.EQUATION
    ),
) -> LayoutBlock:
    return LayoutBlock(
        block_id=block_id,
        page_number=2,
        block_number=block_number,
        block_type=block_type,
        bbox=BoundingBox(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        ),
        text=text,
        source="native",
        confidence=0.65,
    )


class EquationExtractorHeuristicTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.extractor = EquationExtractor(
            api_key=None
        )

    def test_s_double_bang_needs_visual_recovery(
        self,
    ) -> None:
        self.assertTrue(
            self.extractor.needs_visual_recovery(
                "x(t) = 7.40 cm cos(4.16 s!! t - 2.42)"
            )
        )

    def test_repeated_placeholders_need_visual_recovery(
        self,
    ) -> None:
        self.assertTrue(
            self.extractor.needs_visual_recovery(
                "omega = ! ; f = !! ; ! ! !"
            )
        )

    def test_single_bang_math_fragment_needs_recovery(
        self,
    ) -> None:
        self.assertTrue(
            self.extractor.needs_visual_recovery(
                "! ; f ="
            )
        )

    def test_fragmented_multiline_fraction_needs_recovery(
        self,
    ) -> None:
        self.assertTrue(
            self.extractor.needs_visual_recovery(
                "2π = 1\n2π\ng\nL = 1"
            )
        )

    def test_clean_equation_does_not_need_recovery(
        self,
    ) -> None:
        self.assertFalse(
            self.extractor.needs_visual_recovery(
                "F = ma"
            )
        )

    def test_plain_exclamation_prose_does_not_trigger(
        self,
    ) -> None:
        self.assertFalse(
            self.extractor.needs_visual_recovery(
                "Great! This motion is periodic."
            )
        )

    def test_corrupted_math_paragraph_is_candidate(
        self,
    ) -> None:
        block = make_block(
            block_id="p2_b3",
            block_number=3,
            text=(
                "spring constant k; omega = ! ; "
                "f = !! ; T = !"
            ),
            x0=20,
            y0=100,
            x1=400,
            y1=125,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        self.assertTrue(
            self.extractor.is_equation_candidate(
                block
            )
        )

    def test_adjacent_formula_fragments_form_one_region(
        self,
    ) -> None:
        blocks = [
            make_block(
                block_id="p2_b5",
                block_number=5,
                text="ω =",
                x0=50,
                y0=100,
                x1=90,
                y1=120,
            ),
            make_block(
                block_id="p2_b7",
                block_number=7,
                text="! ; f =",
                x0=105,
                y0=100,
                x1=175,
                y1=120,
            ),
            make_block(
                block_id="p2_b9",
                block_number=9,
                text="!! =",
                x0=185,
                y0=100,
                x1=230,
                y1=120,
            ),
        ]

        regions = (
            self.extractor
            ._build_recovery_regions(
                blocks=blocks,
                page_width=612,
                page_height=792,
            )
        )

        self.assertEqual(
            len(regions),
            1,
        )

        self.assertEqual(
            {
                block.block_id
                for block in regions[0]
            },
            {
                "p2_b5",
                "p2_b7",
                "p2_b9",
            },
        )

    def test_distant_equations_are_not_merged(
        self,
    ) -> None:
        blocks = [
            make_block(
                block_id="p2_b5",
                block_number=5,
                text="! ; f =",
                x0=50,
                y0=100,
                x1=150,
                y1=120,
            ),
            make_block(
                block_id="p2_b50",
                block_number=50,
                text="!! =",
                x0=50,
                y0=500,
                x1=150,
                y1=520,
            ),
        ]

        regions = (
            self.extractor
            ._build_recovery_regions(
                blocks=blocks,
                page_width=612,
                page_height=792,
            )
        )

        self.assertEqual(
            len(regions),
            2,
        )

    def test_native_equation_keeps_full_expression(
        self,
    ) -> None:
        equations = (
            self.extractor
            ._native_equations(
                "k = 120 N/m; f = 6.00 Hz"
            )
        )

        self.assertEqual(
            equations,
            [
                "k = 120 N/m; f = 6.00 Hz"
            ],
        )


class EquationChunkerIntegrationTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.chunker = HierarchicalChunker()

    def test_successful_multiblock_region_replaces_once(
        self,
    ) -> None:
        blocks = [
            make_block(
                block_id="p2_b5",
                block_number=5,
                text="ω =",
                x0=50,
                y0=100,
                x1=90,
                y1=120,
            ),
            make_block(
                block_id="p2_b7",
                block_number=7,
                text="! ; f =",
                x0=105,
                y0=100,
                x1=175,
                y1=120,
            ),
            make_block(
                block_id="p2_b9",
                block_number=9,
                text="!! =",
                x0=185,
                y0=100,
                x1=230,
                y1=120,
            ),
        ]

        artifact = EquationArtifact(
            artifact_id="eq_2_region_1",
            page_number=2,
            source_block_id="p2_b5",
            source_block_ids=[
                "p2_b5",
                "p2_b7",
                "p2_b9",
            ],
            original_text=(
                "ω =\n! ; f =\n!! ="
            ),
            transcribed_text=(
                "ω = 2πf; "
                "f = ω/(2π)"
            ),
            equations=[
                "ω = 2πf",
                "f = ω/(2π)",
            ],
            confidence=0.99,
            extraction_method=(
                "openai_vision_region"
            ),
            region_kind="multi_block",
            replacement_safe=True,
            suppress_original_on_failure=True,
            formula_dominant=True,
        )

        artifact_map = {
            block_id: artifact
            for block_id in (
                artifact.source_block_ids
            )
        }

        (
            corrected,
            provenance,
        ) = (
            self.chunker
            ._apply_equation_transcriptions(
                blocks=blocks,
                equations_by_block=(
                    artifact_map
                ),
            )
        )

        self.assertEqual(
            len(corrected),
            1,
        )

        self.assertEqual(
            corrected[0].text,
            "ω = 2πf; f = ω/(2π)",
        )

        self.assertEqual(
            provenance[
                corrected[0].block_id
            ],
            [
                "p2_b5",
                "p2_b7",
                "p2_b9",
            ],
        )

    def test_failed_formula_region_suppresses_garbage(
        self,
    ) -> None:
        blocks = [
            make_block(
                block_id="p2_b7",
                block_number=7,
                text="! ; f =",
                x0=105,
                y0=100,
                x1=175,
                y1=120,
            ),
            make_block(
                block_id="p2_b9",
                block_number=9,
                text="!! =",
                x0=185,
                y0=100,
                x1=230,
                y1=120,
            ),
        ]

        artifact = EquationArtifact(
            artifact_id="eq_2_region_1",
            page_number=2,
            source_block_id="p2_b7",
            source_block_ids=[
                "p2_b7",
                "p2_b9",
            ],
            original_text=(
                "! ; f =\n!! ="
            ),
            transcribed_text="",
            equations=[],
            confidence=0.0,
            extraction_method=(
                "openai_vision_region"
            ),
            region_kind="multi_block",
            replacement_safe=False,
            suppress_original_on_failure=True,
            formula_dominant=True,
        )

        artifact_map = {
            block_id: artifact
            for block_id in (
                artifact.source_block_ids
            )
        }

        (
            corrected,
            provenance,
        ) = (
            self.chunker
            ._apply_equation_transcriptions(
                blocks=blocks,
                equations_by_block=(
                    artifact_map
                ),
            )
        )

        self.assertEqual(
            corrected,
            [],
        )

        self.assertEqual(
            provenance,
            {},
        )


    def test_orphan_equation_fragment_is_dropped(
        self,
    ) -> None:
        block = make_block(
            block_id="p1_b18",
            block_number=18,
            text="𝑇",
            x0=100,
            y0=100,
            x1=110,
            y1=120,
        )

        cleaned = (
            self.chunker
            ._sanitize_unrecovered_block(
                block
            )
        )

        self.assertIsNone(
            cleaned
        )

    def test_short_placeholder_debris_is_dropped(
        self,
    ) -> None:
        block = make_block(
            block_id="p2_b10",
            block_number=10,
            text='!"#',
            x0=100,
            y0=100,
            x1=130,
            y1=120,
        )

        cleaned = (
            self.chunker
            ._sanitize_unrecovered_block(
                block
            )
        )

        self.assertIsNone(
            cleaned
        )

    def test_readable_prose_survives_corrupt_symbol(
        self,
    ) -> None:
        block = make_block(
            block_id="p2_b30",
            block_number=30,
            text=(
                "𝑣!: velocity of mass m at x (m/s)"
            ),
            x0=100,
            y0=100,
            x1=300,
            y1=120,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        cleaned = (
            self.chunker
            ._sanitize_unrecovered_block(
                block
            )
        )

        self.assertIsNotNone(
            cleaned
        )

        self.assertNotIn(
            "!",
            cleaned.text
        )

        self.assertIn(
            "velocity of mass",
            cleaned.text
        )

    def test_child_ids_are_unique_across_sections(
        self,
    ) -> None:
        block = make_block(
            block_id="p2_b1",
            block_number=1,
            text="Simple harmonic motion repeats.",
            x0=50,
            y0=50,
            x1=300,
            y1=80,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        first = (
            self.chunker
            ._create_child_chunks(
                user_id="local-user",
                document_id="doc1",
                page_number=2,
                parent_id="doc1_p2_section1",
                heading="Section 1",
                blocks=[block],
                topics=[],
                grade_min=6,
                grade_max=12,
                linked_figure_ids=[],
                source_ids_by_block={
                    block.block_id: [
                        block.block_id
                    ]
                },
            )
        )

        second = (
            self.chunker
            ._create_child_chunks(
                user_id="local-user",
                document_id="doc1",
                page_number=2,
                parent_id="doc1_p2_section2",
                heading="Section 2",
                blocks=[block],
                topics=[],
                grade_min=6,
                grade_max=12,
                linked_figure_ids=[],
                source_ids_by_block={
                    block.block_id: [
                        block.block_id
                    ]
                },
            )
        )

        self.assertNotEqual(
            first[0].chunk_id,
            second[0].chunk_id,
        )


    def test_single_math_glyph_paragraph_is_dropped(
        self,
    ) -> None:
        block = make_block(
            block_id="p1_b18",
            block_number=18,
            text="𝑇",
            x0=100,
            y0=100,
            x1=110,
            y1=120,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        cleaned = (
            self.chunker
            ._sanitize_unrecovered_block(
                block
            )
        )

        self.assertIsNone(
            cleaned
        )

    def test_decorative_separator_is_dropped(
        self,
    ) -> None:
        block = make_block(
            block_id="p3_b22",
            block_number=22,
            text="-\u00ad-\u00ad",
            x0=100,
            y0=100,
            x1=130,
            y1=120,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        cleaned = (
            self.chunker
            ._sanitize_unrecovered_block(
                block
            )
        )

        self.assertIsNone(
            cleaned
        )

    def test_duplicate_short_label_next_to_vision_is_removed(
        self,
    ) -> None:
        native = make_block(
            block_id="p2_b56",
            block_number=56,
            text="Solution:",
            x0=50,
            y0=100,
            x1=120,
            y1=120,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        vision = make_block(
            block_id="p2_b57",
            block_number=57,
            text=(
                "Solution:\n"
                "k = 120 N/m; f = 6.00 Hz"
            ),
            x0=50,
            y0=122,
            x1=350,
            y1=180,
            block_type=(
                LayoutBlockType.EQUATION
            ),
        ).model_copy(
            update={
                "source": "vision",
                "confidence": 0.99,
            }
        )

        cleaned = (
            self.chunker
            ._deduplicate_overlapping_blocks(
                [
                    native,
                    vision,
                ]
            )
        )

        self.assertEqual(
            len(cleaned),
            1,
        )

        self.assertEqual(
            cleaned[0].source,
            "vision",
        )

    def test_adjacent_duplicate_lines_are_collapsed(
        self,
    ) -> None:
        cleaned = (
            self.chunker
            ._clean_assembled_text(
                "Solution:\n"
                "Solution:\n"
                "k = 120 N/m"
            )
        )

        self.assertEqual(
            cleaned,
            "Solution:\nk = 120 N/m",
        )


class EquationVisionOutputHygieneTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.extractor = EquationExtractor(
            api_key=None
        )
        self.chunker = HierarchicalChunker()

    def test_vision_meta_commentary_is_removed(
        self,
    ) -> None:
        cleaned = (
            self.extractor
            ._strip_vision_meta_commentary(
                "\\\\omega = "
                "\\\\sqrt{g/L} "
                "\\\\text{(and continuation "
                "not shown in image)}"
            )
        )

        self.assertIn(
            "\\\\omega",
            cleaned,
        )

        self.assertNotIn(
            "continuation",
            cleaned.lower(),
        )

        self.assertNotIn(
            "image",
            cleaned.lower(),
        )


    def test_real_unicode_separator_variant_is_dropped(
        self,
    ) -> None:
        block = make_block(
            block_id="p3_b22_real",
            block_number=22,
            text="-\u00ad\u2010-\u00ad\u2010",
            x0=100,
            y0=100,
            x1=140,
            y1=120,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        cleaned = (
            self.chunker
            ._sanitize_unrecovered_block(
                block
            )
        )

        self.assertIsNone(
            cleaned
        )

    def test_native_word_separators_become_spaces(
        self,
    ) -> None:
        block = make_block(
            block_id="p1_b1_ws",
            block_number=1,
            text=(
                "SIMPLE\t\r \u00a0HARMONIC\t\r "
                "\u00a0MOTION"
            ),
            x0=50,
            y0=50,
            x1=300,
            y1=80,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        normalized = (
            self.chunker
            ._normalize_block_text(
                block
            )
        )

        self.assertEqual(
            normalized,
            "SIMPLE HARMONIC MOTION",
        )

    def test_vision_line_breaks_are_preserved(
        self,
    ) -> None:
        block = make_block(
            block_id="p2_b16_vision",
            block_number=16,
            text=(
                "omega = sqrt(k/m)\n"
                "\n"
                "x = A cos(omega t)"
            ),
            x0=50,
            y0=100,
            x1=350,
            y1=180,
            block_type=(
                LayoutBlockType.EQUATION
            ),
        ).model_copy(
            update={
                "source": "vision",
                "confidence": 0.99,
            }
        )

        normalized = (
            self.chunker
            ._normalize_block_text(
                block
            )
        )

        self.assertEqual(
            normalized,
            (
                "omega = sqrt(k/m)\n\n"
                "x = A cos(omega t)"
            ),
        )

    def test_nearby_native_label_inside_vision_is_removed(
        self,
    ) -> None:
        vision = make_block(
            block_id="p2_b16_vision2",
            block_number=16,
            text=(
                "E: \\text{mechanical energy "
                "of the system}\n"
                "\\textbf{B/ Simple pendulum}"
            ),
            x0=50,
            y0=100,
            x1=400,
            y1=220,
            block_type=(
                LayoutBlockType.EQUATION
            ),
        ).model_copy(
            update={
                "source": "vision",
                "confidence": 0.99,
            }
        )

        native_energy = make_block(
            block_id="p2_b30_label",
            block_number=30,
            text=(
                "E: mechanical energy "
                "of the system"
            ),
            x0=50,
            y0=225,
            x1=300,
            y1=245,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        native_heading = make_block(
            block_id="p2_b32_label",
            block_number=32,
            text="B/ Simple pendulum",
            x0=50,
            y0=250,
            x1=220,
            y1=270,
            block_type=(
                LayoutBlockType.PARAGRAPH
            ),
        )

        cleaned = (
            self.chunker
            ._deduplicate_overlapping_blocks(
                [
                    vision,
                    native_energy,
                    native_heading,
                ]
            )
        )

        self.assertEqual(
            [
                block.block_id
                for block in cleaned
            ],
            [
                "p2_b16_vision2",
            ],
        )


if __name__ == "__main__":
    unittest.main()