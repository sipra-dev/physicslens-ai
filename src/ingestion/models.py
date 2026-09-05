from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LayoutBlockType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"

    # Structural list items.
    BULLET_ITEM = "bullet_item"
    NUMBERED_ITEM = "numbered_item"

    EQUATION = "equation"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    WORKED_EXAMPLE = "worked_example"
    QUESTION = "question"
    ANSWER = "answer"
    MARGIN_NOTE = "margin_note"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ParsedBlock(BaseModel):
    block_id: str
    page_number: int
    block_number: int

    block_type: Literal[
        "text",
        "image",
    ]

    bbox: BoundingBox
    text: str = ""

    source: Literal[
        "native",
        "ocr",
        "image",
    ] = "native"


class ParsedPage(BaseModel):
    page_number: int
    width: float
    height: float
    rendered_width: int
    rendered_height: int
    rendered_image_path: str
    native_text: str
    native_text_length: int
    blocks: list[ParsedBlock]
    requires_ocr: bool


class ParsedDocument(BaseModel):
    document_id: str
    source_path: str
    file_extension: str
    page_count: int
    pages: list[ParsedPage]


class LayoutBlock(BaseModel):
    block_id: str
    page_number: int
    block_number: int
    block_type: LayoutBlockType
    bbox: BoundingBox
    text: str
    source: str

    confidence: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )


class PageLayout(BaseModel):
    page_number: int
    blocks: list[LayoutBlock]


class DocumentLayout(BaseModel):
    document_id: str
    pages: list[PageLayout]


class OCRWord(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox


class OCRPageResult(BaseModel):
    page_number: int
    attempted: bool
    available: bool
    used: bool
    text: str

    average_confidence: float | None = None

    words: list[OCRWord] = Field(
        default_factory=list
    )

    error: str | None = None


class OCRDocumentResult(BaseModel):
    document_id: str
    engine: str
    pages: list[OCRPageResult]

    @property
    def combined_text(self) -> str:
        return "\n\n".join(
            page.text.strip()
            for page in self.pages
            if page.text.strip()
        )


class FigureArtifact(BaseModel):
    figure_id: str

    page_number: int = Field(
        ge=1
    )

    bbox: BoundingBox

    # Local path of the canonical saved crop.
    image_path: str

    # Existing chunking/retrieval code may still use this.
    caption: str | None = None

    # Exact source blocks associated with this visual.
    linked_block_ids: list[str] = Field(
        default_factory=list
    )

    extraction_method: Literal[
        "embedded_image_block",
        "full_page_image",
        "rendered_page_visual_region",
        "vision_page_visual_region",
    ]

    # Deterministic figure order.
    document_figure_index: int | None = Field(
        default=None,
        ge=1,
    )

    page_figure_index: int | None = Field(
        default=None,
        ge=1,
    )

    # Only populated when an explicit label is visible.
    exact_source_label: str | None = None

    # What the source visual visibly shows.
    source_description: str | None = None

    # Standard Physics knowledge used for explanation.
    standard_physics_explanation: str | None = None

    # Interpretation based on source evidence and Physics.
    derived_interpretation: str | None = None

    # Labels, symbols, axis names and values seen in the visual.
    visible_labels: list[str] = Field(
        default_factory=list
    )

    vision_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # Model used for visual understanding.
    visual_model: str | None = None

    # Backward-compatible searchable visual description.
    semantic_description: str | None = None

    # Nearby text for retrieval and structural linking.
    nearby_text: str | None = None


class FigureExtractionResult(BaseModel):
    document_id: str

    figures: list[FigureArtifact] = Field(
        default_factory=list
    )


# =============================================================
# STRUCTURAL DOCUMENT MODELS
# =============================================================


class StructuralNodeType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    BULLET_ITEM = "bullet_item"
    NUMBERED_ITEM = "numbered_item"
    DEFINITION = "definition"
    DERIVATION = "derivation"
    WORKED_EXAMPLE = "worked_example"
    PROBLEM = "problem"
    QUESTION = "question"
    ANSWER = "answer"
    SOLUTION = "solution"
    EQUATION = "equation"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    OTHER = "other"


class StructuralSourceSpan(BaseModel):
    # PDF page containing this content.
    page_number: int = Field(
        ge=1
    )

    # Parser/layout blocks used to construct it.
    block_ids: list[str] = Field(
        default_factory=list
    )

    # Exact position on the page.
    bbox: BoundingBox | None = None

    # Full rendered PDF page image.
    rendered_image_path: str | None = None


class PageStructuralElement(BaseModel):
    """
    One structure item detected from a single PDF page.

    Page-level Vision extraction creates these elements before
    the complete cross-page document structure is assembled.
    """

    local_id: str

    reading_order: int = Field(
        ge=1
    )

    node_type: StructuralNodeType

    # Visible source label, such as "Problem 16".
    label: str = ""

    # Position under the current parent on this page.
    ordinal_in_parent: int = Field(
        default=0,
        ge=0,
    )

    # Nearest visible parent heading.
    parent_heading: str = ""

    # Complete extracted source text.
    text: str = ""

    # True when this looks like a numerical question/example.
    is_numerical: bool = False

    # True when the text refers to a visual.
    references_visual: bool = False

    # Labels connected with the visual.
    visual_labels: list[str] = Field(
        default_factory=list
    )

    # Search terms extracted from the content.
    semantic_keywords: list[str] = Field(
        default_factory=list
    )

    # False when the item visibly continues onto another page.
    source_complete: bool = True

    # Hint describing how the item continues.
    continuation_hint: str = ""

    # Original parser/layout block identities.
    source_block_ids: list[str] = Field(
        default_factory=list
    )

    bbox: BoundingBox | None = None


class PageStructure(BaseModel):
    """
    Structural extraction result for one PDF page.
    """

    document_id: str

    page_number: int = Field(
        ge=1
    )

    rendered_image_path: str | None = None

    elements: list[PageStructuralElement] = Field(
        default_factory=list
    )


class PageStructureExtractionResult(BaseModel):
    """
    All page-level structure results before cross-page merging.
    """

    document_id: str

    pages: list[PageStructure] = Field(
        default_factory=list
    )


class StructuralNode(BaseModel):
    """
    One item in the final cross-page document structure.
    """

    node_id: str
    node_type: StructuralNodeType

    # Generic visible source label.
    label: str = ""

    # Parent structural node.
    parent_id: str | None = None

    # Direct children of this node.
    child_ids: list[str] = Field(
        default_factory=list
    )

    # One-based order across the complete document.
    document_order: int = Field(
        ge=1
    )

    # One-based position among all children under the parent.
    ordinal_within_parent: int = Field(
        default=0,
        ge=0,
    )

    # One-based position among the same node type globally.
    global_kind_ordinal: int = Field(
        default=0,
        ge=0,
    )

    # One-based position among the same kind under a heading.
    kind_ordinal_in_heading: int = Field(
        default=0,
        ge=0,
    )

    # Position among real points under a heading.
    #
    # Only bullet_item/numbered_item nodes will receive this
    # ordinal during structural indexing.
    point_ordinal_in_heading: int = Field(
        default=0,
        ge=0,
    )

    # Position among real calculation-based numerical tasks.
    numerical_ordinal: int = Field(
        default=0,
        ge=0,
    )

    # Hierarchy depth.
    depth: int = Field(
        default=0,
        ge=0,
    )

    title: str | None = None

    # Nearest visible heading.
    parent_heading: str | None = None

    # Exact label visibly present in the document.
    #
    # Examples:
    # Problem 16
    # Example 2
    # Figure 4.1
    exact_source_label: str | None = None

    # Exact bullet or number marker.
    #
    # Examples:
    # •
    # 4.
    # (a)
    # (ii)
    list_marker: str | None = None

    # Complete source text belonging to this item.
    text: str = ""

    page_start: int | None = Field(
        default=None,
        ge=1,
    )

    page_end: int | None = Field(
        default=None,
        ge=1,
    )

    # Human-readable heading hierarchy.
    heading_path: list[str] = Field(
        default_factory=list
    )

    # Exact PDF areas from which this node was built.
    source_spans: list[StructuralSourceSpan] = Field(
        default_factory=list
    )

    # True when the node represents a numerical.
    is_numerical: bool = False

    # Stronger signal: this actually asks for a calculation.
    is_calculation_task: bool = False

    # Whether the visual is required to answer correctly.
    requires_visual_to_understand: bool = False

    visual_labels: list[str] = Field(
        default_factory=list
    )

    # Used later for remembered-word and semantic matching.
    semantic_keywords: list[str] = Field(
        default_factory=list
    )

    # False if required source content could not be recovered.
    source_complete: bool = True

    # Structural visual nodes related to this item.
    related_visual_node_ids: list[str] = Field(
        default_factory=list
    )

    # Existing FigureArtifact identities related to this node.
    linked_figure_ids: list[str] = Field(
        default_factory=list
    )

    # Existing semantic parent chunks connected to this node.
    linked_parent_chunk_ids: list[str] = Field(
        default_factory=list
    )

    # Existing FAISS/BM25 searchable chunks connected to it.
    linked_retrieval_chunk_ids: list[str] = Field(
        default_factory=list
    )

    confidence: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )


class DocumentStructure(BaseModel):
    """
    Complete cross-page structure of one document.
    """

    document_id: str
    document_title: str = ""

    # Allows future contract upgrades without silently breaking
    # already processed documents.
    schema_version: Literal["1.0"] = "1.0"

    root_node_ids: list[str] = Field(
        default_factory=list
    )

    # Kept in original document order.
    nodes: list[StructuralNode] = Field(
        default_factory=list
    )

    # Model used to build the structure.
    structural_model: str | None = None


class ScopeDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT_NON_PHYSICS = "REJECT_NON_PHYSICS"
    REJECT_ADVANCED = "REJECT_ADVANCED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ScopeClassification(BaseModel):
    is_physics: bool
    school_level: bool

    estimated_grade_min: int | None = Field(
        default=None,
        ge=1,
        le=12,
    )

    estimated_grade_max: int | None = Field(
        default=None,
        ge=1,
        le=12,
    )

    topics: list[str] = Field(
        default_factory=list
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    decision: ScopeDecision
    reasoning: str

    classifier: Literal[
        "heuristic",
        "openai",
        "heuristic_fallback",
    ]


class ProcessingArtifacts(BaseModel):
    parsed_document: str | None = None
    layout: str | None = None
    ocr: str | None = None
    figures: str | None = None

    # Saved page-level structural extraction.
    page_structures: str | None = None

    # Saved final cross-page structural document.
    structure: str | None = None

    scope: str | None = None


class ParentChunk(BaseModel):
    parent_id: str

    user_id: str
    document_id: str

    page_number: int = Field(
        ge=1
    )

    heading: str | None = None
    text: str

    topics: list[str] = Field(
        default_factory=list
    )

    grade_min: int | None = Field(
        default=None,
        ge=1,
        le=12,
    )

    grade_max: int | None = Field(
        default=None,
        ge=1,
        le=12,
    )

    figures: list[str] = Field(
        default_factory=list
    )

    equations: list[str] = Field(
        default_factory=list
    )

    child_ids: list[str] = Field(
        default_factory=list
    )

    # Structural nodes represented by this semantic parent.
    structural_node_ids: list[str] = Field(
        default_factory=list
    )


class RetrievalChunk(BaseModel):
    chunk_id: str

    user_id: str
    document_id: str

    page_number: int = Field(
        ge=1
    )

    chunk_kind: Literal[
        "child",
        "visual",
    ]

    text: str
    content_type: str

    parent_id: str | None = None

    topics: list[str] = Field(
        default_factory=list
    )

    grade_min: int | None = Field(
        default=None,
        ge=1,
        le=12,
    )

    grade_max: int | None = Field(
        default=None,
        ge=1,
        le=12,
    )

    source_block_ids: list[str] = Field(
        default_factory=list
    )

    # Structural nodes represented by this searchable chunk.
    structural_node_ids: list[str] = Field(
        default_factory=list
    )

    linked_figure_ids: list[str] = Field(
        default_factory=list
    )

    image_path: str | None = None
    caption: str | None = None


class ChunkingResult(BaseModel):
    document_id: str
    user_id: str

    parent_chunks: list[ParentChunk] = Field(
        default_factory=list
    )

    retrieval_chunks: list[RetrievalChunk] = Field(
        default_factory=list
    )