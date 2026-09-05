from __future__ import annotations

import base64
import json
import mimetypes
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.models import (
    BoundingBox,
    DocumentLayout,
    DocumentStructure,
    PageLayout,
    PageStructuralElement,
    PageStructure,
    PageStructureExtractionResult,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    StructuralNode,
    StructuralNodeType,
    StructuralSourceSpan,
)
from src.models.contracts import ModelTask
from src.models.gateway import LLMGateway
from src.models.routing import ModelRoute, ModelRouter


_PAGE_STRUCTURE_SYSTEM_PROMPT = """
You are indexing exactly ONE PAGE of an educational Physics document.
Recover the document's visible structure, not arbitrary text chunks.

Identify items in reading order using only these node types:
- title
- heading
- subheading
- section
- paragraph
- bullet_item
- numbered_item
- definition
- derivation
- worked_example
- problem
- question
- answer
- solution
- equation
- figure
- figure_caption
- table
- other

Rules:
1. Preserve visible source labels exactly when possible, such as "16.",
   "Problem 16", "Example 2", or "Fig. 1". Never invent a label.
2. Keep a complete problem or worked example as one item. Never mix
   neighboring questions merely because their topics are similar.
3. parent_heading is the nearest meaningful visible heading/subheading.
4. ordinal_in_parent is a SOURCE/LIST ordinal, not the position among
   arbitrary page elements.
   - For a visible bullet or numbered list item, use its one-based position
     in that list when clear.
   - For an explicitly numbered problem/example, preserve that visible
     source ordinal when clear.
   - Otherwise use 0.
5. is_numerical=true only for an actual student calculation task where one
   or more quantities must be determined.
   - A definition mentioning symbols, units, numbers, or an equation is not
     a numerical.
   - A formula/equation by itself is not a numerical.
   - A worked solution is not a second numerical; its source problem is the
     calculation task.
   - Represent a calculation question as node_type="problem" even if its
     visible label says "Question"; keep the exact visible label in label.
6. references_visual=true only when the item explicitly or naturally
   depends on a figure, diagram, graph, table, circuit, or other visual.
7. For a figure, describe only what is visibly present. Do not invent labels.
8. source_complete=false if the item is visibly cut off at a page boundary.
9. The page image is authoritative for layout, bullets, numbering, equations,
   and diagrams. Extracted text and layout blocks are noisy aids.
10. Preserve Unicode and mathematical notation faithfully.
11. local_id must be unique within this page and should start with the page
    number, for example p3_item_1.
12. source_block_ids may contain only block IDs supplied in the layout-block
    input. Use an empty list when no supplied block can be linked safely.
13. Return the supplied document_id and page_number unchanged.
""".strip()


_DOCUMENT_STRUCTURE_SYSTEM_PROMPT = """
Consolidate page-level indexes into one clean structural document index.

Rules:
1. Merge an item split across adjacent pages only when it is clearly the same
   source item.
2. Never merge neighboring questions merely because their topics are similar.
3. Preserve visible source labels such as Problem 16, Example 2, and Fig. 1.
   Never invent a source label.
4. Preserve parent headings and the complete available source text.
5. Keep list identity separate from general reading order. A "fourth point"
   means the fourth bullet_item/numbered_item under its heading, never the
   fourth mixture of paragraph, equation, figure, and bullet.
6. is_numerical=true only for the actual calculation problem/worked example.
   Definitions, equations, quantity descriptions, and solution blocks are not
   separate numerical tasks merely because they contain symbols or numbers.
7. Link an item to a visual node only when wording or layout clearly connects
   them.
8. requires_visual_to_understand=true only when a correct explanation or
   solution genuinely needs diagram-only givens, geometry, labels/arrows,
   graph shape, or circuit topology. A nearby visual is not automatically
   mandatory.
9. source_complete=false if the available source remains incomplete.
10. Do not add Physics facts. This task indexes source structure; it does not
    answer the student's question.
11. node_id must be unique and readable.
12. parent_id and related_visual_node_ids may reference only node IDs returned
    in the same result. Use null/an empty list when no safe link exists.
13. source_local_ids may contain only local_id values present in the supplied
    page indexes.
14. Keep document_order one-based and consistent with source reading order.
15. Return the supplied document_id unchanged.
""".strip()


class StructureBuildError(RuntimeError):
    """Raised when verified structural indexing cannot be completed."""


class _StrictStructureModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class _PageElementDraft(_StrictStructureModel):
    local_id: str = Field(min_length=1, max_length=300)
    reading_order: int = Field(ge=1)
    node_type: StructuralNodeType
    label: str = Field(max_length=1000)
    ordinal_in_parent: int = Field(ge=0)
    parent_heading: str = Field(max_length=2000)
    text: str = Field(max_length=30000)
    is_numerical: bool
    references_visual: bool
    visual_labels: list[str]
    semantic_keywords: list[str]
    source_complete: bool
    continuation_hint: str = Field(max_length=2000)
    source_block_ids: list[str]


class _PageStructureDraft(_StrictStructureModel):
    document_id: str = Field(min_length=1, max_length=300)
    page_number: int = Field(ge=1)
    elements: list[_PageElementDraft]


class _DocumentNodeDraft(_StrictStructureModel):
    node_id: str = Field(min_length=1, max_length=500)
    document_order: int = Field(ge=1)
    node_type: StructuralNodeType
    label: str = Field(max_length=1000)
    parent_id: str | None
    parent_heading: str = Field(max_length=2000)
    ordinal_in_parent: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(max_length=60000)
    is_numerical: bool
    requires_visual_to_understand: bool
    visual_labels: list[str]
    related_visual_node_ids: list[str]
    semantic_keywords: list[str]
    source_complete: bool
    source_local_ids: list[str]


class _DocumentStructureDraft(_StrictStructureModel):
    document_id: str = Field(min_length=1, max_length=300)
    document_title: str = Field(max_length=2000)
    nodes: list[_DocumentNodeDraft]


@dataclass(frozen=True, slots=True)
class StructureBuildResult:
    page_structures: PageStructureExtractionResult
    document_structure: DocumentStructure


def add_structural_ordinals(
    structure: DocumentStructure,
) -> DocumentStructure:
    """
    Add deterministic structural addresses.

    `point_ordinal_in_heading` counts only bullet/numbered items.
    `numerical_ordinal` counts only genuine calculation problems/examples.
    """

    ordered_nodes = sorted(
        structure.nodes,
        key=lambda node: (
            node.document_order,
            node.page_start or 0,
            node.node_id,
        ),
    )

    kind_count: defaultdict[StructuralNodeType, int] = defaultdict(int)
    parent_kind_count: defaultdict[
        tuple[str, StructuralNodeType],
        int,
    ] = defaultdict(int)
    parent_point_count: defaultdict[str, int] = defaultdict(int)

    numerical_count = 0
    updated_nodes: list[StructuralNode] = []

    for node in ordered_nodes:
        parent_key = _parent_group_key(node)

        kind_count[node.node_type] += 1
        global_kind_ordinal = kind_count[node.node_type]

        parent_kind_key = (
            parent_key,
            node.node_type,
        )
        parent_kind_count[parent_kind_key] += 1
        kind_ordinal_in_heading = parent_kind_count[parent_kind_key]

        ordinal_in_parent = node.ordinal_within_parent

        if node.node_type in {
            StructuralNodeType.BULLET_ITEM,
            StructuralNodeType.NUMBERED_ITEM,
        }:
            parent_point_count[parent_key] += 1
            point_ordinal_in_heading = parent_point_count[parent_key]

            if ordinal_in_parent <= 0:
                ordinal_in_parent = point_ordinal_in_heading
        else:
            point_ordinal_in_heading = 0

        is_calculation_task = (
            node.is_numerical
            and node.node_type
            in {
                StructuralNodeType.PROBLEM,
                StructuralNodeType.WORKED_EXAMPLE,
            }
        )

        if is_calculation_task:
            numerical_count += 1
            numerical_ordinal = numerical_count
        else:
            numerical_ordinal = 0

        updated_nodes.append(
            node.model_copy(
                update={
                    "ordinal_within_parent": ordinal_in_parent,
                    "global_kind_ordinal": global_kind_ordinal,
                    "kind_ordinal_in_heading": kind_ordinal_in_heading,
                    "point_ordinal_in_heading": point_ordinal_in_heading,
                    "is_calculation_task": is_calculation_task,
                    "numerical_ordinal": numerical_ordinal,
                }
            )
        )

    return structure.model_copy(
        update={
            "nodes": updated_nodes,
        }
    )


class DocumentStructureBuilder:
    """
    Build a page-aware and cross-page structural index.

    This class does not persist artifacts and does not replace semantic
    chunking/retrieval. The ingestion service can save the returned models and
    later link their node IDs with ParentChunk/RetrievalChunk records.
    """

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        model_router: ModelRouter,
        max_page_text_characters: int = 16000,
        max_layout_text_characters: int = 20000,
        max_document_payload_characters: int | None = None,
    ) -> None:
        if max_page_text_characters < 1000:
            raise ValueError(
                "max_page_text_characters must be at least 1000."
            )

        if max_layout_text_characters < 1000:
            raise ValueError(
                "max_layout_text_characters must be at least 1000."
            )

        if (
            max_document_payload_characters is not None
            and max_document_payload_characters < 10000
        ):
            raise ValueError(
                "max_document_payload_characters must be at least 10000 "
                "when provided."
            )

        self.gateway = gateway
        self.model_router = model_router
        self.max_page_text_characters = max_page_text_characters
        self.max_layout_text_characters = max_layout_text_characters
        self.max_document_payload_characters = (
            max_document_payload_characters
        )

    def build(
        self,
        *,
        parsed_document: ParsedDocument,
        document_layout: DocumentLayout,
    ) -> StructureBuildResult:
        self._validate_inputs(
            parsed_document=parsed_document,
            document_layout=document_layout,
        )

        route = self.model_router.route_task(
            ModelTask.DOCUMENT_STRUCTURE
        )

        if not route.requires_vision:
            raise StructureBuildError(
                "The DOCUMENT_STRUCTURE route must support Vision."
            )

        layouts_by_page = {
            page.page_number: page
            for page in document_layout.pages
        }

        page_structures: list[PageStructure] = []

        for parsed_page in sorted(
            parsed_document.pages,
            key=lambda page: page.page_number,
        ):
            page_layout = layouts_by_page.get(
                parsed_page.page_number
            )

            page_structures.append(
                self._index_page(
                    document_id=parsed_document.document_id,
                    parsed_page=parsed_page,
                    page_layout=page_layout,
                    route=route,
                )
            )

        page_result = PageStructureExtractionResult(
            document_id=parsed_document.document_id,
            pages=page_structures,
        )

        document_structure = self._consolidate(
            parsed_document=parsed_document,
            page_result=page_result,
            route=route,
        )

        return StructureBuildResult(
            page_structures=page_result,
            document_structure=document_structure,
        )

    def _index_page(
        self,
        *,
        document_id: str,
        parsed_page: ParsedPage,
        page_layout: PageLayout | None,
        route: ModelRoute,
    ) -> PageStructure:
        image_url = _image_file_to_data_url(
            parsed_page.rendered_image_path
        )

        layout_payload = self._layout_payload(
            parsed_page=parsed_page,
            page_layout=page_layout,
        )

        page_text = parsed_page.native_text.strip()

        if not page_text:
            page_text = "\n".join(
                block.text.strip()
                for block in parsed_page.blocks
                if block.text.strip()
            )

        page_text = page_text[
            : self.max_page_text_characters
        ]

        user_prompt = (
            f"DOCUMENT ID: {document_id}\n"
            f"PAGE NUMBER: {parsed_page.page_number}\n\n"
            "EXTRACTED TEXT (may be noisy):\n"
            f"{page_text}\n\n"
            "LAYOUT BLOCKS (may be noisy):\n"
            f"{json.dumps(layout_payload, ensure_ascii=False)}"
        )

        draft = self.gateway.generate_structured(
            route=route,
            system_prompt=_PAGE_STRUCTURE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=_PageStructureDraft,
            image_urls=[image_url],
        )

        if draft.document_id != document_id:
            raise StructureBuildError(
                "Page structure returned a different document_id."
            )

        if draft.page_number != parsed_page.page_number:
            raise StructureBuildError(
                "Page structure returned a different page_number."
            )

        block_map = {
            block.block_id: block
            for block in parsed_page.blocks
        }

        elements: list[PageStructuralElement] = []
        seen_local_ids: set[str] = set()

        for position, element in enumerate(
            sorted(
                draft.elements,
                key=lambda item: (
                    item.reading_order,
                    item.local_id,
                ),
            ),
            start=1,
        ):
            local_id = _unique_identifier(
                preferred=element.local_id,
                fallback=(
                    f"p{parsed_page.page_number}_item_{position}"
                ),
                used=seen_local_ids,
            )

            source_block_ids = [
                block_id
                for block_id in dict.fromkeys(
                    element.source_block_ids
                )
                if block_id in block_map
            ]

            if not source_block_ids:
                source_block_ids = _match_source_blocks(
                    text=element.text,
                    blocks=parsed_page.blocks,
                )

            elements.append(
                PageStructuralElement(
                    local_id=local_id,
                    reading_order=position,
                    node_type=element.node_type,
                    label=element.label,
                    ordinal_in_parent=(
                        element.ordinal_in_parent
                    ),
                    parent_heading=element.parent_heading,
                    text=element.text,
                    is_numerical=element.is_numerical,
                    references_visual=(
                        element.references_visual
                    ),
                    visual_labels=list(
                        dict.fromkeys(element.visual_labels)
                    ),
                    semantic_keywords=list(
                        dict.fromkeys(
                            element.semantic_keywords
                        )
                    ),
                    source_complete=element.source_complete,
                    continuation_hint=(
                        element.continuation_hint
                    ),
                    source_block_ids=source_block_ids,
                    bbox=_combined_bbox(
                        block_map[block_id].bbox
                        for block_id in source_block_ids
                    ),
                )
            )

        return PageStructure(
            document_id=document_id,
            page_number=parsed_page.page_number,
            rendered_image_path=(
                parsed_page.rendered_image_path
            ),
            elements=elements,
        )

    def _consolidate(
        self,
        *,
        parsed_document: ParsedDocument,
        page_result: PageStructureExtractionResult,
        route: ModelRoute,
    ) -> DocumentStructure:
        page_payload = page_result.model_dump(
            mode="json"
        )

        serialized_payload = json.dumps(
            page_payload,
            ensure_ascii=False,
        )

        if (
            self.max_document_payload_characters is not None
            and len(serialized_payload)
            > self.max_document_payload_characters
        ):
            raise StructureBuildError(
                "The complete page-structure payload exceeds the configured "
                "safe consolidation limit. It was not silently truncated."
            )

        user_prompt = (
            f"DOCUMENT ID: {parsed_document.document_id}\n"
            f"DOCUMENT FILE: {Path(parsed_document.source_path).name}\n\n"
            "PAGE STRUCTURES:\n"
            f"{serialized_payload}"
        )

        draft = self.gateway.generate_structured(
            route=route,
            system_prompt=_DOCUMENT_STRUCTURE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=_DocumentStructureDraft,
        )

        if draft.document_id != parsed_document.document_id:
            raise StructureBuildError(
                "Document consolidation returned a different document_id."
            )

        return self._materialize_document_structure(
            draft=draft,
            page_result=page_result,
            structural_model=route.model_name,
        )

    def _materialize_document_structure(
        self,
        *,
        draft: _DocumentStructureDraft,
        page_result: PageStructureExtractionResult,
        structural_model: str,
    ) -> DocumentStructure:
        ordered_drafts = sorted(
            draft.nodes,
            key=lambda node: (
                node.document_order,
                node.page_start,
                node.node_id,
            ),
        )

        used_node_ids: set[str] = set()
        raw_to_final_id: dict[str, str] = {}
        final_ids: list[str] = []

        for position, node in enumerate(
            ordered_drafts,
            start=1,
        ):
            final_id = _unique_identifier(
                preferred=node.node_id,
                fallback=f"structure_node_{position}",
                used=used_node_ids,
            )
            final_ids.append(final_id)
            raw_to_final_id.setdefault(
                node.node_id,
                final_id,
            )

        valid_final_ids = set(final_ids)
        parent_by_id: dict[str, str | None] = {}

        for final_id, node in zip(
            final_ids,
            ordered_drafts,
        ):
            parent_id = (
                raw_to_final_id.get(node.parent_id)
                if node.parent_id
                else None
            )

            if (
                parent_id == final_id
                or parent_id not in valid_final_ids
            ):
                parent_id = None

            parent_by_id[final_id] = parent_id

        _break_parent_cycles(parent_by_id)

        children_by_id: defaultdict[str, list[str]] = defaultdict(list)

        for node_id, parent_id in parent_by_id.items():
            if parent_id:
                children_by_id[parent_id].append(node_id)

        page_elements = _page_element_lookup(page_result)
        page_image_paths = {
            page.page_number: page.rendered_image_path
            for page in page_result.pages
        }

        nodes: list[StructuralNode] = []

        for document_order, (
            final_id,
            node,
        ) in enumerate(
            zip(final_ids, ordered_drafts),
            start=1,
        ):
            if node.page_end < node.page_start:
                raise StructureBuildError(
                    f"Structural node {node.node_id!r} has page_end before "
                    "page_start."
                )

            related_visual_ids = [
                mapped_id
                for raw_id in dict.fromkeys(
                    node.related_visual_node_ids
                )
                if (
                    mapped_id := raw_to_final_id.get(raw_id)
                )
                and mapped_id in valid_final_ids
                and mapped_id != final_id
            ]

            source_spans = _source_spans_for_node(
                source_local_ids=node.source_local_ids,
                page_start=node.page_start,
                page_end=node.page_end,
                page_elements=page_elements,
                page_image_paths=page_image_paths,
            )

            title = _title_for_node(
                node_type=node.node_type,
                label=node.label,
                text=node.text,
            )

            nodes.append(
                StructuralNode(
                    node_id=final_id,
                    node_type=node.node_type,
                    label=node.label,
                    parent_id=parent_by_id[final_id],
                    child_ids=children_by_id[final_id],
                    document_order=document_order,
                    ordinal_within_parent=(
                        node.ordinal_in_parent
                    ),
                    depth=0,
                    title=title,
                    parent_heading=(
                        node.parent_heading or None
                    ),
                    exact_source_label=(
                        node.label or None
                    ),
                    list_marker=_list_marker(
                        node.label,
                        node.node_type,
                    ),
                    text=node.text,
                    page_start=node.page_start,
                    page_end=node.page_end,
                    heading_path=[],
                    source_spans=source_spans,
                    is_numerical=node.is_numerical,
                    requires_visual_to_understand=(
                        node.requires_visual_to_understand
                    ),
                    visual_labels=list(
                        dict.fromkeys(node.visual_labels)
                    ),
                    semantic_keywords=list(
                        dict.fromkeys(
                            node.semantic_keywords
                        )
                    ),
                    source_complete=node.source_complete,
                    related_visual_node_ids=(
                        related_visual_ids
                    ),
                )
            )

        nodes = _add_hierarchy_metadata(nodes)

        structure = DocumentStructure(
            document_id=draft.document_id,
            document_title=draft.document_title,
            root_node_ids=[
                node.node_id
                for node in nodes
                if node.parent_id is None
            ],
            nodes=nodes,
            structural_model=structural_model,
        )

        return add_structural_ordinals(structure)

    def _layout_payload(
        self,
        *,
        parsed_page: ParsedPage,
        page_layout: PageLayout | None,
    ) -> list[dict[str, object]]:
        if page_layout is not None:
            source_blocks = page_layout.blocks
        else:
            source_blocks = parsed_page.blocks

        payload: list[dict[str, object]] = []
        used_characters = 0

        for block in source_blocks:
            text = block.text.strip()
            remaining = (
                self.max_layout_text_characters
                - used_characters
            )

            if remaining <= 0:
                break

            included_text = text[:remaining]
            used_characters += len(included_text)

            block_type = getattr(
                block.block_type,
                "value",
                block.block_type,
            )

            payload.append(
                {
                    "block_id": block.block_id,
                    "block_number": block.block_number,
                    "block_type": str(block_type),
                    "bbox": block.bbox.model_dump(),
                    "text": included_text,
                }
            )

        return payload

    @staticmethod
    def _validate_inputs(
        *,
        parsed_document: ParsedDocument,
        document_layout: DocumentLayout,
    ) -> None:
        if (
            parsed_document.document_id
            != document_layout.document_id
        ):
            raise ValueError(
                "ParsedDocument and DocumentLayout must have the same "
                "document_id."
            )

        if not parsed_document.pages:
            raise ValueError(
                "ParsedDocument must contain at least one page."
            )

        parsed_page_numbers = [
            page.page_number
            for page in parsed_document.pages
        ]

        if len(parsed_page_numbers) != len(
            set(parsed_page_numbers)
        ):
            raise ValueError(
                "ParsedDocument contains duplicate page numbers."
            )


def _image_file_to_data_url(
    image_path: str,
) -> str:
    path = Path(image_path).expanduser()

    if not path.is_file():
        raise StructureBuildError(
            f"Rendered page image was not found: {path}"
        )

    mime_type, _ = mimetypes.guess_type(path.name)

    if mime_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }:
        raise StructureBuildError(
            f"Unsupported rendered page image type: {path.suffix}"
        )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def _unique_identifier(
    *,
    preferred: str,
    fallback: str,
    used: set[str],
) -> str:
    normalized = re.sub(
        r"[^\w.-]+",
        "_",
        preferred.strip(),
        flags=re.UNICODE,
    ).strip("_.-")

    base = normalized or fallback
    candidate = base
    suffix = 2

    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1

    used.add(candidate)
    return candidate


def _normalize_text(value: str) -> str:
    return " ".join(
        value.casefold().split()
    )


def _match_source_blocks(
    *,
    text: str,
    blocks: Iterable[ParsedBlock],
) -> list[str]:
    target = _normalize_text(text)

    if not target:
        return []

    matches: list[str] = []

    for block in blocks:
        block_text = _normalize_text(block.text)

        if not block_text:
            continue

        if (
            block_text in target
            or target in block_text
        ):
            matches.append(block.block_id)

    return matches


def _combined_bbox(
    boxes: Iterable[BoundingBox],
) -> BoundingBox | None:
    collected = list(boxes)

    if not collected:
        return None

    return BoundingBox(
        x0=min(box.x0 for box in collected),
        y0=min(box.y0 for box in collected),
        x1=max(box.x1 for box in collected),
        y1=max(box.y1 for box in collected),
    )


def _page_element_lookup(
    page_result: PageStructureExtractionResult,
) -> dict[str, list[tuple[PageStructure, PageStructuralElement]]]:
    lookup: defaultdict[
        str,
        list[tuple[PageStructure, PageStructuralElement]],
    ] = defaultdict(list)

    for page in page_result.pages:
        for element in page.elements:
            lookup[element.local_id].append(
                (page, element)
            )

    return dict(lookup)


def _source_spans_for_node(
    *,
    source_local_ids: list[str],
    page_start: int,
    page_end: int,
    page_elements: dict[
        str,
        list[tuple[PageStructure, PageStructuralElement]],
    ],
    page_image_paths: dict[int, str | None],
) -> list[StructuralSourceSpan]:
    spans: list[StructuralSourceSpan] = []
    seen: set[tuple[int, tuple[str, ...]]] = set()

    for local_id in dict.fromkeys(source_local_ids):
        candidates = page_elements.get(local_id, [])

        candidate = next(
            (
                item
                for item in candidates
                if page_start
                <= item[0].page_number
                <= page_end
            ),
            None,
        )

        if candidate is None:
            continue

        page, element = candidate
        identity = (
            page.page_number,
            tuple(element.source_block_ids),
        )

        if identity in seen:
            continue

        seen.add(identity)
        spans.append(
            StructuralSourceSpan(
                page_number=page.page_number,
                block_ids=element.source_block_ids,
                bbox=element.bbox,
                rendered_image_path=(
                    page.rendered_image_path
                ),
            )
        )

    if spans:
        return spans

    for page_number in range(
        page_start,
        page_end + 1,
    ):
        spans.append(
            StructuralSourceSpan(
                page_number=page_number,
                rendered_image_path=(
                    page_image_paths.get(page_number)
                ),
            )
        )

    return spans


def _title_for_node(
    *,
    node_type: StructuralNodeType,
    label: str,
    text: str,
) -> str | None:
    if node_type not in {
        StructuralNodeType.TITLE,
        StructuralNodeType.HEADING,
        StructuralNodeType.SUBHEADING,
        StructuralNodeType.SECTION,
    }:
        return None

    if label.strip():
        return label.strip()

    first_line = next(
        (
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ),
        "",
    )

    return first_line[:500] or None


def _list_marker(
    label: str,
    node_type: StructuralNodeType,
) -> str | None:
    if node_type not in {
        StructuralNodeType.BULLET_ITEM,
        StructuralNodeType.NUMBERED_ITEM,
    }:
        return None

    match = re.match(
        r"^\s*(?:[•◦▪▫‣⁃●○■□◆◇*-]|\(?[A-Za-z0-9ivxlcdmIVXLCDM]+\)?[.)]?)",
        label,
    )

    return match.group(0).strip() if match else None


def _parent_group_key(
    node: StructuralNode,
) -> str:
    if node.parent_id:
        return f"id:{node.parent_id}"

    normalized_heading = _normalize_text(
        node.parent_heading or ""
    )

    if normalized_heading:
        return f"heading:{normalized_heading}"

    return "document:root"


def _break_parent_cycles(
    parent_by_id: dict[str, str | None],
) -> None:
    for start_id in list(parent_by_id):
        seen: set[str] = set()
        current_id: str | None = start_id

        while current_id is not None:
            if current_id in seen:
                parent_by_id[start_id] = None
                break

            seen.add(current_id)
            current_id = parent_by_id.get(current_id)


def _add_hierarchy_metadata(
    nodes: list[StructuralNode],
) -> list[StructuralNode]:
    by_id = {
        node.node_id: node
        for node in nodes
    }

    updated: list[StructuralNode] = []

    for node in nodes:
        depth = 0
        heading_path: list[str] = []
        parent_id = node.parent_id
        visited: set[str] = set()

        while (
            parent_id
            and parent_id in by_id
            and parent_id not in visited
        ):
            visited.add(parent_id)
            parent = by_id[parent_id]
            depth += 1

            if parent.node_type in {
                StructuralNodeType.TITLE,
                StructuralNodeType.HEADING,
                StructuralNodeType.SUBHEADING,
                StructuralNodeType.SECTION,
            }:
                heading_name = (
                    parent.title
                    or parent.label
                )

                if not heading_name and parent.text:
                    heading_name = (
                        parent.text.splitlines()[0]
                    )

                if heading_name:
                    heading_path.append(
                        heading_name.strip()
                    )

            parent_id = parent.parent_id

        heading_path.reverse()

        updated.append(
            node.model_copy(
                update={
                    "depth": depth,
                    "heading_path": heading_path,
                }
            )
        )

    return updated
