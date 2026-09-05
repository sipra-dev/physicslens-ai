from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.models import (
    DocumentStructure,
    StructuralNode,
    StructuralNodeType,
)
from src.models.contracts import (
    ModelTask,
    PendingStructuralClarification,
)
from src.models.gateway import LLMGateway, LLMGatewayError
from src.models.routing import ModelRouter
from src.storage import StorageBackend


logger = logging.getLogger(
    "phymentor.structural_resolver"
)


_RESOLVER_SYSTEM_PROMPT = """
Resolve a student's natural-language request against a STRUCTURED educational
Physics document index. The user may remember only a few words, use typos,
loose grammar, abbreviations, Bengali/English mixed wording, or transliterated
mathematical symbols.

MATCH PRIORITY
1. Explicit visible source identity/label, such as a named problem, example,
   question, figure, equation, or section.
2. Structural address, such as a particular point under a heading or a
   particular numerical task.
3. A few remembered words or partial source text.
4. Broader semantic/contextual match.

STRUCTURAL ADDRESS RULES
- Do not require exact wording when the intended source item is clear.
- Match heading names approximately when evidence supports it.
- A requested numerical ordinal refers only to nodes where
  is_calculation_task=true and numerical_ordinal has that value.
- Never treat a definition, equation, quantity description, paragraph, or
  solution as a numerical merely because it contains symbols or numbers.
- A requested point ordinal under a heading refers only to bullet_item or
  numbered_item nodes with the corresponding point_ordinal_in_heading.
- Never interpret "Nth point" as the Nth arbitrary structural element.
- If the requested bullet/numbered point does not exist, do not substitute a
  paragraph, equation, definition, or figure. Ask for clarification when a
  plausible ambiguity exists; otherwise return no_match.
- Preserve the user's raw wording. Do not mechanically rewrite it in code.

CLARIFICATION FOLLOW-UP RULES
- PENDING_CLARIFICATION may contain candidates from the immediately previous
  turn.
- First consider whether the current query naturally answers that pending
  clarification.
- Interpret replies such as both/all/first/second/previous or a descriptive
  phrase only within the pending candidate set when that reading is natural.
- Do not expand "all" to the whole document when the user is answering a
  clarification about a bounded candidate set.
- If the user clearly changes topic, ignore the pending clarification and
  resolve against the full document.

CONTEXTUAL SYMBOL RULES
- Terms such as mu, theta, omega, eta, alpha, beta, uC, or micro C must be
  interpreted using both the query and candidate source context.
- Never force one universal symbol mapping.
- Record useful contextual interpretations in interpreted_user_terms.

VISUAL RULES
- needs_visual=true only when the requested answer depends on diagram-only
  givens, geometry, arrows, labels, graph shape, circuit topology, or the
  target itself is a requested visual explanation/transcription.
- A linked figure is not automatically mandatory for every numerical.
- If source text is sufficient and a visual is only supporting,
  needs_visual=false.

SAFETY AND SCOPE
- Return only node IDs present in the supplied index.
- If more than one candidate remains genuinely plausible, do not guess. Set
  clarification_needed=true, return the bounded candidate IDs, and ask one
  short clarification question.
- If no safe structural target exists, return no_match with no target IDs.
- Do not answer the Physics question. This task resolves source identity and
  requested action only.
""".strip()


class StructuralResolverError(RuntimeError):
    """Raised for invalid resolver construction or unsafe internal state."""


class StructuralResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    NO_MATCH = "NO_MATCH"
    STRUCTURE_UNAVAILABLE = "STRUCTURE_UNAVAILABLE"


class StructuralResolutionAction(str, Enum):
    EXPLAIN = "explain"
    SOLVE = "solve"
    SUMMARIZE = "summarize"
    TRANSCRIBE = "transcribe"
    COMPARE = "compare"
    ANSWER = "answer"


class StructuralMatchMode(str, Enum):
    EXACT_LABEL = "exact_label"
    ORDINAL_UNDER_HEADING = "ordinal_under_heading"
    GLOBAL_ORDINAL = "global_ordinal"
    PARTIAL_TEXT = "partial_text"
    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


class _StrictResolverModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class InterpretedUserTerm(_StrictResolverModel):
    raw: str = Field(
        min_length=1,
        max_length=300,
    )
    meaning: str = Field(
        min_length=1,
        max_length=1000,
    )


class ClarificationCandidate(_StrictResolverModel):
    node_id: str
    node_type: StructuralNodeType
    label: str
    parent_heading: str | None
    page_start: int | None = Field(
        default=None,
        ge=1,
    )
    text_preview: str


class AnswerScopeContract(_StrictResolverModel):
    requested_action: StructuralResolutionAction
    allowed_target_node_ids: list[str]
    scope_rules: list[str]


class StructuralResolution(_StrictResolverModel):
    status: StructuralResolutionStatus
    document_id: str
    raw_query: str
    action: StructuralResolutionAction
    target_node_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    candidate_node_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    candidates: list[ClarificationCandidate] = Field(
        default_factory=list,
        max_length=20,
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    match_mode: StructuralMatchMode
    needs_visual: bool
    visual_node_ids: list[str] = Field(
        default_factory=list,
        max_length=30,
    )
    linked_retrieval_chunk_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )
    linked_parent_chunk_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )
    linked_figure_ids: list[str] = Field(
        default_factory=list,
        max_length=50,
    )
    source_page_numbers: list[int] = Field(
        default_factory=list,
        max_length=100,
    )
    visual_page_numbers: list[int] = Field(
        default_factory=list,
        max_length=100,
    )
    interpreted_user_terms: list[
        InterpretedUserTerm
    ] = Field(
        default_factory=list,
        max_length=30,
    )
    clarification_needed: bool
    clarification_question: str = Field(
        default="",
        max_length=2000,
    )
    reason: str = Field(
        default="",
        max_length=4000,
    )
    fallback_to_semantic: bool
    structural_warning: str | None = Field(
        default=None,
        max_length=2000,
    )
    answer_scope: AnswerScopeContract | None = None

    def to_pending_clarification(
        self,
    ) -> PendingStructuralClarification | None:
        if (
            self.status
            != StructuralResolutionStatus.NEEDS_CLARIFICATION
            or not self.clarification_question
            or not self.candidate_node_ids
        ):
            return None

        return PendingStructuralClarification(
            document_id=self.document_id,
            original_query=self.raw_query,
            clarification_question=(
                self.clarification_question
            ),
            candidate_node_ids=(
                self.candidate_node_ids
            ),
        )


class _ResolutionDraft(_StrictResolverModel):
    action: StructuralResolutionAction
    target_node_ids: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    match_mode: StructuralMatchMode
    needs_visual: bool
    visual_node_ids: list[str]
    interpreted_user_terms: list[
        InterpretedUserTerm
    ]
    clarification_needed: bool
    clarification_question: str = Field(
        max_length=2000,
    )
    reason: str = Field(
        max_length=4000,
    )


class StructuralResolver:
    """
    Resolve natural-language references against a saved DocumentStructure.

    This service does not answer the question and does not replace hybrid
    retrieval. It either resolves verified structural targets, asks a bounded
    clarification, or explicitly allows semantic fallback.
    """

    def __init__(
        self,
        *,
        storage: StorageBackend,
        gateway: LLMGateway,
        model_router: ModelRouter,
        text_preview_characters: int = 1200,
        max_index_payload_characters: int | None = None,
    ) -> None:
        if text_preview_characters < 200:
            raise ValueError(
                "text_preview_characters must be at least 200."
            )

        if (
            max_index_payload_characters is not None
            and max_index_payload_characters < 10000
        ):
            raise ValueError(
                "max_index_payload_characters must be at least 10000 "
                "when provided."
            )

        self.storage = storage
        self.gateway = gateway
        self.model_router = model_router
        self.text_preview_characters = (
            text_preview_characters
        )
        self.max_index_payload_characters = (
            max_index_payload_characters
        )

    def resolve(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
        pending_clarification: (
            PendingStructuralClarification | None
        ) = None,
    ) -> StructuralResolution:
        normalized_query = query.strip()
        normalized_user_id = user_id.strip()
        normalized_document_id = document_id.strip()

        if not normalized_query:
            return self._fallback_result(
                status=StructuralResolutionStatus.NO_MATCH,
                document_id=normalized_document_id,
                raw_query="",
                reason="EMPTY_QUERY",
            )

        if not normalized_user_id:
            raise StructuralResolverError(
                "user_id cannot be empty."
            )

        if not normalized_document_id:
            raise StructuralResolverError(
                "document_id cannot be empty."
            )

        structure, unavailable_reason = (
            self._load_document_structure(
                user_id=normalized_user_id,
                document_id=normalized_document_id,
            )
        )

        if structure is None:
            return self._fallback_result(
                status=(
                    StructuralResolutionStatus
                    .STRUCTURE_UNAVAILABLE
                ),
                document_id=normalized_document_id,
                raw_query=normalized_query,
                reason=(
                    unavailable_reason
                    or "STRUCTURE_UNAVAILABLE"
                ),
                warning=(
                    "The structural document index is unavailable. "
                    "Semantic retrieval may still be used."
                ),
            )

        compact_index = self._compact_index(
            structure
        )

        serialized_index = json.dumps(
            compact_index,
            ensure_ascii=False,
        )

        if (
            self.max_index_payload_characters is not None
            and len(serialized_index)
            > self.max_index_payload_characters
        ):
            return self._fallback_result(
                status=(
                    StructuralResolutionStatus
                    .STRUCTURE_UNAVAILABLE
                ),
                document_id=normalized_document_id,
                raw_query=normalized_query,
                reason="STRUCTURAL_INDEX_PAYLOAD_TOO_LARGE",
                warning=(
                    "The complete structural index exceeded the configured "
                    "resolver limit and was not silently truncated."
                ),
            )

        pending_payload = self._pending_payload(
            pending_clarification=(
                pending_clarification
            ),
            structure=structure,
        )

        user_prompt = (
            "CURRENT RAW USER QUERY:\n"
            f"{normalized_query}\n\n"
            "PENDING_CLARIFICATION:\n"
            f"{json.dumps(pending_payload, ensure_ascii=False)}\n\n"
            "DOCUMENT TITLE:\n"
            f"{structure.document_title}\n\n"
            "STRUCTURED INDEX:\n"
            f"{serialized_index}"
        )

        route = self.model_router.route_task(
            ModelTask.STRUCTURAL_RESOLUTION
        )

        if route.requires_vision:
            raise StructuralResolverError(
                "STRUCTURAL_RESOLUTION must be a text-only route."
            )

        try:
            draft = self.gateway.generate_structured(
                route=route,
                system_prompt=_RESOLVER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=_ResolutionDraft,
            )

        except LLMGatewayError as exc:
            logger.warning(
                "structural_resolution_model_failed "
                "document_id=%s failure_kind=%s",
                normalized_document_id,
                exc.failure_kind.value,
            )

            return self._fallback_result(
                status=(
                    StructuralResolutionStatus
                    .STRUCTURE_UNAVAILABLE
                ),
                document_id=normalized_document_id,
                raw_query=normalized_query,
                reason=(
                    "STRUCTURAL_RESOLUTION_MODEL_FAILED:"
                    f"{exc.failure_kind.value}"
                ),
                warning=(
                    "The structural resolver model was unavailable. "
                    "Semantic retrieval may still be used."
                ),
            )

        return self._finalize_resolution(
            raw_query=normalized_query,
            structure=structure,
            draft=draft,
        )

    def _load_document_structure(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> tuple[DocumentStructure | None, str | None]:
        try:
            metadata = self.storage.read_document_metadata(
                user_id=user_id,
                document_id=document_id,
            )

            if metadata is None:
                return None, "DOCUMENT_METADATA_NOT_FOUND"

            artifacts = metadata.get("artifacts")

            if not isinstance(artifacts, dict):
                return None, "STRUCTURE_ARTIFACT_NOT_RECORDED"

            artifact_path = artifacts.get("structure")

            if (
                not isinstance(artifact_path, str)
                or not artifact_path.strip()
            ):
                return None, "STRUCTURE_ARTIFACT_NOT_RECORDED"

            raw_structure = (
                self.storage.read_document_json_artifact(
                    user_id=user_id,
                    document_id=document_id,
                    artifact_path=artifact_path,
                )
            )

            if raw_structure is None:
                return None, "STRUCTURE_ARTIFACT_MISSING"

            structure = DocumentStructure.model_validate(
                raw_structure
            )

            if structure.document_id != document_id:
                return None, "STRUCTURE_DOCUMENT_ID_MISMATCH"

            if not structure.nodes:
                return None, "STRUCTURE_HAS_NO_NODES"

            return structure, None

        except Exception:
            logger.exception(
                "structural_artifact_read_failed "
                "user_id=%s document_id=%s",
                user_id,
                document_id,
            )
            return None, "STRUCTURE_ARTIFACT_READ_FAILED"

    def _compact_index(
        self,
        structure: DocumentStructure,
    ) -> list[dict[str, Any]]:
        return [
            {
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "label": node.label,
                "exact_source_label": (
                    node.exact_source_label
                ),
                "parent_id": node.parent_id,
                "parent_heading": node.parent_heading,
                "heading_path": node.heading_path,
                "ordinal_in_parent": (
                    node.ordinal_within_parent
                ),
                "kind_ordinal_in_heading": (
                    node.kind_ordinal_in_heading
                ),
                "point_ordinal_in_heading": (
                    node.point_ordinal_in_heading
                ),
                "global_kind_ordinal": (
                    node.global_kind_ordinal
                ),
                "numerical_ordinal": (
                    node.numerical_ordinal
                ),
                "page_start": node.page_start,
                "page_end": node.page_end,
                "is_numerical": node.is_numerical,
                "is_calculation_task": (
                    node.is_calculation_task
                ),
                "requires_visual_to_understand": (
                    node.requires_visual_to_understand
                ),
                "related_visual_node_ids": (
                    node.related_visual_node_ids
                ),
                "visual_labels": node.visual_labels,
                "semantic_keywords": (
                    node.semantic_keywords
                ),
                "source_complete": node.source_complete,
                "text_preview": node.text[
                    : self.text_preview_characters
                ],
            }
            for node in sorted(
                structure.nodes,
                key=lambda item: (
                    item.document_order,
                    item.node_id,
                ),
            )
        ]

    def _pending_payload(
        self,
        *,
        pending_clarification: (
            PendingStructuralClarification | None
        ),
        structure: DocumentStructure,
    ) -> dict[str, Any] | str:
        if (
            pending_clarification is None
            or pending_clarification.document_id
            != structure.document_id
        ):
            return "NONE"

        by_id = {
            node.node_id: node
            for node in structure.nodes
        }

        candidate_ids = [
            node_id
            for node_id in dict.fromkeys(
                pending_clarification
                .candidate_node_ids
            )
            if node_id in by_id
        ]

        if not candidate_ids:
            return "NONE"

        return {
            "original_query": (
                pending_clarification.original_query
            ),
            "clarification_question": (
                pending_clarification
                .clarification_question
            ),
            "candidate_node_ids": candidate_ids,
            "candidate_nodes": [
                self._candidate_payload(by_id[node_id])
                for node_id in candidate_ids
            ],
        }

    def _finalize_resolution(
        self,
        *,
        raw_query: str,
        structure: DocumentStructure,
        draft: _ResolutionDraft,
    ) -> StructuralResolution:
        by_id = {
            node.node_id: node
            for node in structure.nodes
        }

        raw_target_ids = list(
            dict.fromkeys(draft.target_node_ids)
        )
        invalid_target_ids = [
            node_id
            for node_id in raw_target_ids
            if node_id not in by_id
        ]

        if invalid_target_ids:
            return self._fallback_result(
                status=StructuralResolutionStatus.NO_MATCH,
                document_id=structure.document_id,
                raw_query=raw_query,
                reason="RESOLVER_RETURNED_UNVERIFIED_NODE_IDS",
                warning=(
                    "The structural resolver returned an unverified "
                    "source reference, so it was rejected."
                ),
            )

        target_nodes = [
            by_id[node_id]
            for node_id in raw_target_ids
        ]

        if draft.clarification_needed:
            candidates = [
                self._clarification_candidate(node)
                for node in target_nodes
            ]

            question = draft.clarification_question.strip()

            if not question:
                question = (
                    "Which of these document items did you mean?"
                )

            return StructuralResolution(
                status=(
                    StructuralResolutionStatus
                    .NEEDS_CLARIFICATION
                ),
                document_id=structure.document_id,
                raw_query=raw_query,
                action=draft.action,
                target_node_ids=[],
                candidate_node_ids=raw_target_ids,
                candidates=candidates,
                confidence=draft.confidence,
                match_mode=StructuralMatchMode.AMBIGUOUS,
                needs_visual=False,
                interpreted_user_terms=(
                    draft.interpreted_user_terms
                ),
                clarification_needed=True,
                clarification_question=question,
                reason=draft.reason,
                fallback_to_semantic=False,
            )

        if not target_nodes:
            return self._fallback_result(
                status=StructuralResolutionStatus.NO_MATCH,
                document_id=structure.document_id,
                raw_query=raw_query,
                reason=(
                    draft.reason
                    or "NO_STRUCTURAL_TARGET"
                ),
                action=draft.action,
                match_mode=StructuralMatchMode.NO_MATCH,
                interpreted_user_terms=(
                    draft.interpreted_user_terms
                ),
            )

        raw_visual_ids = list(
            dict.fromkeys(draft.visual_node_ids)
        )
        verified_visual_ids = [
            node_id
            for node_id in raw_visual_ids
            if node_id in by_id
        ]

        for node in target_nodes:
            if node.node_type == StructuralNodeType.FIGURE:
                verified_visual_ids.append(node.node_id)

            verified_visual_ids.extend(
                node.related_visual_node_ids
            )

        verified_visual_ids = [
            node_id
            for node_id in dict.fromkeys(
                verified_visual_ids
            )
            if node_id in by_id
        ]

        needs_visual = (
            draft.needs_visual
            or any(
                node.requires_visual_to_understand
                for node in target_nodes
            )
        )

        visual_nodes = [
            by_id[node_id]
            for node_id in verified_visual_ids
        ]

        evidence_nodes = list(target_nodes)

        if needs_visual:
            evidence_nodes.extend(visual_nodes)

        evidence_nodes = self._deduplicate_nodes(
            evidence_nodes
        )

        source_pages = self._pages_for_nodes(
            target_nodes
        )

        if needs_visual:
            visual_pages = self._pages_for_nodes(
                visual_nodes
            )

            if not visual_pages:
                visual_pages = source_pages
        else:
            visual_pages = []

        linked_retrieval_ids = list(
            dict.fromkeys(
                chunk_id
                for node in evidence_nodes
                for chunk_id in (
                    node.linked_retrieval_chunk_ids
                )
            )
        )

        linked_parent_ids = list(
            dict.fromkeys(
                parent_id
                for node in evidence_nodes
                for parent_id in (
                    node.linked_parent_chunk_ids
                )
            )
        )

        linked_figure_ids = list(
            dict.fromkeys(
                figure_id
                for node in evidence_nodes
                for figure_id in node.linked_figure_ids
            )
        )

        return StructuralResolution(
            status=StructuralResolutionStatus.RESOLVED,
            document_id=structure.document_id,
            raw_query=raw_query,
            action=draft.action,
            target_node_ids=raw_target_ids,
            confidence=draft.confidence,
            match_mode=draft.match_mode,
            needs_visual=needs_visual,
            visual_node_ids=verified_visual_ids,
            linked_retrieval_chunk_ids=(
                linked_retrieval_ids
            ),
            linked_parent_chunk_ids=linked_parent_ids,
            linked_figure_ids=linked_figure_ids,
            source_page_numbers=source_pages,
            visual_page_numbers=visual_pages,
            interpreted_user_terms=(
                draft.interpreted_user_terms
            ),
            clarification_needed=False,
            clarification_question="",
            reason=draft.reason,
            fallback_to_semantic=False,
            answer_scope=self._answer_scope(
                action=draft.action,
                target_node_ids=raw_target_ids,
            ),
        )

    @staticmethod
    def _answer_scope(
        *,
        action: StructuralResolutionAction,
        target_node_ids: list[str],
    ) -> AnswerScopeContract:
        return AnswerScopeContract(
            requested_action=action,
            allowed_target_node_ids=(
                target_node_ids
            ),
            scope_rules=[
                (
                    "Only perform the requested action for the "
                    "resolved targets."
                ),
                (
                    "Treat selected nodes as evidence, not permission "
                    "to answer their entire parent section."
                ),
                "Do not perform a broader operation than requested.",
                (
                    "Use only the minimum surrounding context needed "
                    "for comprehension."
                ),
                "After covering the resolved targets, stop.",
            ],
        )

    def _candidate_payload(
        self,
        node: StructuralNode,
    ) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "label": node.label,
            "parent_heading": node.parent_heading,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "text_preview": node.text[
                : self.text_preview_characters
            ],
        }

    def _clarification_candidate(
        self,
        node: StructuralNode,
    ) -> ClarificationCandidate:
        return ClarificationCandidate(
            node_id=node.node_id,
            node_type=node.node_type,
            label=(
                node.exact_source_label
                or node.label
                or node.title
                or ""
            ),
            parent_heading=node.parent_heading,
            page_start=node.page_start,
            text_preview=node.text[
                : min(
                    self.text_preview_characters,
                    500,
                )
            ],
        )

    @staticmethod
    def _pages_for_nodes(
        nodes: list[StructuralNode],
    ) -> list[int]:
        pages: set[int] = set()

        for node in nodes:
            pages.update(
                span.page_number
                for span in node.source_spans
            )

            if (
                node.page_start is not None
                and node.page_end is not None
                and node.page_end >= node.page_start
            ):
                pages.update(
                    range(
                        node.page_start,
                        node.page_end + 1,
                    )
                )

        return sorted(
            page
            for page in pages
            if page > 0
        )

    @staticmethod
    def _deduplicate_nodes(
        nodes: list[StructuralNode],
    ) -> list[StructuralNode]:
        result: list[StructuralNode] = []
        seen: set[str] = set()

        for node in nodes:
            if node.node_id in seen:
                continue

            seen.add(node.node_id)
            result.append(node)

        return result

    @staticmethod
    def _fallback_result(
        *,
        status: StructuralResolutionStatus,
        document_id: str,
        raw_query: str,
        reason: str,
        warning: str | None = None,
        action: StructuralResolutionAction = (
            StructuralResolutionAction.ANSWER
        ),
        match_mode: StructuralMatchMode = (
            StructuralMatchMode.NO_MATCH
        ),
        interpreted_user_terms: list[
            InterpretedUserTerm
        ] | None = None,
    ) -> StructuralResolution:
        return StructuralResolution(
            status=status,
            document_id=document_id,
            raw_query=raw_query,
            action=action,
            confidence=0.0,
            match_mode=match_mode,
            needs_visual=False,
            interpreted_user_terms=(
                interpreted_user_terms or []
            ),
            clarification_needed=False,
            clarification_question="",
            reason=reason,
            fallback_to_semantic=True,
            structural_warning=warning,
        )
