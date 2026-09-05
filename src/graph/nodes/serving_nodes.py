from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol

from src.agents.tutor_agent import TutorAgent
from src.agents.verifier_agent import VerifierAgent
from src.graph.state import PhysicsTutorState
from src.models.contracts import (
    AnswerType,
    ConversationMessage,
    IntentDecision,
    LanguageCode,
    MemorySnapshot,
    PendingStructuralClarification,
    SessionDocumentReference,
    QueryRewriteResult,
    QueryUnderstandingResult,
    RequestIntent,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)
from src.query.service import QueryUnderstandingService
from src.retrieval.models import (
    ContextBundle,
    ContextItem,
    HybridRetrievalResult,
)
from src.retrieval.service import RetrievalService
from src.retrieval.structural_resolver import (
    StructuralResolution,
    StructuralResolutionStatus,
)


def _safe_debug(
    label: str,
    payload: Any,
) -> None:
    """
    Emit debug output without depending on the Windows console code page.

    ensure_ascii=True escapes arbitrary Unicode before stdout sees it.
    This affects logging only; it never mutates the real user query,
    equations, retrieval text, stored data, prompts, or responses.
    """

    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            default=str,
        )
    except Exception:
        serialized = ascii(payload)

    print(
        f"{label} {serialized}",
        flush=True,
    )


class SessionStoreProtocol(Protocol):
    """
    Minimal interface needed by the LangGraph serving nodes.

    The concrete local implementation may use SQLite/Redis/etc.
    """

    def load(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> MemorySnapshot | None:
        ...

    def save(
        self,
        *,
        user_id: str,
        session_id: str,
        memory: MemorySnapshot,
    ) -> None:
        ...


class LongTermMemoryProtocol(Protocol):
    """
    Durable user-learning memory interface.
    """

    def hydrate_memory(
        self,
        *,
        user_id: str,
        memory: MemorySnapshot,
    ) -> MemorySnapshot:
        ...

    def write_candidates(
        self,
        *,
        user_id: str,
        candidates: list[dict[str, Any]],
    ) -> bool:
        ...


class SemanticLearningMemoryProtocol(Protocol):
    """
    Meaning-based student learning memory.

    READ:
        current question -> relevant old learning
        memory -> Tutor personalization

    WRITE:
        verifier-PASS tutoring turn -> durable
        learning signal -> semantic store
    """

    def recall_for_tutor(
        self,
        *,
        user_id: str,
        query_text: str,
    ) -> str | None:
        ...

    def learn_from_turn(
        self,
        *,
        user_id: str,
        student_text: str,
        tutor_text: str,
        verification_passed: bool,
        verifier_text: str | None = None,
        session_id: str | None = None,
        document_id: str | None = None,
    ) -> Any:
        ...


class QueryCacheProtocol(Protocol):
    """
    Cache interface used by LangGraph serving.

    Cache implementations may perform:
    - exact lookup
    - semantic lookup
    - TTL handling
    - Redis failure fallback
    """

    def get(
        self,
        *,
        user_id: str,
        document_id: str,
        query: str,
        active_page: int | None = None,
        figure_id: str | None = None,
        language: str | None = None,
        grade: int | None = None,
    ) -> dict[str, Any] | None:
        ...

    def set(
        self,
        *,
        user_id: str,
        document_id: str,
        query: str,
        answer: dict[str, Any],
        verification_status: str,
        active_page: int | None = None,
        figure_id: str | None = None,
        language: str | None = None,
        grade: int | None = None,
    ) -> bool:
        ...

class OutputGuardProtocol(Protocol):
    """
    Optional adapter for src/guardrails/output_guard.py.

    The built-in serving guard still fails closed even when this adapter
    is absent.
    """

    def guard(
        self,
        *,
        answer: TutorAnswer,
        verification: VerificationResult | None,
        context: ContextBundle | None,
        strict_document_mode: bool,
    ) -> TutorAnswer:
        ...


class FigureArtifactResolverProtocol(Protocol):
    """
    Narrow chat-time interface for resolving canonical document figure IDs.

    The concrete implementation may be IngestionService or another service
    exposing the same method. ServingNodes does not know storage paths,
    artifact filenames, Physics topics, or figure semantics.
    """

    def resolve_figure_artifacts(
        self,
        *,
        user_id: str,
        document_id: str,
        figure_ids: list[str] | tuple[str, ...],
    ) -> list[Any]:
        ...


class StructuralResolverProtocol(Protocol):
    """Narrow serving-time interface for structural source resolution."""

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
        ...


class ServingNodes:
    """
    Deterministic LangGraph node collection for local PhyMentor AI.

    IMPORTANT ARCHITECTURE RULE:
    - TutorAgent and VerifierAgent are the ONLY two agents.
    - Intent/scope/memory/retrieval/cache/output steps are normal nodes.
    - Phase-4 RetrievalService is reused instead of reimplementing RAG.
    - Maximum Tutor generations per user request = 2.
    """

    MAX_GENERATION_ATTEMPTS = 2
    MAX_RETRIEVAL_QUERIES = 3

    OUT_OF_SCOPE_MESSAGE = (
        "This assistant currently supports school-level Physics for "
        "Classes 1–12. Please upload a Physics page or ask a "
        "school-level Physics question."
    )

    NO_DOCUMENT_MESSAGE = (
        "No uploaded Physics document is available for this "
        "document-grounded request. Please upload the relevant "
        "Physics PDF or image first."
    )

    AMBIGUOUS_DOCUMENT_MESSAGE = (
        "I can see more than one uploaded Physics document, but I cannot "
        "reliably tell which one this question refers to. Please mention "
        "the document name, topic, page, or diagram once; after that I can "
        "continue with it naturally."
    )

    MAX_DOCUMENT_RESOLUTION_PROBES = 12

    RETRY_EXHAUSTED_MESSAGE = (
        "I could not fully verify the answer within the safe retry limit."
    )

    NOT_ENOUGH_VERIFIED_LABEL = (
        "Not enough verified"
    )

    CLEARER_IMAGE_MESSAGE = (
        "The visual evidence is not clear enough for me to verify the "
        "diagram reliably. Please upload a clearer or closer crop."
    )

    INSUFFICIENT_MESSAGE = (
        "I do not have enough reliable evidence to answer this safely "
        "from the available Physics context."
    )

    def __init__(
        self,
        *,
        query_service: QueryUnderstandingService,
        retrieval_service: RetrievalService,
        tutor_agent: TutorAgent,
        verifier_agent: VerifierAgent,
        session_store: SessionStoreProtocol | None = None,
        long_term_memory: LongTermMemoryProtocol | None = None,
        semantic_learning_memory: (
            SemanticLearningMemoryProtocol | None
        ) = None,
        query_cache: QueryCacheProtocol | None = None,
        output_guard: OutputGuardProtocol | None = None,
        figure_artifact_resolver: (
            FigureArtifactResolverProtocol | None
        ) = None,
        structural_resolver: (
            StructuralResolverProtocol | None
        ) = None,
        max_merged_context_items: int = 8,
        max_merged_context_characters: int = 12000,
    ) -> None:
        if max_merged_context_items <= 0:
            raise ValueError(
                "max_merged_context_items must be positive."
            )

        if max_merged_context_characters <= 0:
            raise ValueError(
                "max_merged_context_characters must be positive."
            )

        self.query_service = query_service
        self.retrieval_service = retrieval_service
        self.tutor_agent = tutor_agent
        self.verifier_agent = verifier_agent

        self.session_store = session_store
        self.long_term_memory = long_term_memory
        self.semantic_learning_memory = (
            semantic_learning_memory
        )
        self.query_cache = query_cache
        self.output_guard_runner = output_guard
        self.figure_artifact_resolver = (
            figure_artifact_resolver
        )
        self.structural_resolver = (
            structural_resolver
        )

        self.max_merged_context_items = (
            max_merged_context_items
        )
        self.max_merged_context_characters = (
            max_merged_context_characters
        )

    # =========================================================
    # 1. validate_request
    # =========================================================

    def validate_request(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        request_id = state["request_id"].strip()
        user_id = state["user_id"].strip()
        session_id = state["session_id"].strip()
        raw_query = state["raw_query"].strip()

        normalized_query = " ".join(
            raw_query.split()
        )

        errors: list[str] = []

        if not request_id:
            errors.append(
                "request_id cannot be empty."
            )

        if not user_id:
            errors.append(
                "user_id cannot be empty."
            )

        if not session_id:
            errors.append(
                "session_id cannot be empty."
            )

        if not normalized_query:
            errors.append(
                "query cannot be empty."
            )

        selected_page = state.get(
            "selected_page"
        )

        if (
            selected_page is not None
            and selected_page < 1
        ):
            errors.append(
                "selected_page must be positive."
            )

        if errors:
            raise ValueError(
                " ".join(errors)
            )

        return {
            "normalized_query": (
                normalized_query
            ),
            "validation_passed": True,
            "errors": [],
            "generation_attempts": 0,
            "retrieval_rounds": 0,
            "retry_count": 0,
            "cache_hit": False,
            "broader_retrieval_requested": False,
            "structural_resolution_attempted": False,
            "structural_resolution": None,
            "structural_resolution_status": None,
            "structural_match_mode": None,
            "structural_clarification_required": False,
            "structural_clarification_question": None,
            "structural_target_node_ids": [],
            "structural_candidate_node_ids": [],
            "structural_linked_retrieval_chunk_ids": [],
            "structural_linked_parent_chunk_ids": [],
            "structural_linked_figure_ids": [],
            "structural_source_page_numbers": [],
            "structural_visual_page_numbers": [],
            "structural_needs_visual": False,
            "structural_answer_scope": None,
            "structural_fallback_to_semantic": True,
            "structural_warning": None,
        }

    # =========================================================
    # 2. load_session
    # =========================================================

    def load_session(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        """
        Load the existing session memory, merge the current request's
        lightweight document references into the Redis-backed session
        bookshelf, then apply request-local context overrides before
        query understanding runs.

        Important:
        - actual files/chunks/embeddings are NOT stored here
        - document references are deduplicated by document_id
        - older session documents remain available
        - the newest/current request metadata wins for duplicate IDs
        - explicit_document_id remains only a recent/contextual hint
        """
        memory = state.get("memory")

        if memory is None:
            if self.session_store is not None:
                loaded = self.session_store.load(
                    user_id=state["user_id"],
                    session_id=state["session_id"],
                )
                memory = (
                    loaded
                    if loaded is not None
                    else MemorySnapshot()
                )
            else:
                memory = MemorySnapshot()

        if self.long_term_memory is not None:
            memory = self.long_term_memory.hydrate_memory(
                user_id=state["user_id"],
                memory=memory,
            )

        updates: dict[str, Any] = {}

        explicit_document_id = state.get(
            "explicit_document_id"
        )

        if explicit_document_id:
            explicit_document_id = (
                explicit_document_id.strip()
            )

        # -----------------------------------------------------
        # SESSION DOCUMENT BOOKSHELF
        #
        # Merge:
        #   Redis remembered documents
        #       +
        #   documents supplied by this request
        #
        # This is lightweight metadata only:
        # document_id + name.
        # -----------------------------------------------------

        remembered_signature = [
            (
                item.document_id,
                item.name,
            )
            for item in memory.available_documents
        ]

        available_documents = (
            self._normalized_available_documents(
                state,
                memory=memory,
            )
        )

        # Backward compatibility:
        # an older client may send only document_id.
        if explicit_document_id:
            known_ids = {
                item["document_id"]
                for item in available_documents
            }

            if (
                explicit_document_id
                not in known_ids
            ):
                available_documents.append(
                    {
                        "document_id": (
                            explicit_document_id
                        ),
                        "name": (
                            explicit_document_id
                        ),
                    }
                )

        available_documents = (
            available_documents[-30:]
        )

        available_document_references = [
            SessionDocumentReference(
                document_id=item[
                    "document_id"
                ],
                name=item["name"],
            )
            for item in available_documents
        ]

        current_signature = [
            (
                item.document_id,
                item.name,
            )
            for item
            in available_document_references
        ]

        bookshelf_changed = (
            current_signature
            != remembered_signature
        )

        if bookshelf_changed:
            updates[
                "available_documents"
            ] = (
                available_document_references
            )

        if explicit_document_id:
            updates["active_document_id"] = (
                explicit_document_id
            )

        selected_page = state.get(
            "selected_page"
        )
        if selected_page is not None:
            updates["active_page"] = selected_page

        selected_figure_id = state.get(
            "selected_figure_id"
        )
        if selected_figure_id:
            updates["last_selected_figure_id"] = (
                selected_figure_id.strip()
            )

        # PhyMentor's serving policy is English-only.
        # Keep old enum values readable in stored data, but never carry a
        # non-English preference into the current workflow.
        updates["language"] = (
            LanguageCode.ENGLISH
        )

        if updates:
            memory = memory.model_copy(
                update=updates
            )

        # Save a changed bookshelf immediately.
        #
        # This makes the session remember new document references
        # even when the current turn later takes a fast path and
        # memory_write_decision does not run a normal chat write.
        if (
            bookshelf_changed
            and self.session_store is not None
        ):
            self.session_store.save(
                user_id=state["user_id"],
                session_id=state[
                    "session_id"
                ],
                memory=memory,
            )

        return {
            "memory": memory,
            "pending_structural_clarification": (
                memory.pending_structural_clarification
            ),
            "conversation_summary": (
                state.get(
                    "conversation_summary",
                    "",
                )
            ),
            "recent_messages": [
                {
                    "role": item.role,
                    "content": item.content,
                }
                for item in memory.recent_messages
            ],
        }

    # =========================================================
    # 3. classify_intent
    # =========================================================

    def classify_intent(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        """
        Run the existing PUBLIC query-understanding service once.

        The graph still exposes separate intent/scope/rewrite nodes, but they
        project their own part of this already-validated result. This avoids
        coupling LangGraph to private QueryUnderstandingService methods and
        preserves the existing ServingService test doubles.
        """
        raw_query = state["raw_query"]
        memory = state.get(
            "memory",
            MemorySnapshot(),
        )

        understanding_arguments: dict[
            str,
            Any,
        ] = {
            "query": raw_query,
            "memory": memory,
            "upload_present": bool(
                state.get(
                    "upload_present",
                    False,
                )
            ),
        }

        selected_model = state.get(
            "selected_model"
        )

        if selected_model is not None:
            # Preserve compatibility with older query-service test doubles
            # when no model is explicitly selected. The new keyword is sent
            # only for a real request-level model override.
            understanding_arguments[
                "selected_model"
            ] = selected_model

        understanding = (
            self.query_service.understand(
                **understanding_arguments
            )
        )

        intent = understanding.intent

        # Query understanding may still read historical multilingual data,
        # but the product-facing tutoring workflow is English-only.
        if intent.language != LanguageCode.ENGLISH:
            intent = intent.model_copy(
                update={
                    "language": (
                        LanguageCode.ENGLISH
                    )
                }
            )

            understanding = (
                understanding.model_copy(
                    update={
                        "intent": intent
                    }
                )
            )

        _safe_debug(
            "[PHYMENTOR-DEBUG][intent]",
            {
                "query_length": len(raw_query),
                "intent": intent.intent.value,
                "prefer_visual": bool(intent.prefer_visual),
                "requires_document": intent.requires_document,
                "requires_visual": intent.requires_visual,
                "document_usage": (
                    intent.document_usage.value
                    if intent.document_usage is not None
                    else None
                ),
                "requested_quantities": [
                    item.quantity
                    for item in intent.requested_quantities
                ],
                "given_quantity_count": len(
                    intent.given_quantities
                ),
                "given_equation_count": len(
                    intent.given_equations
                ),
                "estimated_grade": intent.estimated_grade,
            },
        )

        updates: dict[str, Any] = {
            "query_understanding": (
                understanding
            ),
            "intent": intent,
            "language": intent.language,
            "estimated_grade": (
                intent.estimated_grade
                or memory.estimated_grade
            ),
            "prefer_visual": bool(
                intent.prefer_visual
                or intent.requires_visual is True
            ),
        }

        if (
            intent.intent
            == RequestIntent.GREETING
        ):
            updates.update(
                {
                    "answer_draft": (
                        self._greeting_answer(
                            intent=intent
                        )
                    ),
                    "terminal_action": None,
                    "verification_passed": True,
                    "next_pending_structural_clarification": None,
                }
            )

        elif (
            intent.intent
            == RequestIntent.UPLOAD_DOCUMENT
        ):
            updates.update(
                {
                    "answer_draft": (
                        self._direct_answer(
                            (
                                "Please use the document upload flow. "
                                "After the Physics document is ready, "
                                "ask your question about it."
                            )
                        )
                    ),
                    "terminal_action": None,
                    "verification_passed": True,
                    "next_pending_structural_clarification": None,
                }
            )

        return updates

    # =========================================================
    # 4. scope_guard
    # =========================================================

    def scope_guard(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        intent = state["intent"]

        understanding = state.get(
            "query_understanding"
        )

        if understanding is None:
            raise RuntimeError(
                "Query understanding must be available before scope_guard."
            )

        scope = understanding.scope

        if intent.intent in {
            RequestIntent.GREETING,
            RequestIntent.UPLOAD_DOCUMENT,
        }:
            return {
                "scope": None,
                "scope_status": "NOT_REQUIRED",
                "scope_confidence": 1.0,
            }

        if scope is None:
            raise RuntimeError(
                "Scope decision is missing for a non-fast-path request."
            )

        updates: dict[str, Any] = {
            "scope": scope,
            "scope_status": (
                scope.status.value
            ),
            "scope_confidence": (
                scope.confidence
            ),
        }

        if self._is_out_of_scope(
            intent=intent,
            scope=scope,
        ):
            updates.update(
                {
                    "answer_draft": (
                        self._direct_answer(
                            self.OUT_OF_SCOPE_MESSAGE
                        )
                    ),
                    "terminal_action": (
                        VerificationAction
                        .REJECT_OUT_OF_SCOPE
                    ),
                    "verification_passed": False,
                    "next_pending_structural_clarification": None,
                }
            )

        return updates

    # =========================================================
    # 5. resolve_active_document
    # =========================================================

    def resolve_active_document(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        """
        Resolve the document that should be used for THIS turn.

        Multi-document rules:

        - available_documents contains every document known to the
          current frontend chat session.
        - explicit_document_id is only the recent/current hint from
          the client. Its presence alone never forces RAG.
        - The current query first decides whether document grounding
          is needed.
        - If grounding is needed, the resolver can move between older
          and newer uploaded documents automatically.
        - Retrieval still runs against exactly ONE resolved document
          for the turn, preserving user/document isolation.
        """

        memory = state.get(
            "memory",
            MemorySnapshot(),
        )

        understanding = state.get(
            "query_understanding"
        )

        explicit_document_id = state.get(
            "explicit_document_id"
        )

        if explicit_document_id:
            explicit_document_id = (
                explicit_document_id.strip()
            )

        understood_document_id = (
            understanding.active_document_id
            if understanding is not None
            else None
        )

        available_documents = (
            self._normalized_available_documents(
                state,
                memory=memory,
            )
        )

        # Keep a backward-compatible candidate even when an older
        # client sends only document_id and no available_documents.
        known_ids = {
            item["document_id"]
            for item in available_documents
        }

        if (
            explicit_document_id
            and explicit_document_id
            not in known_ids
        ):
            available_documents.append(
                {
                    "document_id": (
                        explicit_document_id
                    ),
                    "name": (
                        explicit_document_id
                    ),
                }
            )
            known_ids.add(
                explicit_document_id
            )

        if (
            memory.active_document_id
            and memory.active_document_id
            not in known_ids
        ):
            available_documents.append(
                {
                    "document_id": (
                        memory.active_document_id
                    ),
                    "name": (
                        memory.active_document_id
                    ),
                }
            )
            known_ids.add(
                memory.active_document_id
            )

        if (
            understood_document_id
            and understood_document_id
            not in known_ids
        ):
            available_documents.append(
                {
                    "document_id": (
                        understood_document_id
                    ),
                    "name": (
                        understood_document_id
                    ),
                }
            )

        recent_document_id = (
            explicit_document_id
            or memory.active_document_id
            or understood_document_id
        )

        use_document_for_turn = (
            self._should_use_document_for_turn(
                state=state,
                explicit_document_id=(
                    recent_document_id
                ),
                available_documents=(
                    available_documents
                ),
            )
        )

        resolved_document_id: (
            str | None
        ) = None

        ambiguous = False

        if use_document_for_turn:
            (
                resolved_document_id,
                ambiguous,
            ) = self._resolve_document_for_turn(
                state=state,
                available_documents=(
                    available_documents
                ),
                recent_document_id=(
                    recent_document_id
                ),
            )

        active_document_id = (
            resolved_document_id
            if use_document_for_turn
            else None
        )

        # Always derive strict mode fresh for this turn.
        # A previous document-grounded question must never make
        # a later general Physics question strict.
        strict_document_mode = bool(
            use_document_for_turn
        )

        updates: dict[str, Any] = {
            "active_document_id": (
                active_document_id
            ),
            "strict_document_mode": (
                strict_document_mode
            ),
        }

        _safe_debug(
            "[PHYMENTOR-DEBUG][document]",
            {
                "use_document_for_turn": use_document_for_turn,
                "resolved_document_id": active_document_id,
                "recent_document_id": recent_document_id,
                "strict_document_mode": strict_document_mode,
                "ambiguous": ambiguous,
                "available_documents": [
                    {
                        "document_id": item.get("document_id"),
                        "name": item.get("name"),
                    }
                    for item in available_documents
                ],
            },
        )

        # A document-grounded question must never silently fall
        # back to a random document or to ungrounded answering.
        if (
            use_document_for_turn
            and not active_document_id
            and not self._has_terminal_answer(
                state
            )
        ):
            message = (
                self.AMBIGUOUS_DOCUMENT_MESSAGE
                if ambiguous
                else self.NO_DOCUMENT_MESSAGE
            )

            updates.update(
                {
                    "answer_draft": (
                        self._insufficient_answer(
                            message
                        )
                    ),
                    "terminal_action": (
                        VerificationAction
                        .INSUFFICIENT_EVIDENCE
                    ),
                    "verification_passed": False,
                }
            )

        return updates

    # =========================================================
    # 6. load_short_term_memory
    # =========================================================

    def load_short_term_memory(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        memory = state.get(
            "memory",
            MemorySnapshot(),
        )

        # English-only serving policy. Historical memory may contain an
        # older language value, but it is not reused for this turn.
        language = LanguageCode.ENGLISH

        turn_document_id = (
            state.get(
                "active_document_id"
            )
        )

        # Page/figure context is request-local.
        # When this turn is general Physics, do not leak
        # an old PDF page/figure into cache/retrieval/Tutor
        # state. The persistent MemorySnapshot still keeps
        # that document context for a later PDF question.
        if turn_document_id:
            active_page = (
                state.get("selected_page")
                or memory.active_page
            )

            selected_figure = (
                state.get(
                    "selected_figure_id"
                )
                or memory
                .last_selected_figure_id
            )
        else:
            active_page = None
            selected_figure = None

        memory_updates: dict[
            str,
            Any,
        ] = {
            "language": language,
            "estimated_grade": (
                state.get(
                    "estimated_grade"
                )
                or memory.estimated_grade
            ),
        }

        if turn_document_id:
            memory_updates.update(
                {
                    "active_document_id": (
                        turn_document_id
                    ),
                    "active_page": (
                        active_page
                    ),
                    "last_selected_figure_id": (
                        selected_figure
                    ),
                }
            )

        updated_memory = (
            memory.model_copy(
                update=memory_updates
            )
        )

        return {
            "memory": updated_memory,
            "pending_structural_clarification": (
                updated_memory
                .pending_structural_clarification
            ),
            "active_page": active_page,
            "referenced_figure_id": (
                selected_figure
            ),
            "language": language,
            "recent_messages": [
                {
                    "role": item.role,
                    "content": item.content,
                }
                for item in (
                    updated_memory
                    .recent_messages
                )
            ],
        }

    # =========================================================
    # 6A. resolve_structural_reference
    # =========================================================

    def resolve_structural_reference(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        """
        Resolve a natural-language source reference before ordinary RAG.

        This node never replaces FAISS/BM25/reranking. A verified structural
        match contributes exact source identity, page/figure preferences and
        a strict answer-scope contract. A safe no-match keeps the old semantic
        retrieval path. Genuine ambiguity stops the flow and asks the student
        one bounded clarification question instead of guessing.
        """

        pending = state.get(
            "pending_structural_clarification"
        )

        if pending is None:
            pending = state.get(
                "memory",
                MemorySnapshot(),
            ).pending_structural_clarification

        base_updates: dict[str, Any] = {
            "structural_resolution_attempted": False,
            "structural_resolution": None,
            "structural_resolution_status": None,
            "structural_match_mode": None,
            "structural_clarification_required": False,
            "structural_clarification_question": None,
            "structural_target_node_ids": [],
            "structural_candidate_node_ids": [],
            "structural_linked_retrieval_chunk_ids": [],
            "structural_linked_parent_chunk_ids": [],
            "structural_linked_figure_ids": [],
            "structural_source_page_numbers": [],
            "structural_visual_page_numbers": [],
            "structural_needs_visual": False,
            "structural_answer_scope": None,
            "structural_fallback_to_semantic": True,
            "structural_warning": None,
        }

        if self._has_terminal_answer(state):
            base_updates[
                "next_pending_structural_clarification"
            ] = None
            return base_updates

        document_id = state.get(
            "active_document_id"
        )

        if not document_id:
            # A non-document turn supersedes a pending document choice.
            base_updates[
                "next_pending_structural_clarification"
            ] = None
            return base_updates

        if (
            pending is not None
            and pending.document_id != document_id
        ):
            pending = None

        if self.structural_resolver is None:
            # Feature wiring is optional during staged migration. The
            # established semantic retrieval path remains available.
            base_updates[
                "next_pending_structural_clarification"
            ] = pending
            return base_updates

        query = (
            state.get("normalized_query")
            or state["raw_query"]
        )

        try:
            resolution = (
                self.structural_resolver.resolve(
                    query=query,
                    user_id=state["user_id"],
                    document_id=document_id,
                    pending_clarification=pending,
                )
            )
        except Exception as exc:
            # Structural enrichment must not destroy the established RAG
            # path. The failure is visible in state/debugging and semantic
            # retrieval remains enabled.
            warning = (
                "Structural source resolution was unavailable; "
                "semantic retrieval will be used."
            )
            _safe_debug(
                "[PHYMENTOR-DEBUG][structural-error]",
                {
                    "document_id": document_id,
                    "error_type": type(exc).__name__,
                },
            )
            base_updates.update(
                {
                    "structural_resolution_attempted": True,
                    "structural_warning": warning,
                    "next_pending_structural_clarification": (
                        pending
                    ),
                }
            )
            return base_updates

        updates = {
            **base_updates,
            "structural_resolution_attempted": True,
            "structural_resolution": resolution,
            "structural_resolution_status": resolution.status,
            "structural_match_mode": resolution.match_mode,
            "structural_clarification_required": (
                resolution.status
                == StructuralResolutionStatus.NEEDS_CLARIFICATION
            ),
            "structural_clarification_question": (
                resolution.clarification_question
                or None
            ),
            "structural_target_node_ids": list(
                resolution.target_node_ids
            ),
            "structural_candidate_node_ids": list(
                resolution.candidate_node_ids
            ),
            "structural_linked_retrieval_chunk_ids": list(
                resolution.linked_retrieval_chunk_ids
            ),
            "structural_linked_parent_chunk_ids": list(
                resolution.linked_parent_chunk_ids
            ),
            "structural_linked_figure_ids": list(
                resolution.linked_figure_ids
            ),
            "structural_source_page_numbers": list(
                resolution.source_page_numbers
            ),
            "structural_visual_page_numbers": list(
                resolution.visual_page_numbers
            ),
            "structural_needs_visual": resolution.needs_visual,
            "structural_answer_scope": resolution.answer_scope,
            "structural_fallback_to_semantic": (
                resolution.fallback_to_semantic
            ),
            "structural_warning": resolution.structural_warning,
        }

        if (
            resolution.status
            == StructuralResolutionStatus.NEEDS_CLARIFICATION
        ):
            next_pending = (
                resolution.to_pending_clarification()
            )
            updates.update(
                {
                    "next_pending_structural_clarification": (
                        next_pending
                    ),
                    "answer_draft": (
                        self._structural_clarification_answer(
                            resolution
                        )
                    ),
                    "terminal_action": None,
                    "verification_passed": True,
                }
            )
            return updates

        if (
            resolution.status
            == StructuralResolutionStatus.RESOLVED
        ):
            preferred_pages = list(
                dict.fromkeys(
                    [
                        *state.get(
                            "preferred_page_numbers",
                            [],
                        ),
                        *resolution.source_page_numbers,
                        *resolution.visual_page_numbers,
                    ]
                )
            )

            updates.update(
                {
                    "next_pending_structural_clarification": None,
                    "preferred_page_numbers": preferred_pages,
                    "prefer_visual": bool(
                        state.get("prefer_visual", False)
                        or resolution.needs_visual
                    ),
                }
            )

            if (
                not state.get("referenced_figure_id")
                and resolution.linked_figure_ids
            ):
                updates["referenced_figure_id"] = (
                    resolution.linked_figure_ids[0]
                )

            return updates

        # NO_MATCH abandons an old clarification because the resolver has
        # determined that this turn does not safely select its candidates.
        # A temporary structure/model failure preserves it for a retry.
        if (
            resolution.status
            == StructuralResolutionStatus.STRUCTURE_UNAVAILABLE
        ):
            updates[
                "next_pending_structural_clarification"
            ] = pending
        else:
            updates[
                "next_pending_structural_clarification"
            ] = None

        return updates


    # =========================================================
    # 7. rewrite_contextual_query
    # =========================================================

    def rewrite_contextual_query(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        intent = state["intent"]
        scope = state.get("scope")

        structural_pages = list(
            dict.fromkeys(
                [
                    *state.get(
                        "structural_source_page_numbers",
                        [],
                    ),
                    *state.get(
                        "structural_visual_page_numbers",
                        [],
                    ),
                ]
            )
        )

        understanding = state.get(
            "query_understanding"
        )

        if understanding is None:
            raise RuntimeError(
                "Query understanding must be available before query rewrite."
            )

        rewrite = understanding.rewrite

        if (
            intent.intent
            in {
                RequestIntent.GREETING,
                RequestIntent.UPLOAD_DOCUMENT,
            }
            or self._is_out_of_scope(
                intent=intent,
                scope=scope,
            )
        ):
            return {
                "rewrite": rewrite,
                "rewritten_query": (
                    understanding
                    .normalized_query
                ),
                "retrieval_queries": [],
                "use_hyde": False,
                "hyde_text": None,
            }

        if rewrite is None:
            return {
                "rewrite": None,
                "rewritten_query": (
                    understanding
                    .normalized_query
                ),
                "retrieval_queries": [],
                "preferred_page_numbers": (
                    structural_pages
                ),
                "referenced_figure_id": (
                    state.get(
                        "referenced_figure_id"
                    )
                ),
                "prefer_visual": bool(
                    state.get(
                        "prefer_visual",
                        False,
                    )
                ),
                "use_hyde": False,
                "hyde_text": None,
            }

        return {
            "rewrite": rewrite,
            "rewritten_query": (
                rewrite.rewritten_query
            ),
            "retrieval_queries": list(
                rewrite.retrieval_queries
            ),
            "preferred_page_numbers": (
                list(
                    dict.fromkeys(
                        [
                            *structural_pages,
                            *rewrite
                            .preferred_page_numbers,
                        ]
                    )
                )
            ),
            "referenced_figure_id": (
                rewrite.referenced_figure_id
                or state.get(
                    "referenced_figure_id"
                )
            ),
            "prefer_visual": bool(
                state.get(
                    "prefer_visual",
                    False,
                )
                or rewrite.prefer_visual
            ),
            "use_hyde": rewrite.use_hyde,
            "hyde_text": rewrite.hyde_text,
        }

    # =========================================================
    # 8. check_query_cache
    # =========================================================

    def check_query_cache(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        cache_key = self._build_cache_key(
            state
        )

        if (
            self.query_cache is None
            or self._should_skip_cache_lookup(
                state
            )
        ):
            return {
                "cache_key": cache_key,
                "cache_hit": False,
                "cached_answer": None,
            }

        query = (
            state.get(
                "rewritten_query"
            )
            or state["normalized_query"]
        )

        document_id = (
            state.get(
                "active_document_id"
            )
            or "no-document"
        )

        language = self._cache_language_value(
            state.get("language")
        )

        cached = self.query_cache.get(
            user_id=state["user_id"],
            document_id=document_id,
            query=query,
            active_page=state.get(
                "active_page"
            ),
            figure_id=state.get(
                "referenced_figure_id"
            ),
            language=language,
            grade=state.get(
                "estimated_grade"
            ),
        )

        if cached is None:
            return {
                "cache_key": cache_key,
                "cache_hit": False,
                "cached_answer": None,
            }

        answer = TutorAnswer.model_validate(
            cached
        )

        return {
            "cache_key": cache_key,
            "cache_hit": True,
            "cached_answer": answer,
            "answer_draft": answer,
            "generation_attempts": 0,
            "retrieval_rounds": 0,
        }

    # =========================================================
    # 9. retrieval_planner
    # =========================================================

    def retrieval_planner(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        if self._has_terminal_answer(
            state
        ):
            return {}

        if state.get(
            "cache_hit",
            False,
        ):
            return {}

        candidates: list[str] = []

        rewrite = state.get("rewrite")

        if rewrite is not None:
            candidates.extend(
                rewrite.retrieval_queries
            )

            if (
                rewrite.use_hyde
                and rewrite.hyde_text
            ):
                candidates.append(
                    rewrite.hyde_text
                )

        if not candidates:
            candidates.append(
                state["normalized_query"]
            )

        cleaned: list[str] = []

        for item in candidates:
            normalized = " ".join(
                item.strip().split()
            )

            if (
                normalized
                and normalized
                not in cleaned
            ):
                cleaned.append(normalized)

            if (
                len(cleaned)
                >= self.MAX_RETRIEVAL_QUERIES
            ):
                break

        return {
            "retrieval_queries": cleaned,
        }

    # =========================================================
    # 10. hybrid_retrieval
    # =========================================================

    def hybrid_retrieval(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        if self._has_terminal_answer(
            state
        ):
            return {}

        if state.get(
            "cache_hit",
            False,
        ):
            return {}

        active_document_id = (
            state.get(
                "active_document_id"
            )
        )

        # Generic school-Physics questions may be answered without a document
        # when strict document mode is false.
        if not active_document_id:
            return {
                "retrieval_results": [],
                "retrieval_rounds": 0,
            }

        queries = (
            state.get(
                "retrieval_queries"
            )
            or [
                state["normalized_query"]
            ]
        )

        results = self._run_retrieval_round(
            queries=queries,
            state=state,
            broader=False,
        )

        return {
            "retrieval_results": results,
            "retrieval_rounds": 1,
        }

    # =========================================================
    # 11. reranking checkpoint
    # Phase-4 RetrievalService has already done CrossEncoder reranking.
    # This graph node records the resulting candidates without re-running it.
    # =========================================================

    def reranking(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        retrieved_chunks: list[
            dict[str, Any]
        ] = []

        for result in state.get(
            "retrieval_results",
            [],
        ):
            for item in (
                result.reranked_hits
            ):
                hit = item.hit

                if not self._same_scope(
                    state=state,
                    user_id=hit.user_id,
                    document_id=(
                        hit.document_id
                    ),
                ):
                    continue

                retrieved_chunks.append(
                    {
                        "chunk_id": (
                            hit.chunk_id
                        ),
                        "page_number": (
                            hit.page_number
                        ),
                        "content_type": (
                            hit.content_type
                        ),
                        "chunk_kind": (
                            hit.chunk_kind
                        ),
                        "parent_id": (
                            hit.parent_id
                        ),
                        "score": (
                            item.rerank_score
                        ),
                        "source_chunk_ids": [
                            hit.chunk_id
                        ],
                    }
                )

        return {
            "retrieved_chunks": (
                retrieved_chunks
            )
        }

    # =========================================================
    # 12. visual_context_resolver
    # =========================================================

    def visual_context_resolver(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        """
        Resolve retrieved canonical figure IDs to actual stored image paths.

        Retrieval/chunking decides which figure IDs are linked to the evidence.
        This node performs identity resolution only. It does not infer Physics
        topics, semantic figure meaning, source labels, or positional phrases.
        """

        figures: list[
            dict[str, Any]
        ] = []

        seen: set[str] = set()

        for result in state.get(
            "retrieval_results",
            [],
        ):
            for item in (
                result.context.items
            ):
                if not self._same_scope(
                    state=state,
                    user_id=item.user_id,
                    document_id=(
                        item.document_id
                    ),
                ):
                    continue

                linked_ids = [
                    figure_id.strip()
                    for figure_id
                    in item.linked_figure_ids
                    if (
                        isinstance(
                            figure_id,
                            str,
                        )
                        and figure_id.strip()
                    )
                ]

                for figure_id in linked_ids:
                    if figure_id in seen:
                        continue

                    seen.add(figure_id)

                    # A ContextItem has only one image_path. It is safe to
                    # associate that path directly only when the item links
                    # exactly one canonical figure. Multi-figure items are
                    # resolved below from the canonical catalogue instead of
                    # guessing which ID the one path belongs to.
                    directly_bound = bool(
                        len(linked_ids) == 1
                        and isinstance(
                            item.image_path,
                            str,
                        )
                        and item.image_path.strip()
                    )

                    figures.append(
                        {
                            "figure_id": (
                                figure_id
                            ),
                            "page_number": (
                                item.page_number
                            ),
                            "image_path": (
                                item.image_path
                                if directly_bound
                                else None
                            ),
                            "caption": (
                                item.caption
                                if directly_bound
                                else None
                            ),
                            "context_id": (
                                item.context_id
                            ),
                        }
                    )

        unresolved_ids = [
            item["figure_id"]
            for item in figures
            if not item.get(
                "image_path"
            )
        ]

        active_document_id = (
            state.get(
                "active_document_id"
            )
        )

        if (
            unresolved_ids
            and active_document_id
            and self.figure_artifact_resolver
            is not None
        ):
            try:
                resolved = (
                    self.figure_artifact_resolver
                    .resolve_figure_artifacts(
                        user_id=state[
                            "user_id"
                        ],
                        document_id=(
                            active_document_id
                        ),
                        figure_ids=(
                            unresolved_ids
                        ),
                    )
                )

                resolved_by_id = {
                    str(
                        figure.figure_id
                    ).strip(): figure
                    for figure in resolved
                    if str(
                        getattr(
                            figure,
                            "figure_id",
                            "",
                        )
                    ).strip()
                }

                for item in figures:
                    if item.get(
                        "image_path"
                    ):
                        continue

                    figure = (
                        resolved_by_id.get(
                            item["figure_id"]
                        )
                    )

                    if figure is None:
                        continue

                    item["page_number"] = (
                        figure.page_number
                    )
                    item["image_path"] = (
                        figure.image_path
                    )
                    item["caption"] = (
                        figure.caption
                    )

            except Exception as exc:
                # Artifact resolution is enrichment. Do not crash the request;
                # the Tutor/Verifier visual hard gate will fail closed if the
                # required actual image still cannot be supplied.
                _safe_debug(
                    "[PHYMENTOR-DEBUG][figure-resolution]",
                    {
                        "document_id": (
                            active_document_id
                        ),
                        "requested_count": len(
                            unresolved_ids
                        ),
                        "resolved": False,
                        "error_type": (
                            type(exc).__name__
                        ),
                    },
                )

        _safe_debug(
            "[PHYMENTOR-DEBUG][figures]",
            {
                "linked_count": len(
                    figures
                ),
                "resolved_image_count": sum(
                    1
                    for item in figures
                    if item.get(
                        "image_path"
                    )
                ),
            },
        )

        return {
            "retrieved_figures": figures,
        }


    # =========================================================
    # 13. context_compression
    # Phase-4 RetrievalService already produced parent-expanded compressed
    # ContextBundle objects. Here we merge up to 3 query results safely.
    # =========================================================

    def context_compression(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        results = state.get(
            "retrieval_results",
            [],
        )

        if not results:
            return {
                "reranked_context": None
            }

        active_document_id = (
            state.get(
                "active_document_id"
            )
        )

        if not active_document_id:
            return {
                "reranked_context": None
            }

        context = self._merge_contexts(
            query=(
                state.get(
                    "rewritten_query"
                )
                or state[
                    "normalized_query"
                ]
            ),
            user_id=state["user_id"],
            document_id=(
                active_document_id
            ),
            results=results,
        )

        context = (
            self._attach_resolved_figures_to_context(
                context=context,
                retrieved_figures=state.get(
                    "retrieved_figures",
                    [],
                ),
                referenced_figure_id=state.get(
                    "referenced_figure_id"
                ),
            )
        )

        return {
            "reranked_context": context
        }

    # =========================================================
    # 14. tutor_agent
    # =========================================================

    def tutor_agent_node(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        if self._has_terminal_answer(
            state
        ):
            return {}

        if state.get(
            "cache_hit",
            False,
        ):
            return {}

        attempts = int(
            state.get(
                "generation_attempts",
                0,
            )
        )

        if (
            attempts
            >= self.MAX_GENERATION_ATTEMPTS
        ):
            return {
                "answer_draft": (
                    self._insufficient_answer(
                        self.RETRY_EXHAUSTED_MESSAGE
                    )
                ),
                "terminal_action": (
                    VerificationAction
                    .INSUFFICIENT_EVIDENCE
                ),
                "verification_passed": False,
            }

        intent = state["intent"]
        scope = state.get("scope")
        context = state.get(
            "reranked_context"
        )
        memory = state.get(
            "memory",
            MemorySnapshot(),
        )

        semantic_memory_context: (
            str | None
        ) = None

        if (
            self.semantic_learning_memory
            is not None
        ):
            semantic_query = (
                state.get(
                    "rewritten_query"
                )
                or state[
                    "normalized_query"
                ]
            )

            try:
                semantic_memory_context = (
                    self.semantic_learning_memory
                    .recall_for_tutor(
                        user_id=state[
                            "user_id"
                        ],
                        query_text=(
                            semantic_query
                        ),
                    )
                )
            except Exception:
                # Semantic personalization is useful
                # but must never block the Tutor.
                semantic_memory_context = None

        tutor_scope_arguments: dict[
            str,
            Any,
        ] = {}

        structural_answer_scope = state.get(
            "structural_answer_scope"
        )

        if structural_answer_scope is not None:
            # Keep the legacy Tutor call byte-compatible when no structural
            # target was resolved. Real TutorAgent receives this optional
            # contract only for a resolver-verified source item.
            tutor_scope_arguments[
                "structural_answer_scope"
            ] = structural_answer_scope

        selected_model = state.get(
            "selected_model"
        )

        if selected_model is not None:
            # Preserve compatibility with older Tutor test doubles when the
            # request did not explicitly choose a model.
            tutor_scope_arguments[
                "selected_model"
            ] = selected_model

        answer = self.tutor_agent.answer(
            query=state[
                "raw_query"
            ],
            intent=intent,
            scope=scope,
            context=context,
            memory=memory,
            semantic_memory_context=(
                semantic_memory_context
            ),
            strict_document_mode=bool(
                state.get(
                    "strict_document_mode",
                    False,
                )
            ),
            verifier_feedback=(
                state.get(
                    "verifier_feedback"
                )
                or None
            ),
            **tutor_scope_arguments,
        )

        attempts += 1

        _safe_debug(
            "[PHYMENTOR-DEBUG][tutor]",
            {
                "attempt": attempts,
                "answer_type": answer.answer_type.value,
                "direct_answer": (
                    answer.direct_answer[:500]
                    if answer.direct_answer
                    else ""
                ),
                "steps": list(answer.steps)[:10],
                "formulae": [
                    {
                        "latex": formula.latex,
                        "meaning": formula.meaning,
                    }
                    for formula in answer.formulae[:10]
                ],
                "final_result": answer.final_result,
                "source_pages": list(answer.source_pages),
                "citations": [
                    citation.model_dump(mode="json")
                    for citation in answer.citations
                ],
                "strict_document_mode": bool(
                    state.get("strict_document_mode", False)
                ),
                "context_items": (
                    len(context.items)
                    if context is not None
                    else 0
                ),
                "context_has_image": bool(
                    context
                    and any(
                        bool(
                            item.image_path
                            and item.image_path.strip()
                        )
                        for item in context.items
                    )
                ),
                "verifier_feedback_in": (
                    state.get("verifier_feedback")
                    or []
                ),
                "structural_scope_applied": (
                    structural_answer_scope
                    is not None
                ),
            },
        )

        return {
            "answer_draft": answer,
            "generation_attempts": attempts,
            "retry_count": max(
                0,
                attempts - 1,
            ),
            "citations": list(
                answer.citations
            ),
        }

    # =========================================================
    # 15. verifier_agent
    # =========================================================

    def verifier_agent_node(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        answer = state.get(
            "answer_draft"
        )

        if answer is None:
            raise RuntimeError(
                "Tutor answer is missing before verification."
            )

        verifier_scope_arguments: dict[
            str,
            Any,
        ] = {}

        structural_answer_scope = state.get(
            "structural_answer_scope"
        )

        if structural_answer_scope is not None:
            # Preserve the established Verifier call when no structural
            # target exists. The optional contract is forwarded only for a
            # resolver-verified source item.
            verifier_scope_arguments[
                "structural_answer_scope"
            ] = structural_answer_scope

        selected_model = state.get(
            "selected_model"
        )

        if selected_model is not None:
            # Preserve compatibility with older Verifier test doubles when the
            # request did not explicitly choose a model.
            verifier_scope_arguments[
                "selected_model"
            ] = selected_model

        verification = (
            self.verifier_agent.verify(
                query=state[
                    "raw_query"
                ],
                intent=state["intent"],
                scope=state.get(
                    "scope"
                ),
                tutor_answer=answer,
                context=state.get(
                    "reranked_context"
                ),
                strict_document_mode=bool(
                    state.get(
                        "strict_document_mode",
                        False,
                    )
                ),
                **verifier_scope_arguments,
            )
        )

        passed = (
            verification.action
            == VerificationAction.PASS
        )

        _safe_debug(
            "[PHYMENTOR-DEBUG][verifier]",
            {
                "action": verification.action.value,
                "grounded": verification.grounded,
                "physics_correct": verification.physics_correct,
                "calculation_correct": verification.calculation_correct,
                "units_correct": verification.units_correct,
                "diagram_claims_supported": (
                    verification.diagram_claims_supported
                ),
                "within_school_scope": (
                    verification.within_school_scope
                ),
                "citation_valid": verification.citation_valid,
                "confidence": verification.confidence,
                "issues": list(verification.issues),
                "generation_attempts": int(
                    state.get("generation_attempts", 0)
                ),
                "retrieval_rounds": int(
                    state.get("retrieval_rounds", 0)
                ),
                "structural_scope_applied": (
                    structural_answer_scope
                    is not None
                ),
            },
        )

        feedback = list(
            verification.issues
        )

        action_note = (
            "Verifier action: "
            f"{verification.action.value}."
        )

        if (
            action_note not in feedback
            and len(feedback) < 20
        ):
            feedback.append(
                action_note
            )

        return {
            "verification_result": (
                verification
            ),
            "verification_passed": passed,
            "terminal_action": (
                verification.action
            ),
            "verifier_feedback": (
                feedback[:20]
            ),
            "broader_retrieval_requested": (
                verification.action
                == VerificationAction
                .RETRY_RETRIEVAL
            ),
        }

    # =========================================================
    # 16. broader_retrieval
    # =========================================================

    def broader_retrieval(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        active_document_id = (
            state.get(
                "active_document_id"
            )
        )

        if not active_document_id:
            return {
                "terminal_action": (
                    VerificationAction
                    .INSUFFICIENT_EVIDENCE
                ),
                "broader_retrieval_requested": False,
            }

        if (
            int(
                state.get(
                    "generation_attempts",
                    0,
                )
            )
            >= self.MAX_GENERATION_ATTEMPTS
        ):
            return {
                "terminal_action": (
                    VerificationAction
                    .INSUFFICIENT_EVIDENCE
                ),
                "broader_retrieval_requested": False,
            }

        queries = (
            state.get(
                "retrieval_queries"
            )
            or [
                state["normalized_query"]
            ]
        )

        results = self._run_retrieval_round(
            queries=queries,
            state=state,
            broader=True,
        )

        context = self._merge_contexts(
            query=(
                state.get(
                    "rewritten_query"
                )
                or state[
                    "normalized_query"
                ]
            ),
            user_id=state["user_id"],
            document_id=(
                active_document_id
            ),
            results=results,
        )

        return {
            "retrieval_results": results,
            "reranked_context": context,
            "retrieval_rounds": (
                int(
                    state.get(
                        "retrieval_rounds",
                        0,
                    )
                )
                + 1
            ),
            "broader_retrieval_requested": False,
        }

    # =========================================================
    # 17. insufficient_evidence_response
    # =========================================================

    def insufficient_evidence_response(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        """
        Mark retry exhaustion and replace every rejected Tutor draft.

        A draft that failed verification must never reach the student merely
        because the bounded retry budget was exhausted. The verifier outcome
        remains available in workflow state for observability, while the
        user-facing answer becomes a deterministic safe response.
        """

        previous = state.get(
            "verification_result"
        )

        verification = (
            self._exhausted_verification(
                previous
            )
        )

        answer = self._insufficient_answer(
            self.RETRY_EXHAUSTED_MESSAGE
        )

        return {
            "answer_draft": answer,
            "verification_result": (
                verification
            ),
            "verification_passed": False,
            "terminal_action": (
                VerificationAction
                .INSUFFICIENT_EVIDENCE
            ),
        }

    # =========================================================
    # 18. output_guard
    # =========================================================

    def output_guard(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        cached = state.get(
            "cached_answer"
        )

        if (
            state.get(
                "cache_hit",
                False,
            )
            and cached is not None
        ):
            safe_answer = cached

            if (
                self.output_guard_runner
                is not None
            ):
                safe_answer = (
                    self.output_guard_runner
                    .guard(
                        answer=safe_answer,
                        verification=None,
                        context=None,
                        strict_document_mode=bool(
                            state.get(
                                "strict_document_mode",
                                False,
                            )
                        ),
                    )
                )

            return {
                "final_answer": safe_answer,
                "output_guard_passed": True,
                "terminal_action": None,
            }

        if (
            state.get(
                "structural_clarification_required",
                False,
            )
            and state.get("answer_draft")
            is not None
        ):
            # This is a deterministic question assembled only from
            # resolver-verified candidates, not an unverified Physics claim.
            return {
                "final_answer": state["answer_draft"],
                "output_guard_passed": True,
                "terminal_action": None,
            }

        answer = state.get(
            "answer_draft"
        )
        verification = state.get(
            "verification_result"
        )
        action = state.get(
            "terminal_action"
        )

        # Deterministic fast-path outputs generated by our own code.
        if (
            verification is None
            and answer is not None
            and state["intent"].intent
            in {
                RequestIntent.GREETING,
                RequestIntent.UPLOAD_DOCUMENT,
            }
        ):
            return {
                "final_answer": answer,
                "output_guard_passed": True,
            }

        if (
            action
            == VerificationAction
            .REJECT_OUT_OF_SCOPE
        ):
            safe_answer = (
                self._direct_answer(
                    self.OUT_OF_SCOPE_MESSAGE
                )
            )

        elif (
            verification is not None
            and verification.action
            == VerificationAction.PASS
            and answer is not None
        ):
            safe_answer = answer

        elif (
            answer is not None
            and self._is_deterministic_safe_answer(
                state
            )
        ):
            safe_answer = answer

        elif (
            answer is not None
            and answer.answer_type
            == AnswerType.INSUFFICIENT_EVIDENCE
        ):
            # Keep a deterministic/Tutor refusal when there is no substantive
            # draft to expose. This includes ambiguous-document and genuinely
            # missing-evidence messages.
            safe_answer = answer

        elif (
            action
            == VerificationAction
            .ASK_FOR_CLEARER_IMAGE
        ):
            safe_answer = (
                self._insufficient_answer(
                    self.CLEARER_IMAGE_MESSAGE
                )
            )

        elif (
            action
            == VerificationAction
            .INSUFFICIENT_EVIDENCE
        ):
            safe_answer = (
                self._insufficient_answer(
                    self.INSUFFICIENT_MESSAGE
                )
            )

        elif (
            answer is not None
            and verification is not None
        ):
            # Any non-PASS verified Tutor draft is unsafe to expose. This is
            # a final defensive barrier in case a future graph branch reaches
            # output_guard without first producing a deterministic refusal.
            safe_answer = (
                self._insufficient_answer(
                    self.INSUFFICIENT_MESSAGE
                )
            )

        else:
            # No verified answer and no Tutor draft exists.
            safe_answer = (
                self._insufficient_answer(
                    self.INSUFFICIENT_MESSAGE
                )
            )

        if (
            self.output_guard_runner
            is not None
        ):
            safe_answer = (
                self.output_guard_runner
                .guard(
                    answer=safe_answer,
                    verification=verification,
                    context=state.get(
                        "reranked_context"
                    ),
                    strict_document_mode=bool(
                        state.get(
                            "strict_document_mode",
                            False,
                        )
                    ),
                )
            )

        return {
            "final_answer": safe_answer,
            "output_guard_passed": True,
        }

    # =========================================================
    # 19. memory_write_decision
    # =========================================================

    def memory_write_decision(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        memory = state.get(
            "memory",
            MemorySnapshot(),
        )

        if (
            "next_pending_structural_clarification"
            in state
        ):
            next_pending_clarification = state.get(
                "next_pending_structural_clarification"
            )
        else:
            next_pending_clarification = (
                memory.pending_structural_clarification
            )

        pending_clarification_changed = (
            next_pending_clarification
            != memory.pending_structural_clarification
        )

        intent = state["intent"]

        # Design: greetings are not permanently stored.
        should_write = (
            intent.intent
            in {
                RequestIntent
                .PHYSICS_QUESTION,
                RequestIntent
                .DIAGRAM_QUESTION,
                RequestIntent
                .NUMERICAL_PROBLEM,
                RequestIntent
                .FOLLOW_UP,
            }
            and state.get(
                "final_answer"
            )
            is not None
        )

        if not should_write:
            next_memory = memory

            if pending_clarification_changed:
                next_memory = memory.model_copy(
                    update={
                        "pending_structural_clarification": (
                            next_pending_clarification
                        )
                    }
                )

                if self.session_store is not None:
                    self.session_store.save(
                        user_id=state["user_id"],
                        session_id=state[
                            "session_id"
                        ],
                        memory=next_memory,
                    )

            return {
                "should_write_memory": (
                    pending_clarification_changed
                ),
                "next_memory": next_memory,
                "pending_structural_clarification": (
                    next_pending_clarification
                ),
                "memory_candidates": [],
            }

        answer = state[
            "final_answer"
        ]

        recent = list(
            memory.recent_messages
        )

        recent.append(
            ConversationMessage(
                role="user",
                content=state[
                    "raw_query"
                ][:12000],
            )
        )

        recent.append(
            ConversationMessage(
                role="assistant",
                content=(
                    self._answer_for_memory(
                        answer
                    )
                ),
            )
        )

        recent = recent[-10:]

        turn_document_id = (
            state.get(
                "active_document_id"
            )
        )

        if turn_document_id:
            active_page = (
                answer.source_pages[0]
                if answer.source_pages
                else (
                    state.get(
                        "active_page"
                    )
                    or memory.active_page
                )
            )

            selected_figure = (
                self._first_figure_id(
                    answer
                )
                or state.get(
                    "referenced_figure_id"
                )
                or memory
                .last_selected_figure_id
            )
        else:
            # This answer was produced in general Physics
            # mode. Keep the old PDF context in persistent
            # memory, but do not expose it as current-turn
            # page/figure state.
            active_page = None
            selected_figure = None

        # Track only the document used by the immediately previous
        # successful tutoring answer. General-Physics answers clear this
        # pointer while active_document_id may still remember an older
        # document for future explicit references. Failed/insufficient
        # document turns also do not become follow-up anchors.
        last_turn_document_id = (
            turn_document_id
            if (
                turn_document_id
                and answer.answer_type
                != AnswerType.INSUFFICIENT_EVIDENCE
            )
            else None
        )

        language = LanguageCode.ENGLISH

        estimated_grade = (
            state.get(
                "estimated_grade"
            )
            or memory.estimated_grade
        )

        next_memory = (
            memory.model_copy(
                update={
                    "active_document_id": (
                        turn_document_id
                        or memory.active_document_id
                    ),
                    "last_turn_document_id": (
                        last_turn_document_id
                    ),
                    "active_page": (
                        active_page
                        if turn_document_id
                        else memory.active_page
                    ),
                    "last_selected_figure_id": (
                        selected_figure
                        if turn_document_id
                        else memory
                        .last_selected_figure_id
                    ),
                    "recent_messages": recent,
                    "language": language,
                    "estimated_grade": (
                        estimated_grade
                    ),
                    "pending_structural_clarification": (
                        next_pending_clarification
                    ),
                }
            )
        )

        if self.session_store is not None:
            self.session_store.save(
                user_id=state["user_id"],
                session_id=state[
                    "session_id"
                ],
                memory=next_memory,
            )

        candidates: list[
            dict[str, Any]
        ] = []

        if (
            language
            != LanguageCode.UNKNOWN
        ):
            candidates.append(
                {
                    "kind": (
                        "language_preference"
                    ),
                    "value": (
                        language.value
                    ),
                }
            )

        if estimated_grade is not None:
            candidates.append(
                {
                    "kind": "grade",
                    "value": (
                        estimated_grade
                    ),
                }
            )

        if (
            self.long_term_memory is not None
            and candidates
        ):
            self.long_term_memory.write_candidates(
                user_id=state["user_id"],
                candidates=candidates,
            )

        verification = state.get(
            "verification_result"
        )

        if (
            self.semantic_learning_memory
            is not None
            and verification is not None
            and verification.action
            == VerificationAction.PASS
            and not state.get(
                "cache_hit",
                False,
            )
        ):
            verifier_parts = [
                (
                    "Verifier action: "
                    f"{verification.action.value}."
                )
            ]

            verifier_parts.extend(
                verification.issues[:20]
            )

            verifier_text = "\n".join(
                verifier_parts
            )

            try:
                (
                    self.semantic_learning_memory
                    .learn_from_turn(
                        user_id=state[
                            "user_id"
                        ],
                        student_text=state[
                            "raw_query"
                        ],
                        tutor_text=(
                            self._answer_for_memory(
                                answer
                            )
                        ),
                        verification_passed=True,
                        verifier_text=(
                            verifier_text
                        ),
                        session_id=state.get(
                            "session_id"
                        ),
                        document_id=state.get(
                            "active_document_id"
                        ),
                    )
                )
            except Exception:
                # Semantic learning enrichment must
                # never break the user response.
                pass

        return {
            "should_write_memory": True,
            "next_memory": next_memory,
            "pending_structural_clarification": (
                next_pending_clarification
            ),
            "memory_candidates": (
                candidates
            ),
            "active_page": active_page,
            "referenced_figure_id": (
                selected_figure
            ),
        }

    # =========================================================
    # 20. cache_write
    # =========================================================

    def cache_write(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        answer = state.get(
            "final_answer"
        )

        verification = state.get(
            "verification_result"
        )

        should_write = bool(
            self.query_cache is not None
            and not state.get(
                "cache_hit",
                False,
            )
            # Keep Auto-mode cache reusable, but never store a response from
            # an explicitly selected model in the shared answer cache.
            and state.get(
                "selected_model"
            ) is None
            and state.get(
                "structural_resolution_status"
            )
            != StructuralResolutionStatus.RESOLVED
            and answer is not None
            and verification is not None
            and verification.action
            == VerificationAction.PASS
            and state["intent"].intent
            not in {
                RequestIntent.GREETING,
                RequestIntent.UPLOAD_DOCUMENT,
                RequestIntent.OUT_OF_SCOPE,
                RequestIntent.UNSUPPORTED,
            }
        )

        if not should_write:
            return {
                "should_write_cache": False
            }

        query = (
            state.get(
                "rewritten_query"
            )
            or state["normalized_query"]
        )

        document_id = (
            state.get(
                "active_document_id"
            )
            or "no-document"
        )

        language = self._cache_language_value(
            state.get("language")
        )

        written = self.query_cache.set(
            user_id=state["user_id"],
            document_id=document_id,
            query=query,
            answer=answer.model_dump(
                mode="json"
            ),
            verification_status=(
                verification.action.value
            ),
            active_page=state.get(
                "active_page"
            ),
            figure_id=state.get(
                "referenced_figure_id"
            ),
            language=language,
            grade=state.get(
                "estimated_grade"
            ),
        )

        return {
            "cache_key": (
                state.get("cache_key")
                or self._build_cache_key(
                    state
                )
            ),
            "should_write_cache": bool(
                written
            ),
        }

    # =========================================================
    # 21. respond
    # =========================================================

    def respond(
        self,
        state: PhysicsTutorState,
    ) -> dict[str, Any]:
        final_answer = state.get(
            "final_answer"
        )

        if final_answer is None:
            raise RuntimeError(
                "No guarded final answer is available."
            )

        return {
            "final_answer": final_answer
        }

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _active_document_is_image(
        self,
        *,
        state: PhysicsTutorState,
        document_id: str,
    ) -> bool:
        """
        Return True when the resolved document for this turn is a
        standalone image upload.

        The session bookshelf stores lightweight document metadata
        (document_id + original name), so source modality can be
        inferred without touching files, indexes, or another service.

        This deliberately does not turn every numerical into a visual
        query. It only changes retrieval preference when THIS turn has
        already resolved to an uploaded image document.
        """
        if not document_id:
            return False

        memory = state.get(
            "memory",
            MemorySnapshot(),
        )

        documents = (
            self._normalized_available_documents(
                state,
                memory=memory,
            )
        )

        image_suffixes = (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        )

        for document in documents:
            if (
                document["document_id"]
                != document_id
            ):
                continue

            name = (
                document.get(
                    "name",
                    "",
                )
                .strip()
                .casefold()
            )

            return name.endswith(
                image_suffixes
            )

        return False

    def _run_retrieval_round(
        self,
        *,
        queries: list[str],
        state: PhysicsTutorState,
        broader: bool,
    ) -> list[
        HybridRetrievalResult
    ]:
        document_id = state[
            "active_document_id"
        ]

        grade = (
            state.get(
                "estimated_grade"
            )
            or state.get(
                "memory",
                MemorySnapshot(),
            ).estimated_grade
        )

        preferred_pages = tuple(
            state.get(
                "preferred_page_numbers",
                [],
            )
        ) or None

        prefer_visual = bool(
            state.get(
                "prefer_visual",
                False,
            )
        )

        # Source-aware multimodal grounding:
        #
        # Query intent and evidence modality are different things.
        # A request can be a NUMERICAL_PROBLEM while its givens,
        # circuit, graph, or geometry live inside an uploaded image.
        #
        # If the resolved document for this turn is itself an image,
        # always prefer real visual evidence for document-grounded
        # retrieval. This does NOT affect:
        #   - general/typed numericals with no active document,
        #   - text/PDF documents,
        #   - the existing DIAGRAM_QUESTION visual path.
        source_is_image = self._active_document_is_image(
            state=state,
            document_id=document_id,
        )

        if (
            not prefer_visual
            and source_is_image
        ):
            prefer_visual = True

        # A verified structural resolution already identifies the exact
        # indexed evidence for this turn. Pass those canonical identities to
        # RetrievalService instead of asking semantic retrieval to rediscover
        # a nearby passage.
        #
        # NO_MATCH / STRUCTURE_UNAVAILABLE deliberately keep this dictionary
        # empty, so the established Dense + BM25 + RRF path receives exactly
        # the same call it received before structural resolution existed.
        structural_lookup_arguments: dict[
            str,
            Any,
        ] = {}

        structural_match_is_verified = (
            state.get(
                "structural_resolution_status"
            )
            == StructuralResolutionStatus.RESOLVED
            and not state.get(
                "structural_fallback_to_semantic",
                True,
            )
        )

        required_chunk_ids: tuple[
            str,
            ...,
        ] = ()
        required_parent_ids: tuple[
            str,
            ...,
        ] = ()

        if structural_match_is_verified:
            required_chunk_ids = tuple(
                dict.fromkeys(
                    str(chunk_id).strip()
                    for chunk_id in state.get(
                        "structural_linked_retrieval_chunk_ids",
                        [],
                    )
                    if str(chunk_id).strip()
                )
            )

            required_parent_ids = tuple(
                dict.fromkeys(
                    str(parent_id).strip()
                    for parent_id in state.get(
                        "structural_linked_parent_chunk_ids",
                        [],
                    )
                    if str(parent_id).strip()
                )
            )

            # A resolved source identity without linked indexed evidence is
            # an incomplete/corrupt structural artifact. Do not silently run
            # a broad semantic search and risk answering from another item.
            if (
                not required_chunk_ids
                and not required_parent_ids
            ):
                fallback_query = (
                    queries[0]
                    if queries
                    else state.get(
                        "normalized_query",
                        state["raw_query"],
                    )
                )

                return [
                    HybridRetrievalResult(
                        query=fallback_query,
                        context=ContextBundle(
                            query=fallback_query,
                            user_id=state["user_id"],
                            document_id=document_id,
                        ),
                        evidence_found=False,
                        failure_reason=(
                            "RESOLVED_STRUCTURAL_TARGET_HAS_NO_"
                            "LINKED_EVIDENCE"
                        ),
                    )
                ]

            structural_lookup_arguments = {
                "required_chunk_ids": (
                    required_chunk_ids
                ),
                "required_parent_ids": (
                    required_parent_ids
                ),
            }

        _safe_debug(
            "[PHYMENTOR-DEBUG][retrieval-start]",
            {
                "document_id": document_id,
                "source_is_image": source_is_image,
                "prefer_visual": prefer_visual,
                "preferred_pages": preferred_pages,
                "broader": broader,
                "structural_match_is_verified": (
                    structural_match_is_verified
                ),
                "required_chunk_ids": list(
                    required_chunk_ids
                ),
                "required_parent_ids": list(
                    required_parent_ids
                ),
                "queries": list(
                    queries[:self.MAX_RETRIEVAL_QUERIES]
                ),
            },
        )

        dense_top_k = (
            30 if broader else 20
        )

        bm25_top_k = (
            30 if broader else 20
        )

        # Normal retrieval keeps Top-30. The one bounded broader retry
        # widens the candidate pool while reranking still returns max 8.
        fused_top_k = (
            40 if broader else 30
        )

        rerank_top_k = 8
        max_contexts = (
            8 if broader else 6
        )

        results: list[
            HybridRetrievalResult
        ] = []

        retrieval_queries = queries[
            :self.MAX_RETRIEVAL_QUERIES
        ]

        # Exact canonical IDs are independent of query rewrites. Running the
        # same exact lookup for every rewrite would only duplicate context.
        if structural_match_is_verified:
            retrieval_queries = retrieval_queries[:1]

        for retrieval_query in (
            retrieval_queries
        ):
            result = (
                self.retrieval_service
                .retrieve(
                    query=retrieval_query,
                    user_id=state["user_id"],
                    document_id=document_id,
                    dense_top_k=dense_top_k,
                    bm25_top_k=bm25_top_k,
                    fused_top_k=fused_top_k,
                    rerank_top_k=rerank_top_k,
                    max_contexts=max_contexts,
                    page_numbers=None,
                    content_types=None,
                    topics=None,
                    grade=grade,
                    include_visual=True,
                    preferred_page_numbers=(
                        preferred_pages
                    ),
                    prefer_visual=(
                        prefer_visual
                    ),
                    **structural_lookup_arguments,
                )
            )

            _safe_debug(
                "[PHYMENTOR-DEBUG][retrieval-result]",
                {
                    "query": retrieval_query,
                    "evidence_found": result.evidence_found,
                    "failure_reason": result.failure_reason,
                    "dense_hits": len(result.dense_hits),
                    "bm25_hits": len(result.bm25_hits),
                    "reranked_hits": [
                        {
                            "chunk_id": item.hit.chunk_id,
                            "chunk_kind": item.hit.chunk_kind,
                            "page": item.hit.page_number,
                            "rerank_score": item.rerank_score,
                            "image_path": item.hit.image_path,
                        }
                        for item in result.reranked_hits
                    ],
                    "context_items": [
                        {
                            "context_id": item.context_id,
                            "page": item.page_number,
                            "content_type": item.content_type,
                            "image_path": item.image_path,
                            "linked_figure_ids": list(
                                item.linked_figure_ids
                            ),
                            "source_chunk_ids": list(
                                item.source_chunk_ids
                            ),
                            "text_preview": item.text[:300],
                        }
                        for item in result.context.items
                    ],
                },
            )

            results.append(result)

        return results

    def _attach_resolved_figures_to_context(
        self,
        *,
        context: ContextBundle,
        retrieved_figures: list[
            dict[str, Any]
        ],
        referenced_figure_id: (
            str | None
        ),
    ) -> ContextBundle:
        """
        Attach an actual resolved image to a ContextItem when identity is clear.

        Rules:
        - preserve an existing image_path;
        - if exactly one linked figure has a resolved image, attach it;
        - if several linked figures are available, use only an explicitly
          referenced canonical figure ID;
        - otherwise do not guess.

        This keeps the bridge generic and fail-closed.
        """

        figure_by_id: dict[
            str,
            dict[str, Any],
        ] = {}

        for raw_figure in (
            retrieved_figures
        ):
            figure_id = str(
                raw_figure.get(
                    "figure_id",
                    "",
                )
            ).strip()

            image_path = raw_figure.get(
                "image_path"
            )

            if (
                not figure_id
                or not isinstance(
                    image_path,
                    str,
                )
                or not image_path.strip()
            ):
                continue

            figure_by_id[
                figure_id
            ] = raw_figure

        if not figure_by_id:
            return context

        explicit_figure_id = (
            referenced_figure_id.strip()
            if isinstance(
                referenced_figure_id,
                str,
            )
            else ""
        )

        enriched_items: list[
            ContextItem
        ] = []

        for item in context.items:
            if (
                isinstance(
                    item.image_path,
                    str,
                )
                and item.image_path.strip()
            ):
                enriched_items.append(
                    item
                )
                continue

            linked_ids = [
                figure_id.strip()
                for figure_id
                in item.linked_figure_ids
                if (
                    isinstance(
                        figure_id,
                        str,
                    )
                    and figure_id.strip()
                    in figure_by_id
                )
            ]

            if not linked_ids:
                enriched_items.append(
                    item
                )
                continue

            selected_figure_id: (
                str | None
            ) = None

            if (
                explicit_figure_id
                and explicit_figure_id
                in linked_ids
            ):
                selected_figure_id = (
                    explicit_figure_id
                )
            elif len(linked_ids) == 1:
                selected_figure_id = (
                    linked_ids[0]
                )

            if selected_figure_id is None:
                # Multiple possible figures and no explicit canonical identity:
                # leave the context unresolved instead of choosing by order.
                enriched_items.append(
                    item
                )
                continue

            resolved_figure = (
                figure_by_id[
                    selected_figure_id
                ]
            )

            resolved_caption = (
                resolved_figure.get(
                    "caption"
                )
            )

            enriched_items.append(
                item.model_copy(
                    update={
                        "image_path": (
                            resolved_figure[
                                "image_path"
                            ]
                        ),
                        "caption": (
                            item.caption
                            or (
                                resolved_caption
                                if isinstance(
                                    resolved_caption,
                                    str,
                                )
                                else None
                            )
                        ),
                    }
                )
            )

        return context.model_copy(
            update={
                "items": enriched_items
            }
        )

    def _merge_contexts(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
        results: list[
            HybridRetrievalResult
        ],
    ) -> ContextBundle:
        candidates: list[
            ContextItem
        ] = []

        any_truncated = False

        for result in results:
            any_truncated = (
                any_truncated
                or result.context.truncated
            )

            for item in (
                result.context.items
            ):
                if (
                    item.user_id
                    != user_id
                    or item.document_id
                    != document_id
                ):
                    continue

                candidates.append(item)

        candidates.sort(
            key=lambda item: (
                -item.rerank_score,
                item.page_number,
                item.context_id,
            )
        )

        selected: list[
            ContextItem
        ] = []

        seen: set[
            tuple[str, int]
        ] = set()

        total_characters = 0
        merged_truncated = (
            any_truncated
        )

        for item in candidates:
            identity = (
                item.parent_id
                or item.context_id,
                item.page_number,
            )

            if identity in seen:
                continue

            item_characters = len(
                item.text
            )

            if (
                selected
                and (
                    total_characters
                    + item_characters
                    > self
                    .max_merged_context_characters
                )
            ):
                merged_truncated = True
                continue

            if (
                not selected
                and item_characters
                > self
                .max_merged_context_characters
            ):
                bounded_text = (
                    item.text[
                        :self
                        .max_merged_context_characters
                    ].rstrip()
                )

                item = item.model_copy(
                    update={
                        "text": bounded_text
                    }
                )

                item_characters = len(
                    bounded_text
                )

                merged_truncated = True

            selected.append(item)
            seen.add(identity)

            total_characters += (
                item_characters
            )

            if (
                len(selected)
                >= self
                .max_merged_context_items
            ):
                if (
                    len(candidates)
                    > len(selected)
                ):
                    merged_truncated = True

                break

        return ContextBundle(
            query=query,
            user_id=user_id,
            document_id=document_id,
            items=selected,
            total_characters=(
                total_characters
            ),
            truncated=merged_truncated,
        )

    @classmethod
    def _should_use_document_for_turn(
        cls,
        *,
        state: PhysicsTutorState,
        explicit_document_id: str | None,
        available_documents: (
            list[dict[str, str]]
            | None
        ) = None,
    ) -> bool:
        """
        Decide whether THIS question requires document grounding.

        Merely having uploaded documents must not force every school
        Physics question into RAG mode.
        """

        if (
            state.get("selected_page")
            is not None
        ):
            return True

        if state.get(
            "selected_figure_id"
        ):
            return True

        intent = state.get("intent")

        # Structured query understanding is the primary semantic signal.
        # A positive requirement is authoritative. Existing deterministic
        # explicit-reference checks below remain as a conservative upgrade
        # path if the model misses an obvious uploaded-source reference.
        if (
            intent is not None
            and intent.requires_document is True
        ):
            return True

        # A modern structured NO-DOCUMENT decision is authoritative too.
        #
        # This prevents an older document/follow-up anchor from upgrading a
        # fresh standalone Physics question back into RAG mode.
        #
        # Explicit request-local page/figure hints were already handled above,
        # so genuine source-specific hints still win before this guard.
        if (
            intent is not None
            and intent.requires_document is False
            and intent.document_usage is not None
            and intent.document_usage.value == "NONE"
        ):
            return False

        # Backward compatibility for older clients/tests that identify the
        # current document but predate the explicit document-usage decision
        # fields. A modern explicit False remains authoritative; only an
        # undecided (None) intent falls back to the request's document ID.
        request_document_id = str(
            state.get(
                "explicit_document_id",
                "",
            )
            or ""
        ).strip()

        if (
            request_document_id
            and (
                intent is None
                or (
                    intent.requires_document
                    is None
                    and intent.document_usage
                    is None
                )
            )
        ):
            return True

        query = str(
            state.get(
                "normalized_query",
                "",
            )
        ).strip()

        if not query:
            return False

        if cls._query_references_document_context(
            query
        ):
            return True

        normalized = " ".join(
            query.casefold().split()
        )

        contextual_document_phrases = (
            "based on this",
            "based on the above",
            "using this",
            "using the above",
            "from this",
            "from the above",
            "according to this",
            "based on that",
            "from that",
            "using that",
        )

        if (
            explicit_document_id
            and any(
                phrase in normalized
                for phrase
                in contextual_document_phrases
            )
        ):
            return True

        documents = (
            available_documents
            or []
        )

        if (
            documents
            and cls._query_mentions_document_name(
                query=query,
                documents=documents,
            )
        ):
            return True

        # A short/elliptical follow-up may omit words such as
        # "document" or "page" entirely (for example:
        # "please explain the 2nd point"). Only the document used by
        # the immediately previous successful tutoring turn may anchor
        # such a follow-up.
        if cls._immediate_followup_document_id(
            state=state,
            query=query,
            documents=documents,
        ):
            return True

        # Natural references such as "earlier fission diagram",
        # "my second PDF", or "previous SHM notes".
        if documents and re.search(
            (
                r"(?i)\b("
                r"my|uploaded|earlier|previous|first|second|third|"
                r"latest|newest|older|old|recent"
                r")\b.*\b("
                r"pdf|document|doc|file|page|figure|fig|diagram|"
                r"image|photo|notes?"
                r")\b"
            ),
            normalized,
        ):
            return True

        return False

    def _resolve_document_for_turn(
        self,
        *,
        state: PhysicsTutorState,
        available_documents: (
            list[dict[str, str]]
        ),
        recent_document_id: str | None,
    ) -> tuple[str | None, bool]:
        """
        Pick exactly one document for a document-grounded turn.

        Returns:
            (document_id, ambiguous)

        ambiguous=True means multiple documents were plausible and
        the resolver intentionally refused to guess.
        """

        if not available_documents:
            return (
                recent_document_id,
                False,
            )

        query = str(
            state.get(
                "normalized_query",
                "",
            )
        ).strip()

        # Page/figure UI context belongs to the client-reported
        # recent document. This remains backward compatible.
        if (
            state.get("selected_page")
            is not None
            or state.get(
                "selected_figure_id"
            )
        ):
            candidate = (
                self._known_document_id(
                    document_id=(
                        recent_document_id
                    ),
                    documents=(
                        available_documents
                    ),
                )
            )

            if candidate:
                return candidate, False

        # 1. Strongest signal: the question mentions a filename,
        # filename stem, or distinctive filename token.
        (
            name_match,
            name_ambiguous,
        ) = self._resolve_document_by_name(
            query=query,
            documents=available_documents,
        )

        if name_match is not None:
            return name_match, False

        # 2. Explicit positional reference:
        # first/second/latest/previous document.
        positional_match = (
            self._resolve_document_by_position(
                query=query,
                documents=available_documents,
                recent_document_id=(
                    recent_document_id
                ),
            )
        )

        if positional_match is not None:
            return positional_match, False

        # 3. Deictic follow-up such as "this PDF" or
        # "based on this" should continue from the recent document.
        if (
            recent_document_id
            and self._query_prefers_recent_document(
                query
            )
        ):
            candidate = (
                self._known_document_id(
                    document_id=(
                        recent_document_id
                    ),
                    documents=(
                        available_documents
                    ),
                )
            )

            if candidate:
                return candidate, False

        # 4. Elliptical follow-up with no explicit document words:
        # continue from the document used by the immediately previous
        # successful tutoring turn. Explicit filename/position/deictic
        # signals above always win, and a generic request such as
        # "please explain the document" remains ambiguous.
        followup_document_id = (
            self._immediate_followup_document_id(
                state=state,
                query=query,
                documents=available_documents,
            )
        )

        if followup_document_id:
            return followup_document_id, False

        # 5. If only one document exists, there is no ambiguity.
        if len(
            available_documents
        ) == 1:
            return (
                available_documents[0][
                    "document_id"
                ],
                False,
            )

        # 6. For a genuine multi-document reference with no useful
        # filename, probe the existing document-scoped RetrievalService
        # against bounded candidates and compare evidence. Use a compact
        # topic-bearing query when query understanding supplied one, so
        # wrappers such as "can you explain what is in the ... document"
        # do not dilute the document-selection signal.
        resolution_query = (
            self._document_resolution_query(
                state=state,
                fallback_query=query,
            )
        )

        semantic_match = (
            self._resolve_document_semantically(
                state=state,
                query=resolution_query,
                documents=(
                    available_documents
                ),
                recent_document_id=(
                    recent_document_id
                ),
            )
        )

        if semantic_match is not None:
            return semantic_match, False

        # A tied filename signal is also ambiguity; semantic probing
        # has already had a chance to disambiguate it.
        if name_ambiguous:
            return None, True

        # Never silently choose a random PDF/image.
        return None, True

    def _resolve_document_semantically(
        self,
        *,
        state: PhysicsTutorState,
        query: str,
        documents: list[dict[str, str]],
        recent_document_id: str | None,
    ) -> str | None:
        """
        Bounded semantic document selection using the existing,
        user-scoped, document-scoped RetrievalService.

        This does NOT merge evidence across documents. It only probes
        candidates, chooses one, then the normal retrieval pipeline
        runs against that one document.
        """

        probe_documents = list(
            documents
        )

        _safe_debug(
            "[PHYMENTOR-DEBUG][resolver-probe]",
            {
                "query": query,
                "recent_document_id": recent_document_id,
                "documents": [
                    {
                        "document_id": item.get("document_id"),
                        "name": item.get("name"),
                    }
                    for item in probe_documents
                ],
            },
        )

        if (
            len(probe_documents)
            > self.MAX_DOCUMENT_RESOLUTION_PROBES
        ):
            # Keep recent context plus the newest bounded candidates.
            recent_item = next(
                (
                    item
                    for item
                    in probe_documents
                    if item["document_id"]
                    == recent_document_id
                ),
                None,
            )

            probe_documents = (
                probe_documents[
                    -self
                    .MAX_DOCUMENT_RESOLUTION_PROBES:
                ]
            )

            if (
                recent_item is not None
                and all(
                    item["document_id"]
                    != recent_item[
                        "document_id"
                    ]
                    for item
                    in probe_documents
                )
            ):
                probe_documents[0] = (
                    recent_item
                )

        scored: list[
            tuple[
                float,
                float,
                str,
            ]
        ] = []

        for document in probe_documents:
            document_id = document[
                "document_id"
            ]

            try:
                result = (
                    self.retrieval_service
                    .retrieve(
                        query=query,
                        user_id=state[
                            "user_id"
                        ],
                        document_id=(
                            document_id
                        ),
                        dense_top_k=6,
                        bm25_top_k=6,
                        fused_top_k=8,
                        rerank_top_k=3,
                        max_contexts=3,
                        page_numbers=None,
                        content_types=None,
                        topics=None,
                        grade=(
                            state.get(
                                "estimated_grade"
                            )
                            or state.get(
                                "memory",
                                MemorySnapshot(),
                            ).estimated_grade
                        ),
                        include_visual=True,
                        preferred_page_numbers=None,
                        prefer_visual=bool(
                            state.get(
                                "prefer_visual",
                                False,
                            )
                        ),
                    )
                )
            except Exception:
                # A processing/deleted/unavailable candidate must
                # not break the entire chat request.
                continue

            if not result.reranked_hits:
                continue

            rerank_scores = [
                float(
                    item.rerank_score
                )
                for item
                in result.reranked_hits[:3]
            ]

            top_score = (
                rerank_scores[0]
            )

            average_score = (
                sum(rerank_scores)
                / len(rerank_scores)
            )

            context_text = " ".join(
                item.text
                for item
                in result.context.items[:3]
            )

            overlap = (
                self._lexical_overlap(
                    query=query,
                    text=context_text,
                )
            )

            # Reranker remains the primary signal.
            # Lexical overlap is only a small tie-breaker.
            combined = (
                top_score
                + (
                    0.15
                    * average_score
                )
                + (
                    1.5
                    * overlap
                )
            )

            _safe_debug(
                "[PHYMENTOR-DEBUG][resolver-score]",
                {
                    "document_id": document_id,
                    "name": document.get("name"),
                    "top_score": top_score,
                    "average_score": average_score,
                    "lexical_overlap": overlap,
                    "combined_score": combined,
                    "rerank_scores": rerank_scores,
                    "context_preview": context_text[:500],
                },
            )

            scored.append(
                (
                    combined,
                    overlap,
                    document_id,
                )
            )

        if not scored:
            _safe_debug(
                "[PHYMENTOR-DEBUG][resolver-decision]",
                {
                    "decision": None,
                    "reason": "no_scored_documents",
                },
            )
            return None

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        best = scored[0]

        if len(scored) == 1:
            # Require at least a little textual support when only
            # one probe returned evidence; otherwise remain cautious.
            selected = (
                best[2]
                if best[1] >= 0.08
                else None
            )

            _safe_debug(
                "[PHYMENTOR-DEBUG][resolver-decision]",
                {
                    "decision": selected,
                    "reason": "single_scored_document",
                    "combined_score": best[0],
                    "lexical_overlap": best[1],
                    "required_overlap": 0.08,
                },
            )

            return selected

        second = scored[1]

        score_gap = (
            best[0]
            - second[0]
        )

        overlap_gap = (
            best[1]
            - second[1]
        )

        dynamic_margin = max(
            0.25,
            abs(best[0]) * 0.06,
        )

        selected = (
            best[2]
            if (
                score_gap
                >= dynamic_margin
                or (
                    best[1] >= 0.20
                    and overlap_gap >= 0.10
                )
            )
            else None
        )

        _safe_debug(
            "[PHYMENTOR-DEBUG][resolver-decision]",
            {
                "decision": selected,
                "best_document_id": best[2],
                "best_combined_score": best[0],
                "best_overlap": best[1],
                "second_document_id": second[2],
                "second_combined_score": second[0],
                "second_overlap": second[1],
                "score_gap": score_gap,
                "overlap_gap": overlap_gap,
                "required_dynamic_margin": dynamic_margin,
                "lexical_override": bool(
                    best[1] >= 0.20
                    and overlap_gap >= 0.10
                ),
            },
        )

        return selected

    @classmethod
    def _resolve_document_by_name(
        cls,
        *,
        query: str,
        documents: list[dict[str, str]],
    ) -> tuple[str | None, bool]:
        scored: list[
            tuple[int, str]
        ] = []

        for document in documents:
            score = (
                cls._document_name_score(
                    query=query,
                    name=document["name"],
                )
            )

            if score > 0:
                scored.append(
                    (
                        score,
                        document[
                            "document_id"
                        ],
                    )
                )

        if not scored:
            return None, False

        scored.sort(
            reverse=True
        )

        best_score = scored[0][0]

        best_ids = [
            document_id
            for score, document_id
            in scored
            if score == best_score
        ]

        if len(best_ids) == 1:
            return best_ids[0], False

        return None, True

    @classmethod
    def _query_mentions_document_name(
        cls,
        *,
        query: str,
        documents: list[dict[str, str]],
    ) -> bool:
        return any(
            cls._document_name_score(
                query=query,
                name=item["name"],
            )
            > 0
            for item in documents
        )

    @staticmethod
    def _document_name_score(
        *,
        query: str,
        name: str,
    ) -> int:
        normalized_query = " ".join(
            re.sub(
                r"[^\w]+",
                " ",
                query.casefold(),
                flags=re.UNICODE,
            ).split()
        )

        normalized_name = " ".join(
            re.sub(
                r"[^\w]+",
                " ",
                name.casefold(),
                flags=re.UNICODE,
            ).split()
        )

        if not normalized_query or not normalized_name:
            return 0

        if normalized_name in normalized_query:
            return 120

        # Remove the final extension before token comparison.
        stem = re.sub(
            r"\.[a-z0-9]{1,8}$",
            "",
            name.casefold().strip(),
        )

        normalized_stem = " ".join(
            re.sub(
                r"[^\w]+",
                " ",
                stem,
                flags=re.UNICODE,
            ).split()
        )

        if (
            len(normalized_stem) >= 3
            and normalized_stem
            in normalized_query
        ):
            return 110

        stop_tokens = {
            "pdf",
            "doc",
            "document",
            "file",
            "image",
            "img",
            "photo",
            "scan",
            "page",
            "pages",
            "notes",
            "note",
            "physics",
            "class",
            "chapter",
            "jpg",
            "jpeg",
            "png",
            "webp",
        }

        name_tokens = {
            token
            for token
            in normalized_stem.split()
            if (
                len(token) >= 3
                and token
                not in stop_tokens
            )
        }

        if not name_tokens:
            return 0

        query_tokens = set(
            normalized_query.split()
        )

        matches = (
            name_tokens
            & query_tokens
        )

        if not matches:
            return 0

        if (
            len(matches) >= 2
            or matches == name_tokens
        ):
            return (
                80
                + min(
                    20,
                    5 * len(matches),
                )
            )

        only = next(
            iter(matches)
        )

        if len(only) >= 5:
            return 60

        # Short acronyms such as SHM are useful only when the
        # query also looks document-related.
        if re.search(
            (
                r"(?i)\b("
                r"pdf|document|doc|file|page|"
                r"figure|diagram|image|notes?"
                r")\b"
            ),
            normalized_query,
        ):
            return 45

        return 0

    @staticmethod
    def _resolve_document_by_position(
        *,
        query: str,
        documents: list[dict[str, str]],
        recent_document_id: str | None,
    ) -> str | None:
        normalized = " ".join(
            query.casefold().split()
        )

        if not documents:
            return None

        # Ordinals are overloaded in tutoring queries:
        #
        #   "use the 2nd document"  -> document position
        #   "explain the 2nd point" -> structure inside the
        #                              already-grounded document
        #
        # Do not let a structural ordinal accidentally switch to
        # another uploaded document before follow-up grounding runs.
        structural_reference = bool(
            re.search(
                (
                    r"(?i)\b("
                    r"point|points|bullet|bullets|item|items|"
                    r"step|steps|line|lines|paragraph|paragraphs|"
                    r"section|sections|heading|headings|"
                    r"question|questions|problem|problems|"
                    r"example|examples|equation|equations|"
                    r"formula|formulae|formulas|page|pages|"
                    r"figure|figures|fig|diagram|diagrams"
                    r")\b"
                ),
                normalized,
            )
        )

        position_patterns = (
            (
                (
                    "first",
                    "1st",
                ),
                0,
            ),
            (
                (
                    "second",
                    "2nd",
                ),
                1,
            ),
            (
                (
                    "third",
                    "3rd",
                ),
                2,
            ),
        )

        if not structural_reference:
            for words, index in (
                position_patterns
            ):
                if any(
                    re.search(
                        rf"(?i)\b{re.escape(word)}\b",
                        normalized,
                    )
                    for word in words
                ):
                    if index < len(documents):
                        return (
                            documents[index][
                                "document_id"
                            ]
                        )

        latest_words = (
            "latest",
            "newest",
            "last one",
            "last document",
            "last pdf",
            "last file",
            "new document",
            "new pdf",
            "recent document",
            "recent pdf",
        )

        if (
            not structural_reference
            and any(
                word in normalized
                for word in latest_words
            )
        ):
            return (
                documents[-1][
                    "document_id"
                ]
            )

        previous_words = (
            "previous",
            "earlier",
            "older",
            "old document",
            "old pdf",
        )

        if (
            not structural_reference
            and any(
                word in normalized
                for word in previous_words
            )
        ):
            if recent_document_id:
                for index, item in enumerate(
                    documents
                ):
                    if (
                        item["document_id"]
                        == recent_document_id
                        and index > 0
                    ):
                        return (
                            documents[
                                index - 1
                            ][
                                "document_id"
                            ]
                        )

            if len(documents) >= 2:
                return (
                    documents[-2][
                        "document_id"
                    ]
                )

        return None

    @staticmethod
    def _query_prefers_recent_document(
        query: str,
    ) -> bool:
        normalized = " ".join(
            query.casefold().split()
        )

        phrases = (
            "this pdf",
            "this document",
            "this file",
            "this page",
            "this figure",
            "this diagram",
            "that pdf",
            "that document",
            "that file",
            "that figure",
            "that diagram",
            "based on this",
            "based on the above",
            "using this",
            "using the above",
            "from this",
            "from the above",
            "according to this",
        )

        return any(
            phrase in normalized
            for phrase in phrases
        )

    @staticmethod
    def _known_document_id(
        *,
        document_id: str | None,
        documents: list[dict[str, str]],
    ) -> str | None:
        if not document_id:
            return None

        for item in documents:
            if (
                item["document_id"]
                == document_id
            ):
                return document_id

        return None

    @classmethod
    def _immediate_followup_document_id(
        cls,
        *,
        state: PhysicsTutorState,
        query: str,
        documents: list[dict[str, str]],
    ) -> str | None:
        """
        Return a safe document anchor for an elliptical follow-up.

        `active_document_id` is intentionally not enough here because it
        may remember an older document across later general-Physics turns.
        `last_turn_document_id` means the immediately previous successful
        tutoring answer actually used that document.

        Explicit document wording is excluded so requests such as
        "please explain the document" can still clarify when several
        documents are available.
        """

        if not documents:
            return None

        if cls._query_references_document_context(
            query
        ):
            return None

        understanding = state.get(
            "query_understanding"
        )

        intent = (
            understanding.intent
            if understanding is not None
            else state.get("intent")
        )

        if (
            intent is None
            or not (
                intent.intent
                == RequestIntent.FOLLOW_UP
                or intent.is_follow_up
            )
        ):
            return None

        memory = state.get(
            "memory",
            MemorySnapshot(),
        )

        return cls._known_document_id(
            document_id=(
                memory.last_turn_document_id
            ),
            documents=documents,
        )

    @staticmethod
    def _normalized_available_documents(
        state: PhysicsTutorState,
        *,
        memory: MemorySnapshot | None = None,
    ) -> list[dict[str, str]]:
        """
        Build one bounded, deduplicated document bookshelf.

        Sources:
        1. documents already remembered in Redis session memory
        2. documents supplied by the current frontend request

        The current request is applied last, so fresher metadata for
        the same document_id replaces the remembered copy and moves
        that document to the end of the recency-ordered registry.
        """

        raw_documents: list[
            dict[str, Any]
        ] = []

        if memory is not None:
            raw_documents.extend(
                {
                    "document_id": (
                        item.document_id
                    ),
                    "name": item.name,
                }
                for item
                in memory.available_documents
            )

        request_documents = state.get(
            "available_documents",
            [],
        )

        for item in request_documents:
            if isinstance(
                item,
                dict,
            ):
                raw_documents.append(
                    item
                )

        registry: dict[
            str,
            dict[str, str],
        ] = {}

        for item in raw_documents:
            document_id = str(
                item.get(
                    "document_id",
                    "",
                )
            ).strip()

            name = str(
                item.get(
                    "name",
                    "",
                )
            ).strip()

            if not document_id:
                continue

            # Newer copy wins and moves to the end.
            registry.pop(
                document_id,
                None,
            )

            registry[
                document_id
            ] = {
                "document_id": (
                    document_id
                ),
                "name": (
                    name
                    or document_id
                ),
            }

        return list(
            registry.values()
        )[-30:]

    @classmethod
    def _document_resolution_query(
        cls,
        *,
        state: PhysicsTutorState,
        fallback_query: str,
    ) -> str:
        """
        Build a compact query used ONLY to compare candidate documents.

        For explicit topic-to-document requests, remove conversational
        wrapper/document words while keeping the user's actual topic
        terms. This is generic text cleanup, not a topic->filename map.

        If cleanup has no meaningful topic left (for example
        "please explain the document"), keep the original wording so
        the resolver can remain ambiguous instead of guessing.
        """

        if cls._query_references_document_context(
            fallback_query
        ):
            wrapper_tokens = {
                "a",
                "an",
                "the",
                "can",
                "could",
                "would",
                "u",
                "you",
                "me",
                "my",
                "please",
                "pls",
                "explain",
                "tell",
                "show",
                "what",
                "whats",
                "there",
                "is",
                "are",
                "in",
                "on",
                "from",
                "of",
                "about",
                "related",
                "regarding",
                "document",
                "doc",
                "pdf",
                "file",
                "image",
                "photo",
                "page",
                "pages",
                "figure",
                "fig",
                "diagram",
                "notes",
                "note",
                "uploaded",
            }

            topic_tokens = [
                token
                for token in re.findall(
                    r"\w+",
                    fallback_query.casefold(),
                    flags=re.UNICODE,
                )
                if (
                    len(token) >= 2
                    and token
                    not in wrapper_tokens
                )
            ]

            if topic_tokens:
                return " ".join(
                    topic_tokens[:12]
                )

            return fallback_query

        scope = state.get("scope")

        topic_values = (
            list(scope.topics)
            if (
                scope is not None
                and scope.topics
            )
            else []
        )

        topics: list[str] = []

        for item in topic_values:
            normalized = " ".join(
                str(item).strip().split()
            )

            if (
                normalized
                and normalized.casefold()
                not in {
                    existing.casefold()
                    for existing in topics
                }
            ):
                topics.append(normalized)

            if len(topics) >= 4:
                break

        if topics:
            return " ".join(topics)

        understanding = state.get(
            "query_understanding"
        )

        rewrite = (
            understanding.rewrite
            if understanding is not None
            else None
        )

        if (
            rewrite is not None
            and rewrite.was_rewritten
            and rewrite.rewritten_query.strip()
        ):
            return rewrite.rewritten_query.strip()

        return fallback_query

    @staticmethod
    def _lexical_overlap(
        *,
        query: str,
        text: str,
    ) -> float:
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "why",
            "how",
            "explain",
            "tell",
            "show",
            "this",
            "that",
            "from",
            "in",
            "on",
            "of",
            "to",
            "and",
            "or",
            "my",
            "pdf",
            "document",
            "file",
            "page",
            "figure",
            "diagram",
            "please",
        }

        query_tokens = {
            token
            for token
            in re.findall(
                r"\w+",
                query.casefold(),
                flags=re.UNICODE,
            )
            if (
                len(token) >= 3
                and token
                not in stop_words
            )
        }

        if not query_tokens:
            return 0.0

        text_tokens = set(
            re.findall(
                r"\w+",
                text.casefold(),
                flags=re.UNICODE,
            )
        )

        matched = (
            query_tokens
            & text_tokens
        )

        return (
            len(matched)
            / len(query_tokens)
        )

    @staticmethod
    def _query_references_document_context(
        query: str,
    ) -> bool:
        """
        Conservative deterministic document-reference detector.

        Bare "this"/"that" still do not count by themselves.
        """

        normalized = " ".join(
            query.casefold().split()
        )

        phrases = (
            # English
            "the pdf",
            "this pdf",
            "that pdf",
            "uploaded pdf",
            "my pdf",
            "the document",
            "this document",
            "that document",
            "uploaded document",
            "my document",
            "the uploaded file",
            "this file",
            "that file",
            "my file",
            "from the file",
            "in the file",
            "from the document",
            "in the document",
            "from the pdf",
            "in the pdf",
            "this page",
            "that page",
            "previous page",
            "next page",
            "this figure",
            "that figure",
            "previous figure",
            "this diagram",
            "that diagram",
            "previous diagram",
            "figure on page",
            "diagram on page",
        )

        if any(
            phrase in normalized
            for phrase in phrases
        ):
            return True

        # Topic-qualified document references do not always contain a
        # fixed phrase such as "the document". Examples:
        # "kinematics related document" or "SHM notes". The noun
        # itself is enough to request document grounding; the resolver
        # will still choose one document conservatively or clarify.
        if re.search(
            (
                r"(?i)\b("
                r"pdf|document|doc|file|notes?"
                r")\b"
            ),
            normalized,
        ):
            return True

        if re.search(
            (
                r"(?i)\b("
                r"page|figure|fig\.?|diagram"
                r")\s*(?:no\.?\s*)?#?\d+\b"
            ),
            normalized,
        ):
            return True

        return False

    @staticmethod
    def _is_out_of_scope(
        *,
        intent: IntentDecision,
        scope,
    ) -> bool:
        if intent.intent in {
            RequestIntent.OUT_OF_SCOPE,
            RequestIntent.UNSUPPORTED,
        }:
            return True

        if scope is None:
            return False

        return (
            not scope.supported
            or not scope.is_physics
            or not scope.school_level
        )

    @staticmethod
    def _has_terminal_answer(
        state: PhysicsTutorState,
    ) -> bool:
        if (
            state.get(
                "structural_clarification_required",
                False,
            )
            and state.get("answer_draft")
            is not None
        ):
            return True

        action = state.get(
            "terminal_action"
        )

        if action in {
            VerificationAction
            .REJECT_OUT_OF_SCOPE,
            VerificationAction
            .ASK_FOR_CLEARER_IMAGE,
            VerificationAction
            .INSUFFICIENT_EVIDENCE,
        }:
            return True

        intent = state.get("intent")

        return bool(
            intent is not None
            and intent.intent
            in {
                RequestIntent.GREETING,
                RequestIntent.UPLOAD_DOCUMENT,
            }
            and state.get(
                "answer_draft"
            )
            is not None
        )

    @staticmethod
    def _same_scope(
        *,
        state: PhysicsTutorState,
        user_id: str,
        document_id: str,
    ) -> bool:
        return bool(
            user_id == state["user_id"]
            and document_id
            == state.get(
                "active_document_id"
            )
        )

    @staticmethod
    def _cache_language_value(
        language: Any,
    ) -> str | None:
        # Keep the parameter for protocol/backward compatibility, while the
        # current product policy intentionally uses one English cache scope.
        _ = language
        return LanguageCode.ENGLISH.value

    def _build_cache_key(
        self,
        state: PhysicsTutorState,
    ) -> str:
        language_value = (
            LanguageCode.ENGLISH.value
        )

        pieces = [
            state["user_id"],
            state.get(
                "active_document_id"
            )
            or "no-document",
            str(
                state.get(
                    "active_page"
                )
                or 0
            ),
            str(
                state.get(
                    "estimated_grade"
                )
                or 0
            ),
            language_value,
            state.get(
                "rewritten_query"
            )
            or state[
                "normalized_query"
            ],
        ]

        raw = "\x1f".join(
            pieces
        ).encode(
            "utf-8"
        )

        digest = hashlib.sha256(
            raw
        ).hexdigest()

        return (
            "phymentor:answer:"
            f"{digest}"
        )

    @staticmethod
    def _should_skip_cache_lookup(
        state: PhysicsTutorState,
    ) -> bool:
        intent = state["intent"]

        # A user-selected model must actually execute for this request.
        # Reusing an answer generated earlier by Auto or by another selected
        # model would make the selector misleading, so selected-model turns
        # bypass answer-cache lookup entirely.
        if state.get(
            "selected_model"
        ) is not None:
            return True

        if state.get(
            "structural_resolution_status"
        ) in {
            StructuralResolutionStatus.RESOLVED,
            StructuralResolutionStatus.NEEDS_CLARIFICATION,
        }:
            return True

        return intent.intent in {
            RequestIntent.GREETING,
            RequestIntent
            .UPLOAD_DOCUMENT,
            RequestIntent.OUT_OF_SCOPE,
            RequestIntent.UNSUPPORTED,
        }

    @staticmethod
    def _is_deterministic_safe_answer(
        state: PhysicsTutorState,
    ) -> bool:
        if (
            state.get(
                "structural_clarification_required",
                False,
            )
            and state.get("answer_draft")
            is not None
        ):
            return True

        intent = state["intent"]

        return intent.intent in {
            RequestIntent.GREETING,
            RequestIntent
            .UPLOAD_DOCUMENT,
            RequestIntent.OUT_OF_SCOPE,
            RequestIntent.UNSUPPORTED,
        }

    @staticmethod
    def _exhausted_verification(
        previous: (
            VerificationResult | None
        ),
    ) -> VerificationResult:
        issue = (
            "Maximum answer-generation attempts "
            "were exhausted."
        )

        if previous is None:
            return VerificationResult(
                grounded=False,
                physics_correct=False,
                calculation_correct=False,
                units_correct=False,
                diagram_claims_supported=False,
                within_school_scope=True,
                citation_valid=False,
                issues=[issue],
                action=(
                    VerificationAction
                    .INSUFFICIENT_EVIDENCE
                ),
                confidence=1.0,
            )

        issues = list(
            previous.issues
        )

        if (
            issue not in issues
            and len(issues) < 20
        ):
            issues.append(issue)

        return VerificationResult(
            **{
                **previous.model_dump(),
                "issues": issues,
                "action": (
                    VerificationAction
                    .INSUFFICIENT_EVIDENCE
                ),
            }
        )

    @staticmethod
    def _first_figure_id(
        answer: TutorAnswer,
    ) -> str | None:
        for citation in (
            answer.citations
        ):
            if citation.figure_id:
                return (
                    citation.figure_id
                )

        return None

    @staticmethod
    def _answer_for_memory(
        answer: TutorAnswer,
    ) -> str:
        parts: list[str] = [
            answer.direct_answer
        ]

        parts.extend(
            answer.steps
        )

        if answer.diagram_explanation:
            parts.append(
                answer.diagram_explanation
            )

        if answer.final_result:
            parts.append(
                (
                    "Final result: "
                    f"{answer.final_result}"
                )
            )

        text = "\n".join(
            part.strip()
            for part in parts
            if part
            and part.strip()
        )

        return (
            text[:12000]
            or answer.direct_answer[
                :12000
            ]
        )

    @staticmethod
    def _greeting_answer(
        *,
        intent: IntentDecision,
    ) -> TutorAnswer:
        # Keep the intent parameter for the existing call contract.
        _ = intent

        text = (
            "Hello! What school-level Physics "
            "question can I help you with?"
        )

        return ServingNodes._direct_answer(
            text
        )

    @classmethod
    def _structural_clarification_answer(
        cls,
        resolution: StructuralResolution,
    ) -> TutorAnswer:
        """Render only resolver-verified choices; never invent an option."""

        lines = [
            resolution.clarification_question
            or "Which document item did you mean?"
        ]

        for index, candidate in enumerate(
            resolution.candidates,
            start=1,
        ):
            label = (
                candidate.label.strip()
                or candidate.text_preview.strip()[
                    :160
                ]
                or candidate.node_type.value.replace(
                    "_",
                    " ",
                )
            )

            location = (
                f" (page {candidate.page_start})"
                if candidate.page_start is not None
                else ""
            )

            lines.append(
                f"{index}. {label}{location}"
            )

        return cls._direct_answer(
            "\n".join(lines)
        )

    @staticmethod
    def _direct_answer(
        text: str,
    ) -> TutorAnswer:
        return TutorAnswer(
            answer_type=(
                AnswerType.DIRECT_ANSWER
            ),
            direct_answer=text,
            steps=[],
            formulae=[],
            diagram_explanation=None,
            problem_statement=None,
            common_mistake=None,
            final_result=None,
            source_pages=[],
            citations=[],
        )

    @classmethod
    def _not_enough_verified_answer(
        cls,
        *,
        answer: TutorAnswer,
        verification: VerificationResult,
    ) -> TutorAnswer:
        """
        Preserve the best available Tutor draft while making its status
        unmistakable.

        Safety properties:
        - answer_type becomes INSUFFICIENT_EVIDENCE, so downstream memory does
          not treat the turn as a successfully grounded document answer;
        - the original draft fields remain available to the user;
        - invalid citations/source pages are stripped rather than surfaced;
        - verifier issues are summarized as warnings, not silently hidden.
        """

        direct_answer = (
            answer.direct_answer.strip()
            if answer.direct_answer
            else ""
        )

        issue_lines = [
            issue.strip()
            for issue in verification.issues[:3]
            if isinstance(
                issue,
                str,
            )
            and issue.strip()
        ]

        prefix_parts = [
            cls.NOT_ENOUGH_VERIFIED_LABEL,
            (
                "The answer below is the best available draft, "
                "but it did not fully pass verification."
            ),
        ]

        if issue_lines:
            prefix_parts.append(
                "Verification note: "
                + "; ".join(
                    issue_lines
                )
            )

        if direct_answer:
            prefix_parts.append(
                direct_answer
            )

        updates: dict[str, Any] = {
            "answer_type": (
                AnswerType
                .INSUFFICIENT_EVIDENCE
            ),
            "direct_answer": (
                "\n\n".join(
                    prefix_parts
                )
            )[:12000],
        }

        if not verification.citation_valid:
            updates.update(
                {
                    "source_pages": [],
                    "citations": [],
                }
            )

        return answer.model_copy(
            update=updates
        )

    @staticmethod
    def _insufficient_answer(
        text: str,
    ) -> TutorAnswer:
        return TutorAnswer(
            answer_type=(
                AnswerType
                .INSUFFICIENT_EVIDENCE
            ),
            direct_answer=text,
            steps=[],
            formulae=[],
            diagram_explanation=None,
            problem_statement=None,
            common_mistake=None,
            final_result=None,
            source_pages=[],
            citations=[],
        )
