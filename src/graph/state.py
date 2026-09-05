from __future__ import annotations

from typing import Any, NotRequired, Required, TypedDict

from src.models.contracts import (
    IntentDecision,
    LanguageCode,
    MemorySnapshot,
    QueryRewriteResult,
    QueryScopeDecision,
    QueryUnderstandingResult,
    SourceCitation,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)
from src.retrieval.models import (
    ContextBundle,
    HybridRetrievalResult,
)
from src.models.routing import (
    UserSelectableModel,
)
from src.retrieval.structural_resolver import (
    AnswerScopeContract,
    PendingStructuralClarification,
    StructuralMatchMode,
    StructuralResolution,
    StructuralResolutionStatus,
)


class PhysicsTutorState(TypedDict, total=False):
    """
    Shared LangGraph state for the local PhyMentor AI serving workflow.

    Important:
    - This is workflow state, not the user's long-term memory.
    - Nodes should return only the keys they update.
    - Tutor and Verifier remain the only two agents.
    - Retrieval, scope, cache, memory and guardrail steps are deterministic
      workflow nodes around those agents.
    """

    # ---------------------------------------------------------
    # 1. REQUEST / SESSION IDENTITY
    # These keys must exist when the graph starts.
    # ---------------------------------------------------------
    request_id: Required[str]
    user_id: Required[str]
    session_id: Required[str]
    raw_query: Required[str]

    # ---------------------------------------------------------
    # REQUEST-LEVEL CONTROLS
    # ---------------------------------------------------------

    # The most recent/current document sent by older clients.
    #
    # We keep this field for backward compatibility.
    explicit_document_id: NotRequired[
        str | None
    ]

    # All documents currently known to this chat session.
    #
    # Only lightweight document identity metadata is stored here.
    # Actual PDF/image contents are NOT carried inside LangGraph state.
    #
    # Example:
    #
    # [
    #     {
    #         "document_id": "abc123",
    #         "name": "nuclear_fission.jpg",
    #     },
    #     {
    #         "document_id": "xyz789",
    #         "name": "shm_notes.pdf",
    #     },
    # ]
    #
    # A later deterministic/LLM-assisted resolver can use this list
    # to decide which document a particular student question refers to.
    available_documents: NotRequired[
        list[dict[str, str]]
    ]

    selected_page: NotRequired[
        int | None
    ]

    selected_figure_id: NotRequired[
        str | None
    ]

    requested_language: NotRequired[
        LanguageCode
    ]

    upload_present: NotRequired[
        bool
    ]

    strict_document_mode: NotRequired[
        bool
    ]

    # Model explicitly selected for this chat turn.
    #
    # It remains optional in workflow state for backward compatibility with
    # older API clients and existing graph tests. The real frontend will
    # require a choice before enabling question submission.
    #
    # Only the four centrally allowlisted UserSelectableModel values can
    # reach model routing; arbitrary provider model names are not accepted.
    selected_model: NotRequired[
        UserSelectableModel | None
    ]

    # ---------------------------------------------------------
    # 2. VALIDATION / NORMALIZATION
    # ---------------------------------------------------------
    normalized_query: NotRequired[
        str
    ]

    validation_passed: NotRequired[
        bool
    ]

    # ---------------------------------------------------------
    # 3. SESSION + SHORT-TERM MEMORY
    # MemorySnapshot is the stable interface to the memory layer.
    # ---------------------------------------------------------
    memory: NotRequired[
        MemorySnapshot
    ]

    next_memory: NotRequired[
        MemorySnapshot
    ]

    conversation_summary: NotRequired[
        str
    ]

    recent_messages: NotRequired[
        list[dict[str, str]]
    ]

    # ---------------------------------------------------------
    # 4. QUERY UNDERSTANDING
    # ---------------------------------------------------------
    query_understanding: NotRequired[
        QueryUnderstandingResult
    ]

    intent: NotRequired[
        IntentDecision
    ]

    scope: NotRequired[
        QueryScopeDecision
    ]

    rewrite: NotRequired[
        QueryRewriteResult
    ]

    rewritten_query: NotRequired[
        str
    ]

    retrieval_queries: NotRequired[
        list[str]
    ]

    use_hyde: NotRequired[
        bool
    ]

    hyde_text: NotRequired[
        str | None
    ]

    language: NotRequired[
        LanguageCode
    ]

    estimated_grade: NotRequired[
        int | None
    ]

    # ---------------------------------------------------------
    # 5. ACTIVE DOCUMENT / PAGE / FIGURE RESOLUTION
    # ---------------------------------------------------------

    # This is the document selected for THIS specific turn.
    #
    # With multi-document support:
    #
    # available_documents
    #         ↓
    # document resolver
    #         ↓
    # active_document_id
    #
    # Retrieval will still operate on only one resolved document
    # at a time unless we deliberately add cross-document retrieval later.
    active_document_id: NotRequired[
        str | None
    ]

    active_page: NotRequired[
        int | None
    ]

    referenced_figure_id: NotRequired[
        str | None
    ]

    preferred_page_numbers: NotRequired[
        list[int]
    ]

    prefer_visual: NotRequired[
        bool
    ]

    # ---------------------------------------------------------
    # 5A. STRUCTURAL DOCUMENT REFERENCE RESOLUTION
    # ---------------------------------------------------------
    #
    # Structural resolution runs before ordinary semantic retrieval for
    # document-dependent questions. It does not replace FAISS/BM25/reranking.
    #
    # RESOLVED:
    # - verified source node/evidence IDs are carried forward;
    # - Tutor receives the strict answer-scope contract.
    #
    # NEEDS_CLARIFICATION:
    # - the bounded candidate set is returned to the student;
    # - semantic retrieval must not silently guess.
    #
    # NO_MATCH / STRUCTURE_UNAVAILABLE:
    # - structural_fallback_to_semantic allows the old retrieval path.
    # ---------------------------------------------------------

    structural_resolution_attempted: NotRequired[
        bool
    ]

    structural_resolution: NotRequired[
        StructuralResolution | None
    ]

    structural_resolution_status: NotRequired[
        StructuralResolutionStatus | None
    ]

    structural_match_mode: NotRequired[
        StructuralMatchMode | None
    ]

    # A pending clarification loaded from short-term session memory.
    # Replies such as "both", "first", or "second" are resolved only
    # against this bounded candidate set.
    pending_structural_clarification: NotRequired[
        PendingStructuralClarification | None
    ]

    # The value that the memory-write node should persist for the next turn.
    # Set it to the new pending clarification, or None after it is resolved,
    # abandoned, or superseded by a clear topic change.
    next_pending_structural_clarification: NotRequired[
        PendingStructuralClarification | None
    ]

    structural_clarification_required: NotRequired[
        bool
    ]

    structural_clarification_question: NotRequired[
        str | None
    ]

    structural_target_node_ids: NotRequired[
        list[str]
    ]

    structural_candidate_node_ids: NotRequired[
        list[str]
    ]

    structural_linked_retrieval_chunk_ids: NotRequired[
        list[str]
    ]

    structural_linked_parent_chunk_ids: NotRequired[
        list[str]
    ]

    structural_linked_figure_ids: NotRequired[
        list[str]
    ]

    structural_source_page_numbers: NotRequired[
        list[int]
    ]

    structural_visual_page_numbers: NotRequired[
        list[int]
    ]

    structural_needs_visual: NotRequired[
        bool
    ]

    structural_answer_scope: NotRequired[
        AnswerScopeContract | None
    ]

    structural_fallback_to_semantic: NotRequired[
        bool
    ]

    structural_warning: NotRequired[
        str | None
    ]

    # Design-friendly scalar scope metadata kept explicitly in state.
    scope_status: NotRequired[
        str
    ]

    scope_confidence: NotRequired[
        float
    ]

    # ---------------------------------------------------------
    # 6. CACHE
    # check_query_cache -> HIT goes toward output guard.
    # ---------------------------------------------------------
    cache_key: NotRequired[
        str | None
    ]

    cache_hit: NotRequired[
        bool
    ]

    cached_answer: NotRequired[
        TutorAnswer | None
    ]

    # ---------------------------------------------------------
    # 7. RETRIEVAL
    # Phase-4 RetrievalService is reused; it is NOT reimplemented here.
    # ---------------------------------------------------------
    retrieval_results: NotRequired[
        list[HybridRetrievalResult]
    ]

    # These three fields mirror the design vocabulary and are useful for
    # observability/debugging. The real grounded answering payload is
    # reranked_context: ContextBundle.
    retrieved_chunks: NotRequired[
        list[dict[str, Any]]
    ]

    retrieved_figures: NotRequired[
        list[dict[str, Any]]
    ]

    reranked_context: NotRequired[
        ContextBundle | None
    ]

    retrieval_rounds: NotRequired[
        int
    ]

    broader_retrieval_requested: NotRequired[
        bool
    ]

    # ---------------------------------------------------------
    # 8. TUTOR AGENT
    # ---------------------------------------------------------
    answer_draft: NotRequired[
        TutorAnswer | None
    ]

    generation_attempts: NotRequired[
        int
    ]

    # Feedback is passed to Tutor only when a bounded regeneration is allowed.
    verifier_feedback: NotRequired[
        list[str]
    ]

    # ---------------------------------------------------------
    # 9. VERIFIER AGENT
    # ---------------------------------------------------------
    verification_result: NotRequired[
        VerificationResult | None
    ]

    terminal_action: NotRequired[
        VerificationAction | None
    ]

    verification_passed: NotRequired[
        bool
    ]

    # Design says maximum answer-generation attempts = 2.
    # retry_count is kept separately because the PDF names it explicitly.
    retry_count: NotRequired[
        int
    ]

    # ---------------------------------------------------------
    # 10. OUTPUT GUARD / SAFE TERMINAL RESPONSE
    # final_answer is what is allowed to leave the serving workflow.
    # A rejected Tutor draft must never be returned directly.
    # ---------------------------------------------------------
    final_answer: NotRequired[
        TutorAnswer | None
    ]

    output_guard_passed: NotRequired[
        bool
    ]

    # ---------------------------------------------------------
    # 11. MEMORY + CACHE WRITE DECISIONS
    # ---------------------------------------------------------
    should_write_memory: NotRequired[
        bool
    ]

    memory_candidates: NotRequired[
        list[dict[str, Any]]
    ]

    should_write_cache: NotRequired[
        bool
    ]

    # ---------------------------------------------------------
    # 12. CITATIONS / ERRORS / OBSERVABILITY
    # ---------------------------------------------------------
    citations: NotRequired[
        list[SourceCitation]
    ]

    errors: NotRequired[
        list[str]
    ]
