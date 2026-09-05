from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class StrictModel(BaseModel):
    """
    Shared base model for Phase-5 contracts.

    extra="forbid" prevents silent schema drift
    between routing, query understanding,
    Tutor, Verifier and model gateway.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


# =========================================================
# REQUEST / ROUTING CONTRACTS
# =========================================================


class RequestIntent(str, Enum):
    GREETING = "GREETING"
    UPLOAD_DOCUMENT = "UPLOAD_DOCUMENT"
    PHYSICS_QUESTION = "PHYSICS_QUESTION"
    DIAGRAM_QUESTION = "DIAGRAM_QUESTION"
    NUMERICAL_PROBLEM = "NUMERICAL_PROBLEM"
    FOLLOW_UP = "FOLLOW_UP"
    VOICE_CONTROL = "VOICE_CONTROL"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNSUPPORTED = "UNSUPPORTED"


class LanguageCode(str, Enum):
    ENGLISH = "en"
    BENGALI = "bn"
    HINDI = "hi"
    BENGALI_ENGLISH_MIXED = "bn_en"
    UNKNOWN = "unknown"


class ScopeStatus(str, Enum):
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNCERTAIN = "UNCERTAIN"


class DocumentUsage(str, Enum):
    """
    How the current request intends to use an uploaded document.
    """

    NONE = "NONE"
    SOURCE_CONTEXT = "SOURCE_CONTEXT"
    SOURCE_SPECIFIC = "SOURCE_SPECIFIC"
    SUMMARY = "SUMMARY"


class DocumentTask(str, Enum):
    """
    What the student wants to do with document evidence.

    This is semantic task metadata produced by query understanding.
    It does not select a document, retrieve chunks, or choose a figure.
    """

    TEXT_QUESTION = "TEXT_QUESTION"
    DIAGRAM_EXPLANATION = "DIAGRAM_EXPLANATION"
    PROOF_EXPLANATION = "PROOF_EXPLANATION"
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    DOCUMENT_NUMERICAL = "DOCUMENT_NUMERICAL"


class FigureReferenceType(str, Enum):
    """
    How the user referred to a figure.

    Physics-specific semantic tags are intentionally
    not hard-coded here.
    """

    EXACT_LABEL = "EXACT_LABEL"
    SEMANTIC = "SEMANTIC"
    POSITIONAL = "POSITIONAL"
    CONTEXTUAL = "CONTEXTUAL"
    PAGE = "PAGE"


class ModelTask(str, Enum):
    INTENT_CLASSIFICATION = (
        "INTENT_CLASSIFICATION"
    )

    QUERY_SCOPE = "QUERY_SCOPE"

    QUERY_REWRITE = "QUERY_REWRITE"

    MEMORY_EXTRACTION = (
        "MEMORY_EXTRACTION"
    )

    # Page-level Vision extraction and cross-page
    # document-structure generation.
    #
    # This stays separate from Tutor calls so logs,
    # token usage, costs and failures are recorded
    # under the correct task.
    DOCUMENT_STRUCTURE = (
        "DOCUMENT_STRUCTURE"
    )

    # Resolve natural-language references against the saved
    # structural document index before semantic retrieval.
    STRUCTURAL_RESOLUTION = (
        "STRUCTURAL_RESOLUTION"
    )

    TUTOR_TEXT = "TUTOR_TEXT"

    TUTOR_MULTIMODAL = (
        "TUTOR_MULTIMODAL"
    )

    TUTOR_NUMERICAL = (
        "TUTOR_NUMERICAL"
    )

    VERIFIER = "VERIFIER"


class VerificationAction(str, Enum):
    PASS = "PASS"

    RETRY_RETRIEVAL = (
        "RETRY_RETRIEVAL"
    )

    REGENERATE = "REGENERATE"

    ASK_FOR_CLEARER_IMAGE = (
        "ASK_FOR_CLEARER_IMAGE"
    )

    INSUFFICIENT_EVIDENCE = (
        "INSUFFICIENT_EVIDENCE"
    )

    REJECT_OUT_OF_SCOPE = (
        "REJECT_OUT_OF_SCOPE"
    )


class AnswerType(str, Enum):
    DIRECT_ANSWER = "direct_answer"

    CONCEPT_EXPLANATION = (
        "concept_explanation"
    )

    FORMULA_EXPLANATION = (
        "formula_explanation"
    )

    NUMERICAL_SOLUTION = (
        "numerical_solution"
    )

    DIAGRAM_EXPLANATION = (
        "diagram_explanation"
    )

    PROOF_EXPLANATION = (
        "proof_explanation"
    )

    DOCUMENT_SUMMARY = (
        "document_summary"
    )

    INSUFFICIENT_EVIDENCE = (
        "insufficient_evidence"
    )


# =========================================================
# MEMORY INPUT CONTRACT
# =========================================================


class ConversationMessage(StrictModel):
    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
        max_length=12000,
    )


class SessionDocumentReference(
    StrictModel
):
    """
    Lightweight reference to one document belonging
    to the current conversation session.

    IMPORTANT:
    This stores only identity metadata.

    It does NOT store:
    - file bytes
    - parsed text
    - chunks
    - embeddings
    - FAISS/BM25 indexes
    """

    document_id: str = Field(
        min_length=1,
        max_length=200,
    )

    name: str = Field(
        min_length=1,
        max_length=500,
    )


class PendingStructuralClarification(
    StrictModel
):
    """
    Bounded structural choices remembered between two chat turns.

    Example:
    - turn 1: two plausible source items remain;
    - PhyMentor asks which one the student means;
    - turn 2: the student replies "first", "second", or "both".

    Only source identity is stored here. PDF bytes, chunks, embeddings,
    model prompts, and generated answers are never copied into memory.
    """

    document_id: str = Field(
        min_length=1,
        max_length=300,
    )

    original_query: str = Field(
        min_length=1,
        max_length=12000,
    )

    clarification_question: str = Field(
        min_length=1,
        max_length=2000,
    )

    candidate_node_ids: list[str] = Field(
        min_length=1,
        max_length=20,
    )

    @field_validator(
        "candidate_node_ids"
    )
    @classmethod
    def validate_candidate_node_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        normalized = [
            node_id.strip()
            for node_id in value
            if node_id.strip()
        ]

        normalized = list(
            dict.fromkeys(normalized)
        )

        if not normalized:
            raise ValueError(
                "candidate_node_ids must contain at least one node ID."
            )

        return normalized[:20]


class MemorySnapshot(StrictModel):
    """
    Stable interface between the memory layer and
    Phase-5 query understanding.

    One session may know several uploaded documents.

    `available_documents` is the session's lightweight
    document bookshelf.

    `active_document_id` remains useful as the most
    recently used/contextual document, but it is NOT the
    only document available to the session.
    """

    available_documents: list[
        SessionDocumentReference
    ] = Field(
        default_factory=list,
        max_length=30,
    )

    active_document_id: (
        str | None
    ) = None

    # Document used by the immediately previous
    # completed turn.
    #
    # This is intentionally separate from
    # `active_document_id`.
    last_turn_document_id: (
        str | None
    ) = None

    active_page: (
        int | None
    ) = Field(
        default=None,
        ge=1,
    )

    last_selected_figure_id: (
        str | None
    ) = None

    recent_messages: list[
        ConversationMessage
    ] = Field(
        default_factory=list,
        max_length=10,
    )

    language: LanguageCode = (
        LanguageCode.UNKNOWN
    )

    estimated_grade: (
        int | None
    ) = Field(
        default=None,
        ge=1,
        le=12,
    )

    explanation_depth: (
        str | None
    ) = None

    problem_solving_state: (
        str | None
    ) = None

    # A clarification is kept only until the immediately following
    # student turn resolves, abandons, or supersedes it.
    pending_structural_clarification: (
        PendingStructuralClarification | None
    ) = None

    @field_validator(
        "available_documents"
    )
    @classmethod
    def validate_available_documents(
        cls,
        value: list[
            SessionDocumentReference
        ],
    ) -> list[
        SessionDocumentReference
    ]:
        """
        Deduplicate documents by document_id while
        preserving their order.
        """

        deduplicated: list[
            SessionDocumentReference
        ] = []

        seen_document_ids: set[str] = set()

        for document in value:
            document_id = (
                document.document_id.strip()
            )

            if (
                not document_id
                or document_id
                in seen_document_ids
            ):
                continue

            seen_document_ids.add(
                document_id
            )

            deduplicated.append(
                document
            )

        return deduplicated[:30]


# =========================================================
# INTENT / QUERY-REQUIREMENT CONTRACTS
# =========================================================


class RequestedQuantity(StrictModel):
    """
    One physical quantity the student is asking
    the Tutor to determine.
    """

    quantity: str = Field(
        min_length=1,
        max_length=200,
    )

    symbol: (
        str | None
    ) = Field(
        default=None,
        max_length=120,
    )

    expected_dimension: (
        str | None
    ) = Field(
        default=None,
        max_length=120,
    )

    raw_reference: (
        str | None
    ) = Field(
        default=None,
        max_length=1000,
    )


class GivenQuantity(StrictModel):
    """
    One quantity/value explicitly supplied
    in the user's text.

    Values and units remain strings so the
    original mathematical/Unicode representation
    is preserved.
    """

    quantity: str = Field(
        min_length=1,
        max_length=200,
    )

    symbol: (
        str | None
    ) = Field(
        default=None,
        max_length=120,
    )

    raw_value: (
        str | None
    ) = Field(
        default=None,
        max_length=500,
    )

    raw_unit: (
        str | None
    ) = Field(
        default=None,
        max_length=300,
    )

    raw_text: (
        str | None
    ) = Field(
        default=None,
        max_length=2000,
    )


class FigureReference(StrictModel):
    """
    Structured description of how the user
    referred to a figure.
    """

    reference_type: FigureReferenceType

    raw_reference: str = Field(
        min_length=1,
        max_length=1000,
    )

    exact_label: (
        str | None
    ) = Field(
        default=None,
        max_length=300,
    )

    semantic_description: (
        str | None
    ) = Field(
        default=None,
        max_length=1500,
    )

    ordinal: (
        int | None
    ) = Field(
        default=None,
        ge=1,
    )

    page_number: (
        int | None
    ) = Field(
        default=None,
        ge=1,
    )


class ProblemReference(StrictModel):
    """
    Structured reference to a problem/exercise
    named by the user.

    Examples:
    - "problem 1"
    - "question 3"
    - "the second numerical"
    - "the exercise on page 12"

    This stores only the user's reference.
    It never invents a problem label.
    """

    raw_reference: str = Field(
        min_length=1,
        max_length=1200,
    )

    exact_label: (
        str | None
    ) = Field(
        default=None,
        max_length=300,
    )

    ordinal: (
        int | None
    ) = Field(
        default=None,
        ge=1,
    )

    page_number: (
        int | None
    ) = Field(
        default=None,
        ge=1,
    )


class IntentDecision(StrictModel):
    intent: RequestIntent

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    language: LanguageCode = (
        LanguageCode.UNKNOWN
    )

    estimated_grade: (
        int | None
    ) = Field(
        default=None,
        ge=1,
        le=12,
    )

    has_physics_request: bool

    is_follow_up: bool

    prefer_visual: bool

    requested_quantities: list[
        RequestedQuantity
    ] = Field(
        default_factory=list,
        max_length=12,
    )

    given_quantities: list[
        GivenQuantity
    ] = Field(
        default_factory=list,
        max_length=40,
    )

    # Equations stay as original strings.
    given_equations: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    # None means "not yet determined"
    # during the staged rollout.
    requires_document: (
        bool | None
    ) = None

    requires_visual: (
        bool | None
    ) = None

    document_usage: (
        DocumentUsage | None
    ) = None

    document_task: (
        DocumentTask | None
    ) = None

    # Document answers may use both retrieved
    # source evidence and established general
    # Physics knowledge.
    wants_document_plus_general_physics: (
        bool
    ) = False

    wants_document_summary: bool = False

    figure_reference: (
        FigureReference | None
    ) = None

    problem_reference: (
        ProblemReference | None
    ) = None


# =========================================================
# QUERY SCOPE CONTRACT
# =========================================================


class QueryScopeDecision(
    StrictModel
):
    is_physics: bool

    school_level: bool

    supported: bool

    status: ScopeStatus

    estimated_grade_range: (
        list[int] | None
    ) = Field(
        default=None,
        min_length=2,
        max_length=2,
    )

    topics: list[str] = Field(
        default_factory=list,
        max_length=12,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: (
        str | None
    ) = Field(
        default=None,
        max_length=600,
    )

    @field_validator(
        "estimated_grade_range"
    )
    @classmethod
    def validate_grade_range(
        cls,
        value: list[int] | None,
    ) -> list[int] | None:

        if value is None:
            return None

        grade_min, grade_max = value

        if not (
            1
            <= grade_min
            <= grade_max
            <= 12
        ):
            raise ValueError(
                "estimated_grade_range "
                "must stay within "
                "Classes 1-12."
            )

        return value


# =========================================================
# QUERY REWRITE CONTRACT
# =========================================================


class QueryRewriteResult(
    StrictModel
):
    original_query: str = Field(
        min_length=1,
        max_length=12000,
    )

    rewritten_query: str = Field(
        min_length=1,
        max_length=12000,
    )

    # PDF allows multi-query retrieval
    # for complex questions.
    retrieval_queries: list[str] = (
        Field(
            min_length=1,
            max_length=3,
        )
    )

    was_rewritten: bool

    prefer_visual: bool

    preferred_page_numbers: list[
        int
    ] = Field(
        default_factory=list,
        max_length=5,
    )

    referenced_figure_id: (
        str | None
    ) = None

    # Conditional only.
    # This does NOT mean every query uses HyDE.
    use_hyde: bool = False

    hyde_text: (
        str | None
    ) = Field(
        default=None,
        max_length=3000,
    )

    @field_validator(
        "preferred_page_numbers"
    )
    @classmethod
    def validate_pages(
        cls,
        value: list[int],
    ) -> list[int]:

        if any(
            page < 1
            for page in value
        ):
            raise ValueError(
                "preferred_page_numbers "
                "must contain positive "
                "page numbers."
            )

        return list(
            dict.fromkeys(value)
        )

    @field_validator(
        "retrieval_queries"
    )
    @classmethod
    def validate_retrieval_queries(
        cls,
        value: list[str],
    ) -> list[str]:

        cleaned: list[str] = []

        for query in value:
            normalized = (
                query.strip()
            )

            if (
                normalized
                and normalized
                not in cleaned
            ):
                cleaned.append(
                    normalized
                )

        if not cleaned:
            raise ValueError(
                "At least one non-empty "
                "retrieval query is required."
            )

        return cleaned[:3]


class QueryUnderstandingResult(
    StrictModel
):
    # Preserve the user's original text separately
    # from normalized/retrieval forms.
    raw_query: (
        str | None
    ) = Field(
        default=None,
        max_length=12000,
    )

    normalized_query: str = Field(
        min_length=1,
        max_length=12000,
    )

    intent: IntentDecision

    scope: (
        QueryScopeDecision | None
    ) = None

    rewrite: (
        QueryRewriteResult | None
    ) = None

    active_document_id: (
        str | None
    ) = None


# =========================================================
# TUTOR OUTPUT CONTRACT
# =========================================================


class FormulaItem(StrictModel):
    latex: str = Field(
        min_length=1,
        max_length=2000,
    )

    meaning: str = Field(
        min_length=1,
        max_length=2000,
    )


class SourceCitation(StrictModel):
    page_number: int = Field(
        ge=1,
    )

    source_chunk_ids: list[
        str
    ] = Field(
        default_factory=list,
        max_length=30,
    )

    figure_id: (
        str | None
    ) = None


class TutorAnswer(StrictModel):
    answer_type: AnswerType

    direct_answer: str = Field(
        min_length=1,
        max_length=12000,
    )

    steps: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    formulae: list[
        FormulaItem
    ] = Field(
        default_factory=list,
        max_length=20,
    )

    diagram_explanation: (
        str | None
    ) = Field(
        default=None,
        max_length=8000,
    )

    # For a numerical retrieved from an uploaded
    # document, preserve the problem statement
    # separately.
    problem_statement: (
        str | None
    ) = Field(
        default=None,
        max_length=12000,
    )

    common_mistake: (
        str | None
    ) = Field(
        default=None,
        max_length=3000,
    )

    final_result: (
        str | None
    ) = Field(
        default=None,
        max_length=3000,
    )

    source_pages: list[
        int
    ] = Field(
        default_factory=list,
        max_length=20,
    )

    citations: list[
        SourceCitation
    ] = Field(
        default_factory=list,
        max_length=30,
    )

    @field_validator(
        "source_pages"
    )
    @classmethod
    def validate_source_pages(
        cls,
        value: list[int],
    ) -> list[int]:

        if any(
            page < 1
            for page in value
        ):
            raise ValueError(
                "source_pages must "
                "contain positive "
                "page numbers."
            )

        return list(
            dict.fromkeys(value)
        )


# =========================================================
# VERIFIER OUTPUT CONTRACT
# =========================================================


class VerificationResult(
    StrictModel
):
    grounded: bool

    physics_correct: bool

    calculation_correct: bool

    units_correct: bool

    diagram_claims_supported: bool

    within_school_scope: bool

    citation_valid: bool

    issues: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    action: VerificationAction

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


# =========================================================
# MODEL GATEWAY METADATA
# =========================================================


class ModelCallMetadata(
    StrictModel
):
    task: ModelTask

    provider: str

    model: str

    # Phase-5 policy:
    # bounded attempts only.
    attempt_count: int = Field(
        ge=1,
        le=2,
    )

    used_fallback: bool = False

    latency_ms: float = Field(
        ge=0.0,
    )

    input_tokens: (
        int | None
    ) = Field(
        default=None,
        ge=0,
    )

    output_tokens: (
        int | None
    ) = Field(
        default=None,
        ge=0,
    )

    total_tokens: (
        int | None
    ) = Field(
        default=None,
        ge=0,
    )
