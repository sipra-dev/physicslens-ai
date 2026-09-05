from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from src.models.contracts import (
    LanguageCode,
    RequestIntent,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)
from src.models.routing import (
    UserSelectableModel,
)


class ChatDocumentReference(BaseModel):
    """
    Lightweight reference to a document that is available
    to the current frontend chat session.

    This does NOT contain document contents.
    It only gives the backend enough identity information
    to resolve which document a student's question refers to.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    document_id: str = Field(
        min_length=1,
        max_length=200,
    )

    name: str = Field(
        min_length=1,
        max_length=500,
    )

    @field_validator(
        "document_id",
        "name",
    )
    @classmethod
    def reject_blank_strings(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "value cannot be blank."
            )

        return normalized


class ChatRequest(BaseModel):
    """
    Public request body for POST /v1/chat.

    user_id is intentionally NOT accepted from the JSON body.
    The API route resolves user identity separately
    through the header/auth layer.

    document_id remains supported for backward compatibility
    and represents the most recently/currently active document.

    available_documents contains the documents known to the
    current frontend session so the backend can resolve a
    document automatically for each question.

    selected_model is optional at the public schema boundary only for
    backward compatibility with older clients and existing automated tests.
    The real frontend will require the student to choose one of the four
    centrally allowlisted models before enabling submission.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    session_id: str = Field(
        min_length=1,
        max_length=200,
    )

    query: str = Field(
        min_length=1,
        max_length=12000,
    )

    # ---------------------------------------------------------
    # BACKWARD-COMPATIBLE CURRENT / RECENT DOCUMENT
    # ---------------------------------------------------------
    document_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    # ---------------------------------------------------------
    # ALL DOCUMENTS AVAILABLE TO THIS CHAT SESSION
    #
    # Example:
    # [
    #     {
    #         "document_id": "abc",
    #         "name": "nuclear_fission.jpg",
    #     },
    #     {
    #         "document_id": "xyz",
    #         "name": "shm_notes.pdf",
    #     },
    # ]
    # ---------------------------------------------------------
    available_documents: list[
        ChatDocumentReference
    ] = Field(
        default_factory=list,
        max_length=30,
    )

    selected_page: int | None = Field(
        default=None,
        ge=1,
    )

    selected_figure_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
    )

    language: LanguageCode = (
        LanguageCode.UNKNOWN
    )

    # Pydantic converts an accepted string to UserSelectableModel and rejects
    # any arbitrary provider model name before the request reaches LangGraph.
    selected_model: (
        UserSelectableModel | None
    ) = None

    @field_validator(
        "session_id",
        "query",
    )
    @classmethod
    def reject_blank_required_strings(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "value cannot be blank."
            )

        return normalized

    @field_validator(
        "document_id",
        "selected_figure_id",
    )
    @classmethod
    def normalize_optional_strings(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "optional string cannot be blank."
            )

        return normalized

    @field_validator(
        "available_documents",
    )
    @classmethod
    def deduplicate_available_documents(
        cls,
        documents: list[
            ChatDocumentReference
        ],
    ) -> list[
        ChatDocumentReference
    ]:
        """
        Never send the same document to the resolver twice.
        Preserve the original order.
        """

        seen: set[str] = set()

        unique_documents: list[
            ChatDocumentReference
        ] = []

        for document in documents:
            if document.document_id in seen:
                continue

            seen.add(
                document.document_id
            )

            unique_documents.append(
                document
            )

        return unique_documents


class ChatResponse(BaseModel):
    """
    Public response contract for POST /v1/chat.

    Internal retrieval context and raw provider details are deliberately not
    exposed. The validated request-level model choice is echoed so the
    frontend can show which selected model produced this turn.

    Grounded source references remain available inside TutorAnswer.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str
    session_id: str
    document_id: str | None

    # Optional for compatibility with responses produced by older tests and
    # clients during this staged integration.
    selected_model: (
        UserSelectableModel | None
    ) = None

    intent: RequestIntent

    answer: TutorAnswer

    verification: (
        VerificationResult | None
    ) = None

    terminal_action: (
        VerificationAction | None
    ) = None

    generation_attempts: int = Field(
        ge=0,
        le=2,
    )

    retrieval_rounds: int = Field(
        ge=0,
        le=2,
    )


class SelectionExplainRequest(BaseModel):
    """
    Request contract for explaining text selected from a rendered
    PhyMentor answer.

    The frontend hover/popover is only a UI trigger. The backend contract
    is named after the actual action: explain a text selection.

    user_id is intentionally NOT accepted from the JSON body.
    The route will resolve user identity through the same header/auth
    boundary used by normal chat.

    selected_text may be a word, phrase, equation text, or full sentence.

    surrounding_text is optional bounded context from the assistant answer.
    It helps disambiguate short selections without sending the whole chat.

    document/page/figure values are contextual hints only. They must never
    make the selection endpoint cross document or user boundaries.

    selected_model is optional so an explicitly selected chat model can be
    preserved for the explanation turn when model generation is actually
    needed.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    session_id: str = Field(
        min_length=1,
        max_length=200,
    )

    selected_text: str = Field(
        min_length=1,
        max_length=2000,
    )

    surrounding_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=6000,
    )

    document_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    selected_page: int | None = Field(
        default=None,
        ge=1,
    )

    selected_figure_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
    )

    selected_model: (
        UserSelectableModel | None
    ) = None

    @field_validator(
        "session_id",
        "selected_text",
    )
    @classmethod
    def reject_blank_required_selection_strings(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "value cannot be blank."
            )

        return normalized

    @field_validator(
        "surrounding_text",
        "document_id",
        "selected_figure_id",
    )
    @classmethod
    def normalize_optional_selection_strings(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "optional string cannot be blank."
            )

        return normalized


class SelectionExplainResponse(BaseModel):
    """
    Public response contract for selected-text explanation.

    found=False is an explicit, user-visible outcome. The backend must not
    invent a memory/context match merely to produce an explanation.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str
    session_id: str

    selected_text: str

    found: bool

    explanation: str = Field(
        min_length=1,
        max_length=6000,
    )

    document_id: str | None = None

    selected_model: (
        UserSelectableModel | None
    ) = None
