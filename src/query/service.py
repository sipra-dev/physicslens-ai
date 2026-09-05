from __future__ import annotations

import re
from typing import Protocol, TypeVar

from pydantic import BaseModel

from src.models.contracts import (
    DocumentUsage,
    FigureReference,
    FigureReferenceType,
    IntentDecision,
    LanguageCode,
    MemorySnapshot,
    ModelTask,
    QueryRewriteResult,
    QueryScopeDecision,
    QueryUnderstandingResult,
    RequestIntent,
    ScopeStatus,
)
from src.models.routing import (
    ModelRoute,
    ModelRouter,
    UserSelectableModel,
)
from src.prompts.intent import (
    INTENT_SYSTEM_PROMPT,
    build_intent_user_prompt,
)
from src.prompts.query_rewrite import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    build_query_rewrite_user_prompt,
)
from src.prompts.query_scope import (
    QUERY_SCOPE_SYSTEM_PROMPT,
    build_query_scope_user_prompt,
)


TModel = TypeVar(
    "TModel",
    bound=BaseModel,
)


class QueryUnderstandingError(Exception):
    """Raised when Phase-5 query understanding cannot complete safely."""


class StructuredModelRunner(Protocol):
    """
    Provider-neutral interface required by the query layer.

    The future LLMGateway will implement this contract.
    Query understanding therefore does not import OpenAI directly.
    """

    def generate_structured(
        self,
        *,
        route: ModelRoute,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
    ) -> TModel:
        ...


_GREETING_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hello there",
    "hi there",
    "good morning",
    "good afternoon",
    "good evening",
    "namaste",
    "নমস্কার",
    "হ্যালো",
    "হাই",
    "সুপ্রভাত",
}

_DEICTIC_PATTERN = re.compile(
    r"(?i)\b("
    r"this|that|these|those|it|"
    r"second one|first one|last one|"
    r"this graph|that graph|"
    r"this diagram|that diagram|"
    r"this figure|that figure|"
    r"this formula|that formula|"
    r"above|below|previous"
    r")\b"
)

_BENGALI_DEICTIC_PATTERN = re.compile(
    r"(এটা|ওটা|এটি|ওটি|এইটা|ওইটা|"
    r"এই গ্রাফ|ওই গ্রাফ|এই ডায়াগ্রাম|"
    r"ওই ডায়াগ্রাম|এই সূত্র|ওই সূত্র|"
    r"আগেরটা|দ্বিতীয়টা|প্রথমটা)"
)

_COMPLEX_QUERY_PATTERN = re.compile(
    r"(?i)("
    r"\bcompare\b|\bcontrast\b|"
    r"\bexplain why\b|"
    r"\brelationship between\b|"
    r"\bdifference between\b|"
    r"\band\b.*\band\b|"
    r"তুলনা|পার্থক্য|এবং|"
    r"तुलना|अंतर|और"
    r")"
)


_DOCUMENT_VISUAL_REFERENCE_PHRASES = (
    # English
    "this diagram",
    "that diagram",
    "the diagram",
    "uploaded diagram",
    "previous diagram",
    "earlier diagram",
    "this figure",
    "that figure",
    "the figure",
    "uploaded figure",
    "previous figure",
    "earlier figure",
    "this image",
    "that image",
    "the image",
    "uploaded image",
    "previous image",
    "earlier image",
    "this graph",
    "that graph",
    "the graph",
    "uploaded graph",
    "previous graph",
    "earlier graph",
    # Banglish
    "ei diagram",
    "oi diagram",
    "ager diagram",
    "aager diagram",
    "ei figure",
    "oi figure",
    "ager figure",
    "aager figure",
    "ei image",
    "oi image",
    "ager image",
    "aager image",
    "ei graph",
    "oi graph",
    "ager graph",
    "aager graph",
    # Bengali
    "এই ডায়াগ্রাম",
    "ওই ডায়াগ্রাম",
    "আগের ডায়াগ্রাম",
    "এই চিত্র",
    "ওই চিত্র",
    "আগের চিত্র",
    "এই গ্রাফ",
    "ওই গ্রাফ",
    "আগের গ্রাফ",
    # Hindi
    "इस डायग्राम",
    "उस डायग्राम",
    "पिछला डायग्राम",
    "इस चित्र",
    "उस चित्र",
    "पिछला चित्र",
    "इस ग्राफ",
    "उस ग्राफ",
    "पिछला ग्राफ",
)


# Explicit references that make a turn depend on the uploaded/session context.
#
# These are structural/conversational markers only. No Physics topic names are
# hard-coded here.
_SESSION_CONTEXT_REFERENCE_PATTERN = re.compile(
    r"(?ix)("
    r"\b(?:document|pdf|file|uploaded|source|notes|material)\b|"
    r"\bq\s*\.?\s*\d+\b|"
    r"\b(?:page|question|problem|exercise|example|part|property|section)"
    r"\s*(?:no\.?\s*)?(?:\d+|\(?[a-zivx]+\)?)\b|"
    r"\b(?:fig(?:ure)?\.?)\s*\d+\b|"
    r"\b(?:from|in|according\s+to|based\s+on|using)\s+"
    r"(?:the\s+|this\s+|that\s+)?"
    r"(?:document|pdf|file|source|notes|material|page|figure|diagram|image|graph)\b|"
    r"\b(?:next|same)\s+one\b|"
    r"\bthe\s+(?:minus|plus)\s+sign\b"
    r")"
)


_SHORT_FOLLOW_UP_PHRASES = {
    "why",
    "how",
    "how so",
    "then",
    "then what",
    "and",
    "what about",
    "continue",
    "continue please",
    "go on",
    "explain further",
    "explain more",
    "again",
    "same",
    "both",
}


class QueryUnderstandingService:
    """
    Deterministic orchestration for Phase-5 query understanding.

    Flow:
        normalize
          -> greeting/upload fast path
          -> structured intent classification
          -> structured query-scope classification
          -> contextual query rewrite when useful
          -> QueryUnderstandingResult

    This service does NOT perform retrieval and does NOT answer questions.
    Phase-4 RetrievalService remains the retrieval implementation.
    """

    def __init__(
        self,
        *,
        model_runner: StructuredModelRunner,
        model_router: ModelRouter,
    ) -> None:
        self.model_runner = model_runner
        self.model_router = model_router

    def understand(
        self,
        *,
        query: str,
        memory: MemorySnapshot | None = None,
        upload_present: bool = False,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> QueryUnderstandingResult:
        # Keep the user's original text separate from the normalized
        # representation used by deterministic routing/retrieval helpers.
        # This is important for equations, units, Unicode and mathematical
        # symbols. We validate the raw text without destructively rewriting it.
        raw_query = self._validate_raw_query(
            query
        )

        normalized_query = self._normalize_query(
            raw_query
        )

        resolved_memory = (
            memory
            if memory is not None
            else MemorySnapshot()
        )

        # -------------------------------------------------
        # QUERY-LOCAL MEMORY VIEW
        # -------------------------------------------------
        #
        # Uploaded documents may remain in Redis for natural follow-ups, but
        # merely having that memory must not make a new self-contained Physics
        # question document-grounded.
        #
        # Examples:
        #
        #     "What are Newton's laws of motion?"
        #         -> classify without old document/conversation anchors
        #
        #     "Explain this diagram again."
        #     "Solve question 16."
        #     "Why?"
        #         -> keep the full session context
        #
        # If the model itself identifies a genuine follow-up after a
        # context-neutral classification, we promote back to full memory before
        # scope classification and rewriting.
        explicit_session_context = (
            self._query_explicitly_uses_session_context(
                raw_query
            )
        )

        query_memory = (
            resolved_memory
            if explicit_session_context
            else self._without_session_context(
                resolved_memory
            )
        )

        if upload_present:
            intent = IntentDecision(
                intent=(
                    RequestIntent.UPLOAD_DOCUMENT
                ),
                confidence=1.0,
                language=self._detect_language(
                    raw_query
                ),
                estimated_grade=(
                    resolved_memory.estimated_grade
                ),
                has_physics_request=False,
                is_follow_up=False,
                prefer_visual=False,
            )

            return QueryUnderstandingResult(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=intent,
                scope=None,
                rewrite=None,
                active_document_id=(
                    resolved_memory
                    .active_document_id
                ),
            )

        greeting = self._greeting_fast_path(
            normalized_query,
            resolved_memory,
        )

        if greeting is not None:
            return QueryUnderstandingResult(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=greeting,
                scope=None,
                rewrite=None,
                active_document_id=(
                    resolved_memory
                    .active_document_id
                ),
            )

        # Let the structured LLM understand the request FIRST.
        #
        # The older visual fast path ran before the LLM. That could turn
        # "solve the numerical in this image" into DIAGRAM_QUESTION merely
        # because the sentence contained "this image". Numericals that need
        # visual evidence must stay NUMERICAL_PROBLEM with requires_visual=True.
        intent = self._classify_intent(
            query=raw_query,
            memory=query_memory,
            selected_model=selected_model,
        )

        # A context-neutral classification is authoritative for ordinary
        # self-contained Physics questions. The model cannot reattach a stale
        # PDF merely because one exists in Redis.
        if (
            not explicit_session_context
            and intent.intent
            in {
                RequestIntent.PHYSICS_QUESTION,
                RequestIntent.NUMERICAL_PROBLEM,
            }
            and not intent.is_follow_up
            and intent.requires_visual is not True
            and not intent.wants_document_summary
        ):
            intent = intent.model_copy(
                update={
                    "requires_document": False,
                    "document_usage": (
                        DocumentUsage.NONE
                    ),
                }
            )

        # Genuine follow-ups are allowed to regain the full session memory
        # after the neutral first pass.
        if self._intent_is_follow_up(
            intent
        ):
            query_memory = resolved_memory

        # Retain the conservative deterministic visual rescue only when the
        # model could not classify an explicit uploaded-visual reference.
        # It is now a fail-safe, not the primary classifier.
        if intent.intent in {
            RequestIntent.OUT_OF_SCOPE,
            RequestIntent.UNSUPPORTED,
        }:
            document_visual_rescue = (
                self._document_visual_fast_path(
                    query=raw_query,
                    memory=resolved_memory,
                    selected_model=selected_model,
                )
            )

            if document_visual_rescue is not None:
                return document_visual_rescue

        if intent.intent in {
            RequestIntent.UPLOAD_DOCUMENT,
            RequestIntent.VOICE_CONTROL,
        }:
            return QueryUnderstandingResult(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=intent,
                scope=None,
                rewrite=None,
                active_document_id=(
                    query_memory
                    .active_document_id
                ),
            )

        if (
            intent.intent
            == RequestIntent.OUT_OF_SCOPE
        ):
            scope = QueryScopeDecision(
                is_physics=(
                    intent.has_physics_request
                ),
                school_level=False,
                supported=False,
                status=(
                    ScopeStatus.OUT_OF_SCOPE
                ),
                estimated_grade_range=None,
                topics=[],
                confidence=intent.confidence,
                reason=(
                    "Intent classifier marked the "
                    "request outside the supported "
                    "Class 1-12 Physics scope."
                ),
            )

            return QueryUnderstandingResult(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=intent,
                scope=scope,
                rewrite=None,
                active_document_id=(
                    query_memory
                    .active_document_id
                ),
            )

        if (
            intent.intent
            == RequestIntent.UNSUPPORTED
        ):
            scope = QueryScopeDecision(
                is_physics=(
                    intent.has_physics_request
                ),
                school_level=False,
                supported=False,
                status=ScopeStatus.UNCERTAIN,
                estimated_grade_range=None,
                topics=[],
                confidence=intent.confidence,
                reason=(
                    "The request could not be "
                    "classified safely."
                ),
            )

            return QueryUnderstandingResult(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=intent,
                scope=scope,
                rewrite=None,
                active_document_id=(
                    query_memory
                    .active_document_id
                ),
            )

        scope = self._classify_scope(
            query=raw_query,
            intent=intent,
            memory=query_memory,
            selected_model=selected_model,
        )

        scope = self._normalize_scope(
            scope
        )

        if (
            scope.status
            != ScopeStatus.IN_SCOPE
            or not scope.supported
        ):
            return QueryUnderstandingResult(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=intent,
                scope=scope,
                rewrite=None,
                active_document_id=(
                    query_memory
                    .active_document_id
                ),
            )

        rewrite = self._rewrite_query(
            query=raw_query,
            intent=intent,
            scope=scope,
            memory=query_memory,
            selected_model=selected_model,
        )

        return QueryUnderstandingResult(
            raw_query=raw_query,
            normalized_query=normalized_query,
            intent=intent,
            scope=scope,
            rewrite=rewrite,
            active_document_id=(
                resolved_memory
                .active_document_id
            ),
        )

    def _document_visual_fast_path(
        self,
        *,
        query: str,
        memory: MemorySnapshot,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> QueryUnderstandingResult | None:
        """
        Conservative fallback for an explicit reference to an uploaded visual.

        IMPORTANT:
        This is intentionally NOT the primary classifier anymore. The
        structured LLM gets first chance to distinguish, for example:

            "solve the numerical in this image"

        from a request whose main purpose is:

            "explain this image"

        That prevents a visual numerical from being incorrectly converted
        into DIAGRAM_QUESTION.

        This fallback is used only when the model returned OUT_OF_SCOPE or
        UNSUPPORTED and the session already has uploaded-document context.
        Document selection still happens later in ServingNodes.
        """

        has_document_context = bool(
            memory.active_document_id
            or memory.available_documents
        )

        if not has_document_context:
            return None

        normalized = " ".join(
            query.casefold().split()
        )

        if not any(
            phrase in normalized
            for phrase in _DOCUMENT_VISUAL_REFERENCE_PHRASES
        ):
            return None

        normalized_query = self._normalize_query(
            query
        )

        intent = IntentDecision(
            intent=RequestIntent.DIAGRAM_QUESTION,
            confidence=1.0,
            language=self._detect_language(
                query
            ),
            estimated_grade=(
                memory.estimated_grade
            ),
            has_physics_request=True,
            is_follow_up=True,
            prefer_visual=True,
            requires_document=True,
            requires_visual=True,
            document_usage=(
                DocumentUsage.SOURCE_SPECIFIC
            ),
            figure_reference=FigureReference(
                reference_type=(
                    FigureReferenceType.CONTEXTUAL
                ),
                raw_reference=query,
            ),
        )

        grade_range = (
            [
                memory.estimated_grade,
                memory.estimated_grade,
            ]
            if memory.estimated_grade is not None
            else None
        )

        scope = QueryScopeDecision(
            is_physics=True,
            school_level=True,
            supported=True,
            status=ScopeStatus.IN_SCOPE,
            estimated_grade_range=(
                grade_range
            ),
            topics=[],
            confidence=1.0,
            reason=(
                "The query explicitly refers to a visual "
                "inside an uploaded Physics document."
            ),
        )

        rewrite = self._rewrite_query(
            query=query,
            intent=intent,
            scope=scope,
            memory=memory,
            selected_model=selected_model,
        )

        return QueryUnderstandingResult(
            raw_query=query,
            normalized_query=normalized_query,
            intent=intent,
            scope=scope,
            rewrite=rewrite,
            active_document_id=(
                memory.active_document_id
            ),
        )

    def _classify_intent(
        self,
        *,
        query: str,
        memory: MemorySnapshot,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> IntentDecision:
        route = self._route_query_task(
            task=ModelTask.INTENT_CLASSIFICATION,
            selected_model=selected_model,
        )

        try:
            result = (
                self.model_runner
                .generate_structured(
                    route=route,
                    system_prompt=(
                        INTENT_SYSTEM_PROMPT
                    ),
                    user_prompt=(
                        build_intent_user_prompt(
                            query=query,
                            memory=memory,
                            upload_present=False,
                        )
                    ),
                    response_model=(
                        IntentDecision
                    ),
                )
            )

        except Exception as exc:
            raise QueryUnderstandingError(
                "Intent classification failed."
            ) from exc

        return self._normalize_intent(
            IntentDecision.model_validate(
                result
            )
        )

    def _classify_scope(
        self,
        *,
        query: str,
        intent: IntentDecision,
        memory: MemorySnapshot,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> QueryScopeDecision:
        route = self._route_query_task(
            task=ModelTask.QUERY_SCOPE,
            selected_model=selected_model,
        )

        try:
            result = (
                self.model_runner
                .generate_structured(
                    route=route,
                    system_prompt=(
                        QUERY_SCOPE_SYSTEM_PROMPT
                    ),
                    user_prompt=(
                        build_query_scope_user_prompt(
                            query=query,
                            intent=intent,
                            memory=memory,
                        )
                    ),
                    response_model=(
                        QueryScopeDecision
                    ),
                )
            )

        except Exception as exc:
            raise QueryUnderstandingError(
                "Query scope classification failed."
            ) from exc

        return QueryScopeDecision.model_validate(
            result
        )

    def _rewrite_query(
        self,
        *,
        query: str,
        intent: IntentDecision,
        scope: QueryScopeDecision,
        memory: MemorySnapshot,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> QueryRewriteResult:
        if not self._needs_model_rewrite(
            query=query,
            intent=intent,
            memory=memory,
        ):
            return self._identity_rewrite(
                query=query,
                intent=intent,
                memory=memory,
            )

        route = self._route_query_task(
            task=ModelTask.QUERY_REWRITE,
            selected_model=selected_model,
        )

        try:
            result = (
                self.model_runner
                .generate_structured(
                    route=route,
                    system_prompt=(
                        QUERY_REWRITE_SYSTEM_PROMPT
                    ),
                    user_prompt=(
                        build_query_rewrite_user_prompt(
                            query=query,
                            intent=intent,
                            scope=scope,
                            memory=memory,
                        )
                    ),
                    response_model=(
                        QueryRewriteResult
                    ),
                )
            )

        except Exception as exc:
            raise QueryUnderstandingError(
                "Contextual query rewrite failed."
            ) from exc

        rewrite = (
            QueryRewriteResult.model_validate(
                result
            )
        )

        return self._normalize_rewrite(
            rewrite=rewrite,
            query=query,
            intent=intent,
            memory=memory,
        )

    def _route_query_task(
        self,
        *,
        task: ModelTask,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ),
    ) -> ModelRoute:
        """
        Route query-understanding work through the request-selected model
        when one is present.

        The None branch intentionally preserves the exact legacy
        `route_task(task)` call shape. This keeps existing tests and older
        callers that use lightweight router fakes compatible.
        """

        if selected_model is None:
            return self.model_router.route_task(
                task
            )

        return self.model_router.route_task(
            task,
            selected_model=selected_model,
        )

    @classmethod
    def _query_explicitly_uses_session_context(
        cls,
        query: str,
    ) -> bool:
        """
        Detect generic source/follow-up wording without hard-coding Physics
        topics or document names.
        """

        if _DEICTIC_PATTERN.search(
            query
        ):
            return True

        if _BENGALI_DEICTIC_PATTERN.search(
            query
        ):
            return True

        normalized = " ".join(
            query.casefold().split()
        )

        if any(
            phrase in normalized
            for phrase
            in _DOCUMENT_VISUAL_REFERENCE_PHRASES
        ):
            return True

        if _SESSION_CONTEXT_REFERENCE_PATTERN.search(
            query
        ):
            return True

        stripped = re.sub(
            r"[^\w\s]+",
            " ",
            normalized,
        )

        stripped = " ".join(
            stripped.split()
        )

        if stripped in _SHORT_FOLLOW_UP_PHRASES:
            return True

        # Tiny discourse continuations are contextual. Longer complete
        # questions beginning with the same words remain standalone.
        words = stripped.split()

        if (
            len(words) <= 4
            and (
                stripped.startswith(
                    "what about "
                )
                or stripped.startswith(
                    "and "
                )
                or stripped.startswith(
                    "then "
                )
            )
        ):
            return True

        return False

    @staticmethod
    def _without_session_context(
        memory: MemorySnapshot,
    ) -> MemorySnapshot:
        """
        Preserve stable student preferences while removing document and
        conversation anchors that could hijack a standalone question.
        """

        return memory.model_copy(
            update={
                "available_documents": [],
                "active_document_id": None,
                "last_turn_document_id": None,
                "active_page": None,
                "last_selected_figure_id": None,
                "recent_messages": [],
                "problem_solving_state": None,
                "pending_structural_clarification": None,
            }
        )

    @staticmethod
    def _intent_is_follow_up(
        intent: IntentDecision,
    ) -> bool:
        return bool(
            intent.is_follow_up
            or intent.intent
            == RequestIntent.FOLLOW_UP
        )

    def _greeting_fast_path(
        self,
        query: str,
        memory: MemorySnapshot,
    ) -> IntentDecision | None:
        normalized = (
            re.sub(
                r"[^\w\u0980-\u09ff"
                r"\u0900-\u097f\s]+",
                " ",
                query.lower(),
            )
        )

        normalized = " ".join(
            normalized.split()
        )

        if normalized not in _GREETING_PHRASES:
            return None

        return IntentDecision(
            intent=RequestIntent.GREETING,
            confidence=1.0,
            language=self._detect_language(
                query
            ),
            estimated_grade=(
                memory.estimated_grade
            ),
            has_physics_request=False,
            is_follow_up=False,
            prefer_visual=False,
        )

    def _needs_model_rewrite(
        self,
        *,
        query: str,
        intent: IntentDecision,
        memory: MemorySnapshot,
    ) -> bool:
        if intent.is_follow_up:
            return True

        if (
            intent.intent
            == RequestIntent.FOLLOW_UP
        ):
            return True

        if (
            intent.intent
            == RequestIntent.DIAGRAM_QUESTION
        ):
            return True

        if _DEICTIC_PATTERN.search(query):
            return True

        if _BENGALI_DEICTIC_PATTERN.search(
            query
        ):
            return True

        if _COMPLEX_QUERY_PATTERN.search(
            query
        ):
            return True

        if len(query) > 180:
            return True

        if (
            intent.intent
            == RequestIntent.PHYSICS_QUESTION
            and self._is_sparse_conceptual(
                query
            )
        ):
            return True

        if (
            memory.last_selected_figure_id
            and intent.prefer_visual
        ):
            return True

        return False

    def _identity_rewrite(
        self,
        *,
        query: str,
        intent: IntentDecision,
        memory: MemorySnapshot,
    ) -> QueryRewriteResult:
        preferred_pages: list[int] = []

        if (
            intent.prefer_visual
            and memory.active_page
            is not None
        ):
            preferred_pages.append(
                memory.active_page
            )

        referenced_figure_id = (
            memory.last_selected_figure_id
            if intent.prefer_visual
            else None
        )

        return QueryRewriteResult(
            original_query=query,
            rewritten_query=query,
            retrieval_queries=[query],
            was_rewritten=False,
            prefer_visual=(
                intent.prefer_visual
            ),
            preferred_page_numbers=(
                preferred_pages
            ),
            referenced_figure_id=(
                referenced_figure_id
            ),
            use_hyde=False,
            hyde_text=None,
        )

    def _normalize_rewrite(
        self,
        *,
        rewrite: QueryRewriteResult,
        query: str,
        intent: IntentDecision,
        memory: MemorySnapshot,
    ) -> QueryRewriteResult:
        rewritten_query = (
            rewrite.rewritten_query.strip()
            or query
        )

        retrieval_queries: list[str] = []

        for candidate in [
            rewritten_query,
            *rewrite.retrieval_queries,
        ]:
            normalized = candidate.strip()

            if (
                normalized
                and normalized
                not in retrieval_queries
            ):
                retrieval_queries.append(
                    normalized
                )

            if len(
                retrieval_queries
            ) >= 3:
                break

        preferred_pages = list(
            rewrite.preferred_page_numbers
        )

        if (
            memory.active_page
            is not None
            and (
                intent.is_follow_up
                or intent.prefer_visual
                or rewrite.prefer_visual
            )
            and memory.active_page
            not in preferred_pages
        ):
            preferred_pages.insert(
                0,
                memory.active_page,
            )

        preferred_pages = (
            preferred_pages[:5]
        )

        referenced_figure_id = (
            rewrite.referenced_figure_id
        )

        if (
            referenced_figure_id
            is None
            and (
                intent.prefer_visual
                or rewrite.prefer_visual
            )
        ):
            referenced_figure_id = (
                memory.last_selected_figure_id
            )

        prefer_visual = (
            intent.prefer_visual
            or rewrite.prefer_visual
        )

        use_hyde = bool(
            rewrite.use_hyde
            and rewrite.hyde_text
            and intent.intent
            == RequestIntent.PHYSICS_QUESTION
            and not prefer_visual
            and self._is_sparse_conceptual(
                query
            )
        )

        hyde_text = (
            rewrite.hyde_text.strip()
            if (
                use_hyde
                and rewrite.hyde_text
            )
            else None
        )

        return rewrite.model_copy(
            update={
                "original_query": query,
                "rewritten_query": (
                    rewritten_query
                ),
                "retrieval_queries": (
                    retrieval_queries
                ),
                "was_rewritten": (
                    rewritten_query
                    != query
                ),
                "prefer_visual": (
                    prefer_visual
                ),
                "preferred_page_numbers": (
                    preferred_pages
                ),
                "referenced_figure_id": (
                    referenced_figure_id
                ),
                "use_hyde": use_hyde,
                "hyde_text": hyde_text,
            }
        )

    @staticmethod
    def _normalize_intent(
        intent: IntentDecision,
    ) -> IntentDecision:
        """
        Apply only safe consistency rules.

        Semantic interpretation remains the LLM's job. This method does not
        guess requested quantities, document identity, figure identity, or
        Physics meaning from raw keywords.
        """

        updates: dict[str, object] = {}

        if (
            intent.intent
            == RequestIntent.GREETING
        ):
            updates.update(
                {
                    "has_physics_request": False,
                    "is_follow_up": False,
                    "prefer_visual": False,
                    "requires_document": False,
                    "requires_visual": False,
                    "document_usage": (
                        DocumentUsage.NONE
                    ),
                }
            )

        elif intent.intent in {
            RequestIntent.PHYSICS_QUESTION,
            RequestIntent.NUMERICAL_PROBLEM,
            RequestIntent.DIAGRAM_QUESTION,
            RequestIntent.FOLLOW_UP,
        }:
            updates[
                "has_physics_request"
            ] = True

        if (
            intent.intent
            == RequestIntent.DIAGRAM_QUESTION
        ):
            updates[
                "prefer_visual"
            ] = True

        if (
            intent.intent
            == RequestIntent.FOLLOW_UP
        ):
            updates[
                "is_follow_up"
            ] = True

        # Hard consistency only: if the model says visual evidence is
        # required, visual evidence must at least be preferred.
        if intent.requires_visual is True:
            updates[
                "prefer_visual"
            ] = True

        # A document-summary request necessarily uses the document.
        if intent.wants_document_summary:
            updates.update(
                {
                    "requires_document": True,
                    "document_usage": (
                        DocumentUsage.SUMMARY
                    ),
                }
            )

        # Any explicit non-NONE document usage requires a document.
        if (
            intent.document_usage is not None
            and intent.document_usage
            != DocumentUsage.NONE
        ):
            updates[
                "requires_document"
            ] = True

        # If the request is explicitly known not to need a document and the
        # model supplied no usage mode, record the neutral NONE mode.
        if (
            intent.requires_document is False
            and intent.document_usage is None
        ):
            updates[
                "document_usage"
            ] = DocumentUsage.NONE

        if not updates:
            return intent

        return intent.model_copy(
            update=updates
        )

    @staticmethod
    def _normalize_scope(
        scope: QueryScopeDecision,
    ) -> QueryScopeDecision:
        if (
            scope.status
            == ScopeStatus.IN_SCOPE
        ):
            return scope.model_copy(
                update={
                    "is_physics": True,
                    "school_level": True,
                    "supported": True,
                }
            )

        if (
            scope.status
            == ScopeStatus.OUT_OF_SCOPE
        ):
            return scope.model_copy(
                update={
                    "supported": False,
                }
            )

        return scope.model_copy(
            update={
                "supported": False,
            }
        )

    @staticmethod
    def _validate_raw_query(
        query: str,
    ) -> str:
        """
        Validate without rewriting the user's mathematical text.

        We deliberately do not collapse internal whitespace, transliterate
        Unicode, replace mathematical characters, or normalize equations.
        A normalized derivative is created separately by _normalize_query().
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if len(query) > 12000:
            raise ValueError(
                "query exceeds the "
                "maximum supported length."
            )

        return query

    @staticmethod
    def _normalize_query(
        query: str,
    ) -> str:
        normalized = " ".join(
            query.strip().split()
        )

        if not normalized:
            raise ValueError(
                "query cannot be empty."
            )

        if len(normalized) > 12000:
            raise ValueError(
                "query exceeds the "
                "maximum supported length."
            )

        return normalized

    @staticmethod
    def _detect_language(
        text: str,
    ) -> LanguageCode:
        has_bengali = bool(
            re.search(
                r"[\u0980-\u09ff]",
                text,
            )
        )

        has_devanagari = bool(
            re.search(
                r"[\u0900-\u097f]",
                text,
            )
        )

        has_latin = bool(
            re.search(
                r"[A-Za-z]",
                text,
            )
        )

        if has_bengali and has_latin:
            return (
                LanguageCode
                .BENGALI_ENGLISH_MIXED
            )

        if has_bengali:
            return LanguageCode.BENGALI

        if has_devanagari:
            return LanguageCode.HINDI

        if has_latin:
            return LanguageCode.ENGLISH

        return LanguageCode.UNKNOWN

    @staticmethod
    def _is_sparse_conceptual(
        query: str,
    ) -> bool:
        normalized = (
            query.strip()
            .rstrip("?？")
            .strip()
        )

        if (
            not normalized
            or len(normalized) > 50
        ):
            return False

        if any(
            character.isdigit()
            for character in normalized
        ):
            return False

        tokens = re.findall(
            r"[^\W\d_]+",
            normalized,
            flags=re.UNICODE,
        )

        return 1 <= len(tokens) <= 6