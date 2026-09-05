from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from src.agents.tutor_agent import TutorAgent
from src.agents.verifier_agent import VerifierAgent
from src.models.contracts import (
    AnswerType,
    ConversationMessage,
    IntentDecision,
    LanguageCode,
    MemorySnapshot,
    QueryScopeDecision,
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


class ServingServiceError(Exception):
    """Raised when the bounded serving workflow cannot execute safely."""


@dataclass(frozen=True, slots=True)
class ServingResult:
    """
    Internal serving result.

    The future /v1/chat API schema can translate this object into the
    public response contract without coupling the core workflow to FastAPI.
    """

    request_id: str
    user_id: str
    session_id: str | None

    query_understanding: QueryUnderstandingResult

    answer: TutorAnswer
    verification: VerificationResult | None

    context: ContextBundle | None
    retrieval_results: tuple[
        HybridRetrievalResult,
        ...,
    ]

    generation_attempts: int
    retrieval_rounds: int

    terminal_action: VerificationAction | None
    next_memory: MemorySnapshot


class _QueryUnderstandingRunner(Protocol):
    def understand(
        self,
        *,
        query: str,
        memory: MemorySnapshot | None = None,
        upload_present: bool = False,
    ) -> QueryUnderstandingResult:
        ...


class _RetrievalRunner(Protocol):
    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        fused_top_k: int = 30,
        rerank_top_k: int = 8,
        max_contexts: int = 6,
        page_numbers: tuple[int, ...] | None = None,
        content_types: tuple[str, ...] | None = None,
        topics: tuple[str, ...] | None = None,
        grade: int | None = None,
        include_visual: bool = True,
        preferred_page_numbers: tuple[int, ...] | None = None,
        prefer_visual: bool = False,
    ) -> HybridRetrievalResult:
        ...


class _TutorRunner(Protocol):
    def answer(
        self,
        *,
        query: str,
        intent: IntentDecision,
        scope: QueryScopeDecision | None,
        context: ContextBundle | None,
        memory: MemorySnapshot | None = None,
        strict_document_mode: bool = True,
        verifier_feedback: list[str] | None = None,
    ) -> TutorAnswer:
        ...


class _VerifierRunner(Protocol):
    def verify(
        self,
        *,
        query: str,
        intent: IntentDecision,
        scope: QueryScopeDecision | None,
        tutor_answer: TutorAnswer,
        context: ContextBundle | None,
        strict_document_mode: bool = True,
    ) -> VerificationResult:
        ...


class ServingService:
    """
    Core Phase-5 serving orchestrator.

    Bounded workflow:
        query understanding
        -> scope / greeting handling
        -> optional document retrieval
        -> Tutor
        -> Verifier
        -> one controlled retry path when required
        -> final answer

    Important:
    - Chat turns are not limited.
    - Tutor generation is limited to two attempts per user request.
    - Provider retry/fallback remains inside LLMGateway and is a separate budget.
    - Phase-4 RetrievalService is reused; it is not reimplemented here.
    - MemorySnapshot is updated in-memory only. Persistence is handled by the
      memory/session layer when the chat API is wired.
    """

    MAX_GENERATION_ATTEMPTS = 2
    MAX_RETRIEVAL_QUERIES = 3

    _OUT_OF_SCOPE_MESSAGE = (
        "This assistant currently supports school-level Physics for "
        "Classes 1–12. Please upload a Physics page or ask a "
        "school-level Physics question."
    )

    def __init__(
        self,
        *,
        query_service: (
            QueryUnderstandingService
            | _QueryUnderstandingRunner
        ),
        retrieval_service: (
            RetrievalService
            | _RetrievalRunner
        ),
        tutor_agent: TutorAgent | _TutorRunner,
        verifier_agent: (
            VerifierAgent
            | _VerifierRunner
        ),
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

        self.max_merged_context_items = (
            max_merged_context_items
        )
        self.max_merged_context_characters = (
            max_merged_context_characters
        )

    def serve(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str | None = None,
        document_id: str | None = None,
        memory: MemorySnapshot | None = None,
        strict_document_mode: bool | None = None,
        upload_present: bool = False,
        request_id: str | None = None,
    ) -> ServingResult:
        normalized_query = " ".join(
            query.strip().split()
        )

        normalized_user_id = user_id.strip()

        normalized_session_id = (
            session_id.strip()
            if session_id
            else None
        )

        explicit_document_id = (
            document_id.strip()
            if document_id
            else None
        )

        normalized_request_id = (
            request_id.strip()
            if request_id
            else uuid4().hex
        )

        if not normalized_query:
            raise ValueError(
                "query cannot be empty."
            )

        if not normalized_user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        if not normalized_request_id:
            raise ValueError(
                "request_id cannot be empty."
            )

        resolved_memory = (
            memory
            if memory is not None
            else MemorySnapshot()
        )

        # An explicit document selection wins for this request and is also
        # exposed to query understanding for follow-up resolution.
        if explicit_document_id is not None:
            resolved_memory = (
                resolved_memory.model_copy(
                    update={
                        "active_document_id": (
                            explicit_document_id
                        )
                    }
                )
            )

        try:
            understanding = (
                self.query_service.understand(
                    query=normalized_query,
                    memory=resolved_memory,
                    upload_present=upload_present,
                )
            )
        except Exception as exc:
            raise ServingServiceError(
                "Query understanding failed."
            ) from exc

        intent = understanding.intent
        scope = understanding.scope

        active_document_id = (
            explicit_document_id
            or understanding.active_document_id
            or resolved_memory.active_document_id
        )

        # Query understanding decides whether THIS turn needs document
        # grounding. A document merely existing in session memory must not
        # force a general Physics question into RAG/document mode.
        document_mode = (
            intent.requires_document is True
            if strict_document_mode is None
            else bool(strict_document_mode)
        )

        # Keep the session's document memory available for later turns, but
        # expose a document id to retrieval/Tutor only when this turn actually
        # requires document grounding.
        turn_document_id = (
            active_document_id
            if document_mode
            else None
        )

        # ---------------------------------------------------------
        # Fast paths: no retrieval, Tutor, or Verifier.
        # ---------------------------------------------------------

        if intent.intent == RequestIntent.GREETING:
            answer = self._greeting_answer(
                intent=intent
            )

            return ServingResult(
                request_id=normalized_request_id,
                user_id=normalized_user_id,
                session_id=normalized_session_id,
                query_understanding=understanding,
                answer=answer,
                verification=None,
                context=None,
                retrieval_results=(),
                generation_attempts=0,
                retrieval_rounds=0,
                terminal_action=None,
                # Greetings are deliberately not written into memory.
                next_memory=resolved_memory,
            )

        if intent.intent == RequestIntent.UPLOAD_DOCUMENT:
            answer = self._direct_answer(
                (
                    "The document upload should be handled by the "
                    "document upload flow. After it is ready, ask a "
                    "school-level Physics question about it."
                )
            )

            return ServingResult(
                request_id=normalized_request_id,
                user_id=normalized_user_id,
                session_id=normalized_session_id,
                query_understanding=understanding,
                answer=answer,
                verification=None,
                context=None,
                retrieval_results=(),
                generation_attempts=0,
                retrieval_rounds=0,
                terminal_action=None,
                next_memory=resolved_memory,
            )

        if self._is_out_of_scope(
            intent=intent,
            scope=scope,
        ):
            answer = self._direct_answer(
                self._OUT_OF_SCOPE_MESSAGE
            )

            return ServingResult(
                request_id=normalized_request_id,
                user_id=normalized_user_id,
                session_id=normalized_session_id,
                query_understanding=understanding,
                answer=answer,
                verification=None,
                context=None,
                retrieval_results=(),
                generation_attempts=0,
                retrieval_rounds=0,
                terminal_action=(
                    VerificationAction
                    .REJECT_OUT_OF_SCOPE
                ),
                next_memory=resolved_memory,
            )

        if (
            document_mode
            and not turn_document_id
        ):
            answer = self._insufficient_answer(
                (
                    "No active Physics document is selected for "
                    "document-grounded answering."
                )
            )

            return ServingResult(
                request_id=normalized_request_id,
                user_id=normalized_user_id,
                session_id=normalized_session_id,
                query_understanding=understanding,
                answer=answer,
                verification=None,
                context=None,
                retrieval_results=(),
                generation_attempts=0,
                retrieval_rounds=0,
                terminal_action=(
                    VerificationAction
                    .INSUFFICIENT_EVIDENCE
                ),
                next_memory=resolved_memory,
            )

        # ---------------------------------------------------------
        # Retrieval planning.
        # ---------------------------------------------------------

        retrieval_queries = (
            self._build_retrieval_queries(
                understanding=understanding
            )
        )

        preferred_pages = (
            tuple(
                understanding
                .rewrite
                .preferred_page_numbers
            )
            if understanding.rewrite
            else None
        )

        prefer_visual = bool(
            intent.prefer_visual
            or (
                understanding.rewrite
                and understanding
                .rewrite
                .prefer_visual
            )
        )

        grade = (
            intent.estimated_grade
            or resolved_memory.estimated_grade
        )

        retrieval_results: tuple[
            HybridRetrievalResult,
            ...
        ] = ()

        context: ContextBundle | None = None
        retrieval_rounds = 0

        if turn_document_id:
            (
                retrieval_results,
                context,
            ) = self._retrieve_round(
                queries=retrieval_queries,
                user_id=normalized_user_id,
                document_id=turn_document_id,
                grade=grade,
                preferred_pages=preferred_pages,
                prefer_visual=prefer_visual,
                broader=False,
            )

            retrieval_rounds = 1

        # ---------------------------------------------------------
        # Bounded Tutor -> Verifier loop.
        # ---------------------------------------------------------

        generation_attempts = 0
        verifier_feedback: list[str] = []

        last_answer: TutorAnswer | None = None
        last_verification: (
            VerificationResult | None
        ) = None

        while (
            generation_attempts
            < self.MAX_GENERATION_ATTEMPTS
        ):
            generation_attempts += 1

            try:
                last_answer = (
                    self.tutor_agent.answer(
                        query=normalized_query,
                        intent=intent,
                        scope=scope,
                        context=context,
                        memory=resolved_memory,
                        strict_document_mode=(
                            document_mode
                        ),
                        verifier_feedback=(
                            verifier_feedback
                            or None
                        ),
                    )
                )
            except Exception as exc:
                raise ServingServiceError(
                    "Tutor generation failed."
                ) from exc

            try:
                last_verification = (
                    self.verifier_agent.verify(
                        query=normalized_query,
                        intent=intent,
                        scope=scope,
                        tutor_answer=last_answer,
                        context=context,
                        strict_document_mode=(
                            document_mode
                        ),
                    )
                )
            except Exception as exc:
                raise ServingServiceError(
                    "Verifier audit failed."
                ) from exc

            action = (
                last_verification.action
            )

            if action == VerificationAction.PASS:
                return self._final_result(
                    request_id=(
                        normalized_request_id
                    ),
                    user_id=normalized_user_id,
                    session_id=(
                        normalized_session_id
                    ),
                    understanding=understanding,
                    answer=last_answer,
                    verification=(
                        last_verification
                    ),
                    context=context,
                    retrieval_results=(
                        retrieval_results
                    ),
                    generation_attempts=(
                        generation_attempts
                    ),
                    retrieval_rounds=(
                        retrieval_rounds
                    ),
                    terminal_action=(
                        VerificationAction.PASS
                    ),
                    previous_memory=(
                        resolved_memory
                    ),
                    active_document_id=(
                        turn_document_id
                    ),
                )

            if action in {
                VerificationAction
                .ASK_FOR_CLEARER_IMAGE,
                VerificationAction
                .INSUFFICIENT_EVIDENCE,
                VerificationAction
                .REJECT_OUT_OF_SCOPE,
            }:
                terminal_answer = last_answer

                if (
                    action
                    == VerificationAction
                    .REJECT_OUT_OF_SCOPE
                ):
                    terminal_answer = (
                        self._direct_answer(
                            self._OUT_OF_SCOPE_MESSAGE
                        )
                    )

                elif (
                    last_answer.answer_type
                    != AnswerType.INSUFFICIENT_EVIDENCE
                ):
                    terminal_answer = (
                        self._not_enough_verified_answer(
                            answer=last_answer,
                            verification=(
                                last_verification
                            ),
                        )
                    )

                return self._final_result(
                    request_id=(
                        normalized_request_id
                    ),
                    user_id=normalized_user_id,
                    session_id=(
                        normalized_session_id
                    ),
                    understanding=understanding,
                    answer=terminal_answer,
                    verification=(
                        last_verification
                    ),
                    context=context,
                    retrieval_results=(
                        retrieval_results
                    ),
                    generation_attempts=(
                        generation_attempts
                    ),
                    retrieval_rounds=(
                        retrieval_rounds
                    ),
                    terminal_action=action,
                    previous_memory=(
                        resolved_memory
                    ),
                    active_document_id=(
                        turn_document_id
                    ),
                )

            # There is no third Tutor generation.
            if (
                generation_attempts
                >= self.MAX_GENERATION_ATTEMPTS
            ):
                break

            verifier_feedback = (
                self._verifier_feedback(
                    verification=(
                        last_verification
                    )
                )
            )

            if (
                action
                == VerificationAction
                .RETRY_RETRIEVAL
            ):
                if not turn_document_id:
                    break

                try:
                    (
                        retrieval_results,
                        context,
                    ) = self._retrieve_round(
                        queries=(
                            retrieval_queries
                        ),
                        user_id=(
                            normalized_user_id
                        ),
                        document_id=(
                            turn_document_id
                        ),
                        grade=grade,
                        preferred_pages=(
                            preferred_pages
                        ),
                        prefer_visual=(
                            prefer_visual
                        ),
                        broader=True,
                    )
                except Exception as exc:
                    raise ServingServiceError(
                        "Broader retrieval failed."
                    ) from exc

                retrieval_rounds += 1
                continue

            if (
                action
                == VerificationAction
                .REGENERATE
            ):
                # Same evidence, but Tutor receives the concrete verifier
                # issues and gets one final generation attempt.
                continue

            # Unknown/unexpected control outcome: fail closed.
            break

        # ---------------------------------------------------------
        # Retry budget exhausted.
        # ---------------------------------------------------------

        terminal_verification = (
            self._exhausted_verification(
                last_verification
            )
        )

        if (
            last_answer is not None
            and last_answer.answer_type
            != AnswerType.INSUFFICIENT_EVIDENCE
        ):
            terminal_answer = (
                self._not_enough_verified_answer(
                    answer=last_answer,
                    verification=(
                        terminal_verification
                    ),
                )
            )
        elif last_answer is not None:
            terminal_answer = last_answer
        else:
            terminal_answer = (
                self._insufficient_answer(
                    (
                        "I could not fully verify an answer "
                        "within the safe retry limit."
                    )
                )
            )

        return self._final_result(
            request_id=normalized_request_id,
            user_id=normalized_user_id,
            session_id=normalized_session_id,
            understanding=understanding,
            answer=terminal_answer,
            verification=(
                terminal_verification
            ),
            context=context,
            retrieval_results=(
                retrieval_results
            ),
            generation_attempts=(
                generation_attempts
            ),
            retrieval_rounds=(
                retrieval_rounds
            ),
            terminal_action=(
                VerificationAction
                .INSUFFICIENT_EVIDENCE
            ),
            previous_memory=resolved_memory,
            active_document_id=(
                turn_document_id
            ),
        )

    def _retrieve_round(
        self,
        *,
        queries: tuple[str, ...],
        user_id: str,
        document_id: str,
        grade: int | None,
        preferred_pages: tuple[int, ...] | None,
        prefer_visual: bool,
        broader: bool,
    ) -> tuple[
        tuple[HybridRetrievalResult, ...],
        ContextBundle,
    ]:
        """
        Run one retrieval round.

        Complex questions may have up to three retrieval queries.
        Each query still uses the unchanged Phase-4 hybrid pipeline.
        Contexts are then merged deterministically and bounded.
        """

        dense_top_k = (
            30 if broader else 20
        )
        bm25_top_k = (
            30 if broader else 20
        )
        fused_top_k = (
            40 if broader else 30
        )

        # The PDF's final reranked context target remains Top 5-8.
        rerank_top_k = 8
        max_contexts = (
            8 if broader else 6
        )

        results: list[
            HybridRetrievalResult
        ] = []

        for retrieval_query in (
            queries[
                :self.MAX_RETRIEVAL_QUERIES
            ]
        ):
            try:
                result = (
                    self.retrieval_service
                    .retrieve(
                        query=retrieval_query,
                        user_id=user_id,
                        document_id=document_id,
                        dense_top_k=dense_top_k,
                        bm25_top_k=bm25_top_k,
                        fused_top_k=fused_top_k,
                        rerank_top_k=(
                            rerank_top_k
                        ),
                        max_contexts=(
                            max_contexts
                        ),
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
                    )
                )
            except Exception as exc:
                raise ServingServiceError(
                    (
                        "Document retrieval failed "
                        f"for query: {retrieval_query}"
                    )
                ) from exc

            results.append(result)

        merged_context = (
            self._merge_contexts(
                query=queries[0],
                user_id=user_id,
                document_id=document_id,
                results=results,
            )
        )

        return (
            tuple(results),
            merged_context,
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
                # Defence in depth. Phase-4 already enforces this,
                # but serving never merges cross-user/doc evidence.
                if (
                    item.user_id != user_id
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
        merged_truncated = any_truncated

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

            # If the very first item is unexpectedly larger than the
            # serving budget, keep only a bounded copy of its text.
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

    def _build_retrieval_queries(
        self,
        *,
        understanding: QueryUnderstandingResult,
    ) -> tuple[str, ...]:
        candidates: list[str] = []

        if understanding.rewrite:
            candidates.extend(
                understanding
                .rewrite
                .retrieval_queries
            )

            if (
                understanding.rewrite.use_hyde
                and understanding
                .rewrite
                .hyde_text
            ):
                candidates.append(
                    understanding
                    .rewrite
                    .hyde_text
                )

        if not candidates:
            candidates.append(
                understanding.normalized_query
            )

        cleaned: list[str] = []

        for candidate in candidates:
            normalized = " ".join(
                candidate.strip().split()
            )

            if (
                normalized
                and normalized
                not in cleaned
            ):
                cleaned.append(
                    normalized
                )

            if (
                len(cleaned)
                >= self
                .MAX_RETRIEVAL_QUERIES
            ):
                break

        if not cleaned:
            cleaned.append(
                understanding.normalized_query
            )

        return tuple(cleaned)

    @staticmethod
    def _is_out_of_scope(
        *,
        intent: IntentDecision,
        scope: QueryScopeDecision | None,
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
    def _verifier_feedback(
        *,
        verification: VerificationResult,
    ) -> list[str]:
        feedback = list(
            verification.issues
        )

        action_note = (
            "Verifier action: "
            f"{verification.action.value}."
        )

        if action_note not in feedback:
            feedback.append(
                action_note
            )

        return feedback[:20]

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
    def _not_enough_verified_answer(
        *,
        answer: TutorAnswer,
        verification: VerificationResult,
    ) -> TutorAnswer:
        """
        Preserve the best available Tutor draft while clearly marking that it
        did not fully pass verification.

        This legacy/core ServingService mirrors the LangGraph serving policy so
        both entry paths behave consistently.
        """

        issues = [
            item.strip()
            for item in verification.issues[:3]
            if isinstance(
                item,
                str,
            )
            and item.strip()
        ]

        parts = [
            "Not enough verified",
            (
                "The answer below is the best available draft, "
                "but it did not fully pass verification."
            ),
        ]

        if issues:
            parts.append(
                "Verification note: "
                + "; ".join(
                    issues
                )
            )

        if answer.direct_answer.strip():
            parts.append(
                answer.direct_answer.strip()
            )

        updates: dict[str, object] = {
            "answer_type": (
                AnswerType
                .INSUFFICIENT_EVIDENCE
            ),
            "direct_answer": (
                "\n\n".join(parts)
            )[:12000],
        }

        # Never expose source references that the verifier already rejected.
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

    def _final_result(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str | None,
        understanding: QueryUnderstandingResult,
        answer: TutorAnswer,
        verification: VerificationResult | None,
        context: ContextBundle | None,
        retrieval_results: tuple[
            HybridRetrievalResult,
            ...,
        ],
        generation_attempts: int,
        retrieval_rounds: int,
        terminal_action: (
            VerificationAction | None
        ),
        previous_memory: MemorySnapshot,
        active_document_id: str | None,
    ) -> ServingResult:
        next_memory = self._next_memory(
            query=(
                understanding
                .normalized_query
            ),
            intent=understanding.intent,
            answer=answer,
            previous_memory=(
                previous_memory
            ),
            active_document_id=(
                active_document_id
            ),
        )

        return ServingResult(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            query_understanding=(
                understanding
            ),
            answer=answer,
            verification=verification,
            context=context,
            retrieval_results=(
                retrieval_results
            ),
            generation_attempts=(
                generation_attempts
            ),
            retrieval_rounds=(
                retrieval_rounds
            ),
            terminal_action=(
                terminal_action
            ),
            next_memory=next_memory,
        )

    def _next_memory(
        self,
        *,
        query: str,
        intent: IntentDecision,
        answer: TutorAnswer,
        previous_memory: MemorySnapshot,
        active_document_id: str | None,
    ) -> MemorySnapshot:
        # The design explicitly avoids permanently saving greetings.
        if intent.intent == RequestIntent.GREETING:
            return previous_memory

        recent = list(
            previous_memory.recent_messages
        )

        recent.append(
            ConversationMessage(
                role="user",
                content=query[
                    :12000
                ],
            )
        )

        assistant_text = (
            self._answer_for_memory(
                answer
            )
        )

        recent.append(
            ConversationMessage(
                role="assistant",
                content=assistant_text,
            )
        )

        recent = recent[-10:]

        active_page = (
            answer.source_pages[0]
            if answer.source_pages
            else previous_memory.active_page
        )

        selected_figure = (
            self._first_figure_id(
                answer
            )
            or previous_memory
            .last_selected_figure_id
        )

        language = (
            intent.language
            if intent.language
            != LanguageCode.UNKNOWN
            else previous_memory.language
        )

        estimated_grade = (
            intent.estimated_grade
            or previous_memory
            .estimated_grade
        )

        return previous_memory.model_copy(
            update={
                "active_document_id": (
                    active_document_id
                    or previous_memory
                    .active_document_id
                ),
                "last_turn_document_id": (
                    active_document_id
                ),
                "active_page": (
                    active_page
                ),
                "last_selected_figure_id": (
                    selected_figure
                ),
                "recent_messages": recent,
                "language": language,
                "estimated_grade": (
                    estimated_grade
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
                return citation.figure_id

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

        if answer.problem_statement:
            parts.append(
                (
                    "Problem statement: "
                    f"{answer.problem_statement}"
                )
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
            if part and part.strip()
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
        if (
            intent.language
            == LanguageCode.BENGALI
        ):
            text = (
                "হ্যালো! স্কুল-লেভেল Physics "
                "নিয়ে কী জানতে চাও?"
            )
        elif (
            intent.language
            == LanguageCode.HINDI
        ):
            text = (
                "नमस्ते! स्कूल-लेवल Physics "
                "के बारे में क्या पूछना है?"
            )
        else:
            text = (
                "Hello! What school-level Physics "
                "question can I help you with?"
            )

        return ServingService._direct_answer(
            text
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