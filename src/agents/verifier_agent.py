from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

from src.models.contracts import (
    AnswerType,
    IntentDecision,
    QueryScopeDecision,
    RequestIntent,
    TutorAnswer,
    VerificationAction,
    VerificationResult,
)
from src.models.gateway import LLMGateway
from src.models.routing import (
    ModelRoute,
    ModelRouter,
    UserSelectableModel,
)
from src.prompts.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    build_verifier_user_prompt,
)
from src.retrieval.models import ContextBundle
from src.verification.numerical import (
    DeterministicNumericalVerifier,
    NumericalVerificationReport,
)


if TYPE_CHECKING:
    from src.retrieval.structural_resolver import (
        AnswerScopeContract,
    )


class VerifierAgentError(Exception):
    """Raised when the Verifier Agent cannot complete a safe audit."""


class VerifierAgent:
    """
    Physics answer-audit agent.

    Responsibilities:
    - audit a TutorAnswer against retrieved evidence;
    - distinguish general answers from document-grounded answers;
    - run deterministic numerical checks before the LLM audit;
    - validate citations defensively;
    - provide real visual evidence whenever visual claims require it;
    - route general verification to the general verifier model;
    - route document verification to the document verifier model;
    - return only VerificationResult.

    It does NOT:
    - retrieve;
    - rewrite the student's question;
    - generate a replacement TutorAnswer;
    - run the Tutor retry loop.
    """

    _SUPPORTED_IMAGE_MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    def __init__(
        self,
        *,
        model_gateway: LLMGateway,
        model_router: ModelRouter,
        numerical_verifier: (
            DeterministicNumericalVerifier | None
        ) = None,
        max_visual_images: int = 4,
        max_image_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if max_visual_images <= 0:
            raise ValueError(
                "max_visual_images must be positive."
            )

        if max_image_bytes <= 0:
            raise ValueError(
                "max_image_bytes must be positive."
            )

        self.model_gateway = model_gateway
        self.model_router = model_router
        self.numerical_verifier = (
            numerical_verifier
            or DeterministicNumericalVerifier()
        )
        self.max_visual_images = max_visual_images
        self.max_image_bytes = max_image_bytes

    def verify(
        self,
        *,
        query: str,
        intent: IntentDecision,
        scope: QueryScopeDecision | None,
        tutor_answer: TutorAnswer,
        context: ContextBundle | None,
        strict_document_mode: bool = True,
        structural_answer_scope: (
            AnswerScopeContract | None
        ) = None,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> VerificationResult:
        raw_query = query

        if not raw_query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Defence in depth. Serving/scope guard should normally
        # stop this before Tutor generation, but Verifier must
        # never PASS a clearly unsupported request.
        if self._is_out_of_scope(
            intent=intent,
            scope=scope,
        ):
            return VerificationResult(
                grounded=False,
                physics_correct=False,
                calculation_correct=False,
                units_correct=False,
                diagram_claims_supported=False,
                within_school_scope=False,
                citation_valid=(
                    self._citations_are_valid(
                        tutor_answer=tutor_answer,
                        context=context,
                        strict_document_mode=(
                            strict_document_mode
                        ),
                    )
                ),
                issues=[
                    (
                        "The request is outside the supported "
                        "school-level Physics scope."
                    )
                ],
                action=(
                    VerificationAction
                    .REJECT_OUT_OF_SCOPE
                ),
                confidence=1.0,
            )

        citation_valid = (
            self._citations_are_valid(
                tutor_answer=tutor_answer,
                context=context,
                strict_document_mode=(
                    strict_document_mode
                ),
            )
        )

        numerical_report = (
            self.numerical_verifier.verify(
                intent=intent,
                tutor_answer=tutor_answer,
            )
        )

        visual_required = (
            self._requires_visual_evidence(
                intent=intent,
                tutor_answer=tutor_answer,
            )
        )

        image_urls = (
            self._collect_visual_evidence(
                context=context
            )
            if visual_required
            else ()
        )

        # A correct refusal caused by genuinely missing document
        # evidence should not require an extra model call.
        if (
            tutor_answer.answer_type
            == AnswerType.INSUFFICIENT_EVIDENCE
            and strict_document_mode
            and not self._has_evidence(context)
        ):
            return VerificationResult(
                grounded=True,
                physics_correct=True,
                calculation_correct=True,
                units_correct=True,
                diagram_claims_supported=(
                    not visual_required
                ),
                within_school_scope=True,
                citation_valid=citation_valid,
                issues=[],
                action=VerificationAction.PASS,
                confidence=1.0,
            )

        # Any request that structurally requires a visual must be audited
        # against the real bounded image evidence. Text/caption alone is not
        # enough, and the Verifier must not infer source-specific visual facts.
        if visual_required and not image_urls:
            return VerificationResult(
                grounded=False,
                physics_correct=True,
                calculation_correct=(
                    numerical_report
                    .calculation_passed
                    is not False
                ),
                units_correct=(
                    numerical_report
                    .units_passed
                    is not False
                ),
                diagram_claims_supported=False,
                within_school_scope=True,
                citation_valid=citation_valid,
                issues=[
                    (
                        "The answer requires visual evidence, "
                        "but no readable resolved image is available."
                    )
                ],
                action=(
                    self._missing_visual_action(
                        context=context
                    )
                ),
                confidence=1.0,
            )

        requires_document = (
            getattr(
                intent,
                "requires_document",
                None,
            )
            is True
        )

        # During staged rollout, strict document mode is a safe fallback only
        # when query understanding did not explicitly decide the requirement.
        if (
            getattr(
                intent,
                "requires_document",
                None,
            )
            is None
            and strict_document_mode
        ):
            requires_document = True

        route = self.model_router.route_verifier(
            requires_document=(
                requires_document
            ),
            visual_context_available=(
                bool(image_urls)
            ),
            selected_model=selected_model,
        )

        routed_images = (
            image_urls
            if route.requires_vision
            else ()
        )

        system_prompt = (
            self._system_prompt_with_structural_scope(
                answer_scope=(
                    structural_answer_scope
                )
            )
        )

        try:
            result = (
                self.model_gateway
                .generate_structured(
                    route=route,
                    system_prompt=(
                        system_prompt
                    ),
                    user_prompt=(
                        build_verifier_user_prompt(
                            query=raw_query,
                            intent=intent,
                            scope=scope,
                            tutor_answer=tutor_answer,
                            context=context,
                            strict_document_mode=(
                                strict_document_mode
                            ),
                            deterministic_numerical_checks=(
                                numerical_report
                                .as_prompt_payload()
                            ),
                        )
                    ),
                    response_model=(
                        VerificationResult
                    ),
                    image_urls=routed_images,
                )
            )

        except Exception as exc:
            raise VerifierAgentError(
                "Verifier model audit failed."
            ) from exc

        verification = (
            VerificationResult
            .model_validate(result)
        )

        structural_scope_violation = (
            structural_answer_scope
            is not None
            and self._reports_structural_scope_violation(
                verification
            )
        )

        return self._apply_deterministic_policy(
            verification=verification,
            tutor_answer=tutor_answer,
            context=context,
            numerical_report=(
                numerical_report
            ),
            citation_valid=citation_valid,
            visual_required=visual_required,
            visual_evidence_present=(
                bool(image_urls)
            ),
            strict_document_mode=(
                strict_document_mode
            ),
            structural_scope_violation=(
                structural_scope_violation
            ),
        )

    @staticmethod
    def _system_prompt_with_structural_scope(
        *,
        answer_scope: AnswerScopeContract | None,
    ) -> str:
        """Add an independent structural-boundary audit when required."""

        if answer_scope is None:
            return VERIFIER_SYSTEM_PROMPT

        requested_action_value = getattr(
            answer_scope.requested_action,
            "value",
            answer_scope.requested_action,
        )
        requested_action = " ".join(
            str(requested_action_value).split()
        )[:100]

        allowed_target_ids = list(
            dict.fromkeys(
                " ".join(
                    str(node_id).split()
                )[:300]
                for node_id in (
                    answer_scope
                    .allowed_target_node_ids
                )
                if str(node_id).strip()
            )
        )[:20]

        if not allowed_target_ids:
            raise VerifierAgentError(
                "Structural answer scope has no allowed target IDs."
            )

        additional_rules = list(
            dict.fromkeys(
                " ".join(
                    str(rule).split()
                )[:1000]
                for rule in answer_scope.scope_rules
                if str(rule).strip()
            )
        )[:20]

        contract_lines = [
            "STRUCTURAL ANSWER-SCOPE AUDIT",
            (
                "Independently verify that the Tutor answered only the "
                "resolver-approved document target or targets."
            ),
            f"Requested action: {requested_action or 'answer'}",
            (
                "Allowed structural target IDs: "
                + ", ".join(allowed_target_ids)
            ),
            "Audit requirements:",
            (
                "- Treat retrieved context as evidence for the resolved "
                "target only, not permission to answer neighboring content."
            ),
            (
                "- Fail the audit if the Tutor unnecessarily includes a "
                "neighboring bullet, question, example, solution, equation, "
                "figure, section, or a broader lesson."
            ),
            (
                "- Fail the audit if the Tutor performs an action broader "
                "than the requested action."
            ),
            (
                "- On any structural-scope breach, set grounded=false, set "
                "action=REGENERATE, and include an issue beginning exactly "
                "with STRUCTURAL_SCOPE_VIOLATION:."
            ),
            (
                "- If the answer stays within the target, do not emit the "
                "STRUCTURAL_SCOPE_VIOLATION marker."
            ),
        ]

        if additional_rules:
            contract_lines.append(
                "Resolver scope rules:"
            )
            contract_lines.extend(
                f"- {rule}"
                for rule in additional_rules
            )

        return (
            VERIFIER_SYSTEM_PROMPT
            + "\n\n"
            + "\n".join(contract_lines)
        )

    @staticmethod
    def _reports_structural_scope_violation(
        verification: VerificationResult,
    ) -> bool:
        marker = "STRUCTURAL_SCOPE_VIOLATION:"

        return any(
            isinstance(issue, str)
            and issue.strip().startswith(marker)
            for issue in verification.issues
        )

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
    def _has_evidence(
        context: ContextBundle | None,
    ) -> bool:
        if context is None:
            return False

        return any(
            bool(item.text.strip())
            or bool(
                item.image_path
                and item.image_path.strip()
            )
            for item in context.items
        )

    @staticmethod
    def _citations_are_valid(
        *,
        tutor_answer: TutorAnswer,
        context: ContextBundle | None,
        strict_document_mode: bool,
    ) -> bool:
        # A general-Physics answer must not cite an uploaded document merely
        # because retrieval/session context happens to exist.
        if (
            not strict_document_mode
            and (
                tutor_answer.source_pages
                or tutor_answer.citations
            )
        ):
            return False

        items = (
            context.items
            if context is not None
            else []
        )

        allowed_pages = {
            item.page_number
            for item in items
        }

        allowed_chunk_ids = {
            chunk_id
            for item in items
            for chunk_id
            in item.source_chunk_ids
        }

        allowed_figure_ids = {
            figure_id
            for item in items
            for figure_id
            in item.linked_figure_ids
        }

        # No citation is valid against no evidence.
        if (
            tutor_answer.source_pages
            or tutor_answer.citations
        ) and not items:
            return False

        if any(
            page not in allowed_pages
            for page
            in tutor_answer.source_pages
        ):
            return False

        for citation in (
            tutor_answer.citations
        ):
            if (
                citation.page_number
                not in allowed_pages
            ):
                return False

            if any(
                chunk_id
                not in allowed_chunk_ids
                for chunk_id
                in citation.source_chunk_ids
            ):
                return False

            if (
                citation.figure_id
                is not None
                and citation.figure_id
                not in allowed_figure_ids
            ):
                return False

        return True

    @staticmethod
    def _requires_visual_evidence(
        *,
        intent: IntentDecision,
        tutor_answer: TutorAnswer,
    ) -> bool:
        """
        Decide whether real visual evidence is mandatory for this audit.

        Structured Query Understanding is authoritative when it explicitly
        requires a visual. Diagram intent/answer type remain conservative
        fallbacks. No Physics topic, figure number, document name, or semantic
        phrase is hard-coded here.
        """

        if (
            getattr(
                intent,
                "requires_visual",
                None,
            )
            is True
        ):
            return True

        has_explicit_diagram_output = bool(
            tutor_answer.diagram_explanation
            and tutor_answer.diagram_explanation.strip()
        )

        cites_figure = any(
            citation.figure_id is not None
            for citation in tutor_answer.citations
        )

        return (
            intent.intent
            == RequestIntent.DIAGRAM_QUESTION
            or tutor_answer.answer_type
            == AnswerType.DIAGRAM_EXPLANATION
            or has_explicit_diagram_output
            or cites_figure
        )

    @staticmethod
    def _missing_visual_action(
        *,
        context: ContextBundle | None,
    ) -> VerificationAction:
        """
        Choose the established fail-closed action for missing visuals.

        If the request requires a diagram but no readable resolved image is
        available, another text-retrieval round cannot verify visual claims.
        Ask for clearer visual evidence instead of repeatedly broadening text
        retrieval. The context argument remains for API compatibility.
        """

        del context

        return (
            VerificationAction
            .ASK_FOR_CLEARER_IMAGE
        )

    def _collect_visual_evidence(
        self,
        *,
        context: ContextBundle | None,
    ) -> tuple[str, ...]:
        if context is None:
            return ()

        image_urls: list[str] = []
        seen_paths: set[str] = set()

        for item in context.items:
            if (
                len(image_urls)
                >= self.max_visual_images
            ):
                break

            raw_path = (
                item.image_path.strip()
                if item.image_path
                else ""
            )

            if not raw_path:
                continue

            path = Path(raw_path)

            try:
                resolved = path.resolve()
            except OSError:
                continue

            key = str(resolved)

            if key in seen_paths:
                continue

            seen_paths.add(key)

            if not resolved.is_file():
                continue

            mime_type = (
                self
                ._SUPPORTED_IMAGE_MIME_TYPES
                .get(
                    resolved.suffix.lower()
                )
            )

            if mime_type is None:
                continue

            try:
                size = (
                    resolved.stat().st_size
                )
            except OSError:
                continue

            if (
                size <= 0
                or size
                > self.max_image_bytes
            ):
                continue

            try:
                encoded = (
                    base64.b64encode(
                        resolved.read_bytes()
                    ).decode("ascii")
                )
            except OSError:
                continue

            image_urls.append(
                (
                    f"data:{mime_type};"
                    f"base64,{encoded}"
                )
            )

        return tuple(image_urls)

    def _apply_deterministic_policy(
        self,
        *,
        verification: VerificationResult,
        tutor_answer: TutorAnswer,
        context: ContextBundle | None,
        numerical_report: NumericalVerificationReport,
        citation_valid: bool,
        visual_required: bool,
        visual_evidence_present: bool,
        strict_document_mode: bool,
        structural_scope_violation: bool = False,
    ) -> VerificationResult:
        updates: dict[str, object] = {}
        issues = list(
            verification.issues
        )

        def add_issue(
            message: str,
        ) -> None:
            if (
                message
                and message not in issues
                and len(issues) < 20
            ):
                issues.append(message)

        # Deterministic citation validation wins over a model PASS.
        if not citation_valid:
            updates[
                "citation_valid"
            ] = False

            add_issue(
                (
                    "Tutor citations contain a page, "
                    "chunk, or figure reference not "
                    "present in retrieved context."
                )
            )

        # Deterministic arithmetic/unit checks are authoritative
        # for the exact facts they successfully parsed.
        if (
            numerical_report
            .calculation_passed
            is False
        ):
            updates[
                "calculation_correct"
            ] = False

        if (
            numerical_report
            .units_passed
            is False
        ):
            updates[
                "units_correct"
            ] = False

        if (
            numerical_report
            .dimensional_consistency_passed
            is False
        ):
            updates[
                "units_correct"
            ] = False

        for numerical_issue in (
            numerical_report.issues
        ):
            add_issue(
                numerical_issue
            )

        if structural_scope_violation:
            # The Verifier identified a generation-boundary failure. Make the
            # grounded flag consistent and ensure normalization cannot turn
            # the requested regeneration into PASS or a retrieval retry.
            updates[
                "grounded"
            ] = False

        if (
            visual_required
            and not visual_evidence_present
        ):
            updates[
                "diagram_claims_supported"
            ] = False

            add_issue(
                (
                    "Required visual evidence "
                    "is unavailable."
                )
            )

        candidate = (
            verification.model_copy(
                update={
                    **updates,
                    "issues": issues,
                }
            )
        )

        normalized_action = (
            self._normalized_action(
                verification=candidate,
                tutor_answer=tutor_answer,
                context=context,
                numerical_report=(
                    numerical_report
                ),
                visual_required=(
                    visual_required
                ),
                visual_evidence_present=(
                    visual_evidence_present
                ),
                strict_document_mode=(
                    strict_document_mode
                ),
                structural_scope_violation=(
                    structural_scope_violation
                ),
            )
        )

        if (
            normalized_action
            != candidate.action
        ):
            candidate = (
                candidate.model_copy(
                    update={
                        "action": (
                            normalized_action
                        )
                    }
                )
            )

        return VerificationResult.model_validate(
            candidate.model_dump()
        )

    @staticmethod
    def _normalized_action(
        *,
        verification: VerificationResult,
        tutor_answer: TutorAnswer,
        context: ContextBundle | None,
        numerical_report: NumericalVerificationReport,
        visual_required: bool,
        visual_evidence_present: bool,
        strict_document_mode: bool,
        structural_scope_violation: bool = False,
    ) -> VerificationAction:
        # Strong deterministic/scope outcomes first.
        if not verification.within_school_scope:
            return (
                VerificationAction
                .REJECT_OUT_OF_SCOPE
            )

        if structural_scope_violation:
            return (
                VerificationAction
                .REGENERATE
            )

        if (
            visual_required
            and (
                not visual_evidence_present
                or not (
                    verification
                    .diagram_claims_supported
                )
            )
        ):
            # If the model specifically judged an answer generation
            # problem with readable visual evidence, preserve REGENERATE.
            if (
                visual_evidence_present
                and verification.action
                == VerificationAction.REGENERATE
            ):
                return (
                    VerificationAction
                    .REGENERATE
                )

            return (
                VerifierAgent
                ._missing_visual_action(
                    context=context
                )
            )

        deterministic_numerical_failure = (
            numerical_report
            .calculation_passed
            is False
            or numerical_report
            .units_passed
            is False
            or numerical_report
            .dimensional_consistency_passed
            is False
        )

        if (
            deterministic_numerical_failure
            or not verification.physics_correct
            or not verification.calculation_correct
            or not verification.units_correct
            or not verification.citation_valid
        ):
            return (
                VerificationAction
                .REGENERATE
            )

        if not verification.grounded:
            # In strict document mode, lack of grounding is an
            # evidence/retrieval problem.
            if strict_document_mode:
                has_context = bool(
                    context
                    and context.items
                )

                if has_context:
                    return (
                        VerificationAction
                        .RETRY_RETRIEVAL
                    )

                return (
                    VerificationAction
                    .INSUFFICIENT_EVIDENCE
                )

            # In general Physics mode, document evidence is not
            # required. If the Verifier still considers the Tutor
            # answer ungrounded, treat that as a generation-quality
            # problem rather than pretending retrieval is needed.
            return (
                VerificationAction
                .REGENERATE
            )

        # A correct insufficient-evidence answer can be returned
        # when the audit itself PASSes it.
        if (
            tutor_answer.answer_type
            == AnswerType.INSUFFICIENT_EVIDENCE
            and verification.action
            == VerificationAction.PASS
        ):
            return VerificationAction.PASS

        # PASS is allowed only after every relevant required
        # invariant is true.
        if (
            verification.grounded
            and verification.physics_correct
            and verification.calculation_correct
            and verification.units_correct
            and verification.within_school_scope
            and verification.citation_valid
            and (
                not visual_required
                or (
                    verification
                    .diagram_claims_supported
                )
            )
        ):
            return VerificationAction.PASS

        # Preserve a safe non-PASS model action when no deterministic
        # rule above requires a more specific action.
        if (
            verification.action
            != VerificationAction.PASS
        ):
            return verification.action

        return (
            VerificationAction.REGENERATE
        )
