from __future__ import annotations

from dataclasses import (
    dataclass,
    replace,
)
from enum import Enum

from src.models.contracts import (
    ModelTask,
    RequestIntent,
)


class UserSelectableModel(str, Enum):
    """
    Models that a PhyMentor user is allowed to select.

    This enum is the one central backend allowlist. It prevents a client
    from sending an arbitrary provider model name.
    """

    GPT_4O = "gpt-4o"
    GPT_5_6_SOL = "gpt-5.6-sol"
    GPT_5_6_TERRA = "gpt-5.6-terra"
    GPT_5_6_LUNA = "gpt-5.6-luna"


SUPPORTED_USER_MODEL_NAMES = frozenset(
    model.value
    for model in UserSelectableModel
)


@dataclass(
    frozen=True,
    slots=True,
)
class ModelRoute:
    """
    Provider-agnostic model route.

    `model_name` may currently point to an OpenAI
    model, but the rest of the application does not
    depend on a specific provider.
    """

    task: ModelTask

    model_name: str

    requires_vision: bool = False

    reasoning_heavy: bool = False

    allow_fallback: bool = True

    maximum_attempts: int = 2

    temperature: float = 0.0

    # True only when the current chat request explicitly selected the
    # model. User-selected routes never silently switch to another model.
    user_selected: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if not (
            1
            <= self.maximum_attempts
            <= 2
        ):
            raise ValueError(
                "maximum_attempts must be "
                "between 1 and 2."
            )

        if not (
            0.0
            <= self.temperature
            <= 2.0
        ):
            raise ValueError(
                "temperature must be "
                "between 0 and 2."
            )


class ModelRouter:
    """
    Deterministic model routing.

    This class DOES NOT call an LLM.

    Core policy:
    - query understanding / classification
      -> strong classifier model
    - document structural indexing
      -> document-capable Vision model
    - general Physics text
      -> strong general text model
    - general numerical
      -> strong reasoning model
    - any document-dependent Tutor answer
      -> document model
    - document visual answer
      -> same document model with vision enabled
    - general verification
      -> general verifier model
    - document verification
      -> document verifier model

    Whether a request actually depends on a document
    is decided semantically upstream by
    QueryUnderstandingService.

    This router never infers document dependence from
    Physics keywords.

    Gateway responsibilities such as provider calls,
    retries, latency, token accounting and schema
    validation stay in gateway.py.
    """

    def __init__(
        self,
        *,
        classifier_model: str,
        text_model: str,
        multimodal_model: str,
        reasoning_model: str,
        verifier_model: str,
        fallback_model: str | None = None,
        document_model: str | None = None,
        document_verifier_model: str | None = None,
    ) -> None:

        self.classifier_model = (
            self._validate_model_name(
                classifier_model,
                "classifier_model",
            )
        )

        self.text_model = (
            self._validate_model_name(
                text_model,
                "text_model",
            )
        )

        self.multimodal_model = (
            self._validate_model_name(
                multimodal_model,
                "multimodal_model",
            )
        )

        self.reasoning_model = (
            self._validate_model_name(
                reasoning_model,
                "reasoning_model",
            )
        )

        self.verifier_model = (
            self._validate_model_name(
                verifier_model,
                "verifier_model",
            )
        )

        # During staged migration, existing construction
        # sites that do not pass document_model remain
        # compatible.
        #
        # The existing multimodal model becomes the
        # document-capable model in that situation.
        self.document_model = (
            self._validate_model_name(
                document_model,
                "document_model",
            )
            if (
                document_model
                and document_model.strip()
            )
            else self.multimodal_model
        )

        # Same compatibility principle for document
        # verification.
        self.document_verifier_model = (
            self._validate_model_name(
                document_verifier_model,
                "document_verifier_model",
            )
            if (
                document_verifier_model
                and document_verifier_model.strip()
            )
            else self.document_model
        )

        self.fallback_model = (
            fallback_model.strip()
            if (
                fallback_model
                and fallback_model.strip()
            )
            else None
        )

    def route_task(
        self,
        task: ModelTask,
        *,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> ModelRoute:
        """
        Route task classes that do not yet depend on
        per-request document context.

        Tutor and verifier call sites that already know
        whether the request requires a document should
        use `route_tutor()` and `route_verifier()`.

        `selected_model` is an optional request-level override. Existing
        callers that omit it receive exactly the established routing policy.

        Upload-time document structural indexing deliberately cannot be
        overridden by a later chat selection. This keeps already-created
        structural artifacts stable and reproducible.
        """

        if (
            task
            == ModelTask.DOCUMENT_STRUCTURE
            and selected_model is not None
        ):
            raise ValueError(
                "selected_model cannot override "
                "DOCUMENT_STRUCTURE."
            )

        route = self._route_task_default(
            task
        )

        return self._apply_user_selection(
            route=route,
            selected_model=selected_model,
        )

    def _route_task_default(
        self,
        task: ModelTask,
    ) -> ModelRoute:
        """
        Return the established application-configured route for a task.

        Keeping this policy separate means a request override can be added
        without duplicating or weakening the original routing rules.
        """

        if task in {
            ModelTask.INTENT_CLASSIFICATION,
            ModelTask.QUERY_SCOPE,
            ModelTask.QUERY_REWRITE,
            ModelTask.MEMORY_EXTRACTION,
        }:
            return ModelRoute(
                task=task,
                model_name=(
                    self.classifier_model
                ),
                requires_vision=False,
                reasoning_heavy=False,
                allow_fallback=True,
                maximum_attempts=2,
                temperature=0.0,
            )

        # Dedicated page-level Vision route for:
        #
        # - headings and subheadings
        # - bullet/numbered items
        # - problem/example labels
        # - numerical-task detection
        # - figures and diagrams
        # - cross-page continuation clues
        #
        # This is separate from Tutor calls.
        if (
            task
            == ModelTask.DOCUMENT_STRUCTURE
        ):
            return ModelRoute(
                task=task,
                model_name=self.document_model,
                requires_vision=True,
                reasoning_heavy=False,

                # The configured document model must be
                # used. Do not silently switch to another
                # model if it is unavailable.
                allow_fallback=False,

                maximum_attempts=2,
                temperature=0.0,
            )

        # Resolve references such as:
        # - "Problem 16"
        # - "the fourth point under ..."
        # - "the first numerical"
        # - a few remembered source words
        #
        # Resolution reads the saved text structure. It does not need
        # an image unless the resolved target later requires one.
        if (
            task
            == ModelTask.STRUCTURAL_RESOLUTION
        ):
            return ModelRoute(
                task=task,
                model_name=self.document_model,
                requires_vision=False,
                reasoning_heavy=True,
                allow_fallback=False,
                maximum_attempts=2,
                temperature=0.0,
            )

        if task == ModelTask.TUTOR_TEXT:
            return ModelRoute(
                task=task,
                model_name=self.text_model,
                requires_vision=False,
                reasoning_heavy=False,
                allow_fallback=True,
                maximum_attempts=2,
                temperature=0.1,
            )

        if (
            task
            == ModelTask.TUTOR_MULTIMODAL
        ):
            return ModelRoute(
                task=task,
                model_name=(
                    self.multimodal_model
                ),
                requires_vision=True,
                reasoning_heavy=False,
                allow_fallback=True,
                maximum_attempts=2,
                temperature=0.1,
            )

        if (
            task
            == ModelTask.TUTOR_NUMERICAL
        ):
            return ModelRoute(
                task=task,
                model_name=(
                    self.reasoning_model
                ),
                requires_vision=False,
                reasoning_heavy=True,
                allow_fallback=True,
                maximum_attempts=2,
                temperature=0.0,
            )

        if task == ModelTask.VERIFIER:
            return ModelRoute(
                task=task,
                model_name=(
                    self.verifier_model
                ),
                requires_vision=False,
                reasoning_heavy=True,
                allow_fallback=True,
                maximum_attempts=2,
                temperature=0.0,
            )

        raise ValueError(
            f"Unsupported model task: {task}"
        )

    def route_tutor(
        self,
        *,
        intent: RequestIntent,
        visual_context_available: bool,
        requires_document: bool = False,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> ModelRoute:
        """
        Decide which Tutor model should handle the
        request after query understanding and retrieval
        context are known.

        IMPORTANT:
        Document dependence has priority over
        numerical-vs-text routing.

        Examples:
        - general conceptual Physics
          -> text/general model
        - general typed numerical
          -> reasoning model
        - document text question
          -> document model without image input
        - document numerical
          -> document model
        - document diagram question
          -> document model with vision enabled

        `visual_context_available` means actual visual
        evidence has been resolved for this turn.
        """

        if requires_document:
            route = ModelRoute(
                # Reuse the existing document-capable
                # task class for Tutor answers.
                task=ModelTask.TUTOR_MULTIMODAL,

                model_name=self.document_model,

                requires_vision=(
                    visual_context_available
                ),

                reasoning_heavy=(
                    intent
                    == RequestIntent.NUMERICAL_PROBLEM
                ),

                allow_fallback=True,

                maximum_attempts=2,

                temperature=(
                    0.0
                    if (
                        intent
                        == RequestIntent.NUMERICAL_PROBLEM
                    )
                    else 0.1
                ),
            )

            return self._apply_user_selection(
                route=route,
                selected_model=selected_model,
            )

        if (
            intent
            == RequestIntent.NUMERICAL_PROBLEM
        ):
            return self.route_task(
                ModelTask.TUTOR_NUMERICAL,
                selected_model=selected_model,
            )

        if (
            intent
            == RequestIntent.DIAGRAM_QUESTION
            or visual_context_available
        ):
            return self.route_task(
                ModelTask.TUTOR_MULTIMODAL,
                selected_model=selected_model,
            )

        return self.route_task(
            ModelTask.TUTOR_TEXT,
            selected_model=selected_model,
        )

    def route_verifier(
        self,
        *,
        requires_document: bool = False,
        visual_context_available: bool = False,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ) = None,
    ) -> ModelRoute:
        """
        Choose the verifier model after the grounding
        mode is known.

        General answers use the general verifier.
        Document-grounded answers use the document
        verifier.

        Vision is enabled only when actual visual
        evidence has been resolved for the verifier.
        """

        if requires_document:
            route = ModelRoute(
                task=ModelTask.VERIFIER,

                model_name=(
                    self.document_verifier_model
                ),

                requires_vision=(
                    visual_context_available
                ),

                reasoning_heavy=True,

                allow_fallback=True,

                maximum_attempts=2,

                temperature=0.0,
            )

            return self._apply_user_selection(
                route=route,
                selected_model=selected_model,
            )

        return self.route_task(
            ModelTask.VERIFIER,
            selected_model=selected_model,
        )

    def fallback_for(
        self,
        route: ModelRoute,
    ) -> str | None:
        """
        Return the configured fallback model only
        when the selected route allows fallback.
        """

        if not route.allow_fallback:
            return None

        return self.fallback_model

    def _apply_user_selection(
        self,
        *,
        route: ModelRoute,
        selected_model: (
            UserSelectableModel
            | str
            | None
        ),
    ) -> ModelRoute:
        """
        Apply a validated request-level model without changing task policy.

        Vision requirements, reasoning classification, temperature and retry
        budget remain those chosen by the original deterministic router.
        Only the model identity and silent-fallback permission change.
        """

        if selected_model is None:
            return route

        model_name = (
            self.validate_user_selected_model(
                selected_model
            )
        )

        return replace(
            route,
            model_name=model_name,
            allow_fallback=False,
            user_selected=True,
        )

    @staticmethod
    def validate_user_selected_model(
        value: UserSelectableModel | str,
    ) -> str:
        """
        Return one canonical allowed model ID or raise a clear error.
        """

        if isinstance(
            value,
            UserSelectableModel,
        ):
            return value.value

        normalized = str(value).strip()

        try:
            return UserSelectableModel(
                normalized
            ).value
        except ValueError as error:
            supported = ", ".join(
                sorted(
                    SUPPORTED_USER_MODEL_NAMES
                )
            )

            raise ValueError(
                "Unsupported user-selected model: "
                f"{normalized!r}. Supported models: "
                f"{supported}."
            ) from error

    @staticmethod
    def _validate_model_name(
        value: str,
        field_name: str,
    ) -> str:

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized
