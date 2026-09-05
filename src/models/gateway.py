from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.models.contracts import ModelCallMetadata
from src.models.fallback import FailureKind, FallbackPolicy, RecoveryAction
from src.models.routing import (
    ModelRoute,
    ModelRouter,
    UserSelectableModel,
)


logger = logging.getLogger("phymentor.model_gateway")

TModel = TypeVar("TModel", bound=BaseModel)


class LLMGatewayError(Exception):
    """Raised when no bounded model-recovery path can return valid output."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: FailureKind,
        task: str,
        model: str,
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.task = task
        self.model = model


class _StructuredResponseError(Exception):
    """Internal error for unusable provider responses."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: FailureKind,
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


@dataclass(frozen=True, slots=True)
class GatewayCallTrace:
    """
    Rich operational trace for one successful gateway request.

    Gateway retry/fallback is infrastructure recovery and is separate
    from the later Tutor-Verifier answer-generation retry loop.
    """

    provider: str
    task: str
    primary_model: str
    final_model: str

    primary_attempt_count: int
    fallback_attempt_count: int
    used_fallback: bool

    latency_ms: float

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    estimated_cost_usd: float | None
    provider_request_id: str | None

    # Request-level model selection metadata. These fields make it possible
    # to audit which user-selected route was used without changing the older
    # ModelCallMetadata contract during this staged integration.
    user_selected: bool = False
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class GatewayStructuredResult(Generic[TModel]):
    value: TModel
    metadata: ModelCallMetadata
    trace: GatewayCallTrace


class LLMGateway:
    """
    Structured OpenAI gateway used by PhyMentor's model-facing modules.

    Responsibilities:
    - execute ModelRouter decisions
    - timeout handling
    - bounded retry
    - configured model fallback
    - Pydantic structured-output validation
    - token and latency tracking
    - optional injected cost estimation

    Not responsible for:
    - intent classification policy
    - Physics scope policy
    - retrieval
    - Tutor/Verifier orchestration
    """

    PROVIDER_NAME = "openai"

    # Provider-specific request compatibility belongs in the gateway.
    # Keep the exact reasoning-capable selector models centralized from the
    # same enum used by ModelRouter instead of accepting arbitrary prefixes.
    GPT_5_6_REASONING_MODELS = frozenset(
        {
            UserSelectableModel.GPT_5_6_SOL.value,
            UserSelectableModel.GPT_5_6_TERRA.value,
            UserSelectableModel.GPT_5_6_LUNA.value,
        }
    )

    def __init__(
        self,
        *,
        model_router: ModelRouter,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        fallback_policy: FallbackPolicy | None = None,
        pricing_per_million_tokens: (
            Mapping[str, tuple[float, float]] | None
        ) = None,
        client: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        `pricing_per_million_tokens` is deliberately injected:

        {
            "model-name": (
                input_cost_per_1m_tokens,
                output_cost_per_1m_tokens,
            )
        }

        No provider price is hard-coded into application logic.
        """

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self.model_router = model_router
        self.timeout_seconds = float(timeout_seconds)
        self.fallback_policy = fallback_policy or FallbackPolicy()
        self.pricing_per_million_tokens = dict(
            pricing_per_million_tokens or {}
        )
        self.sleeper = sleeper

        if client is not None:
            # Allows fast provider-free unit tests.
            self.client = client
            return

        normalized_api_key = api_key.strip() if api_key else ""

        if not normalized_api_key:
            raise ValueError(
                "An OpenAI API key is required when no client is injected."
            )

        # The application owns retry/fallback policy. Disable the SDK's
        # automatic retries so calls are not silently multiplied.
        self.client = OpenAI(
            api_key=normalized_api_key,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def generate_structured(
        self,
        *,
        route: ModelRoute,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
        image_urls: tuple[str, ...] | list[str] | None = None,
    ) -> TModel:
        """
        Exact interface consumed by QueryUnderstandingService.
        """

        return self.generate_structured_with_trace(
            route=route,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            image_urls=image_urls,
        ).value

    def generate_structured_with_trace(
        self,
        *,
        route: ModelRoute,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
        image_urls: tuple[str, ...] | list[str] | None = None,
    ) -> GatewayStructuredResult[TModel]:
        """
        Execute the bounded primary -> retry -> fallback path.
        """

        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()

        if not system_prompt:
            raise ValueError("system_prompt cannot be empty.")

        if not user_prompt:
            raise ValueError("user_prompt cannot be empty.")

        if not isinstance(response_model, type) or not issubclass(
            response_model,
            BaseModel,
        ):
            raise TypeError(
                "response_model must be a Pydantic BaseModel type."
            )

        self._validate_route_selection(
            route
        )

        normalized_image_urls = self._normalize_image_urls(
            image_urls=image_urls,
            route=route,
        )

        primary_model = route.model_name.strip()
        fallback_model = self.model_router.fallback_for(route)

        primary_attempts = 0
        fallback_attempts = 0

        current_model = primary_model
        using_fallback = False

        started_at = time.perf_counter()

        while True:
            if using_fallback:
                fallback_attempts += 1
            else:
                primary_attempts += 1

            try:
                reasoning_effort = (
                    self._reasoning_effort_for(
                        model=current_model,
                        route=route,
                    )
                )

                value, response = self._call_openai(
                    model=current_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                    image_urls=normalized_image_urls,
                    reasoning_effort=(
                        reasoning_effort
                    ),
                )

                latency_ms = (time.perf_counter() - started_at) * 1000.0

                input_tokens, output_tokens, total_tokens = (
                    self._usage_from_response(response)
                )

                estimated_cost_usd = self._estimate_cost(
                    model=current_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

                provider_request_id = getattr(
                    response,
                    "_request_id",
                    None,
                )

                # ModelCallMetadata predates separate fallback-attempt
                # accounting. attempt_count therefore records only the
                # bounded primary-attempt count (1-2).
                metadata = ModelCallMetadata(
                    task=route.task,
                    provider=self.PROVIDER_NAME,
                    model=current_model,
                    attempt_count=max(1, min(primary_attempts, 2)),
                    used_fallback=using_fallback,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )

                trace = GatewayCallTrace(
                    provider=self.PROVIDER_NAME,
                    task=route.task.value,
                    primary_model=primary_model,
                    final_model=current_model,
                    primary_attempt_count=primary_attempts,
                    fallback_attempt_count=fallback_attempts,
                    used_fallback=using_fallback,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost_usd,
                    provider_request_id=(
                        str(provider_request_id)
                        if provider_request_id
                        else None
                    ),
                    user_selected=(
                        route.user_selected
                    ),
                    reasoning_effort=(
                        reasoning_effort
                    ),
                )

                logger.info(
                    "model_call_succeeded "
                    "task=%s model=%s primary_attempts=%s "
                    "fallback_attempts=%s used_fallback=%s "
                    "latency_ms=%.2f input_tokens=%s "
                    "output_tokens=%s request_id=%s",
                    route.task.value,
                    current_model,
                    primary_attempts,
                    fallback_attempts,
                    using_fallback,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    provider_request_id,
                )

                return GatewayStructuredResult(
                    value=value,
                    metadata=metadata,
                    trace=trace,
                )

            except Exception as exc:
                failure_kind = self._failure_kind(exc)

                logger.warning(
                    "model_call_failed "
                    "task=%s model=%s failure_kind=%s "
                    "primary_attempts=%s fallback_attempts=%s",
                    route.task.value,
                    current_model,
                    failure_kind.value,
                    primary_attempts,
                    fallback_attempts,
                )

                # Fallback is terminal. Never recursively retry fallback.
                if using_fallback:
                    raise LLMGatewayError(
                        "Fallback model call failed; no safe recovery "
                        "path remains.",
                        failure_kind=failure_kind,
                        task=route.task.value,
                        model=current_model,
                    ) from exc

                decision = self.fallback_policy.decide(
                    task=route.task,
                    failure_kind=failure_kind,
                    primary_attempt_count=min(
                        max(primary_attempts, 1),
                        self.fallback_policy.MAX_PRIMARY_ATTEMPTS,
                    ),
                    primary_model=primary_model,
                    fallback_model=fallback_model,
                    allow_fallback=route.allow_fallback,
                    fallback_already_attempted=False,
                )

                # ModelRoute is allowed to impose a stricter primary-attempt
                # budget than the global fallback policy.
                if (
                    decision.action == RecoveryAction.RETRY_PRIMARY
                    and primary_attempts >= route.maximum_attempts
                ):
                    decision = self.fallback_policy.decide(
                        task=route.task,
                        failure_kind=failure_kind,
                        primary_attempt_count=(
                            self.fallback_policy.MAX_PRIMARY_ATTEMPTS
                        ),
                        primary_model=primary_model,
                        fallback_model=fallback_model,
                        allow_fallback=route.allow_fallback,
                        fallback_already_attempted=False,
                    )

                if decision.action == RecoveryAction.RETRY_PRIMARY:
                    if decision.retry_delay_seconds > 0:
                        self.sleeper(decision.retry_delay_seconds)

                    current_model = primary_model
                    continue

                if decision.action == RecoveryAction.USE_FALLBACK:
                    next_model = (decision.next_model or "").strip()

                    if not next_model:
                        raise LLMGatewayError(
                            "Fallback policy selected no fallback model.",
                            failure_kind=FailureKind.UNKNOWN,
                            task=route.task.value,
                            model=primary_model,
                        ) from exc

                    using_fallback = True
                    current_model = next_model
                    continue

                raise LLMGatewayError(
                    decision.reason,
                    failure_kind=failure_kind,
                    task=route.task.value,
                    model=current_model,
                ) from exc

    def _call_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
        image_urls: tuple[str, ...],
        reasoning_effort: str | None = None,
    ) -> tuple[TModel, Any]:
        """
        One provider call only. Retry policy lives outside this method.
        """

        user_content: list[dict[str, str]] = [
            {
                "type": "input_text",
                "text": user_prompt,
            }
        ]

        for image_url in image_urls:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": image_url,
                }
            )

        request_arguments: dict[
            str,
            Any,
        ] = {
            "model": model,
            "input": [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "text_format": response_model,
        }

        # The Responses API reasoning object is supported by GPT-5/o-series
        # reasoning models. GPT-4o must keep the established request shape and
        # therefore receives no reasoning parameter.
        if reasoning_effort is not None:
            request_arguments["reasoning"] = {
                "effort": reasoning_effort,
            }

        response = self.client.responses.parse(
            **request_arguments
        )

        parsed = getattr(response, "output_parsed", None)

        if parsed is None:
            if self._contains_refusal(response):
                raise _StructuredResponseError(
                    "Provider refused the structured request.",
                    failure_kind=FailureKind.SAFETY_BLOCK,
                )

            raise _StructuredResponseError(
                "Provider returned no parsed structured output.",
                failure_kind=FailureKind.INVALID_RESPONSE,
            )

        try:
            validated = response_model.model_validate(parsed)

        except ValidationError as exc:
            raise _StructuredResponseError(
                "Structured output failed application schema validation.",
                failure_kind=FailureKind.SCHEMA_VALIDATION,
            ) from exc

        return validated, response

    @classmethod
    def _reasoning_effort_for(
        cls,
        *,
        model: str,
        route: ModelRoute,
    ) -> str | None:
        """
        Translate provider-neutral route intent into GPT-5.6 parameters.

        Normal tutoring/classification uses the documented balanced starting
        point. Numerical, structural-resolution and verifier routes receive a
        higher reasoning budget. Non-GPT-5.6 models keep their original call.
        """

        normalized_model = model.strip()

        if (
            normalized_model
            not in cls.GPT_5_6_REASONING_MODELS
        ):
            return None

        if route.reasoning_heavy:
            return "high"

        return "medium"

    @staticmethod
    def _validate_route_selection(
        route: ModelRoute,
    ) -> None:
        """
        Enforce request-selection invariants again at the provider boundary.

        ModelRouter normally guarantees both rules. Rechecking here prevents
        another call site from manually constructing an unsafe ModelRoute.
        """

        if not route.user_selected:
            return

        ModelRouter.validate_user_selected_model(
            route.model_name
        )

        if route.allow_fallback:
            raise ValueError(
                "A user-selected model route cannot "
                "allow silent model fallback."
            )

    @staticmethod
    def _normalize_image_urls(
        *,
        image_urls: tuple[str, ...] | list[str] | None,
        route: ModelRoute,
    ) -> tuple[str, ...]:
        """
        Validate optional multimodal inputs.

        Image inputs must already be a normal URL or a base64 data URL.
        Local file-path encoding deliberately stays outside the gateway so
        provider code is not coupled to local filesystem storage.
        """

        if not image_urls:
            return ()

        if not route.requires_vision:
            raise ValueError(
                "image_urls were supplied to a route that does not "
                "require vision."
            )

        normalized: list[str] = []

        for raw_value in image_urls:
            value = raw_value.strip()

            if not value:
                continue

            if not (
                value.startswith("https://")
                or value.startswith("http://")
                or value.startswith("data:image/")
            ):
                raise ValueError(
                    "Each image input must be an http(s) URL or "
                    "a base64 data:image/... URL."
                )

            if value not in normalized:
                normalized.append(value)

        if not normalized:
            raise ValueError(
                "At least one valid image URL is required for "
                "multimodal input."
            )

        return tuple(normalized)

    @staticmethod
    def _contains_refusal(response: Any) -> bool:
        """
        Detect refusal content defensively using public response fields.
        """

        for output_item in getattr(response, "output", None) or []:
            for content_item in getattr(output_item, "content", None) or []:
                if getattr(content_item, "type", None) == "refusal":
                    return True

                if getattr(content_item, "refusal", None):
                    return True

        return False

    @staticmethod
    def _failure_kind(exc: Exception) -> FailureKind:
        """
        Convert provider/library errors into the gateway's stable taxonomy.
        """

        if isinstance(exc, _StructuredResponseError):
            return exc.failure_kind

        if isinstance(exc, ValidationError):
            return FailureKind.SCHEMA_VALIDATION

        if isinstance(exc, openai.APITimeoutError):
            return FailureKind.TIMEOUT

        if isinstance(exc, openai.RateLimitError):
            return FailureKind.RATE_LIMIT

        if isinstance(
            exc,
            (openai.AuthenticationError, openai.PermissionDeniedError),
        ):
            return FailureKind.AUTHENTICATION

        if isinstance(exc, openai.InternalServerError):
            return FailureKind.SERVER_ERROR

        if isinstance(exc, openai.APIConnectionError):
            return FailureKind.CONNECTION

        if isinstance(exc, openai.APIResponseValidationError):
            return FailureKind.INVALID_RESPONSE

        if isinstance(exc, openai.APIStatusError):
            status_code = getattr(exc, "status_code", None)

            if status_code == 408:
                return FailureKind.TIMEOUT

            if status_code == 429:
                return FailureKind.RATE_LIMIT

            if status_code in {401, 403}:
                return FailureKind.AUTHENTICATION

            if isinstance(status_code, int) and status_code >= 500:
                return FailureKind.SERVER_ERROR

            if status_code in {400, 422}:
                return FailureKind.INVALID_RESPONSE

        return FailureKind.UNKNOWN

    @staticmethod
    def _usage_from_response(
        response: Any,
    ) -> tuple[int | None, int | None, int | None]:
        usage = getattr(response, "usage", None)

        if usage is None:
            return None, None, None

        return (
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "total_tokens", None),
        )

    def _estimate_cost(
        self,
        *,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        """
        Estimate cost only when pricing is explicitly injected.
        """

        pricing = self.pricing_per_million_tokens.get(model)

        if pricing is None or input_tokens is None or output_tokens is None:
            return None

        input_rate, output_rate = pricing

        if input_rate < 0 or output_rate < 0:
            raise ValueError("Model pricing cannot be negative.")

        return (
            (input_tokens / 1_000_000) * input_rate
            + (output_tokens / 1_000_000) * output_rate
        )
