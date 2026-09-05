from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.models.contracts import ModelTask


class FailureKind(str, Enum):
    """
    Normalized failure categories understood
    by the model gateway.
    """

    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    CONNECTION = "CONNECTION"
    SERVER_ERROR = "SERVER_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    SAFETY_BLOCK = "SAFETY_BLOCK"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(str, Enum):
    RETRY_PRIMARY = "RETRY_PRIMARY"
    USE_FALLBACK = "USE_FALLBACK"
    FAIL_GRACEFULLY = "FAIL_GRACEFULLY"


@dataclass(
    frozen=True,
    slots=True,
)
class FallbackDecision:
    """
    Result of deterministic recovery-policy
    evaluation.
    """

    action: RecoveryAction
    reason: str
    next_model: str | None = None
    retry_delay_seconds: float = 0.0


class FallbackPolicy:
    """
    Deterministic bounded retry/fallback policy.

    Gateway-level provider/model recovery is separate
    from the Tutor-Verifier answer-generation loop.

    Policy:
    - Primary model may be attempted at most twice.
    - After the primary path is exhausted, one configured
      fallback attempt may be used.
    - The fallback itself is never recursively retried.
    - There is never an unbounded retry loop.
    """

    MAX_PRIMARY_ATTEMPTS = 2
    MAX_FALLBACK_ATTEMPTS = 1

    _TRANSIENT_FAILURES = {
        FailureKind.TIMEOUT,
        FailureKind.RATE_LIMIT,
        FailureKind.CONNECTION,
        FailureKind.SERVER_ERROR,
    }

    _NON_RETRYABLE_FAILURES = {
        FailureKind.AUTHENTICATION,
        FailureKind.SAFETY_BLOCK,
    }

    def decide(
        self,
        *,
        task: ModelTask,
        failure_kind: FailureKind,
        primary_attempt_count: int,
        primary_model: str,
        fallback_model: str | None,
        allow_fallback: bool,
        fallback_already_attempted: bool = False,
    ) -> FallbackDecision:
        """
        Decide the next gateway action after a failed call.

        `primary_attempt_count` counts only calls made to
        the primary model. The optional fallback call is
        tracked separately.
        """

        self._validate_inputs(
            primary_attempt_count=(
                primary_attempt_count
            ),
            primary_model=primary_model,
        )

        # A fallback call is the final recovery step.
        # Never retry or recursively fallback from it.
        if fallback_already_attempted:
            return FallbackDecision(
                action=RecoveryAction.FAIL_GRACEFULLY,
                reason=(
                    "The configured fallback has already "
                    "been attempted."
                ),
            )

        # Safety blocks are not transport failures and
        # must not be bypassed by another model.
        if failure_kind == FailureKind.SAFETY_BLOCK:
            return FallbackDecision(
                action=RecoveryAction.FAIL_GRACEFULLY,
                reason=(
                    "The model response was safety-blocked; "
                    "fallback must not be used to bypass it."
                ),
            )

        # Authentication/configuration errors will not be
        # repaired by retrying the same primary call.
        # A separately configured fallback may still be used
        # if the gateway has one available.
        if failure_kind == FailureKind.AUTHENTICATION:
            return self._fallback_or_fail(
                primary_model=primary_model,
                fallback_model=fallback_model,
                allow_fallback=allow_fallback,
                reason=(
                    "Primary model authentication failed."
                ),
            )

        # Transient provider failures receive one bounded
        # retry on the primary route.
        if (
            failure_kind in self._TRANSIENT_FAILURES
            and primary_attempt_count
            < self.MAX_PRIMARY_ATTEMPTS
        ):
            return FallbackDecision(
                action=RecoveryAction.RETRY_PRIMARY,
                reason=(
                    "Transient provider failure; one "
                    "bounded primary retry is allowed."
                ),
                next_model=primary_model,
                retry_delay_seconds=(
                    self._retry_delay(
                        failure_kind,
                        primary_attempt_count,
                    )
                ),
            )

        # Invalid structured output should not keep
        # regenerating indefinitely. After a failed primary
        # structured response, prefer a configured fallback.
        if failure_kind in {
            FailureKind.INVALID_RESPONSE,
            FailureKind.SCHEMA_VALIDATION,
        }:
            return self._fallback_or_fail(
                primary_model=primary_model,
                fallback_model=fallback_model,
                allow_fallback=allow_fallback,
                reason=(
                    "Primary model returned an unusable "
                    "structured response."
                ),
            )

        # Once the primary retry budget is exhausted,
        # attempt the configured fallback exactly once.
        if (
            primary_attempt_count
            >= self.MAX_PRIMARY_ATTEMPTS
        ):
            return self._fallback_or_fail(
                primary_model=primary_model,
                fallback_model=fallback_model,
                allow_fallback=allow_fallback,
                reason=(
                    "Primary retry budget is exhausted."
                ),
            )

        # Unknown/non-transient failures should avoid a
        # blind repeat. Prefer the configured fallback.
        return self._fallback_or_fail(
            primary_model=primary_model,
            fallback_model=fallback_model,
            allow_fallback=allow_fallback,
            reason=(
                f"Primary model failed with "
                f"{failure_kind.value}."
            ),
        )

    def _fallback_or_fail(
        self,
        *,
        primary_model: str,
        fallback_model: str | None,
        allow_fallback: bool,
        reason: str,
    ) -> FallbackDecision:
        if self._fallback_available(
            primary_model=primary_model,
            fallback_model=fallback_model,
            allow_fallback=allow_fallback,
        ):
            return FallbackDecision(
                action=RecoveryAction.USE_FALLBACK,
                reason=reason,
                next_model=fallback_model,
            )

        return FallbackDecision(
            action=RecoveryAction.FAIL_GRACEFULLY,
            reason=(
                f"{reason} No safe fallback "
                "route is available."
            ),
        )

    @staticmethod
    def _fallback_available(
        *,
        primary_model: str,
        fallback_model: str | None,
        allow_fallback: bool,
    ) -> bool:
        if not allow_fallback:
            return False

        if fallback_model is None:
            return False

        normalized_fallback = (
            fallback_model.strip()
        )

        if not normalized_fallback:
            return False

        # The same model is another retry, not fallback.
        if (
            normalized_fallback
            == primary_model.strip()
        ):
            return False

        return True

    @staticmethod
    def _retry_delay(
        failure_kind: FailureKind,
        primary_attempt_count: int,
    ) -> float:
        """
        Small bounded exponential-style backoff.

        There is currently only one primary retry,
        but this keeps the policy explicit.
        """

        base_delay = {
            FailureKind.RATE_LIMIT: 1.0,
            FailureKind.TIMEOUT: 0.5,
            FailureKind.CONNECTION: 0.5,
            FailureKind.SERVER_ERROR: 0.75,
        }.get(
            failure_kind,
            0.0,
        )

        return base_delay * (
            2 ** max(
                primary_attempt_count - 1,
                0,
            )
        )

    @classmethod
    def _validate_inputs(
        cls,
        *,
        primary_attempt_count: int,
        primary_model: str,
    ) -> None:
        if (
            primary_attempt_count < 1
            or primary_attempt_count
            > cls.MAX_PRIMARY_ATTEMPTS
        ):
            raise ValueError(
                "primary_attempt_count must be "
                "between 1 and 2."
            )

        if not primary_model.strip():
            raise ValueError(
                "primary_model cannot be empty."
            )