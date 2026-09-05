from __future__ import annotations

from typing import Literal

from src.graph.state import PhysicsTutorState
from src.models.contracts import (
    RequestIntent,
    VerificationAction,
)


AfterCacheRoute = Literal[
    "output_guard",
    "retrieval_planner",
]

AfterVerifierRoute = Literal[
    "output_guard",
    "broader_retrieval",
    "tutor_agent",
    "insufficient_evidence_response",
]


def route_after_cache(
    state: PhysicsTutorState,
) -> AfterCacheRoute:
    """
    Decide whether the request can safely bypass retrieval/Tutor/Verifier.

    Fast paths:
    - greeting
    - upload-direction response
    - out-of-scope rejection
    - missing-document insufficient response
    - safe answer-cache hit

    Everything else continues to retrieval planning.
    """

    if state.get(
        "cache_hit",
        False,
    ):
        return "output_guard"

    terminal_action = state.get(
        "terminal_action"
    )

    if terminal_action in {
        VerificationAction
        .REJECT_OUT_OF_SCOPE,
        VerificationAction
        .ASK_FOR_CLEARER_IMAGE,
        VerificationAction
        .INSUFFICIENT_EVIDENCE,
    }:
        return "output_guard"

    intent = state.get("intent")

    if (
        intent is not None
        and intent.intent
        in {
            RequestIntent.GREETING,
            RequestIntent
            .UPLOAD_DOCUMENT,
        }
        and state.get(
            "answer_draft"
        )
        is not None
    ):
        return "output_guard"

    return "retrieval_planner"


def route_after_verifier(
    state: PhysicsTutorState,
) -> AfterVerifierRoute:
    """
    Route the bounded Tutor -> Verifier control loop.

    Rules:
    - PASS -> output_guard
    - terminal verifier actions -> output_guard
    - RETRY_RETRIEVAL -> broader retrieval, but only while a second Tutor
      generation is still available
    - REGENERATE -> same-context Tutor retry, again only while the second
      Tutor generation is still available
    - no third Tutor generation is ever allowed
    - unexpected/missing verifier outcome fails closed
    """

    verification = state.get(
        "verification_result"
    )

    if verification is None:
        return (
            "insufficient_evidence_response"
        )

    action = verification.action

    if (
        action
        == VerificationAction.PASS
    ):
        return "output_guard"

    if action in {
        VerificationAction
        .ASK_FOR_CLEARER_IMAGE,
        VerificationAction
        .INSUFFICIENT_EVIDENCE,
        VerificationAction
        .REJECT_OUT_OF_SCOPE,
    }:
        return "output_guard"

    generation_attempts = int(
        state.get(
            "generation_attempts",
            0,
        )
    )

    # Maximum answer-generation attempts = 2.
    if generation_attempts >= 2:
        return (
            "insufficient_evidence_response"
        )

    if (
        action
        == VerificationAction
        .RETRY_RETRIEVAL
    ):
        if state.get(
            "active_document_id"
        ):
            return "broader_retrieval"

        return (
            "insufficient_evidence_response"
        )

    if (
        action
        == VerificationAction.REGENERATE
    ):
        return "tutor_agent"

    # Unknown/unexpected action: fail closed.
    return "insufficient_evidence_response"