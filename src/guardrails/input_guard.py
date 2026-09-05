from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InputGuardDecision:
    blocked: bool
    reason: str | None = None
    message: str | None = None


class InputGuard:
    """
    Deterministic request guard for obvious attempts to obtain
    secrets, credentials, or private contact information.

    This guard intentionally does NOT try to classify Physics,
    scope, retrieval, or answer quality. Those responsibilities
    remain with the existing PhyMentor pipeline.

    The rule is narrow:
    - allow educational questions about security/privacy concepts;
    - block attempts to reveal, retrieve, print, dump, or identify
      actual private credentials or personal contact data.
    """

    BLOCK_MESSAGE = (
        "I can explain security and privacy concepts, but I cannot "
        "reveal private credentials, secrets, tokens, environment "
        "values, or another person's private contact information."
    )

    _EXFILTRATION_VERBS = (
        r"\b("
        r"give|show|reveal|tell|print|display|expose|dump|list|"
        r"provide|send|share|fetch|retrieve|extract|read|return|"
        r"leak|disclose|find|get"
        r")\b"
    )

    _SECRET_TARGETS = (
        r"\b("
        r"api[\s_-]*key|openai[\s_-]*api[\s_-]*key|"
        r"secret[\s_-]*key|client[\s_-]*secret|"
        r"password|passwd|credential|credentials|"
        r"access[\s_-]*token|refresh[\s_-]*token|"
        r"session[\s_-]*token|auth(?:entication)?[\s_-]*token|"
        r"bearer[\s_-]*token|jwt|private[\s_-]*key|"
        r"database[\s_-]*password|db[\s_-]*password|"
        r"redis[\s_-]*password|pinecone[\s_-]*api[\s_-]*key|"
        r"environment[\s_-]*variable|environment[\s_-]*variables|"
        r"env[\s_-]*var|env[\s_-]*vars|\.env"
        r")\b"
    )

    _PRIVATE_CONTACT_TARGETS = (
        r"\b("
        r"phone[\s_-]*number|mobile[\s_-]*number|"
        r"personal[\s_-]*email|email[\s_-]*address|"
        r"home[\s_-]*address|residential[\s_-]*address|"
        r"street[\s_-]*address"
        r")\b"
    )

    _POSSESSIVE_OR_SPECIFIC_PERSON = (
        r"\b("
        r"user(?:'s|s')?|student(?:'s|s')?|person(?:'s|s')?|"
        r"owner(?:'s|s')?|account(?:'s|s')?|"
        r"another[\s_-]*user|other[\s_-]*user|"
        r"someone(?:'s)?|their|his|her"
        r")\b"
    )

    _DIRECT_VALUE_QUESTION = (
        r"\b("
        r"what\s+is|what's|which\s+is|where\s+is|"
        r"tell\s+me|can\s+you\s+give|can\s+you\s+show"
        r")\b"
    )

    _EDUCATIONAL_PREFIXES = (
        "what is an api key",
        "what is a api key",
        "what is an access token",
        "what is a refresh token",
        "what is a jwt",
        "what is a password",
        "what are environment variables",
        "what is an environment variable",
        "what is a phone number",
        "what is an email address",
        "explain api keys",
        "explain access tokens",
        "explain refresh tokens",
        "explain jwt",
        "explain passwords",
        "explain environment variables",
        "explain phone numbers",
        "explain email addresses",
        "how do api keys work",
        "how does an api key work",
        "how do access tokens work",
        "how do refresh tokens work",
        "how does jwt work",
        "how do environment variables work",
    )

    @classmethod
    def check(
        cls,
        text: str,
    ) -> InputGuardDecision:
        normalized = " ".join(
            str(text).casefold().split()
        )

        if not normalized:
            return InputGuardDecision(
                blocked=False
            )

        if cls._looks_educational(
            normalized
        ):
            return InputGuardDecision(
                blocked=False
            )

        secret_target = bool(
            re.search(
                cls._SECRET_TARGETS,
                normalized,
                flags=re.IGNORECASE,
            )
        )

        contact_target = bool(
            re.search(
                cls._PRIVATE_CONTACT_TARGETS,
                normalized,
                flags=re.IGNORECASE,
            )
        )

        exfiltration_verb = bool(
            re.search(
                cls._EXFILTRATION_VERBS,
                normalized,
                flags=re.IGNORECASE,
            )
        )

        direct_value_question = bool(
            re.search(
                cls._DIRECT_VALUE_QUESTION,
                normalized,
                flags=re.IGNORECASE,
            )
        )

        specific_person = bool(
            re.search(
                cls._POSSESSIVE_OR_SPECIFIC_PERSON,
                normalized,
                flags=re.IGNORECASE,
            )
        )

        # Credential / secret requests.
        if secret_target and (
            exfiltration_verb
            or direct_value_question
        ):
            return cls._blocked(
                "secret_or_credential_request"
            )

        # Private contact data should be blocked only when the request
        # is about a concrete person/user, not when discussing the
        # concept in the abstract.
        if (
            contact_target
            and specific_person
            and (
                exfiltration_verb
                or direct_value_question
            )
        ):
            return cls._blocked(
                "private_contact_information_request"
            )

        return InputGuardDecision(
            blocked=False
        )

    @classmethod
    def _blocked(
        cls,
        reason: str,
    ) -> InputGuardDecision:
        return InputGuardDecision(
            blocked=True,
            reason=reason,
            message=cls.BLOCK_MESSAGE,
        )

    @classmethod
    def _looks_educational(
        cls,
        normalized: str,
    ) -> bool:
        return any(
            normalized.startswith(prefix)
            for prefix in cls._EDUCATIONAL_PREFIXES
        )
