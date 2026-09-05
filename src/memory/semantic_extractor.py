from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.memory.semantic_models import (
    SemanticLearningMemoryRecord,
    SemanticMemoryKind,
)


class SemanticMemoryLLMProtocol(Protocol):
    """
    Minimal LLM interface required by the semantic
    memory extractor.

    A runtime adapter will later connect this protocol
    to the project's existing LLMGateway.

    This keeps semantic memory independent of any
    specific provider such as OpenAI.
    """

    def extract_memory(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any] | str:
        ...


class _MemoryExtractionDecision(BaseModel):
    """
    Structured output expected from the LLM.

    This object is deliberately separate from the
    final Pinecone memory record because the LLM must
    first decide whether anything is worth storing.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    should_store: bool

    kind: SemanticMemoryKind | None = None

    topic: str | None = Field(
        default=None,
        max_length=256,
    )

    concept: str | None = Field(
        default=None,
        max_length=256,
    )

    text: str | None = Field(
        default=None,
        max_length=4000,
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class SemanticMemoryExtractor:
    """
    Use an LLM to detect durable learning signals
    from one verified tutoring interaction.

    Examples of memories worth keeping:
    - misconception
    - knowledge gap
    - demonstrated mastery
    - stable support/explanation preference

    Ordinary chat, greetings, temporary confusion,
    or low-confidence guesses must not become
    semantic memory.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.75

    SYSTEM_PROMPT = """
You are the semantic learning-memory extractor for
a school Physics tutoring system.

Your job is NOT to answer the student's Physics
question.

Your only task is to inspect a completed tutoring
interaction and decide whether it reveals a durable,
useful learning signal about the student.

Allowed memory kinds:

1. misconception
   The student expresses or repeatedly demonstrates
   an incorrect Physics belief or mental model.

2. knowledge_gap
   The interaction clearly shows that the student
   lacks an important prerequisite concept or does
   not understand a concept yet.

3. mastery
   The student clearly demonstrates reliable
   understanding of a Physics concept.

4. support_preference
   The student demonstrates a meaningful and likely
   reusable learning preference, such as consistently
   needing visual explanations, simpler wording, or
   step-by-step derivations.

Do NOT store:
- greetings
- casual conversation
- one-off wording choices
- the student's raw question merely because it exists
- facts about the uploaded document
- tutor-generated information that says nothing about
  the student's learning
- uncertain guesses
- sensitive personal information
- temporary conversational context
- redundant information with no educational value

Return exactly one JSON object with this structure:

{
  "should_store": true or false,
  "kind": "misconception" | "knowledge_gap" |
          "mastery" | "support_preference" | null,
  "topic": "Physics topic" | null,
  "concept": "specific concept" | null,
  "text": "short durable learning-memory statement" | null,
  "confidence": number from 0.0 to 1.0
}

Rules:

- If should_store is false, kind/topic/concept/text
  should normally be null.
- Never invent evidence.
- Write memory text in third person, for example:
  "Student confuses speed with acceleration."
- Keep memory text concise and reusable.
- Classify based on the student's demonstrated
  understanding, not merely on what the tutor said.
""".strip()

    def __init__(
        self,
        *,
        llm: SemanticMemoryLLMProtocol,
        confidence_threshold: float = (
            DEFAULT_CONFIDENCE_THRESHOLD
        ),
    ) -> None:
        if not (
            0.0
            <= confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "confidence_threshold must be "
                "between 0.0 and 1.0."
            )

        self.llm = llm

        self.confidence_threshold = (
            confidence_threshold
        )

    def extract(
        self,
        *,
        user_id: str,
        student_text: str,
        tutor_text: str,
        verifier_text: str | None = None,
        session_id: str | None = None,
        document_id: str | None = None,
    ) -> SemanticLearningMemoryRecord | None:
        """
        Analyse one completed tutoring interaction.

        Returns:
            SemanticLearningMemoryRecord:
                A validated memory worth storing.

            None:
                Nothing safe/useful enough should
                be stored.
        """

        normalized_user_id = (
            user_id.strip()
        )

        normalized_student_text = (
            student_text.strip()
        )

        normalized_tutor_text = (
            tutor_text.strip()
        )

        if not normalized_user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        if not normalized_student_text:
            return None

        if not normalized_tutor_text:
            return None

        user_prompt = self._build_user_prompt(
            student_text=(
                normalized_student_text
            ),
            tutor_text=(
                normalized_tutor_text
            ),
            verifier_text=verifier_text,
        )

        try:
            raw_result = (
                self.llm.extract_memory(
                    system_prompt=(
                        self.SYSTEM_PROMPT
                    ),
                    user_prompt=user_prompt,
                )
            )

            decision = (
                self._parse_decision(
                    raw_result
                )
            )

        except Exception:
            # Semantic memory is useful but must never
            # break the main Tutor response path.
            return None

        if decision is None:
            return None

        if not decision.should_store:
            return None

        if (
            decision.confidence
            < self.confidence_threshold
        ):
            return None

        if decision.kind is None:
            return None

        topic = self._clean_optional(
            decision.topic
        )

        text = self._clean_optional(
            decision.text
        )

        if not topic or not text:
            return None

        concept = self._clean_optional(
            decision.concept
        )

        normalized_session_id = (
            self._clean_optional(
                session_id
            )
        )

        normalized_document_id = (
            self._clean_optional(
                document_id
            )
        )

        try:
            return SemanticLearningMemoryRecord(
                user_id=normalized_user_id,
                kind=decision.kind,
                topic=topic,
                concept=concept,
                text=text,
                confidence=(
                    decision.confidence
                ),
                source_session_id=(
                    normalized_session_id
                ),
                source_document_id=(
                    normalized_document_id
                ),
            )

        except ValidationError:
            return None

    @classmethod
    def _parse_decision(
        cls,
        raw_result: dict[str, Any] | str,
    ) -> _MemoryExtractionDecision | None:
        """
        Convert provider output into the strict
        Pydantic contract.

        Invalid or malformed LLM output fails closed.
        """

        payload: Any

        if isinstance(
            raw_result,
            dict,
        ):
            payload = raw_result

        elif isinstance(
            raw_result,
            str,
        ):
            raw_text = (
                raw_result.strip()
            )

            if not raw_text:
                return None

            try:
                payload = json.loads(
                    raw_text
                )

            except json.JSONDecodeError:
                return None

        else:
            return None

        try:
            return (
                _MemoryExtractionDecision
                .model_validate(payload)
            )

        except ValidationError:
            return None

    @staticmethod
    def _clean_optional(
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @staticmethod
    def _build_user_prompt(
        *,
        student_text: str,
        tutor_text: str,
        verifier_text: str | None,
    ) -> str:
        """
        Build the evidence presented to the
        memory-extraction LLM.
        """

        verifier_section = (
            verifier_text.strip()
            if verifier_text
            and verifier_text.strip()
            else "No verifier note provided."
        )

        return (
            "Analyse this completed tutoring "
            "interaction.\n\n"
            "STUDENT:\n"
            f"{student_text}\n\n"
            "TUTOR:\n"
            f"{tutor_text}\n\n"
            "VERIFIER:\n"
            f"{verifier_section}\n\n"
            "Decide whether this interaction "
            "contains a durable educational "
            "memory worth storing."
        )