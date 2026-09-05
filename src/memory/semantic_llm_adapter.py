from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from src.models.contracts import (
    ModelTask,
    StrictModel,
)
from src.models.gateway import LLMGateway
from src.models.routing import ModelRouter


class _SemanticMemoryExtractionOutput(
    StrictModel
):
    """
    Structured output expected from the LLM
    when deciding whether a conversation turn
    contains useful long-term learning memory.
    """

    should_store: bool

    kind: (
        Literal[
            "misconception",
            "knowledge_gap",
            "mastery",
            "support_preference",
        ]
        | None
    ) = None

    topic: str | None = Field(
        default=None,
        max_length=300,
    )

    concept: str | None = Field(
        default=None,
        max_length=300,
    )

    text: str | None = Field(
        default=None,
        max_length=1500,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class SemanticMemoryLLMAdapter:
    """
    Adapter between SemanticMemoryExtractor
    and the application's existing model stack.

    Flow:
        SemanticMemoryExtractor
        -> SemanticMemoryLLMAdapter
        -> ModelRouter
        -> LLMGateway
        -> provider
    """

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        model_router: ModelRouter,
    ) -> None:
        self.gateway = gateway
        self.model_router = model_router

    def extract_memory(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """
        Run one structured semantic-memory
        extraction request through the existing
        router + gateway infrastructure.
        """

        route = self.model_router.route_task(
            ModelTask.MEMORY_EXTRACTION
        )

        result = self.gateway.generate_structured(
            route=route,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=(
                _SemanticMemoryExtractionOutput
            ),
        )

        return result.model_dump(
            mode="json"
        )