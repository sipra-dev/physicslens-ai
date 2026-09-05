from __future__ import annotations

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from ragas.embeddings import (
    LangchainEmbeddingsWrapper,
)
from ragas.llms import (
    LangchainLLMWrapper,
)
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextEntityRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from src.config import settings


EVALUATOR_MODEL = "gpt-4o"

EVALUATOR_EMBEDDING_MODEL = (
    "text-embedding-3-small"
)


def _openai_api_key() -> str:
    """
    Reuse PhyMentor's existing configured OpenAI key.

    RAGAS does not own or store a separate credential.
    """

    api_key = str(
        settings.openai_api_key
        or ""
    ).strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured "
            "in the PhyMentor settings."
        )

    return api_key


def build_evaluator_llm():
    """
    One fixed evaluator LLM for every RAGAS metric.

    This evaluator judges PhyMentor responses.
    It does not replace PhyMentor's Tutor model.
    """

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=EVALUATOR_MODEL,
            temperature=0,
            api_key=_openai_api_key(),
        )
    )


def build_evaluator_embeddings():
    """
    Evaluator-only embeddings used by metrics such
    as Answer Relevancy and Answer Correctness.

    These do NOT replace PhyMentor's retrieval
    embeddings.
    """

    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=(
                EVALUATOR_EMBEDDING_MODEL
            ),
            api_key=_openai_api_key(),
        )
    )


def build_ragas_metrics():
    """
    Build the six RAGAS metrics selected for
    PhyMentor evaluation.
    """

    evaluator_llm = (
        build_evaluator_llm()
    )

    evaluator_embeddings = (
        build_evaluator_embeddings()
    )

    return [
        Faithfulness(
            llm=evaluator_llm,
        ),

        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=(
                evaluator_embeddings
            ),
        ),

        LLMContextPrecisionWithReference(
            llm=evaluator_llm,
        ),

        LLMContextRecall(
            llm=evaluator_llm,
        ),

        ContextEntityRecall(
            llm=evaluator_llm,
        ),

        AnswerCorrectness(
            llm=evaluator_llm,
            embeddings=(
                evaluator_embeddings
            ),
        ),
    ]