from __future__ import annotations

from ragas.dataset_schema import SingleTurnSample

from src.models.contracts import TutorAnswer
from src.retrieval.models import ContextBundle


def tutor_answer_to_text(
    answer: TutorAnswer,
) -> str:
    """
    Convert PhyMentor's structured TutorAnswer into
    the complete substantive answer that a RAGAS
    evaluator should judge.

    Source/citation metadata is deliberately excluded.
    """

    sections: list[str] = []

    direct_answer = answer.direct_answer.strip()

    if direct_answer:
        sections.append(direct_answer)

    if answer.problem_statement:
        problem_statement = (
            answer.problem_statement.strip()
        )

        if problem_statement:
            sections.append(
                "Problem statement:\n"
                + problem_statement
            )

    if answer.steps:
        cleaned_steps = [
            step.strip()
            for step in answer.steps
            if step.strip()
        ]

        if cleaned_steps:
            rendered_steps = "\n".join(
                f"{index}. {step}"
                for index, step in enumerate(
                    cleaned_steps,
                    start=1,
                )
            )

            sections.append(
                "Steps:\n"
                + rendered_steps
            )

    if answer.formulae:
        rendered_formulae: list[str] = []

        for formula in answer.formulae:
            latex = formula.latex.strip()
            meaning = formula.meaning.strip()

            if not latex:
                continue

            if meaning:
                rendered_formulae.append(
                    f"{latex}\n{meaning}"
                )
            else:
                rendered_formulae.append(
                    latex
                )

        if rendered_formulae:
            sections.append(
                "Formulae:\n"
                + "\n\n".join(
                    rendered_formulae
                )
            )

    if answer.diagram_explanation:
        diagram_explanation = (
            answer.diagram_explanation.strip()
        )

        if diagram_explanation:
            sections.append(
                "Diagram explanation:\n"
                + diagram_explanation
            )

    if answer.common_mistake:
        common_mistake = (
            answer.common_mistake.strip()
        )

        if common_mistake:
            sections.append(
                "Common mistake:\n"
                + common_mistake
            )

    if answer.final_result:
        final_result = (
            answer.final_result.strip()
        )

        if final_result:
            sections.append(
                "Final result:\n"
                + final_result
            )

    return "\n\n".join(sections).strip()


def retrieved_contexts_to_text(
    context: ContextBundle | None,
) -> list[str]:
    """
    Extract the exact final compressed/reranked
    textual evidence used by the serving workflow.

    Order is preserved because context precision
    cares about retrieval ranking.
    """

    if context is None:
        return []

    contexts: list[str] = []
    seen: set[str] = set()

    for item in context.items:
        text = item.text.strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        contexts.append(text)

    return contexts


def build_ragas_sample(
    *,
    question: str,
    answer: TutorAnswer,
    context: ContextBundle | None,
    reference_answer: str,
) -> SingleTurnSample:
    """
    Build one RAGAS single-turn evaluation sample
    from actual PhyMentor runtime output.
    """

    normalized_question = question.strip()
    normalized_reference = (
        reference_answer.strip()
    )

    if not normalized_question:
        raise ValueError(
            "question cannot be empty."
        )

    if not normalized_reference:
        raise ValueError(
            "reference_answer cannot be empty."
        )

    response = tutor_answer_to_text(
        answer
    )

    if not response:
        raise ValueError(
            "TutorAnswer produced no evaluable text."
        )

    retrieved_contexts = (
        retrieved_contexts_to_text(
            context
        )
    )

    return SingleTurnSample(
        user_input=normalized_question,
        response=response,
        retrieved_contexts=(
            retrieved_contexts
        ),
        reference=normalized_reference,
    )