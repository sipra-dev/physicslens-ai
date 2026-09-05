from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from ragas import evaluate
from ragas.dataset_schema import (
    EvaluationDataset,
)

from apps.api.routes.chat import chat_graph

from evaluation.ragas_metrics import (
    build_ragas_metrics,
)
from evaluation.ragas_sample import (
    build_ragas_sample,
)

from src.config import settings
from src.graph.checkpointing import (
    get_checkpoint_manager,
)
from src.models.contracts import (
    LanguageCode,
    TutorAnswer,
)
from src.retrieval.models import (
    ContextBundle,
)
from src.runtime_services import (
    get_local_storage,
    get_retrieval_service,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "ragas_dataset.json"
)

DEFAULT_RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)


def _load_json(
    path: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, list):
        raise ValueError(
            "RAGAS dataset must contain "
            "a JSON list."
        )

    return payload


def _load_ready_documents(
    *,
    user_id: str,
) -> list[dict[str, str]]:
    """
    Read existing PhyMentor document metadata.

    Evaluation only reads the normal local storage.
    It does not create, re-index, rename or delete
    any document.
    """

    storage = get_local_storage()

    documents_directory = (
        storage.get_document_directory(
            user_id=user_id,
            document_id="__evaluation_probe__",
        ).parent
    )

    if not documents_directory.is_dir():
        return []

    ready_documents: list[
        dict[str, str]
    ] = []

    for document_directory in sorted(
        (
            path
            for path in documents_directory.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name,
    ):
        metadata = (
            storage.read_document_metadata(
                user_id=user_id,
                document_id=(
                    document_directory.name
                ),
            )
        )

        if metadata is None:
            continue

        status = str(
            metadata.get(
                "status",
                "",
            )
        ).strip().upper()

        if status != "READY":
            continue

        document_id = str(
            metadata.get(
                "document_id",
                "",
            )
        ).strip()

        original_filename = str(
            metadata.get(
                "original_filename",
                "",
            )
        ).strip()

        if (
            not document_id
            or not original_filename
        ):
            continue

        ready_documents.append(
            {
                "document_id": document_id,
                "name": original_filename,
            }
        )

    return ready_documents


def _resolve_document(
    *,
    document_name: str,
    ready_documents: list[
        dict[str, str]
    ],
) -> dict[str, str]:
    normalized_name = (
        document_name.strip().casefold()
    )

    matches = [
        document
        for document in ready_documents
        if (
            document["name"]
            .strip()
            .casefold()
            == normalized_name
        )
    ]

    if not matches:
        raise RuntimeError(
            "No READY document named "
            f"{document_name!r} was found."
        )

    if len(matches) > 1:
        raise RuntimeError(
            "More than one READY document "
            f"is named {document_name!r}. "
            "Evaluation requires an unambiguous "
            "document filename."
        )

    return matches[0]


def _coerce_tutor_answer(
    value: Any,
) -> TutorAnswer:
    if isinstance(
        value,
        TutorAnswer,
    ):
        return value

    return TutorAnswer.model_validate(
        value
    )


def _coerce_context(
    value: Any,
) -> ContextBundle | None:
    if value is None:
        return None

    if isinstance(
        value,
        ContextBundle,
    ):
        return value

    return ContextBundle.model_validate(
        value
    )


def _as_string(
    value: Any,
) -> str | None:
    if value is None:
        return None

    enum_value = getattr(
        value,
        "value",
        None,
    )

    if isinstance(
        enum_value,
        str,
    ):
        return enum_value

    rendered = str(value).strip()

    return rendered or None


def _positive_int_tuple(
    value: Any,
) -> tuple[int, ...] | None:
    if not isinstance(
        value,
        (list, tuple),
    ):
        return None

    numbers: list[int] = []

    for item in value:
        try:
            number = int(item)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            number > 0
            and number not in numbers
        ):
            numbers.append(number)

    return (
        tuple(numbers)
        if numbers
        else None
    )


def _string_tuple(
    value: Any,
) -> tuple[str, ...] | None:
    if not isinstance(
        value,
        (list, tuple),
    ):
        return None

    values: list[str] = []

    for item in value:
        normalized = str(
            item
        ).strip()

        if (
            normalized
            and normalized not in values
        ):
            values.append(
                normalized
            )

    return (
        tuple(values)
        if values
        else None
    )


def _fallback_retrieval_context(
    *,
    result: dict[str, Any],
    question: str,
    user_id: str,
    document_id: str,
) -> ContextBundle:
    """
    Normally the exact Tutor context is available
    as result['reranked_context'].

    If an answer came from semantic cache, the graph
    may legitimately skip retrieval on that request.

    In that case this evaluation-only fallback runs
    the EXISTING RetrievalService against the same
    document so RAGAS can still evaluate current
    retrieval quality.

    No production retrieval logic is reimplemented.
    """

    retrieval_service = (
        get_retrieval_service()
    )

    rewritten_query = str(
        result.get(
            "rewritten_query",
            "",
        )
        or ""
    ).strip()

    retrieval_query = (
        rewritten_query
        or question.strip()
    )

    retrieval_result = (
        retrieval_service.retrieve(
            query=retrieval_query,
            user_id=user_id,
            document_id=document_id,
            preferred_page_numbers=(
                _positive_int_tuple(
                    result.get(
                        "preferred_page_numbers"
                    )
                )
            ),
            prefer_visual=bool(
                result.get(
                    "prefer_visual",
                    False,
                )
            ),
            required_chunk_ids=(
                _string_tuple(
                    result.get(
                        "structural_linked_retrieval_chunk_ids"
                    )
                )
            ),
            required_parent_ids=(
                _string_tuple(
                    result.get(
                        "structural_linked_parent_chunk_ids"
                    )
                )
            ),
        )
    )

    return retrieval_result.context


def _run_phymentor_sample(
    *,
    dataset_row: dict[str, Any],
    ready_documents: list[
        dict[str, str]
    ],
    user_id: str,
) -> tuple[
    Any,
    dict[str, Any],
]:
    sample_id = str(
        dataset_row["id"]
    ).strip()

    question = str(
        dataset_row["question"]
    ).strip()

    reference_answer = str(
        dataset_row[
            "reference_answer"
        ]
    ).strip()

    document_name = str(
        dataset_row[
            "document_name"
        ]
    ).strip()

    target_document = (
        _resolve_document(
            document_name=document_name,
            ready_documents=(
                ready_documents
            ),
        )
    )

    document_id = (
        target_document[
            "document_id"
        ]
    )

    request_id = (
        "ragas-"
        + uuid4().hex
    )

    session_id = (
        "ragas-"
        + sample_id
        + "-"
        + uuid4().hex[:12]
    )

    graph_input = {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "raw_query": question,

        # RAGAS is evaluating answer/retrieval quality,
        # not filename-resolution accuracy.
        #
        # Therefore the dataset's known ground-truth
        # document is supplied explicitly.
        "explicit_document_id": (
            document_id
        ),

        "available_documents": [
            {
                "document_id": (
                    document[
                        "document_id"
                    ]
                ),
                "name": (
                    document["name"]
                ),
            }
            for document
            in ready_documents
        ],

        "selected_page": None,
        "selected_figure_id": None,

        "requested_language": (
            LanguageCode.ENGLISH
        ),

        # None means normal PhyMentor automatic
        # model-routing policy.
        "selected_model": None,

        "upload_present": False,
    }

    checkpoint_config = {
        "configurable": {
            "thread_id": (
                "ragas-"
                + uuid4().hex
            ),
        },
    }

    result = chat_graph.invoke(
        graph_input,
        config=checkpoint_config,
    )

    raw_final_answer = (
        result.get(
            "final_answer"
        )
    )

    if raw_final_answer is None:
        raise RuntimeError(
            f"{sample_id}: graph returned "
            "no final_answer."
        )

    final_answer = (
        _coerce_tutor_answer(
            raw_final_answer
        )
    )

    context = _coerce_context(
        result.get(
            "reranked_context"
        )
    )

    cache_hit = bool(
        result.get(
            "cache_hit",
            False,
        )
    )

    context_source = (
        "graph_state"
    )

    # Only reconstruct retrieval context when the
    # graph legitimately skipped retrieval because
    # the answer came from cache.
    #
    # A genuine graph retrieval failure must remain
    # empty in the benchmark; otherwise evaluation
    # would hide a real production failure.
    if (
        (
            context is None
            or not context.items
        )
        and cache_hit
    ):
        active_document_id = str(
            result.get(
                "active_document_id",
                "",
            )
            or document_id
        ).strip()

        context = (
            _fallback_retrieval_context(
                result=result,
                question=question,
                user_id=user_id,
                document_id=(
                    active_document_id
                ),
            )
        )

        context_source = (
            "retrieval_fallback"
        )

    elif (
        context is None
        or not context.items
    ):
        context_source = (
            "graph_state_empty"
        )

    ragas_sample = (
        build_ragas_sample(
            question=question,
            answer=final_answer,
            context=context,
            reference_answer=(
                reference_answer
            ),
        )
    )

    metadata = {
        "id": sample_id,
        "category": str(
            dataset_row.get(
                "category",
                "",
            )
        ),
        "document_name": (
            document_name
        ),
        "document_id": (
            document_id
        ),
        "question": question,
        "reference_answer": (
            reference_answer
        ),
        "response": (
            ragas_sample.response
        ),
        "retrieved_contexts": (
            ragas_sample.retrieved_contexts
            or []
        ),
        "context_source": (
            context_source
        ),
        "retrieval_evidence_found": bool(
            ragas_sample.retrieved_contexts
        ),
        "insufficient_evidence": (
            _as_string(
                final_answer.answer_type
            )
            == "insufficient_evidence"
        ),
        "cache_hit": cache_hit,
        "active_document_id": (
            result.get(
                "active_document_id"
            )
        ),
        "terminal_action": (
            _as_string(
                result.get(
                    "terminal_action"
                )
            )
        ),
        "selected_model": (
            _as_string(
                result.get(
                    "selected_model"
                )
            )
        ),
        "retrieval_rounds": int(
            result.get(
                "retrieval_rounds",
                0,
            )
            or 0
        ),
        "generation_attempts": int(
            result.get(
                "generation_attempts",
                0,
            )
            or 0
        ),
    }

    return (
        ragas_sample,
        metadata,
    )


def _json_safe_score(
    value: Any,
) -> float | int | str | None:
    if value is None:
        return None

    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(item_method):
        try:
            value = item_method()
        except Exception:
            pass

    if isinstance(
        value,
        float,
    ):
        if math.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        int,
    ):
        return int(value)

    return str(value)


def _write_json(
    *,
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _write_csv(
    *,
    path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def _metric_averages(
    score_rows: list[
        dict[str, Any]
    ],
) -> dict[str, float]:
    values_by_metric: dict[
        str,
        list[float],
    ] = {}

    for row in score_rows:
        for key, raw_value in row.items():
            safe_value = (
                _json_safe_score(
                    raw_value
                )
            )

            if not isinstance(
                safe_value,
                (int, float),
            ):
                continue

            values_by_metric.setdefault(
                key,
                [],
            ).append(
                float(safe_value)
            )

    return {
        metric: (
            sum(values)
            / len(values)
        )
        for metric, values
        in values_by_metric.items()
        if values
    }


def _evaluation_health_summary(
    collected_rows: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    total_samples = len(
        collected_rows
    )

    retrieval_success_rows = [
        row
        for row in collected_rows
        if bool(
            row.get(
                "retrieval_evidence_found",
                False,
            )
        )
    ]

    retrieval_failure_rows = [
        row
        for row in collected_rows
        if not bool(
            row.get(
                "retrieval_evidence_found",
                False,
            )
        )
    ]

    insufficient_evidence_rows = [
        row
        for row in collected_rows
        if bool(
            row.get(
                "insufficient_evidence",
                False,
            )
        )
    ]

    retrieval_success_count = len(
        retrieval_success_rows
    )

    retrieval_failure_count = len(
        retrieval_failure_rows
    )

    retrieval_success_rate = (
        retrieval_success_count
        / total_samples
        if total_samples
        else 0.0
    )

    return {
        "total_samples": total_samples,
        "retrieval_success_count": (
            retrieval_success_count
        ),
        "retrieval_failure_count": (
            retrieval_failure_count
        ),
        "retrieval_success_rate": (
            retrieval_success_rate
        ),
        "insufficient_evidence_count": len(
            insufficient_evidence_rows
        ),
        "failed_sample_ids": [
            str(
                row.get(
                    "id",
                    "",
                )
            )
            for row
            in retrieval_failure_rows
        ],
        "insufficient_evidence_sample_ids": [
            str(
                row.get(
                    "id",
                    "",
                )
            )
            for row
            in insufficient_evidence_rows
        ],
    }


def _metric_valid_counts(
    *,
    score_rows: list[
        dict[str, Any]
    ],
    metric_names: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for metric_name in metric_names:
        valid_count = 0

        for row in score_rows:
            safe_value = (
                _json_safe_score(
                    row.get(
                        metric_name
                    )
                )
            )

            if (
                isinstance(
                    safe_value,
                    (int, float),
                )
                and not isinstance(
                    safe_value,
                    bool,
                )
            ):
                valid_count += 1

        counts[
            metric_name
        ] = valid_count

    return counts


def _print_health_summary(
    health_summary: dict[str, Any],
) -> None:
    total_samples = int(
        health_summary[
            "total_samples"
        ]
    )

    success_count = int(
        health_summary[
            "retrieval_success_count"
        ]
    )

    failure_count = int(
        health_summary[
            "retrieval_failure_count"
        ]
    )

    success_rate = float(
        health_summary[
            "retrieval_success_rate"
        ]
    )

    insufficient_count = int(
        health_summary[
            "insufficient_evidence_count"
        ]
    )

    print()
    print(
        "Evaluation collection health:"
    )
    print(
        "  Retrieval evidence found: "
        f"{success_count}/{total_samples} "
        f"({success_rate:.1%})"
    )
    print(
        "  Retrieval failures:",
        failure_count,
    )
    print(
        "  Insufficient-evidence answers:",
        insufficient_count,
    )

    failed_ids = (
        health_summary[
            "failed_sample_ids"
        ]
    )

    if failed_ids:
        print(
            "  Failed sample IDs:",
            ", ".join(
                failed_ids
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline RAGAS evaluation "
            "against the real PhyMentor graph."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            DEFAULT_DATASET_PATH
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_RESULTS_DIRECTORY
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N "
            "samples."
        ),
    )

    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "Run PhyMentor and collect "
            "answers/contexts without "
            "calling RAGAS evaluator LLMs."
        ),
    )

    args = parser.parse_args()

    dataset_path = (
        args.dataset.resolve()
    )

    output_directory = (
        args.output_dir.resolve()
    )

    dataset_rows = (
        _load_json(
            dataset_path
        )
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        dataset_rows = (
            dataset_rows[
                :args.limit
            ]
        )

    if not dataset_rows:
        raise RuntimeError(
            "No evaluation samples "
            "were loaded."
        )

    user_id = (
        settings
        .default_local_user_id
        .strip()
    )

    ready_documents = (
        _load_ready_documents(
            user_id=user_id
        )
    )

    if not ready_documents:
        raise RuntimeError(
            "No READY PhyMentor "
            "documents were found."
        )

    ragas_samples = []

    collected_rows: list[
        dict[str, Any]
    ] = []

    checkpoint_manager = (
        get_checkpoint_manager()
    )

    print(
        "Setting up LangGraph "
        "checkpointing..."
    )

    checkpoint_manager.setup()

    try:
        total = len(
            dataset_rows
        )

        for index, row in enumerate(
            dataset_rows,
            start=1,
        ):
            sample_id = str(
                row.get(
                    "id",
                    f"sample-{index}",
                )
            )

            print(
                f"[{index}/{total}] "
                f"Running {sample_id}..."
            )

            (
                ragas_sample,
                metadata,
            ) = _run_phymentor_sample(
                dataset_row=row,
                ready_documents=(
                    ready_documents
                ),
                user_id=user_id,
            )

            ragas_samples.append(
                ragas_sample
            )

            collected_rows.append(
                metadata
            )

    finally:
        checkpoint_manager.close()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    collected_path = (
        output_directory
        / "collected_samples.json"
    )

    _write_json(
        path=collected_path,
        payload=collected_rows,
    )

    print(
        "Collected samples saved to:",
        collected_path,
    )

    health_summary = (
        _evaluation_health_summary(
            collected_rows
        )
    )

    _print_health_summary(
        health_summary
    )

    collection_summary_path = (
        output_directory
        / "collection_summary.json"
    )

    _write_json(
        path=collection_summary_path,
        payload=health_summary,
    )

    if args.collect_only:
        print()
        print(
            "Collection summary saved to:",
            collection_summary_path,
        )
        print(
            "Collection complete. "
            "RAGAS scoring was skipped."
        )
        return

    evaluation_dataset = (
        EvaluationDataset(
            samples=ragas_samples
        )
    )

    metrics = (
        build_ragas_metrics()
    )

    print(
        "Running RAGAS with metrics:"
    )

    for metric in metrics:
        print(
            " -",
            metric.name,
        )

    evaluation_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=True,

        # Conservative batching avoids
        # unnecessary evaluator API bursts.
        batch_size=1,
    )

    raw_score_rows = (
        evaluation_result.scores
    )

    score_rows: list[
        dict[str, Any]
    ] = []

    detailed_rows: list[
        dict[str, Any]
    ] = []

    for metadata, raw_scores in zip(
        collected_rows,
        raw_score_rows,
        strict=True,
    ):
        safe_scores = {
            key: _json_safe_score(
                value
            )
            for key, value
            in raw_scores.items()
        }

        score_rows.append(
            safe_scores
        )

        detailed_rows.append(
            {
                **metadata,
                "scores": (
                    safe_scores
                ),
            }
        )

    averages = (
        _metric_averages(
            score_rows
        )
    )

    metric_names = [
        metric.name
        for metric in metrics
    ]

    metric_valid_counts = (
        _metric_valid_counts(
            score_rows=score_rows,
            metric_names=metric_names,
        )
    )

    metric_missing_counts = {
        metric_name: (
            len(
                ragas_samples
            )
            - metric_valid_counts[
                metric_name
            ]
        )
        for metric_name
        in metric_names
    }

    results_json_path = (
        output_directory
        / "ragas_results.json"
    )

    _write_json(
        path=results_json_path,
        payload=detailed_rows,
    )

    csv_rows = []

    for metadata, scores in zip(
        collected_rows,
        score_rows,
        strict=True,
    ):
        csv_rows.append(
            {
                "id": metadata["id"],
                "category": (
                    metadata["category"]
                ),
                "document_name": (
                    metadata[
                        "document_name"
                    ]
                ),
                "context_source": (
                    metadata[
                        "context_source"
                    ]
                ),
                "retrieval_evidence_found": (
                    metadata[
                        "retrieval_evidence_found"
                    ]
                ),
                "insufficient_evidence": (
                    metadata[
                        "insufficient_evidence"
                    ]
                ),
                "cache_hit": (
                    metadata[
                        "cache_hit"
                    ]
                ),
                **scores,
            }
        )

    results_csv_path = (
        output_directory
        / "ragas_results.csv"
    )

    _write_csv(
        path=results_csv_path,
        rows=csv_rows,
    )

    summary_path = (
        output_directory
        / "ragas_summary.json"
    )

    _write_json(
        path=summary_path,
        payload={
            "sample_count": len(
                ragas_samples
            ),
            "evaluator_model": (
                "gpt-4o"
            ),
            "retrieval": (
                health_summary
            ),
            "metric_averages": (
                averages
            ),
            "metric_valid_counts": (
                metric_valid_counts
            ),
            "metric_missing_counts": (
                metric_missing_counts
            ),
        },
    )

    print()
    print(
        "RAGAS evaluation complete."
    )

    print(
        "Samples:",
        len(ragas_samples),
    )

    print(
        "Average scores "
        "(valid RAGAS scores only):"
    )

    for metric in sorted(
        metric_names
    ):
        score = averages.get(
            metric
        )

        valid_count = (
            metric_valid_counts[
                metric
            ]
        )

        if score is None:
            print(
                f"  {metric}: "
                "no valid score "
                f"(0/{len(ragas_samples)})"
            )
            continue

        print(
            f"  {metric}: "
            f"{score:.4f} "
            f"({valid_count}/"
            f"{len(ragas_samples)} valid)"
        )

    print()
    print(
        "Detailed JSON:",
        results_json_path,
    )

    print(
        "CSV:",
        results_csv_path,
    )

    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()
