from __future__ import annotations

import re
import sys
from pathlib import Path


# --------------------------------------------------
# Allow:
# python tests/manual_phase4_equation_retrieval.py
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import settings
from src.retrieval.service import RetrievalService


# ==================================================
# TEST DOCUMENT
# ==================================================

USER_ID = "local-user"

DOCUMENT_ID = (
    "7e814389ea2142bcb7f9b6bfc5f9234b"
)


# ==================================================
# REAL RETRIEVAL SERVICE
# ==================================================

service = RetrievalService(
    vector_store_directory=(
        settings.vector_store_dir
    ),
    bm25_store_directory=(
        settings.bm25_store_dir
    ),
    embedding_model_name=(
        settings.embedding_model_name
    ),
    reranker_model_name=(
        settings.reranker_model_name
    ),
    reranker_batch_size=(
        settings.reranker_batch_size
    ),
    max_context_characters=12000,
    max_item_characters=3000,
    minimum_rerank_score=0.0,
)


def normalize(value: str) -> str:
    """
    Remove whitespace differences so equations such as

        2 \pi
        2\pi

    can be compared safely.
    """

    return re.sub(
        r"\s+",
        "",
        value or "",
    )


def all_context_equations(result) -> list[str]:

    equations: list[str] = []

    for item in result.context.items:
        equations.extend(
            item.equations
        )

    return equations


def equation_exists(
    result,
    expected_fragment: str,
) -> bool:

    expected = normalize(
        expected_fragment
    )

    for equation in all_context_equations(
        result
    ):
        if expected in normalize(
            equation
        ):
            return True

    return False


print("\n")
print("=" * 70)
print("POINT 8 — EQUATION RETRIEVAL VALIDATION")
print("=" * 70)


# ==================================================
# 8A — EQUATION CHUNKS EXIST IN RETRIEVAL INDEX
# ==================================================

print("\n")
print("=" * 70)
print("8A — EQUATION CHUNK RETRIEVAL")
print("=" * 70)


query_1 = (
    "What is the restoring force equation "
    "for an ideal spring in simple harmonic motion?"
)


result_1 = service.retrieve(
    query=query_1,
    user_id=USER_ID,
    document_id=DOCUMENT_ID,
    dense_top_k=30,
    bm25_top_k=30,
    fused_top_k=30,
    rerank_top_k=8,
    max_contexts=6,
)


print(
    "Dense hits:",
    len(result_1.dense_hits),
)

print(
    "BM25 hits:",
    len(result_1.bm25_hits),
)

print(
    "Reranked hits:",
    len(result_1.reranked_hits),
)

print(
    "Context items:",
    len(result_1.context.items),
)


candidate_hits = (
    list(result_1.dense_hits)
    + list(result_1.bm25_hits)
)


equation_candidates = [
    hit
    for hit in candidate_hits
    if hit.content_type == "equation"
]


print(
    "Equation-type retrieval candidates:",
    len(equation_candidates),
)


for hit in equation_candidates[:5]:
    print(
        "\nEquation candidate:",
        hit.chunk_id,
    )
    print(
        "Page:",
        hit.page_number,
    )
    print(
        "Source:",
        hit.retrieval_source,
    )
    print(
        "Text:",
        hit.text[:400],
    )


assert result_1.evidence_found, (
    "8A FAILED: equation question "
    "returned no evidence."
)


assert equation_candidates, (
    "8A FAILED: no equation-type chunk "
    "was retrieved by Dense or BM25."
)


print(
    "\n8A Equation Chunk Retrieval PASS ✅"
)


# ==================================================
# 8B — RESTORING FORCE EQUATION MUST SURVIVE
#      INTO FINAL CONTEXT
# ==================================================

print("\n")
print("=" * 70)
print("8B — RESTORING FORCE EQUATION")
print("=" * 70)


expected_restoring_force = (
    r"F_{restoring} = -kx"
)


equations_1 = all_context_equations(
    result_1
)


print(
    "Final context equations:"
)

for equation in equations_1:
    print(
        " -",
        equation,
    )


assert equations_1, (
    "8B FAILED: final context contains "
    "no equation metadata."
)


assert equation_exists(
    result_1,
    expected_restoring_force,
), (
    "8B FAILED: F_restoring = -kx "
    "was not preserved in final context."
)


print(
    "\n8B Restoring Force Equation PASS ✅"
)


# ==================================================
# 8C — EQUATION MUST ALSO BE PRESENT IN
#      GROUNDED CONTEXT TEXT
# ==================================================

print("\n")
print("=" * 70)
print("8C — EQUATION IN GROUNDED TEXT")
print("=" * 70)


combined_context_text_1 = " ".join(
    item.text
    for item in result_1.context.items
)


assert normalize(
    expected_restoring_force
) in normalize(
    combined_context_text_1
), (
    "8C FAILED: equation exists in metadata "
    "but is missing from grounded context text."
)


print(
    "Found:",
    expected_restoring_force,
)

print(
    "\n8C Grounded Equation Text PASS ✅"
)


# ==================================================
# 8D — SEMANTIC EQUATION QUERY
#
# User does NOT type the formula.
# Retrieval must still find the equation.
# ==================================================

print("\n")
print("=" * 70)
print("8D — SEMANTIC SPRING EQUATION QUERY")
print("=" * 70)


query_2 = (
    "How does the angular frequency of "
    "a mass on an ideal spring depend on "
    "the spring constant and mass?"
)


result_2 = service.retrieve(
    query=query_2,
    user_id=USER_ID,
    document_id=DOCUMENT_ID,
    dense_top_k=30,
    bm25_top_k=30,
    fused_top_k=30,
    rerank_top_k=8,
    max_contexts=6,
)


print(
    "Evidence found:",
    result_2.evidence_found,
)

print(
    "Reranked hits:",
    len(result_2.reranked_hits),
)

print(
    "Context items:",
    len(result_2.context.items),
)


assert result_2.evidence_found, (
    "8D FAILED: semantic equation query "
    "returned no evidence."
)


expected_spring_omega = (
    r"\omega = \sqrt{\frac{k}{m}}"
)


equations_2 = all_context_equations(
    result_2
)


print(
    "\nRetrieved context equations:"
)

for equation in equations_2:
    print(
        " -",
        equation,
    )


assert equation_exists(
    result_2,
    expected_spring_omega,
), (
    "8D FAILED: spring angular-frequency "
    "equation was not recovered."
)


print(
    "\n8D Semantic Equation Retrieval PASS ✅"
)


# ==================================================
# 8E — PAGE / DOCUMENT PROVENANCE
# ==================================================

print("\n")
print("=" * 70)
print("8E — EQUATION PROVENANCE")
print("=" * 70)


contexts_with_spring_equation = []

for item in result_2.context.items:

    joined_equations = " ".join(
        item.equations
    )

    if normalize(
        expected_spring_omega
    ) in normalize(
        joined_equations
    ):
        contexts_with_spring_equation.append(
            item
        )


assert contexts_with_spring_equation, (
    "8E FAILED: could not identify "
    "the source context for the equation."
)


for item in contexts_with_spring_equation:

    print(
        "\nContext ID:",
        item.context_id,
    )

    print(
        "User ID:",
        item.user_id,
    )

    print(
        "Document ID:",
        item.document_id,
    )

    print(
        "Page:",
        item.page_number,
    )

    print(
        "Parent ID:",
        item.parent_id,
    )

    print(
        "Source chunks:",
        item.source_chunk_ids,
    )


assert all(
    item.user_id == USER_ID
    for item in contexts_with_spring_equation
), (
    "8E FAILED: wrong user provenance."
)


assert all(
    item.document_id == DOCUMENT_ID
    for item in contexts_with_spring_equation
), (
    "8E FAILED: wrong document provenance."
)


assert any(
    item.page_number == 2
    for item in contexts_with_spring_equation
), (
    "8E FAILED: spring angular-frequency "
    "equation did not resolve to page 2."
)


print(
    "\n8E Equation Provenance PASS ✅"
)


# ==================================================
# 8F — EQUATION MUST NOT BE LOST DURING
#      CHILD → PARENT EXPANSION
# ==================================================

print("\n")
print("=" * 70)
print("8F — EQUATION THROUGH PARENT EXPANSION")
print("=" * 70)


parent_equation_contexts = [
    item
    for item in result_2.context.items
    if (
        item.context_id.startswith(
            "parent:"
        )
        and item.equations
    )
]


print(
    "Parent contexts containing equations:",
    len(parent_equation_contexts),
)


for item in parent_equation_contexts:

    print(
        "\n",
        item.context_id,
    )

    for equation in item.equations:
        print(
            "  -",
            equation,
        )


assert parent_equation_contexts, (
    "8F FAILED: equations disappeared "
    "during parent expansion."
)


assert any(
    equation_exists(
        result_2,
        expected_spring_omega,
    )
    for _ in [0]
), (
    "8F FAILED: expected equation did "
    "not survive parent expansion."
)


print(
    "\n8F Parent Equation Preservation PASS ✅"
)


# ==================================================
# FINAL
# ==================================================

print("\n")
print("=" * 70)
print("POINT 8 COMPLETE ✅")
print("=" * 70)

print(
    "\n"
    "Equation chunk retrieval ✅\n"
    "Exact restoring-force equation ✅\n"
    "Equation present in grounded text ✅\n"
    "Semantic equation retrieval ✅\n"
    "Equation provenance ✅\n"
    "Equation survives parent expansion ✅"
)