from __future__ import annotations

import sys
from pathlib import Path


# --------------------------------------------------
# Allow direct execution:
# python tests/manual_phase4_visual_smoke.py
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
# REAL DOCUMENTS ALREADY INGESTED
# ==================================================

USER_ID = "local-user"


# Diagram-heavy image:
# velocity + acceleration signs
DIAGRAM_DOCUMENT_ID = (
    "13deb748922e4c1db47924380ae70c76"
)

DIAGRAM_QUERY = (
    "When does an object speed up and when "
    "does it slow down based on the signs of "
    "velocity and acceleration?"
)


# Handwritten Physics image:
# acceleration definition + formula + units
HANDWRITTEN_DOCUMENT_ID = (
    "c027fd8b35bf40719d73c3ff008c4441"
)

HANDWRITTEN_QUERY = (
    "What is acceleration and what are its units? "
    "How is acceleration calculated from change "
    "in velocity and change in time?"
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


def visual_reranked_hits(result):
    return [
        item
        for item in result.reranked_hits
        if item.hit.chunk_kind == "visual"
    ]


def visual_contexts(result):
    return [
        item
        for item in result.context.items
        if item.context_id.startswith("visual:")
    ]


print("\n")
print("=" * 70)
print("POINT 9 — VISUAL / IMAGE SMOKE TEST")
print("=" * 70)


# ==================================================
# 9A — DIAGRAM IMAGE RETRIEVAL
# ==================================================

print("\n")
print("=" * 70)
print("9A — DIAGRAM VISUAL RETRIEVAL")
print("=" * 70)


diagram_result = service.retrieve(
    query=DIAGRAM_QUERY,
    user_id=USER_ID,
    document_id=DIAGRAM_DOCUMENT_ID,
    dense_top_k=30,
    bm25_top_k=30,
    fused_top_k=30,
    rerank_top_k=8,
    max_contexts=6,
    include_visual=True,
)


print(
    "Evidence found:",
    diagram_result.evidence_found,
)

print(
    "Dense hits:",
    len(diagram_result.dense_hits),
)

print(
    "Reranked hits:",
    len(diagram_result.reranked_hits),
)

print(
    "Final contexts:",
    len(diagram_result.context.items),
)


assert diagram_result.evidence_found, (
    "9A FAILED: diagram query found no evidence."
)


diagram_visual_hits = visual_reranked_hits(
    diagram_result
)


print(
    "Visual reranked hits:",
    len(diagram_visual_hits),
)


assert diagram_visual_hits, (
    "9A FAILED: no visual chunk survived reranking."
)


print(
    "\n9A Diagram Visual Retrieval PASS ✅"
)


# ==================================================
# 9B — DIAGRAM SEMANTICS MUST BE PRESERVED
# ==================================================

print("\n")
print("=" * 70)
print("9B — DIAGRAM SEMANTIC CONTENT")
print("=" * 70)


diagram_contexts = visual_contexts(
    diagram_result
)


assert diagram_contexts, (
    "9B FAILED: no visual context was produced."
)


diagram_text = " ".join(
    (
        item.caption or ""
    )
    + " "
    + item.text
    for item in diagram_contexts
).lower()


required_diagram_phrases = [
    "speeding up",
    "slowing down",
    "same signs",
    "opposite signs",
    "velocity",
    "acceleration",
]


for phrase in required_diagram_phrases:

    print(
        "Checking:",
        phrase,
    )

    assert phrase in diagram_text, (
        f"9B FAILED: visual evidence is missing "
        f"semantic phrase: {phrase}"
    )


print(
    "\n9B Diagram Semantics PASS ✅"
)


# ==================================================
# 9C — IMAGE PATH + CAPTION PROVENANCE
# ==================================================

print("\n")
print("=" * 70)
print("9C — IMAGE PROVENANCE")
print("=" * 70)


for item in diagram_contexts:

    print(
        "\nContext:",
        item.context_id,
    )

    print(
        "Document:",
        item.document_id,
    )

    print(
        "Page:",
        item.page_number,
    )

    print(
        "Image path:",
        item.image_path,
    )

    print(
        "Caption exists:",
        bool(item.caption),
    )


assert all(
    item.user_id == USER_ID
    for item in diagram_contexts
), (
    "9C FAILED: wrong user provenance."
)


assert all(
    item.document_id
    == DIAGRAM_DOCUMENT_ID
    for item in diagram_contexts
), (
    "9C FAILED: wrong document provenance."
)


assert all(
    item.image_path
    for item in diagram_contexts
), (
    "9C FAILED: visual context lost image_path."
)


assert all(
    item.caption
    for item in diagram_contexts
), (
    "9C FAILED: visual context lost caption."
)


for item in diagram_contexts:

    image_file = Path(
        item.image_path
    )

    assert image_file.is_file(), (
        "9C FAILED: stored visual image file "
        f"does not exist: {image_file}"
    )


print(
    "\n9C Image Provenance PASS ✅"
)


# ==================================================
# 9D — HANDWRITTEN IMAGE RETRIEVAL
# ==================================================

print("\n")
print("=" * 70)
print("9D — HANDWRITTEN IMAGE RETRIEVAL")
print("=" * 70)


handwritten_result = service.retrieve(
    query=HANDWRITTEN_QUERY,
    user_id=USER_ID,
    document_id=HANDWRITTEN_DOCUMENT_ID,
    dense_top_k=30,
    bm25_top_k=30,
    fused_top_k=30,
    rerank_top_k=8,
    max_contexts=6,
    include_visual=True,
)


print(
    "Evidence found:",
    handwritten_result.evidence_found,
)

print(
    "Reranked hits:",
    len(handwritten_result.reranked_hits),
)

print(
    "Final contexts:",
    len(handwritten_result.context.items),
)


assert handwritten_result.evidence_found, (
    "9D FAILED: handwritten-image query "
    "found no evidence."
)


handwritten_visual_hits = visual_reranked_hits(
    handwritten_result
)


assert handwritten_visual_hits, (
    "9D FAILED: handwritten visual chunk "
    "did not survive reranking."
)


handwritten_contexts = visual_contexts(
    handwritten_result
)


assert handwritten_contexts, (
    "9D FAILED: no handwritten visual "
    "context was produced."
)


print(
    "\n9D Handwritten Image Retrieval PASS ✅"
)


# ==================================================
# 9E — HANDWRITTEN PHYSICS SEMANTICS
# ==================================================

print("\n")
print("=" * 70)
print("9E — HANDWRITTEN PHYSICS SEMANTICS")
print("=" * 70)


handwritten_text = " ".join(
    (
        item.caption or ""
    )
    + " "
    + item.text
    for item in handwritten_contexts
).lower()


required_handwritten_phrases = [
    "acceleration",
    "change in velocity",
    "change in time",
]


for phrase in required_handwritten_phrases:

    print(
        "Checking:",
        phrase,
    )

    assert phrase in handwritten_text, (
        f"9E FAILED: handwritten visual "
        f"evidence is missing: {phrase}"
    )


# Accept common representations of m/s².
unit_present = any(
    value in handwritten_text
    for value in [
        "m/s^2",
        "m/s²",
        "meters per second squared",
        "metres per second squared",
    ]
)


assert unit_present, (
    "9E FAILED: acceleration unit "
    "was not preserved."
)


print(
    "\n9E Handwritten Semantics PASS ✅"
)


# ==================================================
# 9F — VISUAL DISABLE SWITCH
#
# When include_visual=False, no visual chunk
# should escape through retrieval/context.
# ==================================================

print("\n")
print("=" * 70)
print("9F — INCLUDE_VISUAL = FALSE")
print("=" * 70)


visual_disabled_result = service.retrieve(
    query=DIAGRAM_QUERY,
    user_id=USER_ID,
    document_id=DIAGRAM_DOCUMENT_ID,
    dense_top_k=30,
    bm25_top_k=30,
    fused_top_k=30,
    rerank_top_k=8,
    max_contexts=6,
    include_visual=False,
)


disabled_dense_visuals = [
    hit
    for hit in visual_disabled_result.dense_hits
    if hit.chunk_kind == "visual"
]

disabled_bm25_visuals = [
    hit
    for hit in visual_disabled_result.bm25_hits
    if hit.chunk_kind == "visual"
]

disabled_reranked_visuals = [
    item
    for item in visual_disabled_result.reranked_hits
    if item.hit.chunk_kind == "visual"
]

disabled_context_visuals = [
    item
    for item in visual_disabled_result.context.items
    if item.context_id.startswith("visual:")
]


print(
    "Dense visual hits:",
    len(disabled_dense_visuals),
)

print(
    "BM25 visual hits:",
    len(disabled_bm25_visuals),
)

print(
    "Reranked visual hits:",
    len(disabled_reranked_visuals),
)

print(
    "Visual contexts:",
    len(disabled_context_visuals),
)


assert not disabled_dense_visuals, (
    "9F FAILED: Dense returned visual data "
    "despite include_visual=False."
)

assert not disabled_bm25_visuals, (
    "9F FAILED: BM25 returned visual data "
    "despite include_visual=False."
)

assert not disabled_reranked_visuals, (
    "9F FAILED: visual evidence escaped "
    "into reranking."
)

assert not disabled_context_visuals, (
    "9F FAILED: visual evidence escaped "
    "into final context."
)


print(
    "\n9F Visual Disable Switch PASS ✅"
)


# ==================================================
# FINAL
# ==================================================

print("\n")
print("=" * 70)
print("POINT 9 COMPLETE ✅")
print("=" * 70)

print(
    "\n"
    "Diagram visual retrieval ✅\n"
    "Diagram semantics ✅\n"
    "Image/caption provenance ✅\n"
    "Handwritten image retrieval ✅\n"
    "Handwritten Physics semantics ✅\n"
    "Visual include/exclude control ✅"
)