from __future__ import annotations

import json
import sys
from pathlib import Path


# --------------------------------------------------
# Allow direct execution from tests/
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
# TEST DATA
# ==================================================

USER_ID = "local-user"

DOCUMENT_ID = (
    "7e814389ea2142bcb7f9b6bfc5f9234b"
)

QUERY = "What is Simple Harmonic Motion?"


# ==================================================
# CREATE REAL RETRIEVAL SERVICE
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


print("\n")
print("=" * 65)
print("POINT 7 — PARENT CONTEXT VALIDATION")
print("=" * 65)


# ==================================================
# RUN THE REAL END-TO-END RETRIEVAL PIPELINE
# ==================================================

result = service.retrieve(
    query=QUERY,
    user_id=USER_ID,
    document_id=DOCUMENT_ID,
    dense_top_k=30,
    bm25_top_k=30,
    fused_top_k=30,
    rerank_top_k=8,
    max_contexts=6,
)


# ==================================================
# 7A — RETRIEVAL MUST FIND EVIDENCE
# ==================================================

print("\n")
print("=" * 65)
print("7A — RETRIEVAL EVIDENCE")
print("=" * 65)


print(
    "Reranked hits:",
    len(result.reranked_hits),
)

print(
    "Final contexts:",
    len(result.context.items),
)

print(
    "Evidence found:",
    result.evidence_found,
)


assert result.evidence_found, (
    "7A FAILED: retrieval found no evidence."
)

assert result.reranked_hits, (
    "7A FAILED: no reranked hits."
)

assert result.context.items, (
    "7A FAILED: no final context."
)


print(
    "\n7A Retrieval Evidence PASS ✅"
)


# ==================================================
# 7B — A CHILD HIT MUST HAVE A PARENT
# ==================================================

print("\n")
print("=" * 65)
print("7B — CHILD → PARENT LINK")
print("=" * 65)


child_hits = [
    item
    for item in result.reranked_hits
    if (
        item.hit.chunk_kind == "child"
        and item.hit.parent_id
    )
]


print(
    "Child hits with parent_id:",
    len(child_hits),
)


for item in child_hits:
    print(
        "\nChild:",
        item.hit.chunk_id,
    )
    print(
        "Parent:",
        item.hit.parent_id,
    )
    print(
        "Rerank score:",
        item.rerank_score,
    )


assert child_hits, (
    "7B FAILED: no reranked child hit "
    "with a parent_id was found."
)


print(
    "\n7B Child → Parent Link PASS ✅"
)


# ==================================================
# 7C — FINAL CONTEXT MUST EXPAND TO PARENT
# ==================================================

print("\n")
print("=" * 65)
print("7C — PARENT EXPANSION")
print("=" * 65)


parent_contexts = [
    item
    for item in result.context.items
    if item.context_id.startswith("parent:")
]


print(
    "Parent contexts:",
    len(parent_contexts),
)


for context in parent_contexts:
    print(
        "\nContext ID:",
        context.context_id,
    )
    print(
        "Parent ID:",
        context.parent_id,
    )
    print(
        "Page:",
        context.page_number,
    )
    print(
        "Source chunks:",
        context.source_chunk_ids,
    )
    print(
        "Linked figures:",
        context.linked_figure_ids,
    )
    print(
        "Equations:",
        context.equations,
    )
    print(
        "Text preview:",
        context.text[:500],
    )


assert parent_contexts, (
    "7C FAILED: no expanded parent "
    "context was produced."
)


matching_pair = None

for child in child_hits:
    for context in parent_contexts:
        if (
            context.parent_id
            == child.hit.parent_id
            and child.hit.chunk_id
            in context.source_chunk_ids
        ):
            matching_pair = (
                child,
                context,
            )
            break

    if matching_pair is not None:
        break


assert matching_pair is not None, (
    "7C FAILED: retrieved child was not "
    "expanded into its parent context."
)


matched_child, matched_context = (
    matching_pair
)


print(
    "\nMatched child:",
    matched_child.hit.chunk_id,
)

print(
    "Expanded parent:",
    matched_context.parent_id,
)


assert (
    matched_context.context_id
    == f"parent:{matched_context.parent_id}"
), (
    "7C FAILED: parent context_id "
    "has the wrong format."
)


print(
    "\n7C Parent Expansion PASS ✅"
)


# ==================================================
# LOAD STORED PARENT DATA
# ==================================================

parent_chunks_path = (
    settings.vector_store_dir
    / "users"
    / USER_ID
    / "documents"
    / DOCUMENT_ID
    / "parent_chunks.json"
)


assert parent_chunks_path.is_file(), (
    "Stored parent_chunks.json does not exist."
)


with parent_chunks_path.open(
    "r",
    encoding="utf-8",
) as file:
    parent_payload = json.load(file)


raw_parents = (
    parent_payload.get("parents")
    or parent_payload.get("parent_chunks")
    or []
)


parents_by_id = {
    parent["parent_id"]: parent
    for parent in raw_parents
}


stored_parent = parents_by_id.get(
    matched_context.parent_id
)


assert stored_parent is not None, (
    "Stored parent record was not found."
)


# ==================================================
# 7D — ALL PARENT CHILD IDS MUST BE PRESERVED
# ==================================================

print("\n")
print("=" * 65)
print("7D — PARENT SOURCE CHUNKS")
print("=" * 65)


stored_child_ids = (
    stored_parent.get("child_ids", [])
)


print(
    "Stored parent child IDs:",
    stored_child_ids,
)

print(
    "Context source chunk IDs:",
    matched_context.source_chunk_ids,
)


assert stored_child_ids, (
    "7D FAILED: stored parent has "
    "no child IDs."
)


assert (
    set(stored_child_ids)
    == set(
        matched_context.source_chunk_ids
    )
), (
    "7D FAILED: final parent context did "
    "not preserve the parent's child IDs."
)


assert len(
    matched_context.source_chunk_ids
) > 1, (
    "7D FAILED: context looks like only "
    "the retrieved child, not the parent."
)


print(
    "\n7D Parent Source Chunks PASS ✅"
)


# ==================================================
# 7E — PARENT EQUATIONS + FIGURES MUST SURVIVE
# ==================================================

print("\n")
print("=" * 65)
print("7E — LINKED MULTIMODAL METADATA")
print("=" * 65)


stored_equations = (
    stored_parent.get("equations", [])
)

stored_figures = (
    stored_parent.get("figures", [])
)


print(
    "Stored equations:",
    stored_equations,
)

print(
    "Context equations:",
    matched_context.equations,
)

print(
    "Stored figures:",
    stored_figures,
)

print(
    "Context linked figures:",
    matched_context.linked_figure_ids,
)


assert (
    set(stored_equations)
    == set(matched_context.equations)
), (
    "7E FAILED: parent equations were "
    "not preserved in final context."
)


assert set(
    stored_figures
).issubset(
    set(
        matched_context.linked_figure_ids
    )
), (
    "7E FAILED: linked parent figures "
    "were lost during expansion."
)


print(
    "\n7E Multimodal Metadata PASS ✅"
)


# ==================================================
# 7F — PARENT CONTEXT MUST BE RICHER THAN CHILD
# ==================================================

print("\n")
print("=" * 65)
print("7F — CONTEXT RICHNESS")
print("=" * 65)


child_text = (
    matched_child.hit.text.strip()
)

parent_context_text = (
    matched_context.text.strip()
)


print(
    "Retrieved child characters:",
    len(child_text),
)

print(
    "Final parent context characters:",
    len(parent_context_text),
)


assert parent_context_text, (
    "7F FAILED: parent context text is empty."
)


assert (
    len(matched_context.source_chunk_ids)
    > 1
), (
    "7F FAILED: parent expansion did "
    "not provide broader source coverage."
)


assert (
    matched_child.hit.chunk_id
    in matched_context.source_chunk_ids
), (
    "7F FAILED: original retrieved child "
    "lost its provenance."
)


print(
    "\n7F Context Richness PASS ✅"
)


# ==================================================
# FINAL
# ==================================================

print("\n")
print("=" * 65)
print("POINT 7 COMPLETE ✅")
print("=" * 65)

print(
    "\n"
    "Retrieval evidence ✅\n"
    "Child → parent link ✅\n"
    "Parent expansion ✅\n"
    "Parent child provenance ✅\n"
    "Equations / figures preserved ✅\n"
    "Broader parent context ✅"
)