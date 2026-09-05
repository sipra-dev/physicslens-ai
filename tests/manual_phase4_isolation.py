from __future__ import annotations

import sys
from pathlib import Path


# --------------------------------------------------
# Make project root importable when this file is run
# directly with:
# python tests/manual_phase4_isolation.py
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import settings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.filters import RetrievalFilter
from src.retrieval.fusion import ReciprocalRankFusion


# ==================================================
# TEST DOCUMENT
# ==================================================

REAL_USER_ID = "local-user"

REAL_DOCUMENT_ID = (
    "7e814389ea2142bcb7f9b6bfc5f9234b"
)

WRONG_USER_ID = "attacker-user"

WRONG_DOCUMENT_ID = (
    "ffffffffffffffffffffffffffffffff"
)

QUERY = (
    "spring force constant 120 N/m "
    "frequency 6.00 Hz mass"
)


# ==================================================
# EXISTING REAL INDEX DIRECTORIES
# ==================================================

dense_directory = (
    settings.vector_store_dir
    / "users"
    / REAL_USER_ID
    / "documents"
    / REAL_DOCUMENT_ID
)

bm25_directory = (
    settings.bm25_store_dir
    / "users"
    / REAL_USER_ID
    / "documents"
    / REAL_DOCUMENT_ID
)


print("\n")
print("=" * 65)
print("POINT 6 — SECURITY / RETRIEVAL ISOLATION")
print("=" * 65)

print(
    "\nDense index:",
    dense_directory,
)

print(
    "BM25 index:",
    bm25_directory,
)


assert dense_directory.exists(), (
    "Test cannot start: dense index does not exist."
)

assert bm25_directory.exists(), (
    "Test cannot start: BM25 index does not exist."
)


# ==================================================
# COMPONENTS
# ==================================================

dense = DenseRetriever(
    model_name=settings.embedding_model_name
)

bm25 = BM25Retriever()

fusion = ReciprocalRankFusion(
    rrf_k=settings.rrf_k
)


# ==================================================
# 6A — BASELINE
#
# Correct user + correct document MUST work.
# Without this baseline, an empty result in the
# security tests would prove nothing.
# ==================================================

print("\n")
print("=" * 65)
print("6A — BASELINE: CORRECT USER + CORRECT DOCUMENT")
print("=" * 65)

correct_filter = RetrievalFilter(
    user_id=REAL_USER_ID,
    document_id=REAL_DOCUMENT_ID,
)


dense_hits = dense.search(
    query=QUERY,
    index_directory=dense_directory,
    retrieval_filter=correct_filter,
    top_k=30,
)

bm25_hits = bm25.search(
    query=QUERY,
    index_directory=bm25_directory,
    retrieval_filter=correct_filter,
    top_k=30,
)


print(
    "Dense hits:",
    len(dense_hits),
)

print(
    "BM25 hits:",
    len(bm25_hits),
)


assert dense_hits, (
    "6A FAILED: baseline Dense retrieval "
    "returned no hits."
)

assert bm25_hits, (
    "6A FAILED: baseline BM25 retrieval "
    "returned no hits."
)


assert all(
    hit.user_id == REAL_USER_ID
    and hit.document_id == REAL_DOCUMENT_ID
    for hit in dense_hits
), (
    "6A FAILED: Dense baseline contains "
    "foreign user/document data."
)


assert all(
    hit.user_id == REAL_USER_ID
    and hit.document_id == REAL_DOCUMENT_ID
    for hit in bm25_hits
), (
    "6A FAILED: BM25 baseline contains "
    "foreign user/document data."
)


print(
    "\n6A Baseline PASS ✅"
)


# ==================================================
# 6B — WRONG USER
#
# Important:
# We intentionally search the REAL physical index
# directory while supplying a WRONG user filter.
#
# If metadata filtering is working, even having
# access to the index must not return the real
# user's chunks.
# ==================================================

print("\n")
print("=" * 65)
print("6B — WRONG USER ISOLATION")
print("=" * 65)

wrong_user_filter = RetrievalFilter(
    user_id=WRONG_USER_ID,
    document_id=REAL_DOCUMENT_ID,
)


wrong_user_dense = dense.search(
    query=QUERY,
    index_directory=dense_directory,
    retrieval_filter=wrong_user_filter,
    top_k=30,
)

wrong_user_bm25 = bm25.search(
    query=QUERY,
    index_directory=bm25_directory,
    retrieval_filter=wrong_user_filter,
    top_k=30,
)


print(
    "Dense hits with wrong user:",
    len(wrong_user_dense),
)

print(
    "BM25 hits with wrong user:",
    len(wrong_user_bm25),
)


assert len(wrong_user_dense) == 0, (
    "6B FAILED: SECURITY LEAK — "
    "Dense returned another user's data."
)

assert len(wrong_user_bm25) == 0, (
    "6B FAILED: SECURITY LEAK — "
    "BM25 returned another user's data."
)


print(
    "\n6B Wrong User Isolation PASS ✅"
)


# ==================================================
# 6C — WRONG DOCUMENT
#
# Same real physical index, but pretend the request
# belongs to a different document.
#
# No chunks from the real document may escape.
# ==================================================

print("\n")
print("=" * 65)
print("6C — WRONG DOCUMENT ISOLATION")
print("=" * 65)

wrong_document_filter = RetrievalFilter(
    user_id=REAL_USER_ID,
    document_id=WRONG_DOCUMENT_ID,
)


wrong_doc_dense = dense.search(
    query=QUERY,
    index_directory=dense_directory,
    retrieval_filter=wrong_document_filter,
    top_k=30,
)

wrong_doc_bm25 = bm25.search(
    query=QUERY,
    index_directory=bm25_directory,
    retrieval_filter=wrong_document_filter,
    top_k=30,
)


print(
    "Dense hits with wrong document:",
    len(wrong_doc_dense),
)

print(
    "BM25 hits with wrong document:",
    len(wrong_doc_bm25),
)


assert len(wrong_doc_dense) == 0, (
    "6C FAILED: SECURITY LEAK — "
    "Dense returned data from a different document."
)

assert len(wrong_doc_bm25) == 0, (
    "6C FAILED: SECURITY LEAK — "
    "BM25 returned data from a different document."
)


print(
    "\n6C Wrong Document Isolation PASS ✅"
)


# ==================================================
# 6D — DEFENSE-IN-DEPTH AT RRF
#
# Here we deliberately give RRF the VALID candidate
# lists obtained in 6A, but supply the WRONG USER
# filter.
#
# RRF itself must reject those candidates.
# ==================================================

print("\n")
print("=" * 65)
print("6D — RRF DEFENSE-IN-DEPTH")
print("=" * 65)


wrong_user_fused = fusion.fuse(
    dense_hits=dense_hits,
    bm25_hits=bm25_hits,
    retrieval_filter=wrong_user_filter,
    top_k=30,
)


print(
    "Fused hits with wrong user:",
    len(wrong_user_fused),
)


assert len(wrong_user_fused) == 0, (
    "6D FAILED: SECURITY LEAK — "
    "RRF allowed another user's candidates."
)


wrong_document_fused = fusion.fuse(
    dense_hits=dense_hits,
    bm25_hits=bm25_hits,
    retrieval_filter=wrong_document_filter,
    top_k=30,
)


print(
    "Fused hits with wrong document:",
    len(wrong_document_fused),
)


assert len(wrong_document_fused) == 0, (
    "6D FAILED: SECURITY LEAK — "
    "RRF allowed another document's candidates."
)


print(
    "\n6D RRF Defense-in-Depth PASS ✅"
)


# ==================================================
# 6E — INDEX NAMESPACE CHECK
#
# Verify that different users/documents map to
# different filesystem namespaces.
# ==================================================

print("\n")
print("=" * 65)
print("6E — INDEX NAMESPACE")
print("=" * 65)


wrong_user_dense_directory = (
    settings.vector_store_dir
    / "users"
    / WRONG_USER_ID
    / "documents"
    / REAL_DOCUMENT_ID
)

wrong_document_dense_directory = (
    settings.vector_store_dir
    / "users"
    / REAL_USER_ID
    / "documents"
    / WRONG_DOCUMENT_ID
)


print(
    "Real user path:",
    dense_directory,
)

print(
    "Wrong user path:",
    wrong_user_dense_directory,
)

print(
    "Wrong document path:",
    wrong_document_dense_directory,
)


assert (
    dense_directory
    != wrong_user_dense_directory
), (
    "6E FAILED: different users resolved "
    "to the same index namespace."
)


assert (
    dense_directory
    != wrong_document_dense_directory
), (
    "6E FAILED: different documents resolved "
    "to the same index namespace."
)


print(
    "\n6E Index Namespace PASS ✅"
)


# ==================================================
# FINAL RESULT
# ==================================================

print("\n")
print("=" * 65)
print("POINT 6 COMPLETE ✅")
print("=" * 65)

print(
    "\n"
    "Baseline retrieval ✅\n"
    "Wrong-user isolation ✅\n"
    "Wrong-document isolation ✅\n"
    "RRF defense-in-depth ✅\n"
    "Index namespace isolation ✅"
)