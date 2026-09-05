from __future__ import annotations

import ast
import inspect
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import settings
from src.retrieval.fusion import (
    ReciprocalRankFusion,
)
from src.retrieval.service import (
    RetrievalService,
    RetrievalServiceError,
)


class Phase4RetrievalContractTests(
    unittest.TestCase
):

    def _bare_service(
        self,
        root: Path,
    ) -> RetrievalService:
        """
        Build a RetrievalService object without
        loading SentenceTransformer/CrossEncoder.

        These tests only need deterministic
        path/config/helper behaviour.
        """

        service = RetrievalService.__new__(
            RetrievalService
        )

        service.vector_store_directory = (
            root / "vector_store"
        ).resolve()

        service.bm25_store_directory = (
            root / "bm25_store"
        ).resolve()

        return service

    def test_phase4_settings_are_valid(
        self,
    ) -> None:
        self.assertGreater(
            settings.rrf_k,
            0,
        )

        self.assertGreater(
            settings.hybrid_candidate_pool_size,
            0,
        )

        self.assertGreater(
            settings.retrieval_per_source_top_k,
            0,
        )

        self.assertGreater(
            settings.reranker_top_k,
            0,
        )

        self.assertLessEqual(
            settings.reranker_top_k,
            8,
        )

        self.assertGreater(
            settings.final_context_count,
            0,
        )

        self.assertLessEqual(
            settings.final_context_count,
            settings.reranker_top_k,
        )

    def test_rrf_k_is_configurable(
        self,
    ) -> None:
        fusion = ReciprocalRankFusion(
            rrf_k=17,
        )

        self.assertEqual(
            fusion.rrf_k,
            17,
        )

    def test_service_constructor_exposes_rrf_k(
        self,
    ) -> None:
        signature = inspect.signature(
            RetrievalService.__init__
        )

        self.assertIn(
            "rrf_k",
            signature.parameters,
        )

        self.assertEqual(
            signature.parameters[
                "rrf_k"
            ].default,
            60,
        )

    def test_shared_runtime_wires_settings_rrf_k(
        self,
    ) -> None:
        """
        RetrievalService is created in the shared runtime
        layer rather than independently inside the API route.

        Verify that central settings.rrf_k is passed to that
        shared service constructor.
        """

        runtime_path = (
            PROJECT_ROOT
            / "src"
            / "runtime_services.py"
        )

        source = runtime_path.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(source)

        found_rrf_wiring = False

        for node in ast.walk(tree):
            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            if not (
                isinstance(
                    node.func,
                    ast.Name,
                )
                and node.func.id
                == "RetrievalService"
            ):
                continue

            for keyword in node.keywords:
                if keyword.arg != "rrf_k":
                    continue

                value = keyword.value

                if (
                    isinstance(
                        value,
                        ast.Attribute,
                    )
                    and value.attr == "rrf_k"
                    and isinstance(
                        value.value,
                        ast.Name,
                    )
                    and value.value.id
                    == "settings"
                ):
                    found_rrf_wiring = True

        self.assertTrue(
            found_rrf_wiring,
            (
                "The shared RetrievalService must receive "
                "rrf_k=settings.rrf_k."
            ),
        )

    def test_document_paths_are_isolated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            service = self._bare_service(
                root
            )

            (
                dense_a,
                bm25_a,
                parent_a,
            ) = service._document_paths(
                user_id="user-a",
                document_id="doc-a",
            )

            (
                dense_b,
                bm25_b,
                parent_b,
            ) = service._document_paths(
                user_id="user-b",
                document_id="doc-a",
            )

            self.assertNotEqual(
                dense_a,
                dense_b,
            )

            self.assertNotEqual(
                bm25_a,
                bm25_b,
            )

            self.assertNotEqual(
                parent_a,
                parent_b,
            )

            self.assertEqual(
                dense_a,
                (
                    service
                    .vector_store_directory
                    / "users"
                    / "user-a"
                    / "documents"
                    / "doc-a"
                ),
            )

            self.assertEqual(
                bm25_a,
                (
                    service
                    .bm25_store_directory
                    / "users"
                    / "user-a"
                    / "documents"
                    / "doc-a"
                ),
            )

            self.assertEqual(
                parent_a,
                dense_a
                / "parent_chunks.json",
            )

    def test_unsafe_path_components_cannot_escape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            service = self._bare_service(
                root
            )

            (
                dense_path,
                bm25_path,
                _,
            ) = service._document_paths(
                user_id="../../outside",
                document_id="../secret",
            )

            self.assertIn(
                service.vector_store_directory,
                dense_path.parents,
            )

            self.assertIn(
                service.bm25_store_directory,
                bm25_path.parents,
            )

            self.assertNotIn(
                "..",
                dense_path.parts,
            )

            self.assertNotIn(
                "..",
                bm25_path.parts,
            )

    def test_missing_index_files_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            service = self._bare_service(
                root
            )

            (
                dense_directory,
                bm25_directory,
                _,
            ) = service._document_paths(
                user_id="user-a",
                document_id="doc-a",
            )

            dense_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            bm25_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.assertRaises(
                RetrievalServiceError
            ):
                service._validate_index_files(
                    dense_directory=(
                        dense_directory
                    ),
                    bm25_directory=(
                        bm25_directory
                    ),
                )

    def test_complete_index_layout_is_accepted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            service = self._bare_service(
                root
            )

            (
                dense_directory,
                bm25_directory,
                _,
            ) = service._document_paths(
                user_id="user-a",
                document_id="doc-a",
            )

            dense_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            bm25_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                dense_directory
                / "dense.faiss"
            ).write_bytes(b"test")

            (
                dense_directory
                / "dense_metadata.json"
            ).write_text(
                "{}",
                encoding="utf-8",
            )

            (
                bm25_directory
                / "bm25_corpus.json"
            ).write_text(
                "{}",
                encoding="utf-8",
            )

            # Must NOT raise.
            service._validate_index_files(
                dense_directory=(
                    dense_directory
                ),
                bm25_directory=(
                    bm25_directory
                ),
            )

    def test_empty_result_preserves_scope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._bare_service(
                Path(tmp)
            )

            result = service._empty_result(
                query="test query",
                user_id="user-a",
                document_id="doc-a",
                reason="TEST_REASON",
            )

            self.assertFalse(
                result.evidence_found
            )

            self.assertEqual(
                result.failure_reason,
                "TEST_REASON",
            )

            self.assertEqual(
                result.context.user_id,
                "user-a",
            )

            self.assertEqual(
                result.context.document_id,
                "doc-a",
            )

            self.assertEqual(
                result.context.items,
                [],
            )


if __name__ == "__main__":
    unittest.main()