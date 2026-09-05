from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)

from src.ingestion.models import (
    RetrievalChunk,
)
from src.retrieval.filters import (
    RetrievalFilter,
    chunk_matches_filter,
)
from src.retrieval.models import (
    RetrievalHit,
)


class DenseIndexError(Exception):
    pass


class DenseRetriever:
    """
    Local dense retrieval using:
    SentenceTransformers + FAISS cosine similarity.
    """

    def __init__(
        self,
        *,
        model_name: str,
    ) -> None:
        self.model_name = model_name

        self._model: (
            SentenceTransformer | None
        ) = None

    @property
    def model(
        self,
    ) -> SentenceTransformer:
        """
        Lazy-load embedding model.

        Model ekbar load hole same instance reuse hobe.
        """

        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name
            )

        return self._model

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Create one normalized embedding.

        Retrieval and semantic cache dujonei
        same loaded embedding model reuse korte parbe.
        """

        text = text.strip()

        if not text:
            raise DenseIndexError(
                "Cannot embed empty text."
            )

        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if (
            embedding.ndim != 2
            or embedding.shape[0] != 1
        ):
            raise DenseIndexError(
                "Embedding model returned "
                "an invalid embedding."
            )

        return embedding[0]

    def build(
        self,
        *,
        chunks: list[RetrievalChunk],
        index_directory: Path,
    ) -> None:
        if not chunks:
            raise DenseIndexError(
                "Cannot build a dense index "
                "without chunks."
            )

        index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise DenseIndexError(
                "Embedding model returned an "
                "invalid embedding matrix."
            )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )

        faiss.write_index(
            index,
            str(
                index_directory
                / "dense.faiss"
            ),
        )

        metadata = {
            "model_name": self.model_name,
            "dimension": dimension,
            "chunk_count": len(chunks),
            "chunks": [
                chunk.model_dump(
                    mode="json"
                )
                for chunk in chunks
            ],
        }

        metadata_path = (
            index_directory
            / "dense_metadata.json"
        )

        with metadata_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def search(
        self,
        *,
        query: str,
        index_directory: Path,
        retrieval_filter: RetrievalFilter,
        top_k: int = 20,
    ) -> list[RetrievalHit]:
        query = query.strip()

        if not query:
            return []

        index_path = (
            index_directory
            / "dense.faiss"
        )

        metadata_path = (
            index_directory
            / "dense_metadata.json"
        )

        if (
            not index_path.is_file()
            or not metadata_path.is_file()
        ):
            raise DenseIndexError(
                "Dense index files are missing."
            )

        index = faiss.read_index(
            str(index_path)
        )

        with metadata_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

        chunks = [
            RetrievalChunk.model_validate(
                item
            )
            for item in metadata.get(
                "chunks",
                [],
            )
        ]

        if not chunks:
            return []

        query_embedding = (
            self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # Oversample so metadata filtering does not
        # accidentally remove every top result.
        search_k = min(
            len(chunks),
            max(
                top_k * 4,
                top_k,
            ),
        )

        scores, indices = index.search(
            query_embedding,
            search_k,
        )

        results: list[
            RetrievalHit
        ] = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            if index_position >= len(chunks):
                continue

            chunk = chunks[
                int(index_position)
            ]

            if not chunk_matches_filter(
                chunk=chunk,
                retrieval_filter=(
                    retrieval_filter
                ),
            ):
                continue

            results.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    user_id=chunk.user_id,
                    document_id=(
                        chunk.document_id
                    ),
                    page_number=(
                        chunk.page_number
                    ),
                    text=chunk.text,
                    content_type=(
                        chunk.content_type
                    ),
                    chunk_kind=(
                        chunk.chunk_kind
                    ),
                    parent_id=(
                        chunk.parent_id
                    ),
                    topics=chunk.topics,
                    grade_min=chunk.grade_min,
                    grade_max=chunk.grade_max,
                    linked_figure_ids=(
                        chunk
                        .linked_figure_ids
                    ),
                    image_path=(
                        chunk.image_path
                    ),
                    caption=chunk.caption,
                    score=float(
                        score
                    ),
                    retrieval_source=(
                        "dense"
                    ),
                )
            )

            if len(results) >= top_k:
                break

        return results