from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

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


class BM25IndexError(Exception):
    pass


_APOSTROPHES = {
    "'",
    "’",
    "ʼ",
}

_STRUCTURAL_MATH_PUNCTUATION = {
    "/",
    "\\",
    "^",
    "_",
}


def _is_word_character(
    character: str,
) -> bool:
    """
    Return True for Unicode letters, numbers, and combining marks.

    The original source/query text is never modified. This helper is used
    only to build a derived BM25 token stream.
    """

    category = unicodedata.category(
        character
    )

    return (
        category.startswith("L")
        or category.startswith("N")
        or category.startswith("M")
    )


def _is_symbol_token(
    character: str,
) -> bool:
    """
    Keep generic mathematical/symbol characters as standalone BM25 tokens.

    Unicode symbol categories cover mathematical operators, arrows, relation
    signs, roots, integrals, and other symbols without requiring a fixed list.

    A few structural ASCII equation characters that Unicode classifies as
    punctuation are also retained. No Physics-topic-specific mapping is used.
    """

    category = unicodedata.category(
        character
    )

    return (
        category.startswith("S")
        or character
        in _STRUCTURAL_MATH_PUNCTUATION
    )


def tokenize_for_physics(
    text: str,
) -> list[str]:
    """
    Unicode-safe tokenizer for school Physics retrieval.

    Design rules:
    - preserve arbitrary valid Unicode in the original text;
    - case-fold only the derived token stream;
    - keep words/numbers across scripts;
    - keep combining marks with their word;
    - keep apostrophes inside words;
    - keep mathematical and symbolic characters as exact tokens;
    - do not hard-code Physics topics or a small fixed Greek-letter list.
    """

    if not text:
        return []

    tokens: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if not current:
            return

        token = "".join(
            current
        ).casefold()

        if token:
            tokens.append(token)

        current.clear()

    length = len(text)

    for index, character in enumerate(
        text
    ):
        if _is_word_character(
            character
        ):
            current.append(
                character
            )
            continue

        if character in _APOSTROPHES:
            previous_is_word = bool(
                current
            )

            next_is_word = (
                index + 1 < length
                and _is_word_character(
                    text[index + 1]
                )
            )

            if (
                previous_is_word
                and next_is_word
            ):
                current.append(
                    character
                )
                continue

        flush_current()

        if _is_symbol_token(
            character
        ):
            tokens.append(
                character.casefold()
            )

    flush_current()

    return tokens


class BM25Retriever:
    """
    Local BM25 retriever.

    Corpus is stored as JSON and BM25 is rebuilt
    from that corpus when searching. This avoids
    unsafe pickle persistence.
    """

    def build(
        self,
        *,
        chunks: list[RetrievalChunk],
        index_directory: Path,
    ) -> None:
        if not chunks:
            raise BM25IndexError(
                "Cannot build BM25 index "
                "without chunks."
            )

        index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "chunk_count": len(chunks),
            "chunks": [
                chunk.model_dump(
                    mode="json"
                )
                for chunk in chunks
            ],
        }

        path = (
            index_directory
            / "bm25_corpus.json"
        )

        with path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
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

        path = (
            index_directory
            / "bm25_corpus.json"
        )

        if not path.is_file():
            raise BM25IndexError(
                "BM25 corpus file is missing."
            )

        with path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )

        chunks = [
            RetrievalChunk.model_validate(
                item
            )
            for item in payload.get(
                "chunks",
                [],
            )
        ]

        if not chunks:
            return []

        tokenized_corpus = [
            tokenize_for_physics(
                chunk.text
            )
            for chunk in chunks
        ]

        bm25 = BM25Okapi(
            tokenized_corpus
        )

        query_tokens = (
            tokenize_for_physics(
                query
            )
        )

        if not query_tokens:
            return []

        scores = bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda index: scores[
                index
            ],
            reverse=True,
        )

        results: list[
            RetrievalHit
        ] = []

        for index_position in ranked_indices:
            score = float(
                scores[index_position]
            )

            if score <= 0:
                continue

            chunk = chunks[
                index_position
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
                    score=score,
                    retrieval_source=(
                        "bm25"
                    ),
                )
            )

            if len(results) >= top_k:
                break

        return results