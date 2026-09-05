from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from src.ingestion.models import ParentChunk
from src.retrieval.models import (
    ContextBundle,
    ContextItem,
    RerankedRetrievalHit,
)
from src.retrieval.text_quality import TextQualityChecker


_APOSTROPHES = {
    "'",
    "’",
    "ʼ",
}

_STRUCTURAL_MATH_PUNCTUATION = {
    "=",
    "/",
    "\\",
    "^",
    "_",
}


_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?।！？])\s+|\n+"
)


def _is_word_character(
    character: str,
) -> bool:
    """
    Unicode-aware word-character test used only for derived
    compression matching tokens.
    """

    category = unicodedata.category(
        character
    )

    return (
        category.startswith("L")
        or category.startswith("N")
        or category.startswith("M")
    )


def _is_search_symbol(
    character: str,
) -> bool:
    """
    Keep generic mathematical/symbol characters as exact overlap tokens.
    """

    category = unicodedata.category(
        character
    )

    return (
        category.startswith("S")
        or character
        in _STRUCTURAL_MATH_PUNCTUATION
    )


def _tokenize_for_overlap(
    text: str,
) -> list[str]:
    """
    Build a Unicode-safe token stream for extractive compression scoring.

    The original query/evidence is never rewritten. This is only a derived
    representation used to rank already-present source sentences.
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
            next_is_word = (
                index + 1 < length
                and _is_word_character(
                    text[index + 1]
                )
            )

            if current and next_is_word:
                current.append(
                    character
                )
                continue

        flush_current()

        if _is_search_symbol(
            character
        ):
            tokens.append(
                character.casefold()
            )

    flush_current()

    return tokens


def _contains_math_signal(
    text: str,
) -> bool:
    """
    Detect generic mathematical notation without maintaining a small,
    language- or topic-specific list of symbols.
    """

    for character in text:
        category = unicodedata.category(
            character
        )

        if (
            category == "Sm"
            or character
            in _STRUCTURAL_MATH_PUNCTUATION
        ):
            return True

    # Also preserve common textual math markup in retrieved equations.
    return bool(
        re.search(
            r"\\[A-Za-z]+",
            text,
        )
    )


class ContextCompressionError(Exception):
    """Raised when stored parent context cannot be safely loaded."""


class ContextCompressor:
    """
    Parent-document expansion plus deterministic
    extractive context compression.

    Important:
    - This stage does NOT generate answers.
    - It does NOT reconstruct damaged source text.
    - If a parent chunk looks unreliable, it falls
      back to the precise retrieved child chunk.
    """

    def __init__(
        self,
        *,
        max_context_characters: int = 12000,
        max_item_characters: int = 3000,
    ) -> None:
        if max_context_characters <= 0:
            raise ValueError(
                "max_context_characters must be positive."
            )

        if max_item_characters <= 0:
            raise ValueError(
                "max_item_characters must be positive."
            )

        self.max_context_characters = (
            max_context_characters
        )

        self.max_item_characters = (
            max_item_characters
        )

        self.text_quality = TextQualityChecker()

    def compress(
        self,
        *,
        query: str,
        reranked_hits: list[
            RerankedRetrievalHit
        ],
        parent_chunks_path: Path,
        user_id: str,
        document_id: str,
        max_contexts: int = 6,
    ) -> ContextBundle:
        """
        Convert reranked child/visual hits into a compact,
        grounded context bundle for the future Tutor Agent.
        """

        normalized_query = query.strip()

        if max_contexts <= 0:
            return self._empty_bundle(
                query=normalized_query,
                user_id=user_id,
                document_id=document_id,
            )

        parents = self._load_parents(
            parent_chunks_path=(
                parent_chunks_path
            ),
            expected_user_id=user_id,
            expected_document_id=(
                document_id
            ),
        )

        items: list[ContextItem] = []
        seen_contexts: set[str] = set()

        total_characters = 0
        bundle_truncated = False

        # We inspect a few more reranked hits than the
        # final number of contexts because multiple child
        # hits can belong to the same parent section.
        candidate_limit = max_contexts * 3

        for reranked in reranked_hits[
            :candidate_limit
        ]:
            hit = reranked.hit

            # Defence in depth:
            # never allow evidence from another user or
            # another document into the final context.
            if (
                hit.user_id != user_id
                or hit.document_id
                != document_id
            ):
                continue

            resolved_context = (
                self._resolve_context_source(
                    hit=hit,
                    parents=parents,
                )
            )

            # If both parent and child evidence are too noisy,
            # skip this candidate completely.
            if resolved_context is None:
                continue

            (
                context_key,
                base_text,
                source_chunk_ids,
                linked_figure_ids,
                equations,
                parent_id,
            ) = resolved_context

            if context_key in seen_contexts:
                continue

            normalized_text = (
                self.text_quality.clean_spacing(
                    base_text
                )
            )

            if not normalized_text:
                continue

            compressed_text, was_compressed = (
                self._compress_text(
                    query=normalized_query,
                    text=normalized_text,
                    maximum_characters=(
                        self.max_item_characters
                    ),
                )
            )

            if was_compressed:
                bundle_truncated = True

            remaining_budget = (
                self.max_context_characters
                - total_characters
            )

            if remaining_budget <= 0:
                bundle_truncated = True
                break

            if (
                len(compressed_text)
                > remaining_budget
            ):
                compressed_text = (
                    compressed_text[
                        :remaining_budget
                    ].rstrip()
                )

                bundle_truncated = True

            if not compressed_text:
                continue

            # Only mark the context as seen once we know
            # it will actually be added.
            seen_contexts.add(
                context_key
            )

            items.append(
                ContextItem(
                    context_id=context_key,
                    user_id=user_id,
                    document_id=document_id,
                    page_number=(
                        hit.page_number
                    ),
                    source_chunk_ids=(
                        source_chunk_ids
                    ),
                    parent_id=parent_id,
                    text=compressed_text,
                    content_type=(
                        hit.content_type
                    ),
                    linked_figure_ids=(
                        linked_figure_ids
                    ),
                    equations=equations,
                    image_path=(
                        hit.image_path
                    ),
                    caption=hit.caption,
                    rerank_score=(
                        reranked.rerank_score
                    ),
                )
            )

            total_characters += len(
                compressed_text
            )

            if len(items) >= max_contexts:
                break

        return ContextBundle(
            query=normalized_query,
            user_id=user_id,
            document_id=document_id,
            items=items,
            total_characters=(
                total_characters
            ),
            truncated=bundle_truncated,
        )

    def _resolve_context_source(
        self,
        *,
        hit,
        parents: dict[
            str,
            ParentChunk,
        ],
    ) -> tuple[
        str,
        str,
        list[str],
        list[str],
        list[str],
        str | None,
    ] | None:
        """
        Decide whether to use:
        1. the visual hit itself,
        2. a trustworthy larger parent section, or
        3. the precise retrieved child as fallback.
        """

        if hit.chunk_kind == "visual":
            return (
                f"visual:{hit.chunk_id}",
                hit.text,
                [hit.chunk_id],
                list(
                    hit.linked_figure_ids
                ),
                [],
                hit.parent_id,
            )

        parent = (
            parents.get(hit.parent_id)
            if hit.parent_id
            else None
        )

        if parent is not None:
            parent_text = (
                self.text_quality.clean_spacing(
                    parent.text
                )
            )

            parent_quality = (
                self.text_quality.evaluate(
                    parent_text
                )
            )

            if parent_quality.is_usable:
                return (
                    f"parent:{parent.parent_id}",
                    parent_text,
                    list(
                        parent.child_ids
                    ),
                    list(
                        dict.fromkeys(
                            list(
                                parent.figures
                            )
                            + list(
                                hit.linked_figure_ids
                            )
                        )
                    ),
                    list(
                        parent.equations
                    ),
                    parent.parent_id,
                )

        # Parent missing or too noisy:
        # use the exact reranked child rather than
        # trying to reconstruct damaged source content.
        child_text = (
            self.text_quality.clean_spacing(
                hit.text
            )
        )

        child_quality = (
            self.text_quality.evaluate(
                child_text
            )
        )

        # Parent is already unusable at this point.
        # If the precise retrieved child is also corrupted,
        # do not pass this evidence farther down the RAG pipeline.
        if not child_quality.is_usable:
            return None

        return (
            f"chunk:{hit.chunk_id}",
            child_text,
            [hit.chunk_id],
            list(
                hit.linked_figure_ids
            ),
            [],
            hit.parent_id,
        )

    def _load_parents(
        self,
        *,
        parent_chunks_path: Path,
        expected_user_id: str,
        expected_document_id: str,
    ) -> dict[str, ParentChunk]:
        if not parent_chunks_path.is_file():
            return {}

        try:
            with parent_chunks_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

        except Exception as exc:
            raise ContextCompressionError(
                "Parent chunk store could not be read."
            ) from exc

        if not isinstance(payload, dict):
            raise ContextCompressionError(
                "Parent chunk store has an invalid format."
            )

        if (
            payload.get("user_id")
            != expected_user_id
        ):
            raise ContextCompressionError(
                "Parent context user mismatch."
            )

        if (
            payload.get("document_id")
            != expected_document_id
        ):
            raise ContextCompressionError(
                "Parent context document mismatch."
            )

        # Current chunking format stores parent chunks
        # under the "parent_chunks" key.
        raw_parents = payload.get(
            "parent_chunks"
        )

        # Backward compatibility:
        # older stored files may use "parents".
        if raw_parents is None:
            raw_parents = payload.get(
                "parents",
                [],
            )

        if not isinstance(
            raw_parents,
            list,
        ):
            raise ContextCompressionError(
                "Parent chunk list has an invalid format."
            )

        parents: dict[
            str,
            ParentChunk,
        ] = {}

        for item in raw_parents:
            try:
                parent = (
                    ParentChunk.model_validate(
                        item
                    )
                )

            except Exception as exc:
                raise ContextCompressionError(
                    "A stored parent chunk is invalid."
                ) from exc

            if (
                parent.user_id
                != expected_user_id
                or parent.document_id
                != expected_document_id
            ):
                continue

            parents[
                parent.parent_id
            ] = parent

        return parents

    def _compress_text(
        self,
        *,
        query: str,
        text: str,
        maximum_characters: int,
    ) -> tuple[str, bool]:
        """
        Deterministic extractive compression.

        It keeps original sentences only.
        It does not paraphrase or invent evidence.
        """

        normalized = text.strip()

        if not normalized:
            return "", False

        if (
            len(normalized)
            <= maximum_characters
        ):
            return normalized, False

        query_tokens = set(
            _tokenize_for_overlap(
                query
            )
        )

        sentences = [
            sentence.strip()
            for sentence
            in _SENTENCE_SPLIT.split(
                normalized
            )
            if sentence.strip()
        ]

        if not sentences:
            return (
                normalized[
                    :maximum_characters
                ].rstrip(),
                True,
            )

        scored_sentences: list[
            tuple[int, int, str]
        ] = []

        for index, sentence in enumerate(
            sentences
        ):
            sentence_tokens = set(
                _tokenize_for_overlap(
                    sentence
                )
            )

            overlap = len(
                query_tokens
                & sentence_tokens
            )

            # Preserve mathematical relationships and
            # equation-bearing sentences when possible.
            math_bonus = (
                1
                if _contains_math_signal(
                    sentence
                )
                else 0
            )

            # Opening sentences often contain the
            # definition of a section.
            opening_bonus = (
                1
                if index < 2
                else 0
            )

            score = (
                overlap * 3
                + math_bonus
                + opening_bonus
            )

            scored_sentences.append(
                (
                    score,
                    index,
                    sentence,
                )
            )

        ranked = sorted(
            scored_sentences,
            key=lambda item: (
                item[0],
                -item[1],
            ),
            reverse=True,
        )

        selected_indices: set[int] = set()
        used_characters = 0

        for _, index, sentence in ranked:
            projected = (
                used_characters
                + len(sentence)
                + (
                    1
                    if used_characters
                    else 0
                )
            )

            if (
                projected
                > maximum_characters
            ):
                continue

            selected_indices.add(
                index
            )

            used_characters = projected

        if not selected_indices:
            return (
                normalized[
                    :maximum_characters
                ].rstrip(),
                True,
            )

        selected = [
            sentences[index]
            for index
            in sorted(
                selected_indices
            )
        ]

        result = " ".join(
            selected
        ).strip()

        return result, True

    def _empty_bundle(
        self,
        *,
        query: str,
        user_id: str,
        document_id: str,
    ) -> ContextBundle:
        return ContextBundle(
            query=query,
            user_id=user_id,
            document_id=document_id,
            items=[],
            total_characters=0,
            truncated=False,
        )