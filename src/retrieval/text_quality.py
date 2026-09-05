from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_STANDALONE_EXCLAMATION_PATTERN = re.compile(
    r"(?<!\w)!+(?!\w)"
)

_SUSPICIOUS_SYMBOL_RUN_PATTERN = re.compile(
    r'["#$%&]{2,}'
)


@dataclass(frozen=True)
class TextQualityResult:
    is_usable: bool
    score: float
    reason: str


class TextQualityChecker:
    """
    Detect corrupted or heavily fragmented extracted text.

    Important:
    - This class never reconstructs or invents source content.
    - Unicode letters, numbers, combining marks, and mathematical symbols
      are treated generically rather than through a Physics-symbol whitelist.
    - Whitespace cleanup preserves intentional line breaks so recovered
      equations are not flattened.
    """

    def __init__(
        self,
        *,
        minimum_score: float = 0.55,
        minimum_characters: int = 40,
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0 and 1."
            )

        if minimum_characters <= 0:
            raise ValueError(
                "minimum_characters must be positive."
            )

        self.minimum_score = minimum_score
        self.minimum_characters = (
            minimum_characters
        )

    def evaluate(
        self,
        text: str,
    ) -> TextQualityResult:
        normalized = self._normalize(
            text
        )

        if (
            len(normalized)
            < self.minimum_characters
        ):
            return TextQualityResult(
                is_usable=False,
                score=0.0,
                reason="Text is too short.",
            )

        tokens = normalized.split()

        if not tokens:
            return TextQualityResult(
                is_usable=False,
                score=0.0,
                reason=(
                    "Text contains no usable tokens."
                ),
            )

        # -----------------------------------------
        # Severe extraction-corruption checks
        # -----------------------------------------

        standalone_exclamations = len(
            _STANDALONE_EXCLAMATION_PATTERN.findall(
                normalized
            )
        )

        suspicious_symbol_runs = len(
            _SUSPICIOUS_SYMBOL_RUN_PATTERN.findall(
                normalized
            )
        )

        replacement_character_count = (
            normalized.count("�")
        )

        if standalone_exclamations >= 8:
            return TextQualityResult(
                is_usable=False,
                score=0.0,
                reason=(
                    "Text contains too many "
                    "equation-extraction placeholders."
                ),
            )

        if suspicious_symbol_runs >= 2:
            return TextQualityResult(
                is_usable=False,
                score=0.0,
                reason=(
                    "Text contains suspicious "
                    "corrupted symbol sequences."
                ),
            )

        if replacement_character_count >= 3:
            return TextQualityResult(
                is_usable=False,
                score=0.0,
                reason=(
                    "Text contains too many "
                    "replacement characters."
                ),
            )

        # -----------------------------------------
        # General text-quality scoring
        # -----------------------------------------

        word_like_count = 0
        fragment_count = 0
        noisy_token_count = 0
        symbol_only_count = 0

        for token in tokens:
            cleaned = self._strip_edge_punctuation(
                token
            )

            if not cleaned:
                if token.strip():
                    symbol_only_count += 1

                continue

            if self._is_word_like_token(
                cleaned
            ):
                word_like_count += 1

            if self._is_fragment_like_token(
                cleaned
            ):
                fragment_count += 1

            if self._looks_noisy(
                cleaned
            ):
                noisy_token_count += 1

            if not any(
                character.isalnum()
                for character in cleaned
            ):
                symbol_only_count += 1

        token_count = max(
            len(tokens),
            1,
        )

        word_like_ratio = (
            word_like_count
            / token_count
        )

        fragment_ratio = (
            fragment_count
            / token_count
        )

        noise_ratio = (
            noisy_token_count
            / token_count
        )

        symbol_only_ratio = (
            symbol_only_count
            / token_count
        )

        invalid_character_ratio = (
            self._invalid_character_ratio(
                normalized
            )
        )

        score = 1.0

        score -= min(
            noise_ratio * 0.60,
            0.60,
        )

        score -= min(
            invalid_character_ratio * 0.60,
            0.60,
        )

        # Prose-dominant text should contain at least some word-like tokens.
        # Equation-heavy text is allowed to be symbol-heavy without being
        # treated as corruption merely because it contains valid notation.
        if (
            word_like_ratio < 0.15
            and not self._contains_math_structure(
                normalized
            )
        ):
            score -= 0.20

        if fragment_ratio > 0.45:
            score -= 0.20

        if (
            symbol_only_ratio > 0.45
            and not self._contains_math_structure(
                normalized
            )
        ):
            score -= 0.25

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        is_usable = (
            score
            >= self.minimum_score
        )

        if is_usable:
            reason = (
                "Text quality is acceptable."
            )
        else:
            reason = (
                "Text appears fragmented "
                "or corrupted."
            )

        return TextQualityResult(
            is_usable=is_usable,
            score=score,
            reason=reason,
        )

    def is_usable(
        self,
        text: str,
    ) -> bool:
        return self.evaluate(
            text
        ).is_usable

    def clean_spacing(
        self,
        text: str,
    ) -> str:
        """
        Safe whitespace normalization only.

        Preserves intentional line boundaries so equations, matrices,
        multi-line derivations, and vision-recovered notation are not
        flattened into one line.
        """

        if not text:
            return ""

        normalized = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        lines: list[str] = []

        for raw_line in normalized.split(
            "\n"
        ):
            line = re.sub(
                r"[ \t\f\v]+",
                " ",
                raw_line,
            ).strip()

            if line:
                lines.append(
                    line
                )
                continue

            if (
                lines
                and lines[-1] != ""
            ):
                lines.append("")

        while (
            lines
            and lines[-1] == ""
        ):
            lines.pop()

        return "\n".join(
            lines
        ).strip()

    def _normalize(
        self,
        text: str,
    ) -> str:
        return self.clean_spacing(
            text
        )

    @staticmethod
    def _strip_edge_punctuation(
        token: str,
    ) -> str:
        """
        Strip ordinary edge punctuation without maintaining a language-
        specific or Physics-symbol-specific whitelist.
        """

        start = 0
        end = len(token)

        while start < end:
            category = unicodedata.category(
                token[start]
            )

            if not category.startswith(
                "P"
            ):
                break

            start += 1

        while end > start:
            category = unicodedata.category(
                token[end - 1]
            )

            if not category.startswith(
                "P"
            ):
                break

            end -= 1

        return token[start:end]

    @staticmethod
    def _is_word_like_token(
        token: str,
    ) -> bool:
        """
        Unicode-aware word detection.

        Supports Latin, Greek, Bengali, Devanagari, and other scripts
        generically through Unicode categories.
        """

        letters = sum(
            unicodedata.category(
                character
            ).startswith("L")
            for character in token
        )

        marks = sum(
            unicodedata.category(
                character
            ).startswith("M")
            for character in token
        )

        return (
            letters >= 2
            or (
                letters >= 1
                and marks >= 1
            )
        )

    @staticmethod
    def _is_fragment_like_token(
        token: str,
    ) -> bool:
        """
        Detect short alphabetic fragments without assuming ASCII.
        Single-symbol variables are intentionally not counted as fragments.
        """

        if not 1 <= len(token) <= 4:
            return False

        if not all(
            (
                unicodedata.category(
                    character
                ).startswith("L")
                or unicodedata.category(
                    character
                ).startswith("M")
            )
            for character in token
        ):
            return False

        # One-letter variables are normal in Physics/math and should not
        # contribute to the fragmentation penalty.
        letter_count = sum(
            unicodedata.category(
                character
            ).startswith("L")
            for character in token
        )

        return letter_count >= 2

    @staticmethod
    def _contains_math_structure(
        text: str,
    ) -> bool:
        """
        Generic mathematical-notation detection using Unicode categories.

        No fixed list such as pi/omega/theta is required.
        """

        for character in text:
            category = unicodedata.category(
                character
            )

            if (
                category == "Sm"
                or character
                in {
                    "=",
                    "^",
                    "_",
                    "\\",
                }
            ):
                return True

        return bool(
            re.search(
                r"\\[A-Za-z]+",
                text,
            )
        )

    def _looks_noisy(
        self,
        token: str,
    ) -> bool:
        if len(token) <= 1:
            return False

        letters_or_numbers = sum(
            (
                unicodedata.category(
                    character
                ).startswith("L")
                or unicodedata.category(
                    character
                ).startswith("N")
            )
            for character in token
        )

        marks = sum(
            unicodedata.category(
                character
            ).startswith("M")
            for character in token
        )

        invalid = sum(
            self._is_invalid_character(
                character
            )
            for character in token
        )

        if invalid > 0:
            return True

        if (
            letters_or_numbers == 0
            and marks == 0
        ):
            # Pure mathematical/symbol tokens are legitimate evidence.
            return not self._contains_math_structure(
                token
            )

        return False

    @staticmethod
    def _is_invalid_character(
        character: str,
    ) -> bool:
        """
        Treat control/unassigned/surrogate/private-use code points as
        suspicious. Normal punctuation, Unicode math, arrows, Greek,
        Indic scripts, superscripts, subscripts, etc. remain valid.
        """

        if character == "�":
            return True

        category = unicodedata.category(
            character
        )

        return category in {
            "Cc",
            "Cs",
            "Co",
            "Cn",
        }

    def _invalid_character_ratio(
        self,
        text: str,
    ) -> float:
        if not text:
            return 1.0

        invalid = sum(
            self._is_invalid_character(
                character
            )
            for character in text
        )

        return (
            invalid
            / len(text)
        )