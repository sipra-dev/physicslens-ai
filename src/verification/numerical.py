from __future__ import annotations

import ast
import math
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.models.contracts import (
    AnswerType,
    IntentDecision,
    TutorAnswer,
)


class NumericalCheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_CHECKED = "NOT_CHECKED"


class NumericalCheck(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str
    status: NumericalCheckStatus
    details: str
    evidence: list[str] = Field(
        default_factory=list,
        max_length=20,
    )


class NumericalVerificationReport(BaseModel):
    """
    Conservative deterministic numerical-verification output.

    None means the checker could not prove the property either way.
    That is intentionally different from True.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    available: bool
    calculation_passed: bool | None = None
    units_passed: bool | None = None
    dimensional_consistency_passed: bool | None = None
    sign_convention_passed: bool | None = None
    significant_figures_passed: bool | None = None
    checks: list[NumericalCheck] = Field(
        default_factory=list,
        max_length=20,
    )
    issues: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    def as_prompt_payload(
        self,
    ) -> dict[str, Any]:
        return self.model_dump(
            mode="json"
        )


class DeterministicNumericalVerifier:
    """
    Python-side conservative numerical checker.

    Architecture rule:
    natural-language intent is NOT inferred here.

    Query Understanding decides what the learner asked for and stores that in
    IntentDecision.requested_quantities / given_quantities / given_equations.

    This deterministic layer only checks facts it can prove mechanically:
    - fully numeric arithmetic equalities;
    - explicit equalities whose units are textually identical after generic
      structural normalization.

    It deliberately does NOT contain:
    - Physics-topic keyword maps;
    - quantity-name keyword maps;
    - document/figure mappings;
    - Physics formula whitelists;
    - a hand-written unit ontology or conversion table.

    Anything that needs semantic Physics knowledge remains NOT_CHECKED and is
    left to the LLM Verifier.
    """

    _NUMBER_ONLY = re.compile(
        r"""
        ^\s*
        (?P<value>
            [-+]?
            (?:
                (?:\d+(?:\.\d*)?)
                |
                (?:\.\d+)
            )
            (?:[eE][-+]?\d+)?
        )
        \s*
        (?P<unit>.+?)
        \s*$
        """,
        flags=re.VERBOSE,
    )

    # These are structural arithmetic glyph variants, not Physics semantics.
    # Original user/Tutor text is never mutated; this is a derived parser copy.
    _MATH_TRANSLATION = str.maketrans(
        {
            "×": "*",
            "÷": "/",
            "−": "-",
            "–": "-",
            "—": "-",
        }
    )

    def verify(
        self,
        *,
        intent: IntentDecision,
        tutor_answer: TutorAnswer,
    ) -> NumericalVerificationReport:
        if (
            tutor_answer.answer_type
            != AnswerType.NUMERICAL_SOLUTION
        ):
            return NumericalVerificationReport(
                available=False
            )

        combined_text = (
            self._combined_answer_text(
                tutor_answer
            )
        )

        arithmetic_check = (
            self._check_arithmetic(
                combined_text
            )
        )

        unit_identity_check = (
            self._check_same_unit_equalities(
                combined_text
            )
        )

        requested_result_unit_check = (
            self._check_requested_result_units(
                intent=intent,
            )
        )

        dimensional_check = NumericalCheck(
            name="dimensional_consistency",
            status=(
                NumericalCheckStatus
                .NOT_CHECKED
            ),
            details=(
                "No generic deterministic dimensional proof was available. "
                "The checker does not use a hard-coded Physics formula list."
            ),
        )

        sign_check = NumericalCheck(
            name="sign_convention",
            status=(
                NumericalCheckStatus
                .NOT_CHECKED
            ),
            details=(
                "Sign convention requires problem semantics and was left to "
                "the Verifier Agent."
            ),
        )

        significant_figures_check = (
            NumericalCheck(
                name="significant_figures",
                status=(
                    NumericalCheckStatus
                    .NOT_CHECKED
                ),
                details=(
                    "Significant figures were not deterministically checked "
                    "because the required precision rule is semantic."
                ),
            )
        )

        checks = [
            arithmetic_check,
            unit_identity_check,
            requested_result_unit_check,
            dimensional_check,
            sign_check,
            significant_figures_check,
        ]

        issues = [
            check.details
            for check in checks
            if (
                check.status
                == NumericalCheckStatus.FAIL
            )
        ]

        return NumericalVerificationReport(
            available=any(
                check.status
                != NumericalCheckStatus
                .NOT_CHECKED
                for check in checks
            ),
            calculation_passed=(
                self._status_to_optional_bool(
                    arithmetic_check.status
                )
            ),
            units_passed=(
                self._aggregate_statuses(
                    [
                        unit_identity_check.status,
                        requested_result_unit_check.status,
                    ]
                )
            ),
            dimensional_consistency_passed=(
                self._status_to_optional_bool(
                    dimensional_check.status
                )
            ),
            sign_convention_passed=None,
            significant_figures_passed=None,
            checks=checks,
            issues=issues,
        )

    @staticmethod
    def _combined_answer_text(
        tutor_answer: TutorAnswer,
    ) -> str:
        parts: list[str] = [
            tutor_answer.direct_answer,
            *tutor_answer.steps,
        ]

        for formula in tutor_answer.formulae:
            parts.extend(
                [
                    formula.latex,
                    formula.meaning,
                ]
            )

        if tutor_answer.final_result:
            parts.append(
                tutor_answer.final_result
            )

        return "\n".join(
            part
            for part in parts
            if part and part.strip()
        )

    def _check_arithmetic(
        self,
        text: str,
    ) -> NumericalCheck:
        """
        Check adjacent numeric expressions inside an equality or assignment
        chain.

        Examples of the generic structure handled here include:
            result = numeric_expression
            variable = numeric_expression = claimed_numeric_value
            numeric_expression = numeric_expression = numeric_expression

        Symbolic expressions, variables, units, prose, functions, and anything
        the AST evaluator cannot prove are skipped rather than guessed. A
        symbolic assignment prefix is therefore allowed, but only adjacent
        sides that are independently evaluable as pure numeric arithmetic are
        compared.
        """

        checked_evidence: list[str] = []
        failures: list[str] = []

        for raw_line in text.splitlines():
            line = (
                self._normalize_math_for_parser(
                    raw_line
                )
            )

            equality_parts = [
                part.strip()
                for part in line.split("=")
            ]

            if len(equality_parts) < 2:
                continue

            if any(
                not part
                for part in equality_parts
            ):
                continue

            line_was_checked = False
            line_failed = False

            for (
                left_text,
                right_text,
            ) in zip(
                equality_parts,
                equality_parts[1:],
            ):
                left_value = (
                    self._safe_eval_numeric(
                        left_text
                    )
                )
                right_value = (
                    self._safe_eval_numeric(
                        right_text
                    )
                )

                if (
                    left_value is None
                    or right_value is None
                ):
                    continue

                line_was_checked = True

                if not math.isclose(
                    left_value,
                    right_value,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    line_failed = True

            if not line_was_checked:
                continue

            evidence = raw_line.strip()
            if evidence:
                checked_evidence.append(
                    evidence
                )

            if line_failed:
                failures.append(
                    (
                        "Arithmetic mismatch: adjacent numeric sides of "
                        f"the equality chain '{evidence}' are not equal."
                    )
                )

        unique_evidence = list(
            dict.fromkeys(
                checked_evidence
            )
        )[:20]

        if failures:
            return NumericalCheck(
                name="arithmetic",
                status=(
                    NumericalCheckStatus.FAIL
                ),
                details=" ".join(
                    dict.fromkeys(
                        failures
                    )
                ),
                evidence=unique_evidence,
            )

        if unique_evidence:
            return NumericalCheck(
                name="arithmetic",
                status=(
                    NumericalCheckStatus.PASS
                ),
                details=(
                    "All fully numeric arithmetic equalities that could be "
                    "parsed safely are internally consistent."
                ),
                evidence=unique_evidence,
            )

        return NumericalCheck(
            name="arithmetic",
            status=(
                NumericalCheckStatus
                .NOT_CHECKED
            ),
            details=(
                "No fully numeric arithmetic equality could be proved safely. "
                "Symbolic algebra was left to the Verifier Agent."
            ),
        )

    def _check_same_unit_equalities(
        self,
        text: str,
    ) -> NumericalCheck:
        """
        Check only explicit scalar equalities where both sides use the same unit
        expression after generic structural normalization.

        Example shape:
            5 <unit> = 5 <same unit>

        Different units are NOT converted here because doing that safely would
        require a unit ontology/conversion registry. Without one, the checker
        returns NOT_CHECKED rather than hard-coding Physics units.
        """

        checked_evidence: list[str] = []
        failures: list[str] = []

        for raw_line in text.splitlines():
            if raw_line.count("=") != 1:
                continue

            left_text, right_text = (
                part.strip()
                for part in raw_line.split(
                    "=",
                    maxsplit=1,
                )
            )

            left = self._parse_scalar_unit(
                left_text
            )
            right = self._parse_scalar_unit(
                right_text
            )

            if (
                left is None
                or right is None
            ):
                continue

            (
                left_value,
                left_unit,
            ) = left

            (
                right_value,
                right_unit,
            ) = right

            if left_unit != right_unit:
                continue

            evidence = raw_line.strip()
            if evidence:
                checked_evidence.append(
                    evidence
                )

            if not math.isclose(
                left_value,
                right_value,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                failures.append(
                    (
                        "Same-unit equality mismatch: "
                        f"'{evidence}' has unequal numeric values."
                    )
                )

        unique_evidence = list(
            dict.fromkeys(
                checked_evidence
            )
        )[:20]

        if failures:
            return NumericalCheck(
                name="unit_consistency",
                status=(
                    NumericalCheckStatus.FAIL
                ),
                details=" ".join(
                    dict.fromkeys(
                        failures
                    )
                ),
                evidence=unique_evidence,
            )

        if unique_evidence:
            return NumericalCheck(
                name="unit_consistency",
                status=(
                    NumericalCheckStatus.PASS
                ),
                details=(
                    "All safely parsed equalities with identical unit "
                    "expressions are numerically consistent."
                ),
                evidence=unique_evidence,
            )

        return NumericalCheck(
            name="unit_consistency",
            status=(
                NumericalCheckStatus
                .NOT_CHECKED
            ),
            details=(
                "No same-unit scalar equality could be checked safely. "
                "Cross-unit conversion was not guessed."
            ),
        )

    @staticmethod
    def _check_requested_result_units(
        *,
        intent: IntentDecision,
    ) -> NumericalCheck:
        """
        Consume upstream structured requested_quantities without re-parsing the
        student's natural-language question.

        IntentDecision currently supplies semantic quantity/dimension metadata,
        but this deterministic module intentionally has no hand-written map from
        dimensions or quantity names to allowed units. Therefore semantic final
        unit compatibility remains NOT_CHECKED until a generic unit registry is
        provided upstream.
        """

        requested = list(
            getattr(
                intent,
                "requested_quantities",
                [],
            )
            or []
        )

        if not requested:
            return NumericalCheck(
                name="requested_result_unit",
                status=(
                    NumericalCheckStatus
                    .NOT_CHECKED
                ),
                details=(
                    "No requested quantity was supplied by Query Understanding. "
                    "The deterministic checker did not infer one from raw text."
                ),
            )

        labels: list[str] = []

        for item in requested:
            quantity = (
                getattr(
                    item,
                    "quantity",
                    None,
                )
                or getattr(
                    item,
                    "symbol",
                    None,
                )
                or getattr(
                    item,
                    "raw_reference",
                    None,
                )
            )

            if quantity:
                labels.append(
                    str(quantity)
                )

        description = (
            ", ".join(labels)
            if labels
            else f"{len(requested)} structured requested quantity item(s)"
        )

        return NumericalCheck(
            name="requested_result_unit",
            status=(
                NumericalCheckStatus
                .NOT_CHECKED
            ),
            details=(
                "Structured requested quantities were received upstream "
                f"({description}), but semantic unit compatibility was left "
                "to the Verifier Agent because no hard-coded unit ontology is "
                "used in this deterministic checker."
            ),
        )

    def _parse_scalar_unit(
        self,
        text: str,
    ) -> tuple[float, str] | None:
        match = self._NUMBER_ONLY.fullmatch(
            text
        )

        if match is None:
            return None

        try:
            value = float(
                match.group(
                    "value"
                )
            )
        except ValueError:
            return None

        if not math.isfinite(value):
            return None

        raw_unit = match.group(
            "unit"
        )

        normalized_unit = (
            self._normalize_unit_structure(
                raw_unit
            )
        )

        if not normalized_unit:
            return None

        # A unit expression must contain something other than digits,
        # decimal punctuation, or signs. This is structural only.
        if not any(
            not (
                character.isdigit()
                or character
                in {
                    ".",
                    "+",
                    "-",
                }
            )
            for character
            in normalized_unit
        ):
            return None

        return (
            value,
            normalized_unit,
        )

    @classmethod
    def _normalize_math_for_parser(
        cls,
        text: str,
    ) -> str:
        return text.translate(
            cls._MATH_TRANSLATION
        )

    @staticmethod
    def _normalize_unit_structure(
        text: str,
    ) -> str:
        """
        Generic comparison normalization only.

        It does not translate a unit into another unit and does not assign
        Physics meaning. Original text is preserved elsewhere.
        """

        return "".join(
            character.casefold()
            for character in text
            if not character.isspace()
        )

    @staticmethod
    def _status_to_optional_bool(
        status: NumericalCheckStatus,
    ) -> bool | None:
        if (
            status
            == NumericalCheckStatus.PASS
        ):
            return True

        if (
            status
            == NumericalCheckStatus.FAIL
        ):
            return False

        return None

    @staticmethod
    def _aggregate_statuses(
        statuses: list[
            NumericalCheckStatus
        ],
    ) -> bool | None:
        if (
            NumericalCheckStatus.FAIL
            in statuses
        ):
            return False

        if (
            NumericalCheckStatus.PASS
            in statuses
        ):
            return True

        return None

    @staticmethod
    def _safe_eval_numeric(
        expression: str,
    ) -> float | None:
        normalized = (
            expression
            .replace(
                "^",
                "**",
            )
            .strip()
        )

        if not normalized:
            return None

        try:
            parsed = ast.parse(
                normalized,
                mode="eval",
            )
        except SyntaxError:
            return None

        allowed_binary = (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
        )

        allowed_unary = (
            ast.UAdd,
            ast.USub,
        )

        def evaluate(
            node: ast.AST,
        ) -> float:
            if isinstance(
                node,
                ast.Expression,
            ):
                return evaluate(
                    node.body
                )

            if isinstance(
                node,
                ast.Constant,
            ):
                if (
                    not isinstance(
                        node.value,
                        (int, float),
                    )
                    or isinstance(
                        node.value,
                        bool,
                    )
                ):
                    raise ValueError

                return float(
                    node.value
                )

            if (
                isinstance(
                    node,
                    ast.UnaryOp,
                )
                and isinstance(
                    node.op,
                    allowed_unary,
                )
            ):
                value = evaluate(
                    node.operand
                )

                if isinstance(
                    node.op,
                    ast.USub,
                ):
                    return -value

                return value

            if (
                isinstance(
                    node,
                    ast.BinOp,
                )
                and isinstance(
                    node.op,
                    allowed_binary,
                )
            ):
                left = evaluate(
                    node.left
                )

                right = evaluate(
                    node.right
                )

                if isinstance(
                    node.op,
                    ast.Add,
                ):
                    return left + right

                if isinstance(
                    node.op,
                    ast.Sub,
                ):
                    return left - right

                if isinstance(
                    node.op,
                    ast.Mult,
                ):
                    return left * right

                if isinstance(
                    node.op,
                    ast.Div,
                ):
                    if right == 0:
                        raise ZeroDivisionError

                    return left / right

                if isinstance(
                    node.op,
                    ast.Mod,
                ):
                    if right == 0:
                        raise ZeroDivisionError

                    return left % right

                if isinstance(
                    node.op,
                    ast.Pow,
                ):
                    if abs(right) > 12:
                        raise ValueError

                    value = left ** right

                    if not math.isfinite(
                        value
                    ):
                        raise ValueError

                    return value

            raise ValueError

        try:
            result = evaluate(
                parsed
            )
        except (
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ):
            return None

        if not math.isfinite(
            result
        ):
            return None

        return result
