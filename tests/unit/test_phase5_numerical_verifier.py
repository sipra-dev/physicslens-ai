from __future__ import annotations

import unittest

from src.models.contracts import (
    AnswerType,
    FormulaItem,
    IntentDecision,
    LanguageCode,
    RequestIntent,
    RequestedQuantity,
    TutorAnswer,
)
from src.verification.numerical import (
    DeterministicNumericalVerifier,
    NumericalCheckStatus,
)


class Phase5NumericalVerifierTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.verifier = (
            DeterministicNumericalVerifier()
        )

    def make_intent(
        self,
        *,
        requested_quantity: str | None = None,
        expected_dimension: str | None = None,
    ) -> IntentDecision:
        requested_quantities = []

        if requested_quantity:
            requested_quantities = [
                RequestedQuantity(
                    quantity=requested_quantity,
                    expected_dimension=(
                        expected_dimension
                    ),
                    raw_reference=(
                        requested_quantity
                    ),
                )
            ]

        return IntentDecision(
            intent=(
                RequestIntent.PHYSICS_QUESTION
            ),
            confidence=0.99,
            language=LanguageCode.ENGLISH,
            estimated_grade=9,
            has_physics_request=True,
            is_follow_up=False,
            prefer_visual=False,
            requested_quantities=(
                requested_quantities
            ),
        )

    def make_answer(
        self,
        direct_answer: str,
        *,
        steps: list[str] | None = None,
        formulae: list[FormulaItem] | None = None,
        final_result: str | None = None,
    ) -> TutorAnswer:
        return TutorAnswer(
            answer_type=(
                AnswerType.NUMERICAL_SOLUTION
            ),
            direct_answer=direct_answer,
            steps=steps or [],
            formulae=formulae or [],
            diagram_explanation=None,
            common_mistake=None,
            final_result=final_result,
            source_pages=[],
            citations=[],
        )

    def status(
        self,
        report,
        name: str,
    ) -> NumericalCheckStatus:
        for check in report.checks:
            if check.name == name:
                return check.status

        raise AssertionError(
            f"Missing numerical check: {name}"
        )

    def test_non_numerical_answer_is_not_checked(
        self,
    ) -> None:
        answer = TutorAnswer(
            answer_type=(
                AnswerType.CONCEPT_EXPLANATION
            ),
            direct_answer=(
                "Force is a push or pull."
            ),
            steps=[],
            formulae=[],
            diagram_explanation=None,
            common_mistake=None,
            final_result=None,
            source_pages=[],
            citations=[],
        )

        report = self.verifier.verify(
            intent=self.make_intent(),
            tutor_answer=answer,
        )

        self.assertFalse(
            report.available
        )

        self.assertEqual(
            report.checks,
            [],
        )

    def test_correct_arithmetic_case_passes(
        self,
    ) -> None:
        answer = self.make_answer(
            "The acceleration is 5 m/s^2.",
            steps=[
                (
                    "Given: F = 10 N and "
                    "m = 2 kg."
                ),
                "a = 10 / 2 = 5",
            ],
            formulae=[
                FormulaItem(
                    latex="F = ma",
                    meaning=(
                        "Force equals mass "
                        "times acceleration."
                    ),
                )
            ],
            final_result="5 m/s^2",
        )

        report = self.verifier.verify(
            intent=self.make_intent(
                requested_quantity=(
                    "acceleration"
                ),
                expected_dimension="L T^-2",
            ),
            tutor_answer=answer,
        )

        self.assertTrue(
            report.available
        )

        self.assertTrue(
            report.calculation_passed
        )

        self.assertEqual(
            self.status(
                report,
                "arithmetic",
            ),
            NumericalCheckStatus.PASS,
        )

        # No hard-coded Physics unit/dimension ontology
        # exists in the deterministic verifier.
        self.assertIsNone(
            report.dimensional_consistency_passed
        )

    def test_wrong_arithmetic_is_detected(
        self,
    ) -> None:
        answer = self.make_answer(
            "The acceleration is 6 m/s^2.",
            steps=[
                "a = 10 / 2 = 6"
            ],
            final_result="6 m/s^2",
        )

        report = self.verifier.verify(
            intent=self.make_intent(
                requested_quantity=(
                    "acceleration"
                ),
            ),
            tutor_answer=answer,
        )

        self.assertFalse(
            report.calculation_passed
        )

        self.assertEqual(
            self.status(
                report,
                "arithmetic",
            ),
            NumericalCheckStatus.FAIL,
        )

    def test_same_unit_equality_passes(
        self,
    ) -> None:
        answer = self.make_answer(
            "The two values agree.",
            steps=[
                "5 m/s = 5 m/s"
            ],
            final_result="5 m/s",
        )

        report = self.verifier.verify(
            intent=self.make_intent(),
            tutor_answer=answer,
        )

        self.assertTrue(
            report.units_passed
        )

        self.assertEqual(
            self.status(
                report,
                "unit_consistency",
            ),
            NumericalCheckStatus.PASS,
        )

    def test_same_unit_mismatch_is_detected(
        self,
    ) -> None:
        answer = self.make_answer(
            "The two values disagree.",
            steps=[
                "5 m/s = 6 m/s"
            ],
            final_result="6 m/s",
        )

        report = self.verifier.verify(
            intent=self.make_intent(),
            tutor_answer=answer,
        )

        self.assertFalse(
            report.units_passed
        )

        self.assertEqual(
            self.status(
                report,
                "unit_consistency",
            ),
            NumericalCheckStatus.FAIL,
        )

    def test_cross_unit_conversion_is_not_guessed(
        self,
    ) -> None:
        answer = self.make_answer(
            "The speed is 10 m/s.",
            steps=[
                "36 km/h = 10 m/s"
            ],
            final_result="10 m/s",
        )

        report = self.verifier.verify(
            intent=self.make_intent(
                requested_quantity="speed",
                expected_dimension="L T^-1",
            ),
            tutor_answer=answer,
        )

        self.assertIsNone(
            report.units_passed
        )

        self.assertEqual(
            self.status(
                report,
                "unit_consistency",
            ),
            NumericalCheckStatus.NOT_CHECKED,
        )

        self.assertEqual(
            self.status(
                report,
                "requested_result_unit",
            ),
            NumericalCheckStatus.NOT_CHECKED,
        )

    def test_requested_result_unit_is_not_guessed(
        self,
    ) -> None:
        answer = self.make_answer(
            "The acceleration is 5 N.",
            steps=[
                "10 / 2 = 5"
            ],
            final_result="5 N",
        )

        report = self.verifier.verify(
            intent=self.make_intent(
                requested_quantity=(
                    "acceleration"
                ),
                expected_dimension="L T^-2",
            ),
            tutor_answer=answer,
        )

        # The structured requested quantity reaches
        # the verifier, but semantic unit compatibility
        # remains the LLM Verifier Agent's responsibility
        # until a generic upstream unit registry exists.
        self.assertEqual(
            self.status(
                report,
                "requested_result_unit",
            ),
            NumericalCheckStatus.NOT_CHECKED,
        )

    def test_sign_and_sig_figs_are_not_guessed(
        self,
    ) -> None:
        answer = self.make_answer(
            "The acceleration is 5 m/s^2.",
            steps=[
                "10 / 2 = 5"
            ],
            final_result="5 m/s^2",
        )

        report = self.verifier.verify(
            intent=self.make_intent(
                requested_quantity=(
                    "acceleration"
                ),
            ),
            tutor_answer=answer,
        )

        self.assertIsNone(
            report.sign_convention_passed
        )

        self.assertIsNone(
            report.significant_figures_passed
        )

        self.assertEqual(
            self.status(
                report,
                "sign_convention",
            ),
            NumericalCheckStatus.NOT_CHECKED,
        )

        self.assertEqual(
            self.status(
                report,
                "significant_figures",
            ),
            NumericalCheckStatus.NOT_CHECKED,
        )

    def test_prompt_payload_is_json_safe(
        self,
    ) -> None:
        answer = self.make_answer(
            "The speed is 10 m/s.",
            steps=[
                "36 km/h = 10 m/s"
            ],
            final_result="10 m/s",
        )

        payload = self.verifier.verify(
            intent=self.make_intent(
                requested_quantity="speed",
            ),
            tutor_answer=answer,
        ).as_prompt_payload()

        self.assertIsInstance(
            payload,
            dict,
        )

        self.assertIn(
            "available",
            payload,
        )

        self.assertIn(
            "checks",
            payload,
        )


if __name__ == "__main__":
    unittest.main()
