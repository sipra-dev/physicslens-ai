from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from src.ingestion.models import (
    OCRDocumentResult,
    ParsedDocument,
    ScopeClassification,
    ScopeDecision,
)
from src.prompts.scope import (
    SCOPE_CLASSIFIER_SYSTEM_PROMPT,
)


# Canonical topic name → phrases that indicate the topic.
# Final metadata will contain canonical names, not every matched phrase.
_PHYSICS_TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "measurement_and_units": (
        "measurement",
        "physical quantity",
        "si unit",
        "units and dimensions",
        "dimensional analysis",
    ),
    "motion_and_kinematics": (
        "motion",
        "distance",
        "displacement",
        "speed",
        "velocity",
        "acceleration",
        "kinematics",
    ),
    "force_and_newton_laws": (
        "force",
        "newton's law",
        "newton law",
        "momentum",
        "impulse",
        "friction",
        "inertia",
    ),
    "work_energy_power": (
        "work",
        "kinetic energy",
        "potential energy",
        "mechanical energy",
        "power",
        "conservation of energy",
    ),
    "gravitation": (
        "gravitation",
        "gravity",
        "gravitational force",
        "free fall",
    ),
    "matter_pressure_fluids": (
        "density",
        "pressure",
        "buoyancy",
        "fluid",
        "archimedes",
        "surface tension",
    ),
    "heat_and_thermodynamics": (
        "heat",
        "temperature",
        "specific heat",
        "thermal expansion",
        "thermodynamics",
        "calorimetry",
    ),
    "oscillations_and_shm": (
        "oscillation",
        "simple harmonic motion",
        "shm",
        "restoring force",
        "amplitude",
        "time period",
        "angular frequency",
        "spring constant",
    ),
    "waves_and_sound": (
        "wave",
        "wavelength",
        "frequency",
        "wave speed",
        "sound",
        "doppler effect",
        "resonance",
    ),
    "light_and_optics": (
        "light",
        "reflection",
        "refraction",
        "lens",
        "mirror",
        "optics",
        "refractive index",
        "total internal reflection",
    ),
    "electricity_and_circuits": (
        "electricity",
        "electric current",
        "electric charge",
        "voltage",
        "potential difference",
        "resistance",
        "ohm's law",
        "electric circuit",
    ),
    "magnetism": (
        "magnetism",
        "magnetic field",
        "magnetic force",
        "bar magnet",
        "electromagnet",
    ),
    "electromagnetic_effects": (
        "electromagnetic induction",
        "electromagnetic effect",
        "faraday's law",
        "lenz's law",
        "electric motor",
        "electric generator",
    ),
    "atomic_and_nuclear_physics": (
        "atomic physics",
        "atom",
        "nucleus",
        "radioactivity",
        "half life",
        "nuclear fission",
        "nuclear fusion",
    ),
    "basic_electronics": (
        "semiconductor",
        "diode",
        "transistor",
        "logic gate",
        "rectifier",
    ),
}


_NON_PHYSICS_PATTERNS: dict[str, tuple[str, ...]] = {
    "biology": (
        "photosynthesis",
        "cell division",
        "mitosis",
        "meiosis",
        "dna",
        "respiration",
        "digestive system",
        "human anatomy",
    ),
    "chemistry": (
        "chemical reaction",
        "organic chemistry",
        "periodic table",
        "acid and base",
        "covalent bond",
        "ionic bond",
        "mole concept",
    ),
    "computer_science": (
        "python programming",
        "source code",
        "database management",
        "machine learning algorithm",
        "software engineering",
    ),
    "history_or_social_science": (
        "world war",
        "political science",
        "historical event",
        "geography",
        "constitution",
    ),
    "language_or_literature": (
        "grammar",
        "poetry",
        "literature",
        "novel",
        "essay writing",
    ),
}


_ADVANCED_PHYSICS_PATTERNS: dict[str, tuple[str, ...]] = {
    "quantum_field_theory": (
        "quantum field theory",
        "lagrangian density",
        "path integral",
        "renormalization",
        "feynman propagator",
    ),
    "advanced_general_relativity": (
        "riemann tensor",
        "ricci tensor",
        "einstein field equation",
        "differential geometry",
        "tensor calculus",
    ),
    "advanced_quantum_mechanics": (
        "hilbert space",
        "density operator",
        "dirac notation",
        "perturbation theory",
        "quantum electrodynamics",
    ),
    "advanced_electrodynamics": (
        "electromagnetic field tensor",
        "maxwell stress tensor",
        "covariant electrodynamics",
    ),
    "advanced_statistical_physics": (
        "partition function",
        "statistical field theory",
        "grand canonical ensemble",
    ),
}


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """
    Build a boundary-aware regex for a word or multi-word phrase.

    Examples:
    - 'light' will not match 'slightly'
    - 'magnetic' will not match the magnetic part of 'electromagnetic'
    - multiple spaces and hyphens are tolerated
    """

    escaped = re.escape(phrase.strip())

    escaped = escaped.replace(
        r"\ ",
        r"[\s\-–—]+",
    )

    return re.compile(
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )


def _contains_phrase(
    text: str,
    phrase: str,
) -> bool:
    return bool(
        _phrase_pattern(phrase).search(text)
    )


def _match_categories(
    *,
    text: str,
    category_patterns: dict[str, tuple[str, ...]],
) -> tuple[list[str], list[str]]:
    """
    Return:
    1. Matched canonical category names
    2. Actual phrases that caused matches
    """

    matched_categories: list[str] = []
    matched_phrases: list[str] = []

    for category, phrases in category_patterns.items():
        category_matched = False

        for phrase in phrases:
            if _contains_phrase(text, phrase):
                matched_phrases.append(phrase)
                category_matched = True

        if category_matched:
            matched_categories.append(category)

    return (
        sorted(set(matched_categories)),
        sorted(set(matched_phrases)),
    )


class ScopeClassifier:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float = 30.0,
        use_llm: bool = True,
    ) -> None:
        self.model = model

        self.use_llm = bool(
            use_llm and api_key
        )

        self.client = (
            OpenAI(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=1,
            )
            if self.use_llm
            else None
        )

    def classify(
        self,
        *,
        parsed_document: ParsedDocument,
        ocr_result: OCRDocumentResult,
    ) -> ScopeClassification:
        combined_text = self._combine_text(
            parsed_document=parsed_document,
            ocr_result=ocr_result,
        )

        heuristic_result = (
            self._heuristic_classification(
                combined_text
            )
        )

        should_use_llm = (
            self.use_llm
            and (
                heuristic_result.decision
                == ScopeDecision.NEEDS_REVIEW
                or heuristic_result.confidence < 0.85
                or not combined_text.strip()
            )
        )

        if not should_use_llm:
            return heuristic_result

        try:
            return self._classify_with_openai(
                text=combined_text,
                image_paths=[
                    Path(page.rendered_image_path)
                    for page in parsed_document.pages[:3]
                ],
            )

        except Exception:
            fallback_data = (
                heuristic_result.model_dump()
            )

            fallback_data["classifier"] = (
                "heuristic_fallback"
            )

            fallback_data["reasoning"] = (
                f"{heuristic_result.reasoning} "
                "The LLM classifier was unavailable, "
                "so the heuristic result was retained."
            )

            return ScopeClassification(
                **fallback_data
            )

    def _combine_text(
        self,
        *,
        parsed_document: ParsedDocument,
        ocr_result: OCRDocumentResult,
    ) -> str:
        text_parts: list[str] = []

        ocr_by_page = {
            page.page_number: page
            for page in ocr_result.pages
        }

        for parsed_page in parsed_document.pages:
            native_text = (
                parsed_page.native_text.strip()
            )

            if native_text:
                text_parts.append(
                    f"[Page {parsed_page.page_number}]\n"
                    f"{native_text}"
                )
                continue

            ocr_page = ocr_by_page.get(
                parsed_page.page_number
            )

            if (
                ocr_page is not None
                and ocr_page.text.strip()
            ):
                text_parts.append(
                    f"[Page {parsed_page.page_number} OCR]\n"
                    f"{ocr_page.text.strip()}"
                )

        return "\n\n".join(text_parts)

    def _heuristic_classification(
        self,
        text: str,
    ) -> ScopeClassification:
        normalized_text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        (
            physics_topics,
            physics_phrases,
        ) = _match_categories(
            text=normalized_text,
            category_patterns=(
                _PHYSICS_TOPIC_PATTERNS
            ),
        )

        (
            non_physics_categories,
            non_physics_phrases,
        ) = _match_categories(
            text=normalized_text,
            category_patterns=(
                _NON_PHYSICS_PATTERNS
            ),
        )

        (
            advanced_topics,
            advanced_phrases,
        ) = _match_categories(
            text=normalized_text,
            category_patterns=(
                _ADVANCED_PHYSICS_PATTERNS
            ),
        )

        if advanced_topics:
            return ScopeClassification(
                is_physics=True,
                school_level=False,
                estimated_grade_min=None,
                estimated_grade_max=None,
                topics=advanced_topics,
                confidence=min(
                    0.98,
                    0.90
                    + 0.02 * len(advanced_topics),
                ),
                decision=(
                    ScopeDecision.REJECT_ADVANCED
                ),
                reasoning=(
                    "Advanced university-level Physics "
                    f"content was detected: "
                    f"{', '.join(advanced_phrases[:5])}."
                ),
                classifier="heuristic",
            )

        physics_score = len(physics_topics)
        non_physics_score = len(
            non_physics_categories
        )

        if (
            non_physics_score >= 2
            and non_physics_score > physics_score
        ):
            return ScopeClassification(
                is_physics=False,
                school_level=False,
                estimated_grade_min=None,
                estimated_grade_max=None,
                topics=non_physics_categories,
                confidence=min(
                    0.97,
                    0.82
                    + 0.04 * non_physics_score,
                ),
                decision=(
                    ScopeDecision.REJECT_NON_PHYSICS
                ),
                reasoning=(
                    "The document is dominated by "
                    "non-Physics subject content: "
                    f"{', '.join(non_physics_phrases[:5])}."
                ),
                classifier="heuristic",
            )

        if physics_score >= 3:
            return ScopeClassification(
                is_physics=True,
                school_level=True,
                estimated_grade_min=6,
                estimated_grade_max=12,
                topics=physics_topics,
                confidence=min(
                    0.96,
                    0.80
                    + 0.035 * physics_score,
                ),
                decision=ScopeDecision.ACCEPT,
                reasoning=(
                    "Multiple distinct school-level "
                    "Physics topics were detected."
                ),
                classifier="heuristic",
            )

        if physics_score in {1, 2}:
            return ScopeClassification(
                is_physics=True,
                school_level=True,
                estimated_grade_min=6,
                estimated_grade_max=12,
                topics=physics_topics,
                confidence=0.68,
                decision=(
                    ScopeDecision.NEEDS_REVIEW
                ),
                reasoning=(
                    "Some school-level Physics content "
                    "was detected, but confidence is not "
                    "high enough for a deterministic decision."
                ),
                classifier="heuristic",
            )

        if (
            non_physics_score >= 1
            and physics_score == 0
        ):
            return ScopeClassification(
                is_physics=False,
                school_level=False,
                estimated_grade_min=None,
                estimated_grade_max=None,
                topics=non_physics_categories,
                confidence=0.72,
                decision=(
                    ScopeDecision.NEEDS_REVIEW
                ),
                reasoning=(
                    "Some non-Physics content was detected, "
                    "but confidence is not high enough for "
                    "an automatic rejection."
                ),
                classifier="heuristic",
            )

        return ScopeClassification(
            is_physics=False,
            school_level=False,
            estimated_grade_min=None,
            estimated_grade_max=None,
            topics=[],
            confidence=0.35,
            decision=ScopeDecision.NEEDS_REVIEW,
            reasoning=(
                "There is not enough reliable text "
                "for a deterministic scope decision."
            ),
            classifier="heuristic",
        )

    def _classify_with_openai(
        self,
        *,
        text: str,
        image_paths: Iterable[Path],
    ) -> ScopeClassification:
        if self.client is None:
            raise RuntimeError(
                "OpenAI client is unavailable."
            )

        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Classify this uploaded document "
                    "for PhyMentor AI.\n\n"
                    "Extracted text:\n"
                    f"{text[:14000] or '[No reliable text extracted]'}"
                ),
            }
        ]

        for image_path in image_paths:
            if not image_path.is_file():
                continue

            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._to_data_url(
                            image_path
                        ),
                        "detail": "low",
                    },
                }
            )

        completion = (
            self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            SCOPE_CLASSIFIER_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                response_format=(
                    ScopeClassification
                ),
                temperature=0,
            )
        )

        message = completion.choices[0].message

        if message.parsed is None:
            raise RuntimeError(
                message.refusal
                or "The classifier returned no result."
            )

        parsed_result = message.parsed

        result_data = parsed_result.model_dump()

        result_data["classifier"] = "openai"

        return ScopeClassification(
            **result_data
        )

    def _to_data_url(
        self,
        image_path: Path,
    ) -> str:
        mime_type = (
            mimetypes.guess_type(
                image_path.name
            )[0]
            or "image/png"
        )

        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

        return (
            f"data:{mime_type};base64,{encoded}"
        )