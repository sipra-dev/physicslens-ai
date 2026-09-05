from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv


PROJECT_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[2]
)

load_dotenv(
    PROJECT_ROOT / ".env"
)


def _read_positive_int(
    name: str,
    default: int,
) -> int:
    raw_value = os.getenv(name)

    if (
        raw_value is None
        or not raw_value.strip()
    ):
        return default

    try:
        value = int(
            raw_value
        )

    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} "
            "must be an integer."
        ) from exc

    if value <= 0:
        raise ValueError(
            f"Environment variable {name} "
            "must be greater than zero."
        )

    return value


def _read_bool(
    name: str,
    default: bool,
) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized_value = (
        raw_value.strip().lower()
    )

    if normalized_value in {
        "true",
        "1",
        "yes",
        "on",
    }:
        return True

    if normalized_value in {
        "false",
        "0",
        "no",
        "off",
    }:
        return False

    raise ValueError(
        f"Environment variable {name} "
        "must be true or false."
    )


def _read_csv(
    name: str,
    default: list[str],
) -> list[str]:
    raw_value = os.getenv(name)

    if (
        raw_value is None
        or not raw_value.strip()
    ):
        return default.copy()

    values = [
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    ]

    return (
        values
        or default.copy()
    )


def _resolve_path(
    raw_path: str | None,
    default_path: Path,
) -> Path:
    if (
        raw_path is None
        or not raw_path.strip()
    ):
        return (
            default_path.resolve()
        )

    path = Path(
        raw_path
    ).expanduser()

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


class Settings:
    """
    Central source of application configuration.

    Other modules should use the shared
    `settings` object instead of reading
    environment variables directly.
    """

    def __init__(self) -> None:
        # -----------------------------------------
        # Core application settings
        # -----------------------------------------

        self.app_name = os.getenv(
            "APP_NAME",
            "PhyMentor AI",
        )

        self.app_version = os.getenv(
            "APP_VERSION",
            "0.1.0",
        )

        self.environment = os.getenv(
            "ENVIRONMENT",
            "local",
        )

        self.debug = _read_bool(
            "APP_DEBUG",
            True,
        )

        self.api_prefix = os.getenv(
            "API_PREFIX",
            "/v1",
        ).rstrip("/")

        self.project_root = (
            PROJECT_ROOT
        )

        # -----------------------------------------
        # Phase 1:
        # Upload / validation / storage
        # -----------------------------------------

        self.upload_dir = (
            _resolve_path(
                os.getenv(
                    "UPLOAD_DIR"
                ),
                PROJECT_ROOT
                / "uploads",
            )
        )

        self.max_upload_size_mb = (
            _read_positive_int(
                "MAX_UPLOAD_SIZE_MB",
                20,
            )
        )

        self.max_upload_size_bytes = (
            self.max_upload_size_mb
            * 1024
            * 1024
        )

        self.max_pdf_pages = (
            _read_positive_int(
                "MAX_PDF_PAGES",
                100,
            )
        )

        self.max_image_pixels = (
            _read_positive_int(
                "MAX_IMAGE_PIXELS",
                40_000_000,
            )
        )

        self.allowed_extensions = tuple(
            value.lower().lstrip(".")
            for value in _read_csv(
                "ALLOWED_EXTENSIONS",
                [
                    "pdf",
                    "png",
                    "jpg",
                    "jpeg",
                    "webp",
                ],
            )
        )

        self.allowed_mime_types = tuple(
            value.lower()
            for value in _read_csv(
                "ALLOWED_MIME_TYPES",
                [
                    "application/pdf",
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    (
                        "application/"
                        "octet-stream"
                    ),
                ],
            )
        )

        self.cors_origins = (
            _read_csv(
                "CORS_ORIGINS",
                [
                    (
                        "http://"
                        "localhost:8501"
                    ),
                    (
                        "http://"
                        "127.0.0.1:8501"
                    ),
                ],
            )
        )

        self.default_local_user_id = (
            os.getenv(
                "DEFAULT_LOCAL_USER_ID",
                "local-user",
            ).strip()
        )

        self.default_rate_limit_per_minute = (
            _read_positive_int(
                (
                    "DEFAULT_RATE_LIMIT_"
                    "PER_MINUTE"
                ),
                60,
            )
        )

        self.upload_rate_limit_per_minute = (
            _read_positive_int(
                (
                    "UPLOAD_RATE_LIMIT_"
                    "PER_MINUTE"
                ),
                10,
            )
        )

        self.log_level = os.getenv(
            "LOG_LEVEL",
            "INFO",
        ).upper()

        # -----------------------------------------
        # Phase 2:
        # Parsing / OCR / scope classification
        # -----------------------------------------

        self.render_dpi = (
            _read_positive_int(
                "RENDER_DPI",
                180,
            )
        )

        self.minimum_native_text_characters = (
            _read_positive_int(
                (
                    "MINIMUM_NATIVE_TEXT_"
                    "CHARACTERS"
                ),
                40,
            )
        )

        self.ocr_languages = os.getenv(
            "OCR_LANGUAGES",
            "eng",
        ).strip()

        self.ocr_minimum_confidence = (
            float(
                os.getenv(
                    (
                        "OCR_MINIMUM_"
                        "CONFIDENCE"
                    ),
                    "25",
                )
            )
        )

        raw_tesseract_command = (
            os.getenv(
                "TESSERACT_COMMAND",
                "",
            ).strip()
        )

        self.tesseract_command = (
            raw_tesseract_command
            or None
        )

        self.openai_api_key = (
            os.getenv(
                "OPENAI_API_KEY",
                "",
            ).strip()
            or None
        )

        # -----------------------------------------
        # Document visual understanding
        # -----------------------------------------
        #
        # Document-involved visual/equation work uses GPT-4o.
        # These remain environment-configurable so production
        # can pin or change models without touching code.
        self.figure_vision_model = (
            os.getenv(
                "FIGURE_VISION_MODEL",
                "gpt-4o",
            ).strip()
        )

        self.equation_vision_model = (
            os.getenv(
                "EQUATION_VISION_MODEL",
                "gpt-4o",
            ).strip()
        )

        self.scope_classifier_model = (
            os.getenv(
                "SCOPE_CLASSIFIER_MODEL",
                "gpt-5.6",
            ).strip()
        )

        self.llm_timeout_seconds = (
            float(
                os.getenv(
                    "LLM_TIMEOUT_SECONDS",
                    "30",
                )
            )
        )

        self.use_llm_scope_classifier = (
            _read_bool(
                (
                    "USE_LLM_SCOPE_"
                    "CLASSIFIER"
                ),
                True,
            )
        )

        self.process_uploads_in_background = (
            _read_bool(
                (
                    "PROCESS_UPLOADS_"
                    "IN_BACKGROUND"
                ),
                True,
            )
        )

        # -----------------------------------------
        # Phase 3:
        # Chunking / FAISS / BM25
        # -----------------------------------------

        self.vector_store_dir = (
            _resolve_path(
                os.getenv(
                    "VECTOR_STORE_DIR"
                ),
                PROJECT_ROOT
                / "storage"
                / "vector_store",
            )
        )

        self.bm25_store_dir = (
            _resolve_path(
                os.getenv(
                    "BM25_STORE_DIR"
                ),
                PROJECT_ROOT
                / "storage"
                / "bm25_store",
            )
        )

        self.embedding_model_name = (
            os.getenv(
                "EMBEDDING_MODEL_NAME",
                (
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                ),
            ).strip()
        )

        self.child_chunk_max_characters = (
            _read_positive_int(
                (
                    "CHILD_CHUNK_MAX_"
                    "CHARACTERS"
                ),
                900,
            )
        )

        self.child_chunk_overlap_characters = (
            _read_positive_int(
                (
                    "CHILD_CHUNK_OVERLAP_"
                    "CHARACTERS"
                ),
                120,
            )
        )

        self.minimum_child_chunk_characters = (
            _read_positive_int(
                (
                    "MINIMUM_CHILD_CHUNK_"
                    "CHARACTERS"
                ),
                80,
            )
        )

        # -----------------------------------------
        # Phase 4:
        # Fusion / reranking / compression
        # -----------------------------------------

        self.rrf_k = (
            _read_positive_int(
                "RRF_K",
                60,
            )
        )

        self.hybrid_candidate_pool_size = (
            _read_positive_int(
                (
                    "HYBRID_CANDIDATE_"
                    "POOL_SIZE"
                ),
                30,
            )
        )

        self.retrieval_per_source_top_k = (
            _read_positive_int(
                (
                    "RETRIEVAL_PER_SOURCE_"
                    "TOP_K"
                ),
                30,
            )
        )

        self.reranker_model_name = (
            os.getenv(
                "RERANKER_MODEL_NAME",
                (
                    "cross-encoder/"
                    "ms-marco-MiniLM-L6-v2"
                ),
            ).strip()
        )

        self.reranker_batch_size = (
            _read_positive_int(
                "RERANKER_BATCH_SIZE",
                8,
            )
        )

        self.reranker_top_k = (
            _read_positive_int(
                "RERANKER_TOP_K",
                8,
            )
        )

        self.final_context_count = (
            _read_positive_int(
                "FINAL_CONTEXT_COUNT",
                6,
            )
        )

        self.max_context_characters = (
            _read_positive_int(
                "MAX_CONTEXT_CHARACTERS",
                12000,
            )
        )

        self.max_context_item_characters = (
            _read_positive_int(
                (
                    "MAX_CONTEXT_ITEM_"
                    "CHARACTERS"
                ),
                3000,
            )
        )

        # -----------------------------------------
        # Phase 5:
        # Query understanding / model gateway /
        # bounded Tutor-Verifier model routing
        # -----------------------------------------

        # Strong model for semantic routing/query understanding.
        self.query_understanding_model = (
            os.getenv(
                "QUERY_UNDERSTANDING_MODEL",
                "gpt-5.6",
            ).strip()
        )

        # Backward-compatible Phase-5 classifier setting.
        self.phase5_classifier_model = (
            os.getenv(
                "PHASE5_CLASSIFIER_MODEL",
                "gpt-5.6",
            ).strip()
        )

        # -----------------------------------------
        # High-level answer routing
        # -----------------------------------------
        #
        # Intended policy:
        #
        #   no document dependency -> GPT-5.6
        #   document dependency    -> GPT-4o
        #
        # Existing tutor_* settings remain below for
        # compatibility while routing is migrated gradually.
        self.general_reasoning_model = (
            os.getenv(
                "GENERAL_REASONING_MODEL",
                "gpt-5.6",
            ).strip()
        )

        self.document_reasoning_model = (
            os.getenv(
                "DOCUMENT_REASONING_MODEL",
                "gpt-4o",
            ).strip()
        )

        self.tutor_text_model = (
            os.getenv(
                "TUTOR_TEXT_MODEL",
                "gpt-5.6",
            ).strip()
        )

        self.tutor_multimodal_model = (
            os.getenv(
                "TUTOR_MULTIMODAL_MODEL",
                "gpt-4o",
            ).strip()
        )

        self.tutor_reasoning_model = (
            os.getenv(
                "TUTOR_REASONING_MODEL",
                "gpt-5.6",
            ).strip()
        )

        self.verifier_model = (
            os.getenv(
                "VERIFIER_MODEL",
                "gpt-5.6",
            ).strip()
        )

        # Explicit verifier policy for the new routing design.
        self.general_verifier_model = (
            os.getenv(
                "GENERAL_VERIFIER_MODEL",
                "gpt-5.6",
            ).strip()
        )

        self.document_verifier_model = (
            os.getenv(
                "DOCUMENT_VERIFIER_MODEL",
                "gpt-4o",
            ).strip()
        )

        raw_model_fallback = (
            os.getenv(
                "MODEL_FALLBACK_MODEL",
                "gpt-4o",
            ).strip()
        )

        self.model_fallback_model = (
            raw_model_fallback
            or None
        )

        # The existing Phase-2 LLM timeout remains
        # the single application-wide timeout source.
        # Phase-5 gateway will reuse this value rather
        # than introducing a second conflicting timeout.
        self.model_gateway_timeout_seconds = (
            self.llm_timeout_seconds
        )

        # -----------------------------------------
        # Phase 7:
        # Redis / cache / short-term memory
        # -----------------------------------------

        self.redis_url = (
            os.getenv(
                "REDIS_URL",
                "",
            ).strip()
            or None
        )

        # -----------------------------------------
        # Phase 7:
        # PostgreSQL durable long-term memory
        # -----------------------------------------

        self.database_url = (
            os.getenv(
                "DATABASE_URL",
                "",
            ).strip()
            or None
        )

        # -----------------------------------------
        # Phase 7:
        # Pinecone semantic learning memory
        # -----------------------------------------

        self.pinecone_api_key = (
            os.getenv(
                "PINECONE_API_KEY",
                "",
            ).strip()
            or None
        )

        self.pinecone_learning_memory_index = (
            os.getenv(
                "PINECONE_LEARNING_MEMORY_INDEX",
                "phymentor-learning-memory",
            ).strip()
        )

    def create_required_directories(
        self,
    ) -> None:
        """
        Create local directories required by
        the application.

        This method must only create directories.
        Configuration values are initialized
        inside __init__.
        """

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.bm25_store_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


settings = Settings()