from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import Settings
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.equations import EquationExtractor
from src.ingestion.figures import FigureExtractor
from src.ingestion.indexer import LocalDocumentIndexer
from src.ingestion.layout import LayoutAnalyzer
from src.ingestion.models import (
    FigureArtifact,
    FigureExtractionResult,
    ScopeDecision,
)
from src.ingestion.ocr import OCRService
from src.ingestion.parser import DocumentParser
from src.ingestion.scope_classifier import ScopeClassifier
from src.ingestion.structure import DocumentStructureBuilder
from src.ingestion.validation import validate_upload
from src.models.gateway import LLMGateway
from src.models.routing import ModelRouter
from src.storage import StorageBackend


logger = logging.getLogger("phymentor.ingestion")


class DocumentNotFoundError(Exception):
    """Raised when a user's document cannot be found."""


class DocumentProcessingError(Exception):
    """Raised when the ingestion pipeline fails."""


class IngestionService:
    def __init__(
        self,
        *,
        storage: StorageBackend,
        application_settings: Settings,
        llm_gateway: LLMGateway | None = None,
        model_router: ModelRouter | None = None,
    ) -> None:
        if (llm_gateway is None) != (model_router is None):
            raise ValueError(
                "llm_gateway and model_router must either both be provided "
                "or both be omitted."
            )

        self.storage = storage
        self.settings = application_settings

        self.structure_builder = (
            DocumentStructureBuilder(
                gateway=llm_gateway,
                model_router=model_router,
            )
            if llm_gateway is not None
            and model_router is not None
            else None
        )

        self.parser = DocumentParser(
            render_dpi=application_settings.render_dpi,
            minimum_native_text_characters=(
                application_settings.minimum_native_text_characters
            ),
        )

        self.layout_analyzer = LayoutAnalyzer()

        self.ocr_service = OCRService(
            languages=application_settings.ocr_languages,
            minimum_confidence=application_settings.ocr_minimum_confidence,
            tesseract_command=application_settings.tesseract_command,
        )

        self.equation_extractor = EquationExtractor(
            api_key=application_settings.openai_api_key,
            model=getattr(
                application_settings,
                "equation_vision_model",
                "gpt-4.1-mini",
            ),
            timeout_seconds=application_settings.llm_timeout_seconds,
        )

        self.figure_extractor = FigureExtractor(
            api_key=application_settings.openai_api_key,
            model=getattr(
                application_settings,
                "figure_vision_model",
                getattr(
                    application_settings,
                    "equation_vision_model",
                    "gpt-4.1-mini",
                ),
            ),
            timeout_seconds=application_settings.llm_timeout_seconds,
        )

        self.scope_classifier = ScopeClassifier(
            api_key=application_settings.openai_api_key,
            model=application_settings.scope_classifier_model,
            timeout_seconds=application_settings.llm_timeout_seconds,
            use_llm=application_settings.use_llm_scope_classifier,
        )

        self.chunker = HierarchicalChunker(
            child_max_characters=(
                application_settings.child_chunk_max_characters
            ),
            child_overlap_characters=(
                application_settings.child_chunk_overlap_characters
            ),
            minimum_child_characters=(
                application_settings.minimum_child_chunk_characters
            ),
        )

        self.indexer = LocalDocumentIndexer(
            vector_store_directory=application_settings.vector_store_dir,
            bm25_store_directory=application_settings.bm25_store_dir,
            embedding_model_name=application_settings.embedding_model_name,
        )

    # =========================================================
    # UPLOAD DOCUMENT
    # =========================================================

    def upload_document(
        self,
        *,
        user_id: str,
        filename: str | None,
        content_type: str | None,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        """
        Validate and save a document.

        Phase 7:
        If the same user already uploaded a document
        with the same SHA-256, reuse the existing one.
        """

        normalized_user_id = user_id.strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty.")

        validation_result = validate_upload(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            application_settings=self.settings,
        )

        # -----------------------------------------------------
        # DOCUMENT DEDUPLICATION
        # -----------------------------------------------------

        existing_metadata = self.storage.find_document_by_sha256(
            user_id=normalized_user_id,
            sha256=validation_result.sha256,
        )

        if existing_metadata is not None:
            result = dict(existing_metadata)

            existing_document_id = result.get("document_id")

            if (
                not isinstance(existing_document_id, str)
                or not existing_document_id.strip()
            ):
                raise DocumentProcessingError(
                    "Stored duplicate document metadata is invalid."
                )

            result["document_id"] = existing_document_id.strip()
            result["_deduplicated"] = True

            result["message"] = (
                "This file was already uploaded. "
                "The existing document is being reused."
            )

            return result

        # -----------------------------------------------------
        # NEW DOCUMENT
        # -----------------------------------------------------

        document_id = uuid4().hex

        stored_file = self.storage.save_original_file(
            user_id=normalized_user_id,
            document_id=document_id,
            original_filename=validation_result.original_filename,
            file_bytes=file_bytes,
        )

        uploaded_at = datetime.now(timezone.utc)

        metadata: dict[str, Any] = {
            "document_id": document_id,
            "user_id": normalized_user_id,
            "status": "UPLOADED",
            "processing_stage": "UPLOADED",
            "original_filename": validation_result.original_filename,
            "stored_filename": stored_file.stored_filename,
            "content_type": validation_result.content_type,
            "file_extension": validation_result.extension,
            "size_bytes": validation_result.size_bytes,
            "sha256": validation_result.sha256,
            "page_count": validation_result.page_count,
            "image_width": validation_result.image_width,
            "image_height": validation_result.image_height,
            "storage_path": stored_file.relative_path,
            "uploaded_at": uploaded_at.isoformat(),
            "scope_classification": None,
            "artifacts": {},
            "structural_index_status": "NOT_STARTED",
            "structural_index_error": None,
            "processing_error": None,
            "message": "File uploaded and validated successfully.",
        }

        try:
            self.storage.write_document_metadata(
                user_id=normalized_user_id,
                document_id=document_id,
                metadata=metadata,
            )

        except Exception:
            try:
                self.storage.delete_document(
                    user_id=normalized_user_id,
                    document_id=document_id,
                )
            except Exception:
                pass

            raise

        result = dict(metadata)
        result["_deduplicated"] = False

        return result

    # =========================================================
    # QUEUE DOCUMENT PROCESSING
    # =========================================================

    def queue_document_processing(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        metadata = self._get_metadata(
            user_id=user_id,
            document_id=document_id,
        )

        metadata["status"] = "PROCESSING"
        metadata["processing_stage"] = "QUEUED"
        metadata["processing_error"] = None

        metadata["message"] = (
            "Document upload completed. "
            "Parsing and scope classification are running in the background."
        )

        self.storage.write_document_metadata(
            user_id=user_id,
            document_id=document_id,
            metadata=metadata,
        )

        return metadata

    # =========================================================
    # BACKGROUND PROCESSING
    # =========================================================

    def process_document_background(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> None:
        """
        Background wrapper.

        Normal processing failures are already recorded
        by process_document(), so they must not crash the
        completed HTTP upload request.
        """

        try:
            self.process_document(
                user_id=user_id,
                document_id=document_id,
            )

        except (DocumentProcessingError, DocumentNotFoundError):
            return

    # =========================================================
    # PROCESS DOCUMENT
    # =========================================================

    def process_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        metadata = self._get_metadata(
            user_id=user_id,
            document_id=document_id,
        )

        try:
            # -------------------------------------------------
            # DOCUMENT PATH
            # -------------------------------------------------

            document_directory = self.storage.get_document_directory(
                user_id=user_id,
                document_id=document_id,
            )

            storage_path = metadata.get("storage_path")

            if not isinstance(storage_path, str):
                raise DocumentProcessingError(
                    "Document storage metadata is invalid."
                )

            source_path = (
                self.settings.upload_dir / storage_path
            ).resolve()

            upload_root = self.settings.upload_dir.resolve()

            try:
                source_path.relative_to(upload_root)

            except ValueError as exc:
                raise DocumentProcessingError(
                    "Document storage path is outside "
                    "the configured upload directory."
                ) from exc

            analysis_directory = document_directory / "analysis"

            analysis_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata.setdefault("artifacts", {})
            metadata.setdefault(
                "structural_index_status",
                "NOT_STARTED",
            )
            metadata.setdefault(
                "structural_index_error",
                None,
            )

            # =================================================
            # PARSING
            # =================================================

            metadata = self._update_stage(
                metadata=metadata,
                user_id=user_id,
                document_id=document_id,
                status="PROCESSING",
                stage="PARSING",
            )

            parsed_document = self.parser.parse(
                document_id=document_id,
                source_path=source_path,
                output_directory=document_directory,
            )

            parsed_path = (
                analysis_directory / "parsed_document.json"
            )

            self._write_json_atomic(
                path=parsed_path,
                data=parsed_document.model_dump(mode="json"),
            )

            metadata["artifacts"]["parsed_document"] = (
                self._relative_to_upload_root(parsed_path)
            )

            # =================================================
            # LAYOUT ANALYSIS
            # =================================================

            metadata = self._update_stage(
                metadata=metadata,
                user_id=user_id,
                document_id=document_id,
                status="PROCESSING",
                stage="LAYOUT_ANALYSIS",
            )

            layout_result = self.layout_analyzer.analyze(
                parsed_document
            )

            layout_path = analysis_directory / "layout.json"

            self._write_json_atomic(
                path=layout_path,
                data=layout_result.model_dump(mode="json"),
            )

            metadata["artifacts"]["layout"] = (
                self._relative_to_upload_root(layout_path)
            )

            # =================================================
            # OCR
            # =================================================

            metadata = self._update_stage(
                metadata=metadata,
                user_id=user_id,
                document_id=document_id,
                status="PROCESSING",
                stage="OCR",
            )

            ocr_result = self.ocr_service.process(
                parsed_document
            )

            ocr_path = analysis_directory / "ocr.json"

            self._write_json_atomic(
                path=ocr_path,
                data=ocr_result.model_dump(mode="json"),
            )

            metadata["artifacts"]["ocr"] = (
                self._relative_to_upload_root(ocr_path)
            )

            # =================================================
            # EQUATION EXTRACTION
            # =================================================

            metadata = self._update_stage(
                metadata=metadata,
                user_id=user_id,
                document_id=document_id,
                status="PROCESSING",
                stage="EQUATION_EXTRACTION",
            )

            equation_result = self.equation_extractor.process(
                parsed_document=parsed_document,
                document_layout=layout_result,
                output_directory=document_directory,
            )

            equations_path = (
                analysis_directory / "equations.json"
            )

            self._write_json_atomic(
                path=equations_path,
                data=equation_result.model_dump(mode="json"),
            )

            metadata["artifacts"]["equations"] = (
                self._relative_to_upload_root(
                    equations_path
                )
            )

            # =================================================
            # FIGURE EXTRACTION
            # =================================================

            metadata = self._update_stage(
                metadata=metadata,
                user_id=user_id,
                document_id=document_id,
                status="PROCESSING",
                stage="FIGURE_EXTRACTION",
            )

            figure_result = self.figure_extractor.extract(
                parsed_document=parsed_document,
                document_layout=layout_result,
                output_directory=document_directory,
            )

            figures_path = (
                analysis_directory / "figures.json"
            )

            self._write_json_atomic(
                path=figures_path,
                data=figure_result.model_dump(mode="json"),
            )

            metadata["artifacts"]["figures"] = (
                self._relative_to_upload_root(
                    figures_path
                )
            )

            # =================================================
            # SCOPE CLASSIFICATION
            # =================================================

            metadata = self._update_stage(
                metadata=metadata,
                user_id=user_id,
                document_id=document_id,
                status="PROCESSING",
                stage="SCOPE_CLASSIFICATION",
            )

            scope_result = self.scope_classifier.classify(
                parsed_document=parsed_document,
                ocr_result=ocr_result,
            )

            scope_path = analysis_directory / "scope.json"

            self._write_json_atomic(
                path=scope_path,
                data=scope_result.model_dump(mode="json"),
            )

            metadata["artifacts"]["scope"] = (
                self._relative_to_upload_root(scope_path)
            )

            metadata["scope_classification"] = (
                scope_result.model_dump(mode="json")
            )

            # =================================================
            # SCOPE DECISION
            # =================================================

            if scope_result.decision in {
                ScopeDecision.REJECT_NON_PHYSICS,
                ScopeDecision.REJECT_ADVANCED,
            }:
                metadata["status"] = "REJECTED"
                metadata["processing_stage"] = (
                    "SCOPE_REJECTED"
                )

                metadata["message"] = (
                    "The document is outside the "
                    "supported Class 1–12 Physics scope."
                )

            elif (
                scope_result.decision
                == ScopeDecision.NEEDS_REVIEW
            ):
                metadata["status"] = "PROCESSING"
                metadata["processing_stage"] = (
                    "SCOPE_REVIEW_REQUIRED"
                )

                metadata["message"] = (
                    "The document could not be classified "
                    "with enough confidence."
                )

            else:
                # =============================================
                # STRUCTURAL DOCUMENT INDEXING
                # =============================================

                structure_result = None
                document_structure_path = None
                structure_links_ready = False

                if self.structure_builder is None:
                    # Backward-compatible staged rollout:
                    # existing construction sites continue through the
                    # original semantic ingestion path until the shared
                    # gateway/router are injected.
                    metadata["structural_index_status"] = (
                        "NOT_CONFIGURED"
                    )
                    metadata["structural_index_error"] = None

                else:
                    metadata = self._update_stage(
                        metadata=metadata,
                        user_id=user_id,
                        document_id=document_id,
                        status="PROCESSING",
                        stage="STRUCTURAL_INDEXING",
                    )

                    metadata["structural_index_status"] = (
                        "PROCESSING"
                    )
                    metadata["structural_index_error"] = None

                    try:
                        structure_result = (
                            self.structure_builder.build(
                                parsed_document=parsed_document,
                                document_layout=layout_result,
                            )
                        )

                        page_structures_path = (
                            analysis_directory
                            / "page_structures.json"
                        )

                        document_structure_path = (
                            analysis_directory
                            / "document_structure.json"
                        )

                        self._write_json_atomic(
                            path=page_structures_path,
                            data=(
                                structure_result
                                .page_structures
                                .model_dump(mode="json")
                            ),
                        )

                        self._write_json_atomic(
                            path=document_structure_path,
                            data=(
                                structure_result
                                .document_structure
                                .model_dump(mode="json")
                            ),
                        )

                        metadata["artifacts"][
                            "page_structures"
                        ] = self._relative_to_upload_root(
                            page_structures_path
                        )

                        metadata["artifacts"][
                            "structure"
                        ] = self._relative_to_upload_root(
                            document_structure_path
                        )

                        metadata["structural_index_status"] = (
                            "READY"
                        )
                        metadata["structural_index_error"] = None

                    except Exception as structure_exc:
                        # Structural indexing is an additive capability.
                        # Record its failure visibly, then preserve the
                        # existing semantic chunking and retrieval path.
                        metadata["structural_index_status"] = (
                            "FAILED"
                        )
                        metadata["structural_index_error"] = (
                            f"{type(structure_exc).__name__}: "
                            f"{structure_exc}"
                        )

                        logger.exception(
                            "structural_indexing_failed "
                            "user_id=%s document_id=%s",
                            user_id,
                            document_id,
                        )

                        structure_result = None
                        document_structure_path = None

                # =============================================
                # CHUNKING
                # =============================================

                metadata = self._update_stage(
                    metadata=metadata,
                    user_id=user_id,
                    document_id=document_id,
                    status="PROCESSING",
                    stage="CHUNKING",
                )

                if structure_result is None:
                    chunking_result = self.chunker.chunk(
                        user_id=user_id,
                        parsed_document=parsed_document,
                        document_layout=layout_result,
                        ocr_result=ocr_result,
                        figure_result=figure_result,
                        equation_result=equation_result,
                        scope_result=scope_result,
                    )

                else:
                    try:
                        chunking_result = self.chunker.chunk(
                            user_id=user_id,
                            parsed_document=parsed_document,
                            document_layout=layout_result,
                            ocr_result=ocr_result,
                            figure_result=figure_result,
                            equation_result=equation_result,
                            scope_result=scope_result,
                            document_structure=(
                                structure_result
                                .document_structure
                            ),
                        )

                        structure_links_ready = True

                    except Exception as linking_exc:
                        # Preserve the established semantic chunking path if
                        # optional structure-to-chunk linking fails.
                        metadata["structural_index_status"] = (
                            "LINKING_FAILED"
                        )
                        metadata["structural_index_error"] = (
                            f"{type(linking_exc).__name__}: "
                            f"{linking_exc}"
                        )

                        logger.exception(
                            "structural_semantic_linking_failed "
                            "user_id=%s document_id=%s",
                            user_id,
                            document_id,
                        )

                        chunking_result = self.chunker.chunk(
                            user_id=user_id,
                            parsed_document=parsed_document,
                            document_layout=layout_result,
                            ocr_result=ocr_result,
                            figure_result=figure_result,
                            equation_result=equation_result,
                            scope_result=scope_result,
                        )

                chunks_path = (
                    analysis_directory / "chunks.json"
                )

                self._write_json_atomic(
                    path=chunks_path,
                    data=chunking_result.model_dump(
                        mode="json"
                    ),
                )

                metadata["artifacts"]["chunks"] = (
                    self._relative_to_upload_root(
                        chunks_path
                    )
                )

                if (
                    structure_links_ready
                    and structure_result is not None
                    and document_structure_path is not None
                ):
                    # Chunker updates the supplied DocumentStructure with
                    # bidirectional structural/semantic IDs. Persist that
                    # linked version atomically over the pre-link artifact.
                    self._write_json_atomic(
                        path=document_structure_path,
                        data=(
                            structure_result
                            .document_structure
                            .model_dump(mode="json")
                        ),
                    )

                # =============================================
                # FAISS + BM25 INDEXING
                # =============================================

                metadata = self._update_stage(
                    metadata=metadata,
                    user_id=user_id,
                    document_id=document_id,
                    status="INDEXING",
                    stage="FAISS_BM25_INDEXING",
                )

                index_manifest = self.indexer.index_document(
                    chunking_result=chunking_result
                )

                metadata["index_manifest"] = (
                    index_manifest
                )

                metadata["status"] = "READY"
                metadata["processing_stage"] = "READY"

                if (
                    metadata["structural_index_status"]
                    in {
                        "FAILED",
                        "LINKING_FAILED",
                    }
                ):
                    metadata["message"] = (
                        "Document processing and local FAISS/BM25 "
                        "indexing completed successfully. Structural "
                        "indexing was unavailable, so semantic retrieval "
                        "remains active for this document."
                    )
                else:
                    metadata["message"] = (
                        "Document processing and local "
                        "FAISS/BM25 indexing completed "
                        "successfully. The document is "
                        "ready for question answering."
                    )

            # =================================================
            # SUCCESS METADATA COMMIT
            # =================================================

            metadata["processing_error"] = None

            self.storage.write_document_metadata(
                user_id=user_id,
                document_id=document_id,
                metadata=metadata,
            )

            return metadata

        except Exception as exc:
            # =================================================
            # GRACEFUL FAILURE
            # =================================================

            metadata["status"] = "FAILED"
            metadata["processing_stage"] = "FAILED"

            metadata["processing_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            metadata["message"] = (
                "Document processing failed. "
                "Check processing_error and server logs."
            )

            # Try to record FAILED state, but never
            # hide the original processing exception.
            try:
                self.storage.write_document_metadata(
                    user_id=user_id,
                    document_id=document_id,
                    metadata=metadata,
                )
            except Exception:
                pass

            raise DocumentProcessingError(
                "The document processing pipeline failed."
            ) from exc

    # =========================================================
    # GET STATUS
    # =========================================================

    def get_document_status(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        return self._get_metadata(
            user_id=user_id,
            document_id=document_id,
        )

    # =========================================================
    # FIGURE CATALOGUE
    # =========================================================

    def get_figure_catalogue(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> FigureExtractionResult:
        """
        Load the canonical figure catalogue recorded for one document.

        The artifact location comes from document metadata and is read through
        the configured StorageBackend. No upload-directory layout, figure
        number, Physics topic, or document-specific path is guessed here.
        """

        normalized_user_id = user_id.strip()
        normalized_document_id = document_id.strip()

        if not normalized_user_id:
            raise ValueError(
                "user_id cannot be empty."
            )

        if not normalized_document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        metadata = self._get_metadata(
            user_id=normalized_user_id,
            document_id=normalized_document_id,
        )

        artifacts = metadata.get(
            "artifacts"
        )

        if not isinstance(
            artifacts,
            dict,
        ):
            return FigureExtractionResult(
                document_id=normalized_document_id,
                figures=[],
            )

        artifact_path = artifacts.get(
            "figures"
        )

        if (
            not isinstance(
                artifact_path,
                str,
            )
            or not artifact_path.strip()
        ):
            return FigureExtractionResult(
                document_id=normalized_document_id,
                figures=[],
            )

        raw_catalogue = (
            self.storage
            .read_document_json_artifact(
                user_id=normalized_user_id,
                document_id=normalized_document_id,
                artifact_path=(
                    artifact_path
                ),
            )
        )

        if raw_catalogue is None:
            raise DocumentProcessingError(
                "The recorded figure catalogue "
                "artifact is missing."
            )

        try:
            catalogue = (
                FigureExtractionResult
                .model_validate(
                    raw_catalogue
                )
            )
        except Exception as exc:
            raise DocumentProcessingError(
                "The stored figure catalogue "
                "has an invalid format."
            ) from exc

        if (
            catalogue.document_id
            != normalized_document_id
        ):
            raise DocumentProcessingError(
                "The stored figure catalogue "
                "belongs to a different document."
            )

        seen_figure_ids: set[str] = set()

        for figure in catalogue.figures:
            figure_id = (
                figure.figure_id.strip()
            )

            if not figure_id:
                raise DocumentProcessingError(
                    "The stored figure catalogue "
                    "contains an empty figure_id."
                )

            if figure_id in seen_figure_ids:
                raise DocumentProcessingError(
                    "The stored figure catalogue "
                    "contains duplicate figure IDs."
                )

            seen_figure_ids.add(
                figure_id
            )

        return catalogue

    def resolve_figure_artifacts(
        self,
        *,
        user_id: str,
        document_id: str,
        figure_ids: list[str]
        | tuple[str, ...],
    ) -> list[FigureArtifact]:
        """
        Resolve exact canonical figure IDs to readable document-owned images.

        Resolution is intentionally identity-based only. Semantic, positional,
        page, and conversational reference interpretation belongs upstream.
        This method never maps a Physics phrase/topic to a figure.

        Requested order is preserved and duplicate requested IDs are ignored.
        Missing catalogue entries or missing image files are simply unresolved;
        callers can then fail closed or retry retrieval.
        """

        requested_ids: list[str] = []

        for raw_figure_id in figure_ids:
            figure_id = (
                raw_figure_id.strip()
                if isinstance(
                    raw_figure_id,
                    str,
                )
                else ""
            )

            if (
                figure_id
                and figure_id
                not in requested_ids
            ):
                requested_ids.append(
                    figure_id
                )

        if not requested_ids:
            return []

        catalogue = (
            self.get_figure_catalogue(
                user_id=user_id,
                document_id=document_id,
            )
        )

        by_id = {
            figure.figure_id: figure
            for figure in catalogue.figures
        }

        document_directory = (
            self.storage
            .get_document_directory(
                user_id=user_id,
                document_id=document_id,
            )
            .resolve()
        )

        resolved_figures: list[
            FigureArtifact
        ] = []

        for figure_id in requested_ids:
            figure = by_id.get(
                figure_id
            )

            if figure is None:
                continue

            raw_image_path = (
                figure.image_path.strip()
            )

            if not raw_image_path:
                continue

            try:
                image_path = Path(
                    raw_image_path
                ).resolve()
            except OSError:
                continue

            try:
                image_path.relative_to(
                    document_directory
                )
            except ValueError as exc:
                raise DocumentProcessingError(
                    "A stored figure image path "
                    "is outside the requested "
                    "document directory."
                ) from exc

            if not image_path.is_file():
                continue

            resolved_figures.append(
                figure.model_copy(
                    update={
                        "image_path": str(
                            image_path
                        )
                    }
                )
            )

        return resolved_figures

    # =========================================================
    # DELETE DOCUMENT
    # =========================================================

    def delete_document(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bool:
        """
        Delete:
        - FAISS index
        - BM25 index
        - uploaded source file
        - metadata
        - OCR/layout/chunks/analysis artifacts

        Indexes are deleted first.

        If index deletion fails, the document itself
        is not deleted.
        """

        # Verify that document exists.
        self._get_metadata(
            user_id=user_id,
            document_id=document_id,
        )

        # -----------------------------------------------------
        # DELETE FAISS + BM25 INDEXES
        # -----------------------------------------------------

        self.indexer.delete_document_indexes(
            user_id=user_id,
            document_id=document_id,
        )

        # -----------------------------------------------------
        # DELETE ORIGINAL DOCUMENT + LOCAL ARTIFACTS
        # -----------------------------------------------------

        deleted = self.storage.delete_document(
            user_id=user_id,
            document_id=document_id,
        )

        if not deleted:
            raise DocumentNotFoundError(
                "The requested document was not found."
            )

        return True

    # =========================================================
    # GET METADATA
    # =========================================================

    def _get_metadata(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        metadata = self.storage.read_document_metadata(
            user_id=user_id,
            document_id=document_id,
        )

        if metadata is None:
            raise DocumentNotFoundError(
                "The requested document was not found."
            )

        return metadata

    # =========================================================
    # UPDATE PROCESSING STAGE
    # =========================================================

    def _update_stage(
        self,
        *,
        metadata: dict[str, Any],
        user_id: str,
        document_id: str,
        status: str,
        stage: str,
    ) -> dict[str, Any]:
        metadata["status"] = status
        metadata["processing_stage"] = stage

        self.storage.write_document_metadata(
            user_id=user_id,
            document_id=document_id,
            metadata=metadata,
        )

        return metadata

    # =========================================================
    # ATOMIC JSON WRITE
    # =========================================================

    def _write_json_atomic(
        self,
        *,
        path: Path,
        data: Any,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        try:
            with temporary_path.open(
                mode="w",
                encoding="utf-8",
            ) as output_file:
                json.dump(
                    data,
                    output_file,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

                output_file.flush()
                os.fsync(output_file.fileno())

            os.replace(
                temporary_path,
                path,
            )

        except Exception:
            temporary_path.unlink(
                missing_ok=True
            )
            raise

    # =========================================================
    # RELATIVE ARTIFACT PATH
    # =========================================================

    def _relative_to_upload_root(
        self,
        path: Path,
    ) -> str:
        return (
            path.resolve()
            .relative_to(
                self.settings.upload_dir.resolve()
            )
            .as_posix()
        )
