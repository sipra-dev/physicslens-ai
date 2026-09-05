from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)

from src.ingestion.service import (
    DocumentNotFoundError,
    DocumentProcessingError,
)
from src.ingestion.validation import (
    FileValidationError,
)
from src.retrieval.service import (
    RetrievalServiceError,
)
from src.storage import StorageError


logger = logging.getLogger(
    "phymentor.errors"
)


def _get_request_id(
    request: Request,
) -> str | None:
    return getattr(
        request.state,
        "request_id",
        None,
    )


def register_exception_handlers(
    app: FastAPI,
) -> None:

    # --------------------------------------------------
    # FASTAPI / ROUTE VALIDATION ERRORS
    # --------------------------------------------------

    @app.exception_handler(
        RequestValidationError
    )
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    "The request payload is invalid."
                ),
                "error_code": (
                    "VALIDATION_ERROR"
                ),
                "errors": jsonable_encoder(
                    exc.errors()
                ),
                "request_id": _get_request_id(
                    request
                ),
            },
        )

    # --------------------------------------------------
    # NORMAL HTTP ERRORS
    # --------------------------------------------------

    @app.exception_handler(
        StarletteHTTPException
    )
    async def handle_http_error(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": str(
                    exc.detail
                ),
                "error_code": (
                    f"HTTP_{exc.status_code}"
                ),
                "request_id": _get_request_id(
                    request
                ),
            },
            headers=exc.headers,
        )

    # --------------------------------------------------
    # FILE VALIDATION
    # --------------------------------------------------

    @app.exception_handler(
        FileValidationError
    )
    async def handle_file_validation_error(
        request: Request,
        exc: FileValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.message,
                "error_code": (
                    "FILE_VALIDATION_ERROR"
                ),
                "request_id": _get_request_id(
                    request
                ),
            },
        )

    # --------------------------------------------------
    # DOCUMENT NOT FOUND
    # --------------------------------------------------

    @app.exception_handler(
        DocumentNotFoundError
    )
    async def handle_document_not_found(
        request: Request,
        exc: DocumentNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exc),
                "error_code": (
                    "DOCUMENT_NOT_FOUND"
                ),
                "request_id": _get_request_id(
                    request
                ),
            },
        )

    # --------------------------------------------------
    # RETRIEVAL ERRORS
    # --------------------------------------------------

    @app.exception_handler(
        RetrievalServiceError
    )
    async def handle_retrieval_error(
        request: Request,
        exc: RetrievalServiceError,
    ) -> JSONResponse:

        message = str(exc)
        normalized = message.lower()

        # Bad retrieval parameters
        if (
            "cannot be empty" in normalized
            or "must be positive" in normalized
        ):
            status_code = 400
            error_code = (
                "INVALID_RETRIEVAL_REQUEST"
            )
            response_detail = message

        # Document has no usable indexes
        elif (
            "indexes are missing"
            in normalized
            or "missing or incomplete"
            in normalized
            or "may not be ready"
            in normalized
        ):
            status_code = 404
            error_code = (
                "DOCUMENT_INDEX_NOT_FOUND"
            )
            response_detail = (
                "The requested document is not "
                "ready for retrieval or its "
                "retrieval index is unavailable."
            )

        # Internal retrieval problem
        else:
            status_code = 503
            error_code = (
                "RETRIEVAL_SERVICE_UNAVAILABLE"
            )
            response_detail = (
                "The retrieval service is "
                "temporarily unavailable."
            )

        logger.warning(
            "retrieval_error "
            "request_id=%s "
            "status=%s "
            "error=%s",
            _get_request_id(request),
            status_code,
            message,
        )

        return JSONResponse(
            status_code=status_code,
            content={
                "detail": response_detail,
                "error_code": error_code,
                "request_id": _get_request_id(
                    request
                ),
            },
        )

    # --------------------------------------------------
    # DOCUMENT PROCESSING
    # --------------------------------------------------

    @app.exception_handler(
        DocumentProcessingError
    )
    async def handle_processing_error(
        request: Request,
        exc: DocumentProcessingError,
    ) -> JSONResponse:
        logger.exception(
            "document_processing_error "
            "request_id=%s",
            _get_request_id(request),
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    "Document processing failed."
                ),
                "error_code": (
                    "DOCUMENT_PROCESSING_ERROR"
                ),
                "request_id": _get_request_id(
                    request
                ),
            },
        )

    # --------------------------------------------------
    # STORAGE
    # --------------------------------------------------

    @app.exception_handler(
        StorageError
    )
    async def handle_storage_error(
        request: Request,
        exc: StorageError,
    ) -> JSONResponse:
        logger.exception(
            "storage_error request_id=%s",
            _get_request_id(request),
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    "A local storage operation "
                    "failed."
                ),
                "error_code": (
                    "STORAGE_ERROR"
                ),
                "request_id": _get_request_id(
                    request
                ),
            },
        )

    # --------------------------------------------------
    # VALUE ERROR
    # --------------------------------------------------

    @app.exception_handler(
        ValueError
    )
    async def handle_value_error(
        request: Request,
        exc: ValueError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(exc),
                "error_code": (
                    "INVALID_VALUE"
                ),
                "request_id": _get_request_id(
                    request
                ),
            },
        )

    # --------------------------------------------------
    # LAST-RESORT ERROR HANDLER
    # --------------------------------------------------

    @app.exception_handler(
        Exception
    )
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "unexpected_error request_id=%s",
            _get_request_id(request),
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    "An unexpected server "
                    "error occurred."
                ),
                "error_code": (
                    "INTERNAL_SERVER_ERROR"
                ),
                "request_id": _get_request_id(
                    request
                ),
            },
        )