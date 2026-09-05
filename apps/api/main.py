from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from apps.api.middleware import (
    LocalRateLimitMiddleware,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    configure_logging,
    register_exception_handlers,
)
from apps.api.routes import (
    chat_router,
    documents_router,
    retrieval_router,
)
from src.config import settings
from src.graph.checkpointing import (
    get_checkpoint_manager,
)


configure_logging(
    settings.log_level
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Application startup / shutdown lifecycle.

    Startup:
    - create required local directories
    - open PostgreSQL checkpoint pool
    - create/migrate LangGraph checkpoint tables

    Shutdown:
    - close checkpoint PostgreSQL pool cleanly
    """

    settings.create_required_directories()

    checkpoint_manager = (
        get_checkpoint_manager()
    )

    # setup() performs synchronous PostgreSQL work,
    # so keep it off FastAPI's async event loop.
    await run_in_threadpool(
        checkpoint_manager.setup
    )

    # Make the shared manager available to
    # application-level diagnostics if needed.
    app.state.checkpoint_manager = (
        checkpoint_manager
    )

    try:
        yield

    finally:
        # Cleanly release pooled PostgreSQL
        # connections during application shutdown.
        await run_in_threadpool(
            checkpoint_manager.close
        )


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Local FastAPI backend for the "
        "PhyMentor AI school-level Physics tutor."
    ),
    debug=settings.debug,
    lifespan=lifespan,
)


allow_all_origins = (
    "*" in settings.cors_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware added last executes first.
# Request ID is therefore added after the others below.

app.add_middleware(
    RequestLoggingMiddleware
)

app.add_middleware(
    LocalRateLimitMiddleware,
    default_requests_per_minute=(
        settings.default_rate_limit_per_minute
    ),
    upload_requests_per_minute=(
        settings.upload_rate_limit_per_minute
    ),
)

app.add_middleware(
    RequestIDMiddleware
)


register_exception_handlers(
    app
)


# ---------------------------------------------------------
# API ROUTERS
# ---------------------------------------------------------

app.include_router(
    documents_router,
    prefix=settings.api_prefix,
)

app.include_router(
    retrieval_router,
    prefix=settings.api_prefix,
)

app.include_router(
    chat_router,
    prefix=settings.api_prefix,
)


# ---------------------------------------------------------
# SYSTEM ENDPOINTS
# ---------------------------------------------------------

@app.get(
    "/",
    tags=["system"],
)
async def root() -> dict[str, str]:
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "message": (
            "PhyMentor AI local backend is running."
        ),
    }


@app.get(
    "/health/live",
    tags=["health"],
)
async def liveness_check() -> dict[str, str]:
    return {
        "status": "alive",
    }


@app.get(
    "/health/ready",
    tags=["health"],
)
async def readiness_check() -> dict[str, str]:
    settings.create_required_directories()

    upload_directory_ready = (
        settings.upload_dir.exists()
        and settings.upload_dir.is_dir()
    )

    if not upload_directory_ready:
        return {
            "status": "not_ready",
            "reason": (
                "The local upload directory "
                "is unavailable."
            ),
        }

    checkpoint_manager = (
        get_checkpoint_manager()
    )

    checkpoint_ready = (
        await run_in_threadpool(
            checkpoint_manager.health_check
        )
    )

    if not checkpoint_ready:
        return {
            "status": "not_ready",
            "reason": (
                "LangGraph checkpoint "
                "PostgreSQL is unavailable."
            ),
        }

    return {
        "status": "ready",
    }