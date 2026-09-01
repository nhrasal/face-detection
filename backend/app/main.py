"""FastAPI application factory and inference-engine lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.face import create_face_router, limiter
from app.api.v1.users import create_users_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.http_security import RequestSizeLimitMiddleware, SecurityHeadersMiddleware
from app.core.logging import configure_logging, get_logger
from app.db.session import create_database_engine, create_session_factory
from app.engine.factory import build_engine
from app.engine.pool import EnginePool
from app.services.profile_storage import ProfileImageStore

log = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.LOG_LEVEL, json_output=settings.LOG_JSON)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = EnginePool(lambda: build_engine(settings), size=settings.INFERENCE_WORKERS)
        database_engine = create_database_engine(settings.DATABASE_URL)
        app.state.engine_pool = pool
        app.state.database_engine = database_engine
        app.state.session_factory = create_session_factory(database_engine)
        app.state.profile_store = ProfileImageStore(settings.UPLOAD_DIR, settings)
        try:
            yield
        finally:
            pool.close()
            database_engine.dispose()

    app = FastAPI(
        title="Face Verification Service",
        version="0.1.0",
        description="V1: compare an uploaded photo against a stored profile image.",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, error: AppError) -> JSONResponse:
        headers = {"Retry-After": "1"} if error.http_status == 503 else None
        return JSONResponse(
            status_code=error.http_status,
            headers=headers,
            content={
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "detail": error.detail,
                },
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
        # Never include exception text: decoder/model failures can contain paths
        # and implementation details. The traceback stays in server logs.
        log.exception("request.failed", path=request.url.path, error_type=type(error).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "PROCESSING_ERROR",
                    "message": "The image could not be processed.",
                    "detail": None,
                },
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,  # explicit allowlist, never ["*"]
        allow_credentials=False,  # no cookie auth in V1
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.MAX_REQUEST_BYTES)
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, Any]:
        """Liveness. Must not touch the database or the models."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    async def readyz() -> dict[str, Any]:
        """Readiness.

        From the phase that introduces the engine pool onward this reports 503
        until the models are loaded and warmed. Today there is nothing to warm,
        so it mirrors liveness.
        """
        pool = getattr(app.state, "engine_pool", None)
        return {
            "status": "ready" if pool is not None else "starting",
            "engine": settings.FACE_ENGINE,
            "warm": pool is not None,
        }

    app.include_router(create_face_router(settings, limiter), prefix=settings.API_PREFIX)
    app.include_router(create_users_router(settings, limiter), prefix=settings.API_PREFIX)

    log.info("app.created", env=settings.ENV, engine=settings.FACE_ENGINE)
    return app


app = create_app()
