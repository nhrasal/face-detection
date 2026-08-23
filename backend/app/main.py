"""FastAPI application factory.

Phase 1 scope: health only. The engine pool, database session, error handlers,
rate limiting and the v1 routers arrive in later phases — see the build plan.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.LOG_LEVEL, json_output=settings.LOG_JSON)

    app = FastAPI(
        title="Face Verification Service",
        version="0.1.0",
        description="V1: compare an uploaded photo against a stored profile image.",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,  # explicit allowlist, never ["*"]
        allow_credentials=False,  # no cookie auth in V1
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

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
        return {"status": "ready", "engine": settings.FACE_ENGINE, "warm": False}

    log.info("app.created", env=settings.ENV, engine=settings.FACE_ENGINE)
    return app


app = create_app()
