"""Application settings.

Every tunable in the service lives here and is overridable by environment
variable or `.env`. Nothing else in the codebase reads `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EngineName = Literal["opencv_zoo", "insightface", "fake"]


class Settings(BaseSettings):
    # `protected_namespaces=()` is required: pydantic v2 reserves the `model_`
    # prefix, and MODEL_DIR / the model-version fields collide with it. Without
    # this the app warns today and fails on a future pydantic release.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- runtime -----------------------------------------------------------
    ENV: Literal["local", "dev", "uat", "prod"] = "local"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False  # human-readable locally, JSON everywhere else
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3320"]

    # --- persistence -------------------------------------------------------
    DATABASE_URL: str = "postgresql+psycopg://face:face@localhost:55432/face_verification"
    REDIS_URL: str | None = None

    # --- face engine -------------------------------------------------------
    FACE_ENGINE: EngineName = "opencv_zoo"
    MODEL_DIR: Path = Path("/models")
    ALLOW_NONCOMMERCIAL_MODELS: bool = False
    INFERENCE_WORKERS: int = Field(default=2, ge=1, le=16)
    DETECT_MAX_SIDE: int = 1280  # downscale before detection; align from full res

    # --- decision ----------------------------------------------------------
    # None => use the engine's own calibrated default from ModelInfo.
    # This MUST end up as a measured value (see docs/THRESHOLD_CALIBRATION.md).
    MATCH_THRESHOLD: float | None = Field(default=None, ge=-1.0, le=1.0)
    REVIEW_MARGIN: float = Field(default=0.0, ge=0.0, le=1.0)
    THRESHOLD_VERSION: str = "sface-2021dec-default"

    # --- upload limits -----------------------------------------------------
    MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024
    MAX_IMAGE_PIXELS: int = 40_000_000
    MAX_IMAGE_SIDE: int = 8000
    ALLOWED_MIME: set[str] = {"image/jpeg", "image/png", "image/webp"}

    RATE_LIMIT_COMPARE: str = "10/minute"
    RATE_LIMIT_DETECT: str = "30/minute"

    # --- storage / retention ----------------------------------------------
    # V1 deliberately keeps no candidate images: scores and metadata only.
    STORE_UPLOADS: bool = False
    UPLOAD_DIR: Path = Path("/data/uploads")
    VERIFICATION_RETENTION_DAYS: int = 180

    @model_validator(mode="after")
    def _forbid_fake_engine_in_prod(self) -> Settings:
        """A fake biometric engine reaching production is catastrophic-class.

        The `fake` adapter does not exist yet (it arrives with the HTTP layer),
        but the guard is written now so it is impossible to ever ship without.
        """
        if self.ENV == "prod" and self.FACE_ENGINE == "fake":
            raise ValueError("FACE_ENGINE=fake is forbidden when ENV=prod")
        return self

    @model_validator(mode="after")
    def _gate_noncommercial_models(self) -> Settings:
        """InsightFace buffalo_l weights are non-commercial-research licensed.

        Using them in production needs a deliberate, explicit opt-in rather
        than an env var someone copied between deployments.
        """
        if (
            self.ENV == "prod"
            and self.FACE_ENGINE == "insightface"
            and not self.ALLOW_NONCOMMERCIAL_MODELS
        ):
            raise ValueError(
                "FACE_ENGINE=insightface uses non-commercially-licensed weights. "
                "Set ALLOW_NONCOMMERCIAL_MODELS=true only with legal sign-off."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Tests override env then call `get_settings.cache_clear()`."""
    return Settings()
