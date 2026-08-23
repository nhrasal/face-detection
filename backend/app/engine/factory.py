"""Select and construct the configured face engine."""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.engine.base import FaceEngine

log = get_logger(__name__)


def _build_opencv_zoo(settings: Settings) -> FaceEngine:
    from app.engine.adapters.opencv_zoo import OpenCvZooEngine

    return OpenCvZooEngine(settings.MODEL_DIR, detect_max_side=settings.DETECT_MAX_SIDE)


def _build_insightface(settings: Settings) -> FaceEngine:
    # Arrives in the benchmarking phase. The licence gate lives in Settings, so
    # by the time this runs the ENV=prod refusal has already applied.
    raise AppError("The insightface engine is not implemented yet.")


def _build_fake(settings: Settings) -> FaceEngine:
    # Arrives with the HTTP layer, where it makes the API tests hermetic. The
    # ENV=prod refusal in Settings already guards it.
    raise AppError("The fake engine is not implemented yet.")


_REGISTRY: dict[str, Callable[[Settings], FaceEngine]] = {
    "opencv_zoo": _build_opencv_zoo,
    "insightface": _build_insightface,
    "fake": _build_fake,
}


def build_engine(settings: Settings) -> FaceEngine:
    try:
        builder = _REGISTRY[settings.FACE_ENGINE]
    except KeyError:
        raise AppError(f"Unknown FACE_ENGINE: {settings.FACE_ENGINE!r}") from None

    engine = builder(settings)
    if "non-commercial" in engine.info.license_note.lower():
        log.warning(
            "engine.noncommercial_weights",
            engine=settings.FACE_ENGINE,
            license=engine.info.license_note,
            detail="benchmark use only; not cleared for production",
        )
    return engine
