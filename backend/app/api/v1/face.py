from __future__ import annotations

import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.schemas import BoundingBoxResponse, CompareResponse, DetectResponse
from app.core.config import Settings
from app.core.errors import ImageTooLargeError
from app.engine.decode import decode_image
from app.engine.pipeline import FaceSelectionPolicy, ImageAnalysis, analyse
from app.engine.pool import EnginePool
from app.services.decision import decide_comparison


async def _read_upload(
    upload: UploadFile, max_bytes: int, *, detail: str = "MAX_UPLOAD_BYTES"
) -> bytes:
    data = await upload.read(max_bytes + 1)
    await upload.close()
    if len(data) > max_bytes:
        raise ImageTooLargeError(f"Image exceeds the {max_bytes}-byte upload limit.", detail=detail)
    return data


def _analyse_bytes_with_image(
    pool: EnginePool,
    data: bytes,
    settings: Settings,
    *,
    max_bytes: int | None = None,
    multi_face: FaceSelectionPolicy = FaceSelectionPolicy.REJECT,
    timeout: float = 10.0,
) -> tuple[ImageAnalysis, np.ndarray]:
    """As `_analyse_bytes`, but hands back the decoded pixels too.

    The liveness check measures eye patches, which needs the image and not just
    the analysis. Returning it from the one call that already decoded is what
    keeps that from costing a second decode per frame on a live socket.
    """
    image = decode_image(
        data,
        allowed_mime=settings.ALLOWED_MIME,
        max_bytes=settings.MAX_UPLOAD_BYTES if max_bytes is None else max_bytes,
        max_pixels=settings.MAX_IMAGE_PIXELS,
        max_side=settings.MAX_IMAGE_SIDE,
    )
    with pool.acquire(timeout=timeout) as engine:
        return analyse(engine, image, multi_face=multi_face), image


def _analyse_bytes(
    pool: EnginePool,
    data: bytes,
    settings: Settings,
    *,
    max_bytes: int | None = None,
    multi_face: FaceSelectionPolicy = FaceSelectionPolicy.REJECT,
    timeout: float = 10.0,
) -> ImageAnalysis:
    analysis, _ = _analyse_bytes_with_image(
        pool,
        data,
        settings,
        max_bytes=max_bytes,
        multi_face=multi_face,
        timeout=timeout,
    )
    return analysis


def _detect_response(analysis: ImageAnalysis) -> DetectResponse:
    face = analysis.face
    return DetectResponse(
        status=analysis.status,
        face_detected=analysis.face_count > 0,
        face_count=analysis.face_count,
        quality_score=analysis.quality.score if analysis.quality else None,
        quality_issues=[issue.value for issue in analysis.quality.issues]
        if analysis.quality
        else [],
        bounding_box=BoundingBoxResponse(
            x=face.bbox.x, y=face.bbox.y, width=face.bbox.w, height=face.bbox.h
        )
        if face
        else None,
    )


def create_face_router(settings: Settings, limiter: Limiter) -> APIRouter:
    router = APIRouter(prefix="/face", tags=["face"])

    @router.post("/detect", response_model=DetectResponse)
    @limiter.limit(settings.RATE_LIMIT_DETECT)
    async def detect_face(request: Request, image: Annotated[UploadFile, File()]) -> DetectResponse:
        data = await _read_upload(image, settings.MAX_UPLOAD_BYTES)
        analysis = await run_in_threadpool(
            _analyse_bytes, request.app.state.engine_pool, data, settings
        )
        return _detect_response(analysis)

    @router.post("/detect/frame", response_model=DetectResponse)
    @limiter.limit(settings.RATE_LIMIT_DETECT_FRAME)
    async def detect_frame(
        request: Request, frame: Annotated[UploadFile, File()]
    ) -> DetectResponse:
        """Detect on one live camera frame, for preview guidance only.

        Differs from /detect in three deliberate ways: a much smaller byte cap, a
        rate limit sized for 2-5 FPS, and the LARGEST multi-face policy, so a
        bystander wandering through frame does not blank the operator's overlay.
        Nothing here is recorded — the frame the operator finally captures goes
        through the normal REJECT path, where a second face is still an error.
        """
        data = await _read_upload(frame, settings.MAX_FRAME_BYTES, detail="MAX_FRAME_BYTES")
        analysis = await run_in_threadpool(
            _analyse_bytes,
            request.app.state.engine_pool,
            data,
            settings,
            max_bytes=settings.MAX_FRAME_BYTES,
            multi_face=FaceSelectionPolicy.LARGEST,
        )
        return _detect_response(analysis)

    @router.post("/compare", response_model=CompareResponse)
    @limiter.limit(settings.RATE_LIMIT_COMPARE)
    async def compare_faces(
        request: Request,
        reference_image: Annotated[UploadFile, File()],
        candidate_image: Annotated[UploadFile, File()],
    ) -> CompareResponse:
        reference_data = await _read_upload(reference_image, settings.MAX_UPLOAD_BYTES)
        candidate_data = await _read_upload(candidate_image, settings.MAX_UPLOAD_BYTES)
        started = time.perf_counter()
        pool: EnginePool = request.app.state.engine_pool
        reference = await run_in_threadpool(_analyse_bytes, pool, reference_data, settings)
        candidate = await run_in_threadpool(_analyse_bytes, pool, candidate_data, settings)
        threshold = settings.MATCH_THRESHOLD or pool.info.default_threshold
        outcome = decide_comparison(
            reference, candidate, threshold=threshold, review_margin=settings.REVIEW_MARGIN
        )
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return CompareResponse(
            face_detected=reference.face_count > 0 and candidate.face_count > 0,
            matched=outcome.matched,
            similarity=outcome.similarity,
            confidence=outcome.confidence,
            threshold=outcome.threshold,
            threshold_version=settings.THRESHOLD_VERSION,
            decision=outcome.decision,
            reason_code=outcome.reason_code,
            quality_issues=[issue.value for issue in outcome.issues],
            reference_status=outcome.reference_status,
            candidate_status=outcome.candidate_status,
            reference_face_count=outcome.reference_face_count,
            candidate_face_count=outcome.candidate_face_count,
            processing_time_ms=elapsed_ms,
            detector_version=f"{pool.info.detector_name}@{pool.info.detector_version}",
            recognition_model_version=f"{pool.info.recognizer_name}@{pool.info.recognizer_version}",
        )

    return router


limiter = Limiter(key_func=get_remote_address)
