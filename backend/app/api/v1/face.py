from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.schemas import BoundingBoxResponse, CompareResponse, DetectResponse
from app.core.config import Settings
from app.core.errors import ImageTooLargeError
from app.engine.decode import decode_image
from app.engine.pipeline import ImageAnalysis, analyse
from app.engine.pool import EnginePool
from app.services.decision import decide_comparison


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    data = await upload.read(max_bytes + 1)
    await upload.close()
    if len(data) > max_bytes:
        raise ImageTooLargeError(
            f"Image exceeds the {max_bytes}-byte upload limit.", detail="MAX_UPLOAD_BYTES"
        )
    return data


def _analyse_bytes(pool: EnginePool, data: bytes, settings: Settings) -> ImageAnalysis:
    image = decode_image(
        data,
        allowed_mime=settings.ALLOWED_MIME,
        max_bytes=settings.MAX_UPLOAD_BYTES,
        max_pixels=settings.MAX_IMAGE_PIXELS,
        max_side=settings.MAX_IMAGE_SIDE,
    )
    with pool.acquire() as engine:
        return analyse(engine, image)


def create_face_router(settings: Settings, limiter: Limiter) -> APIRouter:
    router = APIRouter(prefix="/face", tags=["face"])

    @router.post("/detect", response_model=DetectResponse)
    @limiter.limit(settings.RATE_LIMIT_DETECT)
    async def detect_face(request: Request, image: Annotated[UploadFile, File()]) -> DetectResponse:
        data = await _read_upload(image, settings.MAX_UPLOAD_BYTES)
        analysis = await run_in_threadpool(
            _analyse_bytes, request.app.state.engine_pool, data, settings
        )
        face = analysis.face
        bbox = (
            BoundingBoxResponse(
                x=face.bbox.x,
                y=face.bbox.y,
                width=face.bbox.w,
                height=face.bbox.h,
            )
            if face
            else None
        )
        return DetectResponse(
            status=analysis.status,
            face_detected=analysis.face_count > 0,
            face_count=analysis.face_count,
            quality_score=analysis.quality.score if analysis.quality else None,
            quality_issues=[issue.value for issue in analysis.quality.issues]
            if analysis.quality
            else [],
            bounding_box=bbox,
        )

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
