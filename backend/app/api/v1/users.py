from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from slowapi import Limiter
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.api.schemas import (
    CompareResponse,
    UserResponse,
    VerificationHistoryItem,
    VerificationHistoryResponse,
)
from app.api.v1.face import _analyse_bytes
from app.core.config import Settings
from app.core.errors import ConflictError, ResourceNotFoundError
from app.engine.pool import EnginePool
from app.models.user import User
from app.repositories.face_repository import FaceVerificationRepository, UserRepository
from app.services.decision import Decision, decide_comparison
from app.services.profile_storage import ProfileImageStore

DbSession = Annotated[Session, Depends(get_db)]


def _read_upload_sync(upload: UploadFile, max_bytes: int) -> bytes:
    data = upload.file.read(max_bytes + 1)
    upload.file.close()
    if len(data) > max_bytes:
        from app.core.errors import ImageTooLargeError

        raise ImageTooLargeError(
            f"Image exceeds the {max_bytes}-byte upload limit.", detail="MAX_UPLOAD_BYTES"
        )
    return data


def _get_user(session: Session, user_id: uuid.UUID) -> User:
    user = UserRepository(session).get(user_id)
    if user is None:
        raise ResourceNotFoundError("User not found.")
    return user


def create_users_router(settings: Settings, limiter: Limiter) -> APIRouter:
    router = APIRouter(prefix="/users", tags=["users"])

    @router.post("", response_model=UserResponse, status_code=201)
    def create_user(
        request: Request,
        external_id: Annotated[str, Form(min_length=1, max_length=255)],
        name: Annotated[str, Form(min_length=1, max_length=255)],
        profile_image: Annotated[UploadFile, File()],
        session: DbSession,
    ) -> User:
        if UserRepository(session).get_by_external_id(external_id):
            raise ConflictError("A user with this external ID already exists.")
        data = _read_upload_sync(profile_image, settings.MAX_UPLOAD_BYTES)
        user_id = uuid.uuid4()
        store: ProfileImageStore = request.app.state.profile_store
        key = store.save(user_id, data)
        try:
            return UserRepository(session).create(
                user_id=user_id,
                external_id=external_id,
                name=name,
                profile_image_url=key,
            )
        except IntegrityError as exc:
            store.delete(key)
            raise ConflictError("A user with this external ID already exists.") from exc
        except Exception:
            store.delete(key)
            raise

    @router.get("/{user_id}", response_model=UserResponse)
    def get_user(user_id: uuid.UUID, session: DbSession) -> User:
        return _get_user(session, user_id)

    @router.put("/{user_id}/profile-image", response_model=UserResponse)
    def replace_profile_image(
        request: Request,
        user_id: uuid.UUID,
        profile_image: Annotated[UploadFile, File()],
        session: DbSession,
    ) -> User:
        user = _get_user(session, user_id)
        data = _read_upload_sync(profile_image, settings.MAX_UPLOAD_BYTES)
        store: ProfileImageStore = request.app.state.profile_store
        new_key = store.save(user_id, data)
        old_key = user.profile_image_url
        try:
            UserRepository(session).update_profile_image(user, new_key)
            session.commit()
        except Exception:
            store.delete(new_key)
            raise
        store.delete(old_key)
        return user

    @router.post("/{user_id}/verify", response_model=CompareResponse)
    @limiter.limit(settings.RATE_LIMIT_COMPARE)
    def verify_user(
        request: Request,
        user_id: uuid.UUID,
        candidate_image: Annotated[UploadFile, File()],
        session: DbSession,
    ) -> CompareResponse:
        user = _get_user(session, user_id)
        store: ProfileImageStore = request.app.state.profile_store
        reference_data = store.read(user.profile_image_url)
        candidate_data = _read_upload_sync(candidate_image, settings.MAX_UPLOAD_BYTES)
        started = time.perf_counter()
        pool: EnginePool = request.app.state.engine_pool
        reference = _analyse_bytes(pool, reference_data, settings)
        candidate = _analyse_bytes(pool, candidate_data, settings)
        threshold = settings.MATCH_THRESHOLD or pool.info.default_threshold
        outcome = decide_comparison(
            reference, candidate, threshold=threshold, review_margin=settings.REVIEW_MARGIN
        )
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        FaceVerificationRepository(session).record(
            user_id=user.id,
            outcome=outcome,
            model=pool.info,
            threshold_version=settings.THRESHOLD_VERSION,
            processing_time_ms=elapsed_ms,
        )
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

    @router.get("/{user_id}/verifications", response_model=VerificationHistoryResponse)
    def verification_history(
        user_id: uuid.UUID,
        session: DbSession,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> VerificationHistoryResponse:
        _get_user(session, user_id)
        rows = FaceVerificationRepository(session).list_for_user(
            user_id, limit=limit, offset=offset
        )
        return VerificationHistoryResponse(
            user_id=user_id,
            limit=limit,
            offset=offset,
            items=[
                VerificationHistoryItem(
                    id=row.id,
                    decision=Decision(row.decision),
                    matched=row.decision == Decision.MATCH.value,
                    similarity=row.similarity_score,
                    confidence=row.confidence,
                    threshold=row.threshold,
                    threshold_version=row.threshold_version,
                    reason_code=row.reason_code,
                    quality_issues=row.quality_issues,
                    processing_time_ms=row.processing_time_ms,
                    detector_version=row.detector_version,
                    recognition_model_version=row.recognition_model_version,
                    created_at=row.created_at,
                )
                for row in rows
            ],
        )

    return router
