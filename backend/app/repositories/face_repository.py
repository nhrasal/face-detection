"""Persistence operations for V1 users and verification audit rows."""

from __future__ import annotations

import uuid

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.engine.types import ModelInfo
from app.models.face_verification import FaceVerification
from app.models.user import User
from app.services.decision import ComparisonOutcome


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, user_id: uuid.UUID) -> User | None:
        return self.session.get(User, user_id)

    def get_by_external_id(self, external_id: str) -> User | None:
        return self.session.scalar(select(User).where(User.external_id == external_id))

    def search(self, query: str, *, limit: int = 20) -> list[User]:
        """Substring match over external_id and name, newest-registered first.

        LIKE wildcards in the query are escaped so a user typing "%" searches for a
        literal percent sign rather than matching every row.
        """
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        statement = (
            select(User)
            .where(
                or_(
                    User.external_id.ilike(pattern, escape="\\"),
                    User.name.ilike(pattern, escape="\\"),
                )
            )
            .order_by(User.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(statement))

    def create(
        self,
        *,
        external_id: str,
        name: str,
        profile_image_url: str,
        user_id: uuid.UUID | None = None,
    ) -> User:
        user = User(
            id=user_id or uuid.uuid4(),
            external_id=external_id,
            name=name,
            profile_image_url=profile_image_url,
        )
        self.session.add(user)
        self.session.flush()
        return user

    def update_profile_image(self, user: User, profile_image_url: str) -> User:
        user.profile_image_url = profile_image_url
        self.session.flush()
        return user


class FaceVerificationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, verification_id: uuid.UUID) -> FaceVerification | None:
        return self.session.get(FaceVerification, verification_id)

    def list_for_user(
        self, user_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[FaceVerification]:
        statement = (
            select(FaceVerification)
            .where(FaceVerification.user_id == user_id)
            .order_by(FaceVerification.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(statement))

    def record(
        self,
        *,
        user_id: uuid.UUID,
        outcome: ComparisonOutcome,
        model: ModelInfo,
        threshold_version: str,
        processing_time_ms: int,
    ) -> FaceVerification:
        row = FaceVerification(
            user_id=user_id,
            similarity_score=outcome.similarity,
            confidence=outcome.confidence,
            threshold=outcome.threshold,
            threshold_version=threshold_version,
            decision=outcome.decision.value,
            reason_code=outcome.reason_code,
            quality_issues=[issue.value for issue in outcome.issues],
            reference_status=outcome.reference_status.value,
            candidate_status=outcome.candidate_status.value,
            reference_face_count=outcome.reference_face_count,
            candidate_face_count=outcome.candidate_face_count,
            detector_version=f"{model.detector_name}@{model.detector_version}",
            recognition_model_version=f"{model.recognizer_name}@{model.recognizer_version}",
            processing_time_ms=processing_time_ms,
        )
        self.session.add(row)
        self.session.flush()
        return row
