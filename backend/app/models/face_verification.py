from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.services.decision import Decision

if TYPE_CHECKING:
    from app.models.user import User

DECISION_VALUES = ", ".join(f"'{item.value}'" for item in Decision)


class FaceVerification(Base):
    __tablename__ = "face_verifications"
    __table_args__ = (
        CheckConstraint(f"decision IN ({DECISION_VALUES})", name="decision_valid"),
        CheckConstraint(
            "similarity_score IS NULL OR (similarity_score >= -1 AND similarity_score <= 1)",
            name="similarity_range",
        ),
        CheckConstraint("threshold >= -1 AND threshold <= 1", name="threshold_range"),
        CheckConstraint("processing_time_ms >= 0", name="processing_time_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    verification_type: Mapped[str] = mapped_column(String(32), default="PHOTO_COMPARE")
    reference_source: Mapped[str] = mapped_column(String(32), default="PROFILE_IMAGE")
    similarity_score: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    threshold_version: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(32), index=True)
    reason_code: Mapped[str | None] = mapped_column(String(64))
    quality_issues: Mapped[list[str]] = mapped_column(JSON, default=list)
    reference_status: Mapped[str] = mapped_column(String(32))
    candidate_status: Mapped[str] = mapped_column(String(32))
    reference_face_count: Mapped[int] = mapped_column(Integer)
    candidate_face_count: Mapped[int] = mapped_column(Integer)
    detector_version: Mapped[str] = mapped_column(String(128))
    recognition_model_version: Mapped[str] = mapped_column(String(128))
    processing_time_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    user: Mapped[User] = relationship(back_populates="verifications")
