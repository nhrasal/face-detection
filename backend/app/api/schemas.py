from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.engine.pipeline import FaceStatus
from app.services.decision import Decision


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody


class BoundingBoxResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectResponse(BaseModel):
    success: bool = True
    status: FaceStatus
    face_detected: bool
    face_count: int
    quality_score: float | None = None
    quality_issues: list[str] = Field(default_factory=list)
    bounding_box: BoundingBoxResponse | None = None


class CompareResponse(BaseModel):
    success: bool = True
    face_detected: bool
    matched: bool
    similarity: float | None
    confidence: float | None
    threshold: float
    threshold_version: str
    decision: Decision
    reason_code: str | None = None
    quality_issues: list[str] = Field(default_factory=list)
    reference_status: FaceStatus
    candidate_status: FaceStatus
    reference_face_count: int
    candidate_face_count: int
    processing_time_ms: int
    detector_version: str
    recognition_model_version: str


class UserResponse(BaseModel):
    id: uuid.UUID
    external_id: str
    name: str
    profile_image_url: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class VerificationHistoryItem(BaseModel):
    id: uuid.UUID
    decision: Decision
    matched: bool
    similarity: float | None
    confidence: float | None
    threshold: float
    threshold_version: str
    reason_code: str | None
    quality_issues: list[str]
    processing_time_ms: int
    detector_version: str
    recognition_model_version: str
    created_at: datetime


class VerificationHistoryResponse(BaseModel):
    user_id: uuid.UUID
    items: list[VerificationHistoryItem]
    limit: int
    offset: int
