from __future__ import annotations

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
