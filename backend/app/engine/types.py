"""Value types shared across the engine layer.

Nothing here imports HTTP, the database, or a specific model backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

# Index positions within a canonical (5, 2) landmark array.
LM_EYE_LEFT = 0
LM_EYE_RIGHT = 1
LM_NOSE = 2
LM_MOUTH_LEFT = 3
LM_MOUTH_RIGHT = 4


class ReasonCode(StrEnum):
    """Sub-codes of the LOW_QUALITY decision.

    The API returns the roadmap-compatible `decision` plus these, so the UI can
    say "the photo is out of focus, retake it" instead of a dead-end
    "low quality".
    """

    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    LOW_DETECTION_CONFIDENCE = "LOW_DETECTION_CONFIDENCE"
    IMAGE_BLURRY = "IMAGE_BLURRY"
    TOO_DARK = "TOO_DARK"
    TOO_BRIGHT = "TOO_BRIGHT"
    LOW_CONTRAST = "LOW_CONTRAST"
    EXTREME_POSE = "EXTREME_POSE"
    EXTREME_ROLL = "EXTREME_ROLL"
    FACE_TOO_SMALL_IN_FRAME = "FACE_TOO_SMALL_IN_FRAME"
    FACE_FILLS_FRAME = "FACE_FILLS_FRAME"
    FACE_OUT_OF_FRAME = "FACE_OUT_OF_FRAME"
    # Something is in front of the face. See app/engine/occlusion.py for what
    # these can and cannot see — notably, a bare hand is skin and slips through.
    FACE_COVERED = "FACE_COVERED"
    EYES_COVERED = "EYES_COVERED"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


def canonicalise_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Force landmarks into image-relative order, left-to-right.

    Detectors document their 5 points SUBJECT-relative ("right eye" = the
    subject's right = image-LEFT). The ArcFace destination template is
    IMAGE-relative. Taking either at face value produces a horizontally
    mirrored crop: no exception, no warning, roughly halved accuracy.

    Rather than reason about whose convention is whose, sort by x position.
    Cheapest insurance in the whole pipeline.
    """
    lm = np.array(landmarks, dtype=np.float32, copy=True)
    if lm.shape != (5, 2):
        raise ValueError(f"landmarks must be (5, 2), got {lm.shape}")
    if lm[LM_EYE_LEFT, 0] > lm[LM_EYE_RIGHT, 0]:
        lm[[LM_EYE_LEFT, LM_EYE_RIGHT]] = lm[[LM_EYE_RIGHT, LM_EYE_LEFT]]
    if lm[LM_MOUTH_LEFT, 0] > lm[LM_MOUTH_RIGHT, 0]:
        lm[[LM_MOUTH_LEFT, LM_MOUTH_RIGHT]] = lm[[LM_MOUTH_RIGHT, LM_MOUTH_LEFT]]
    return lm


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """One detected face.

    `landmarks` is always canonical: the constructor enforces it, so it is
    impossible to hold a DetectedFace with mirrored eyes.

    `native` is the backend's private payload (YuNet's 1x15 row) and must never
    cross out of the adapter that produced it.
    """

    bbox: BoundingBox
    landmarks: np.ndarray
    score: float
    native: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "landmarks", canonicalise_landmarks(self.landmarks))

    @property
    def interocular(self) -> float:
        """Distance between eye centres — the honest proxy for usable resolution."""
        return float(np.linalg.norm(self.landmarks[LM_EYE_RIGHT] - self.landmarks[LM_EYE_LEFT]))


@dataclass(frozen=True, slots=True)
class QualityReport:
    passed: bool
    score: float
    issues: tuple[ReasonCode, ...]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    detector_name: str
    detector_version: str
    recognizer_name: str
    recognizer_version: str
    embedding_dim: int
    default_threshold: float
    license_note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector": f"{self.detector_name}@{self.detector_version}",
            "recognizer": f"{self.recognizer_name}@{self.recognizer_version}",
            "embedding_dim": self.embedding_dim,
            "default_threshold": self.default_threshold,
            "license": self.license_note,
        }
