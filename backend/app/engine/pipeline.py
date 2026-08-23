"""Single-image analysis: decode -> detect -> select -> quality -> align -> embed.

Pure with respect to I/O: no HTTP, no database, no logging of image content.
It reports what it found and never decides what that means — mapping a score to
MATCH or NO_MATCH is the business layer's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from app.engine.base import FaceEngine
from app.engine.quality import DEFAULT_THRESHOLDS, QualityThresholds, assess_quality
from app.engine.types import DetectedFace, QualityReport


class FaceStatus(StrEnum):
    """Per-image outcome. Not a verdict about identity."""

    OK = "OK"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_QUALITY = "LOW_QUALITY"


class FaceSelectionPolicy(StrEnum):
    """What to do when an image contains more than one face."""

    REJECT = "reject"
    LARGEST = "largest"


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    status: FaceStatus
    face_count: int
    face: DetectedFace | None = None
    quality: QualityReport | None = None
    # Never crosses the API boundary. An embedding is a biometric identifier
    # that supports partial face reconstruction; it is dropped by the service
    # layer as soon as the similarity has been computed.
    embedding: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.status is FaceStatus.OK


def analyse(
    engine: FaceEngine,
    image_bgr: np.ndarray,
    *,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
    multi_face: FaceSelectionPolicy = FaceSelectionPolicy.REJECT,
) -> ImageAnalysis:
    faces = engine.detect(image_bgr)

    if not faces:
        return ImageAnalysis(status=FaceStatus.NO_FACE, face_count=0)

    if len(faces) > 1:
        # Default REJECT rather than silently taking the biggest face.
        # Auto-selecting in a verification flow is how the wrong person gets
        # verified from a group photo — make the caller resolve the ambiguity.
        # V3 live capture, where a bystander in frame is routine, is what the
        # LARGEST policy exists for.
        if multi_face is FaceSelectionPolicy.REJECT:
            return ImageAnalysis(status=FaceStatus.MULTIPLE_FACES, face_count=len(faces))
        face = max(faces, key=lambda f: f.bbox.area)
    else:
        face = faces[0]

    quality = assess_quality(image_bgr, face, thresholds)
    if not quality.passed:
        return ImageAnalysis(
            status=FaceStatus.LOW_QUALITY,
            face_count=len(faces),
            face=face,
            quality=quality,
        )

    embedding = engine.embed(engine.align(image_bgr, face))
    return ImageAnalysis(
        status=FaceStatus.OK,
        face_count=len(faces),
        face=face,
        quality=quality,
        embedding=embedding,
    )
