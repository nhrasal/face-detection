"""Image quality assessment for a detected face.

Every metric yields a value AND a reason code, so a rejection can tell the user
what to fix. "Low quality" with no explanation guarantees they retry with the
same bad photo.

Starting thresholds are informed guesses, not measurements. They must be
re-tuned against real photos during the calibration phase — the GEMS profile
images this will eventually compare against are capped at 500px / 150KB, so
thresholds tuned on clean images will reject valid inputs. `metrics` is
persisted even when a face passes, because six weeks of production values is
what turns these nine guesses into nine measured numbers, and that data cannot
be recovered retroactively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.engine.types import (
    LM_EYE_LEFT,
    LM_EYE_RIGHT,
    LM_NOSE,
    DetectedFace,
    QualityReport,
    ReasonCode,
)

NORMALISED_SIZE = 112


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_face_px: float = 80.0
    min_detection_score: float = 0.90
    min_interocular_px: float = 32.0
    # DELIBERATELY PERMISSIVE — effectively off until calibrated. Absolute
    # sharpness is strongly content-dependent (synthetic noise and face-like
    # structure differ by orders of magnitude at identical face size), so no
    # honest default exists before measuring real photos. An over-strict blur
    # gate converts would-be MATCHes into LOW_QUALITY rejections and reads as
    # "the model doesn't work" — the single largest recall risk in the pipeline.
    min_sharpness: float = 0.005
    min_brightness: float = 60.0
    max_brightness: float = 200.0
    min_contrast: float = 25.0
    min_yaw_symmetry: float = 0.55
    max_roll_degrees: float = 45.0
    min_face_area_ratio: float = 0.015
    max_face_area_ratio: float = 0.90
    edge_margin_px: int = 5
    crop_padding: float = 0.20


DEFAULT_THRESHOLDS = QualityThresholds()


def _crop_face(
    image_bgr: np.ndarray, bbox_xywh: tuple[int, int, int, int], pad: float
) -> np.ndarray:
    """Face crop padded by `pad`, clipped to the image."""
    height, width = image_bgr.shape[:2]
    x, y, w, h = bbox_xywh
    dx, dy = int(w * pad), int(h * pad)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(width, x + w + dx)
    y1 = min(height, y + h + dy)
    if x1 <= x0 or y1 <= y0:
        return image_bgr
    return image_bgr[y0:y1, x0:x1]


def _headroom(value: float, threshold: float) -> float:
    """Sub-score for a higher-is-better metric: 1.0 once the threshold is met."""
    if threshold <= 0:
        return 1.0
    return float(min(max(value / threshold, 0.0), 1.0))


def _band(value: float, low: float, high: float) -> float:
    """Sub-score for a metric that must sit inside a band."""
    if low <= value <= high:
        return 1.0
    span = max(high - low, 1e-6)
    distance = (low - value) if value < low else (value - high)
    return float(max(0.0, 1.0 - distance / span))


def assess_quality(
    image_bgr: np.ndarray,
    face: DetectedFace,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
) -> QualityReport:
    height, width = image_bgr.shape[:2]
    bbox = face.bbox
    issues: list[ReasonCode] = []

    crop = _crop_face(image_bgr, (bbox.x, bbox.y, bbox.w, bbox.h), thresholds.crop_padding)
    gray_native = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Sharpness is measured on a fixed 112x112 crop so that face size does not
    # drive it. Resizing alone is not enough — see the note below.
    gray_norm = cv2.resize(
        gray_native, (NORMALISED_SIZE, NORMALISED_SIZE), interpolation=cv2.INTER_AREA
    )

    face_px = float(min(bbox.w, bbox.h))
    detection_score = float(face.score)
    interocular = face.interocular
    # Raw variance of Laplacian is scale-dependent even after resizing to a
    # fixed 112x112: a larger source crop carries more real detail per output
    # pixel. Measured on identical content across 90-250px face sizes, raw
    # variance spans 5.7x. Dividing by the crop's own intensity variance
    # decouples contrast and part of the scale effect, narrowing that to 4.0x
    # while keeping sharp-vs-blurred separation at ~15x.
    #
    # 4x is still too wide for a confidently-set absolute threshold, which is
    # why `min_sharpness` ships permissive and must be calibrated. Both the
    # normalised and raw values are persisted so calibration can choose.
    # CAVEAT: the ratio is unreliable below a usable contrast floor. In a
    # near-black frame both terms collapse and 8-bit quantisation noise
    # dominates, so a dark blurred image can report HIGHER sharpness than a
    # clean one. Such images are always caught by the contrast check below —
    # that coupling is load-bearing, and `test_sharpness_is_unreliable_below_
    # the_contrast_floor` exists so nobody relaxes LOW_CONTRAST without
    # noticing it opens a hole here.
    blur_variance = float(cv2.Laplacian(gray_norm, cv2.CV_64F).var())
    sharpness = blur_variance / max(float(gray_norm.std()) ** 2, 1.0)
    brightness = float(gray_native.mean())
    contrast = float(gray_native.std())

    # Yaw proxy: a frontal face puts the nose midway between the eyes, so the
    # two horizontal distances match. A turned head makes them diverge. If the
    # nose falls outside the eyes entirely, one distance goes negative and the
    # ratio fails on its own.
    eye_l, eye_r, nose = (
        face.landmarks[LM_EYE_LEFT],
        face.landmarks[LM_EYE_RIGHT],
        face.landmarks[LM_NOSE],
    )
    d_left = float(nose[0] - eye_l[0])
    d_right = float(eye_r[0] - nose[0])
    denom = max(abs(d_left), abs(d_right), 1e-6)
    yaw_symmetry = float(min(d_left, d_right) / denom)

    # Roll is corrected perfectly by alignment, so its threshold is deliberately
    # loose. It matters only as a signal that the detector is confused.
    roll_degrees = abs(
        math.degrees(math.atan2(float(eye_r[1] - eye_l[1]), float(eye_r[0] - eye_l[0])))
    )
    if roll_degrees > 180.0:
        roll_degrees = 360.0 - roll_degrees

    frame_area = max(width * height, 1)
    face_area_ratio = float(bbox.area / frame_area)

    m = thresholds.edge_margin_px
    contained = (
        bbox.x >= -m and bbox.y >= -m and bbox.right <= width + m and bbox.bottom <= height + m
    )

    if face_px < thresholds.min_face_px or interocular < thresholds.min_interocular_px:
        issues.append(ReasonCode.FACE_TOO_SMALL)
    if detection_score < thresholds.min_detection_score:
        issues.append(ReasonCode.LOW_DETECTION_CONFIDENCE)
    if sharpness < thresholds.min_sharpness:
        issues.append(ReasonCode.IMAGE_BLURRY)
    if brightness < thresholds.min_brightness:
        issues.append(ReasonCode.TOO_DARK)
    elif brightness > thresholds.max_brightness:
        issues.append(ReasonCode.TOO_BRIGHT)
    if contrast < thresholds.min_contrast:
        issues.append(ReasonCode.LOW_CONTRAST)
    if yaw_symmetry < thresholds.min_yaw_symmetry:
        issues.append(ReasonCode.EXTREME_POSE)
    if roll_degrees > thresholds.max_roll_degrees:
        issues.append(ReasonCode.EXTREME_ROLL)
    if face_area_ratio < thresholds.min_face_area_ratio:
        issues.append(ReasonCode.FACE_TOO_SMALL_IN_FRAME)
    elif face_area_ratio > thresholds.max_face_area_ratio:
        issues.append(ReasonCode.FACE_FILLS_FRAME)
    if not contained:
        issues.append(ReasonCode.FACE_OUT_OF_FRAME)

    # A coarse indicator for ranking and for the UI meter. The `issues` list is
    # what is authoritative — never gate on this number.
    sub_scores = [
        _headroom(face_px, thresholds.min_face_px),
        _headroom(detection_score, thresholds.min_detection_score),
        _headroom(interocular, thresholds.min_interocular_px),
        _headroom(sharpness, thresholds.min_sharpness),
        _band(brightness, thresholds.min_brightness, thresholds.max_brightness),
        _headroom(contrast, thresholds.min_contrast),
        _headroom(max(yaw_symmetry, 0.0), thresholds.min_yaw_symmetry),
        _band(roll_degrees, 0.0, thresholds.max_roll_degrees),
        _band(face_area_ratio, thresholds.min_face_area_ratio, thresholds.max_face_area_ratio),
    ]
    score = float(np.clip(float(np.mean(sub_scores)), 0.0, 1.0))

    return QualityReport(
        passed=not issues,
        score=score,
        issues=tuple(issues),
        metrics={
            "face_px": round(face_px, 2),
            "detection_score": round(detection_score, 4),
            "interocular_px": round(interocular, 2),
            "sharpness": round(sharpness, 5),
            # Raw value kept alongside: calibration wants both, and
            # persisting it now costs nothing.
            "blur_variance": round(blur_variance, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "yaw_symmetry": round(yaw_symmetry, 4),
            "roll_degrees": round(roll_degrees, 2),
            "face_area_ratio": round(face_area_ratio, 5),
            "contained": float(contained),
        },
    )
