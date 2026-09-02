"""Detection of things placed in front of a face: masks, scarves, sunglasses, hands.

WHY THIS IS RELATIVE, NEVER ABSOLUTE
------------------------------------
A covered region is not identified by being dark, or grey, or blue — it is
identified by *not looking like the rest of this person's face*. Every measure
here is therefore a ratio between two regions of the same face in the same
photograph, which cancels out skin tone, exposure, white balance and camera.
An absolute "is this skin?" threshold would fail the darkest and lightest
subjects first, which is exactly the failure a verification system must not
have.

WHAT THIS CAN AND CANNOT SEE
----------------------------
Catches: surgical and cloth masks, scarves and high collars, sunglasses, an
opaque hand held over the mouth or eyes — anything that replaces facial skin
with a different colour.

Misses, by construction:
  * A bare hand over the face. It is skin, and this looks for the absence of
    skin. Only the detector's own confidence score speaks to that case.
  * Both halves covered at once (balaclava, full veil). Each half is judged
    against the other, so when neither is skin there is no reference left.
    A face that hidden usually fails detection or LOW_DETECTION_CONFIDENCE.
  * Clear prescription glasses, correctly — they do not obstruct recognition
    and rejecting them would be a false alarm the operator cannot act on.

FAIL-OPEN BY DESIGN
-------------------
When the reference half does not itself read as skin — a greyscale scan, a
heavy colour cast, a synthetic image — no comparison is trustworthy, so no
issue is raised. A missed covering costs one retry; a false "remove your mask"
aimed at someone wearing nothing is unactionable and teaches operators to
ignore the gate.

Thresholds are informed guesses like the rest of the quality gate, and are
deliberately permissive. `metrics` is persisted on every assessment, pass or
fail, so calibration can set them from real photographs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.engine.types import (
    LM_EYE_LEFT,
    LM_EYE_RIGHT,
    LM_MOUTH_LEFT,
    LM_MOUTH_RIGHT,
    DetectedFace,
    ReasonCode,
)

# Classic Cr/Cb skin locus. Chroma carries far less skin-tone variation than
# luma does, which is why the test is made here and not on brightness.
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)


@dataclass(frozen=True, slots=True)
class OcclusionThresholds:
    # Below this, the half being used as the reference does not read as skin
    # itself, so it cannot judge the other half. See FAIL-OPEN above.
    min_reference_skin: float = 0.35
    # A covered half must show less than this fraction of the skin the
    # reference half shows. 0.35 is a large, deliberate gap: a mask or a lens
    # drops the ratio close to zero, while shadow, stubble and lipstick move it
    # only a little. Set wide on purpose until calibration measures the spread.
    max_relative_skin: float = 0.35


DEFAULT_OCCLUSION = OcclusionThresholds()


@dataclass(frozen=True, slots=True)
class OcclusionReport:
    issues: tuple[ReasonCode, ...]
    metrics: dict[str, float]


def _region(
    image_bgr: np.ndarray, centre: np.ndarray, width: float, height: float
) -> np.ndarray | None:
    """An axis-aligned patch of the image, clipped to its bounds.

    Axis-aligned rather than rotated: beyond ~45 degrees of roll the bands stop
    lining up with the features, but that is already rejected as EXTREME_ROLL
    before anyone reads these numbers.
    """
    frame_h, frame_w = image_bgr.shape[:2]
    # float() first: round() on a numpy scalar returns a numpy scalar, which is
    # not a valid slice index.
    cx, cy = float(centre[0]), float(centre[1])
    x0 = max(0, round(cx - width / 2.0))
    x1 = min(frame_w, round(cx + width / 2.0))
    y0 = max(0, round(cy - height / 2.0))
    y1 = min(frame_h, round(cy + height / 2.0))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return image_bgr[y0:y1, x0:x1]


def _skin_ratio(patch: np.ndarray) -> float:
    """Fraction of the patch whose chroma falls inside the skin locus."""
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin = (cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1]) & (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1])
    return float(np.mean(skin))


def _texture(patch: np.ndarray) -> float:
    """Normalised edge energy. Recorded for calibration; nothing gates on it yet.

    A mask is not only the wrong colour, it is flatter than the face it covers.
    That is a second, independent signal — but its spread across real fabrics
    and real skin is unmeasured, so it is persisted rather than acted on.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var() / max(float(gray.std()) ** 2, 1.0))


def assess_occlusion(
    image_bgr: np.ndarray,
    face: DetectedFace,
    thresholds: OcclusionThresholds = DEFAULT_OCCLUSION,
) -> OcclusionReport:
    """Compare the eye half and the mouth half of a face against each other."""
    landmarks = face.landmarks
    eye_centre = (landmarks[LM_EYE_LEFT] + landmarks[LM_EYE_RIGHT]) / 2.0
    mouth_centre = (landmarks[LM_MOUTH_LEFT] + landmarks[LM_MOUTH_RIGHT]) / 2.0

    # Every dimension is a multiple of interocular distance, so the bands track
    # the face through any scale without a pixel constant to re-tune.
    # Half-heights sum to 0.6 * interocular, below the eye-to-mouth separation
    # of even the most compressed face geometry, so the two bands can never
    # overlap and quietly sample each other.
    unit = face.interocular
    if unit <= 0:
        return OcclusionReport(issues=(), metrics={})

    eyes = _region(image_bgr, eye_centre, width=unit * 2.2, height=unit * 0.6)
    mouth = _region(image_bgr, mouth_centre, width=unit * 1.8, height=unit * 0.6)
    if eyes is None or mouth is None:
        return OcclusionReport(issues=(), metrics={})

    eye_skin = _skin_ratio(eyes)
    mouth_skin = _skin_ratio(mouth)
    metrics = {
        "eye_skin_ratio": round(eye_skin, 4),
        "mouth_skin_ratio": round(mouth_skin, 4),
        "eye_texture": round(_texture(eyes), 5),
        "mouth_texture": round(_texture(mouth), 5),
    }

    issues: list[ReasonCode] = []
    # Each half is judged only when the other is skin enough to be believed.
    if (
        eye_skin >= thresholds.min_reference_skin
        and mouth_skin / max(eye_skin, 1e-6) < thresholds.max_relative_skin
    ):
        issues.append(ReasonCode.FACE_COVERED)
    if (
        mouth_skin >= thresholds.min_reference_skin
        and eye_skin / max(mouth_skin, 1e-6) < thresholds.max_relative_skin
    ):
        issues.append(ReasonCode.EYES_COVERED)

    return OcclusionReport(issues=tuple(issues), metrics=metrics)
