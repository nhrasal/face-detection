"""Eye measurement against real portraits and real weights.

The hermetic tier proves the state machine with scripted numbers. This proves
the NUMBERS mean what the state machine assumes — that a shut eye reads lower
than an open one, and that a blurred frame does not impersonate a shut one.

Closed eyes are SYNTHESISED, because every subject in the test set has their
eyes open. How they are synthesised is the whole value of this file, and an
earlier version of it got this wrong in a way worth recording: it flattened the
ENTIRE eye patch. That erases brow, socket and cheek along with the eye, which
no blink does — it modelled a hole in the face rather than a closed lid, passed
comfortably, and hid the fact that the check could not fire on a real face.

What is modelled here instead is a lid drawn across the eye APERTURE only, with
skin tone sampled from just below the eye, leaving everything around it intact.
That is a much weaker signal, which is the point: it is the signal a real blink
actually produces.

Honest limit on what remains unproved: a still photograph cannot show how far an
OPEN eye wanders between consecutive frames, and that noise floor is what sets
the safe distance for the closed threshold. It needs video of real faces. Until
then the threshold is deliberately loose. That gap is stated in
app/engine/liveness.py too.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.engine.liveness import (
    DEFAULT_PRESENCE_THRESHOLDS,
    eye_openness,
    face_sharpness,
)
from app.engine.types import LM_EYE_LEFT, LM_EYE_RIGHT, DetectedFace

pytestmark = pytest.mark.models

THRESHOLDS = DEFAULT_PRESENCE_THRESHOLDS


def _largest_face(engine, image: np.ndarray) -> DetectedFace:
    faces = engine.detect(image)
    assert faces, "portrait should contain a detectable face"
    return max(faces, key=lambda face: face.bbox.area)


# The eye opening itself, as fractions of interocular distance — NOT the patch
# the metric reads. Covering more than this would model something no blink does.
APERTURE_WIDTH = 0.35
APERTURE_HEIGHT = 0.12


def _shut_eyes(image: np.ndarray, face: DetectedFace) -> np.ndarray:
    """Draw a lid across each eye aperture, in skin tone taken from below it.

    Everything outside the aperture — brow, socket, cheek — is left untouched,
    because a blink leaves all of it untouched.
    """
    out = image.copy()
    half_width = max(2, round(face.interocular * APERTURE_WIDTH / 2))
    half_height = max(1, round(face.interocular * APERTURE_HEIGHT / 2))
    for index in (LM_EYE_LEFT, LM_EYE_RIGHT):
        x = round(float(face.landmarks[index][0]))
        y = round(float(face.landmarks[index][1]))
        left, right = max(0, x - half_width), min(out.shape[1], x + half_width + 1)
        top, bottom = max(0, y - half_height), min(out.shape[0], y + half_height + 1)
        # Skin sampled from a band under the eye: that is what the lid that
        # covers it looks like.
        skin_top = min(out.shape[0] - 1, bottom + half_height)
        skin_bottom = min(out.shape[0], bottom + 3 * half_height)
        if skin_bottom <= skin_top:
            continue
        out[top:bottom, left:right] = out[skin_top:skin_bottom, left:right].mean(axis=(0, 1))
    return out


def _readings(engine, path):
    image = cv2.imread(str(path))
    face = _largest_face(engine, image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = max(3, round(face.interocular * 0.25) | 1)
    blurred = cv2.cvtColor(cv2.GaussianBlur(image, (kernel, kernel), 0), cv2.COLOR_BGR2GRAY)
    shut = cv2.cvtColor(_shut_eyes(image, face), cv2.COLOR_BGR2GRAY)
    return {
        "open": eye_openness(gray, face),
        "shut": eye_openness(shut, face),
        "blurred": eye_openness(blurred, face),
        "sharp_open": face_sharpness(gray, face),
        "sharp_shut": face_sharpness(shut, face),
        "sharp_blurred": face_sharpness(blurred, face),
    }


@pytest.fixture(scope="module")
def readings(engine, faces_dir):
    portraits = sorted(faces_dir.glob("*.jpg"))
    assert portraits, "no portraits to measure"
    return [_readings(engine, path) for path in portraits]


class TestOpenEyes:
    def test_every_portrait_yields_a_measurement(self, readings) -> None:
        # A None here would mean the check silently never runs.
        assert all(r["open"] is not None for r in readings)

    def test_open_eyes_read_well_above_zero(self, readings) -> None:
        assert min(r["open"] for r in readings) > 0.15


class TestShutEyesReadLower:
    def test_a_shut_eye_falls_below_the_closed_threshold(self, readings) -> None:
        """Every one of the eleven, not merely most of them.

        This is the assertion the square-patch version passed while the feature
        did not work: with a realistic closure, "most faces" is the same as
        "does not fire", because the one face it misses is somebody's.
        """
        worst = max(r["shut"] / r["open"] for r in readings)
        assert worst <= THRESHOLDS.eyes_closed_ratio

    def test_the_patch_geometry_leaves_headroom_below_the_threshold(self, readings) -> None:
        """A blink must clear the bar, not graze it.

        The frame that happens to be sampled may catch the lid halfway, and a
        half-closed eye dips less than a shut one. If this margin disappears,
        the check starts missing real blinks long before any test fails.
        """
        worst = max(r["shut"] / r["open"] for r in readings)
        assert worst < THRESHOLDS.eyes_closed_ratio * 0.9

    def test_shutting_the_eyes_barely_moves_face_sharpness(self, readings) -> None:
        # This is what lets the blur guard stay out of a real blink's way.
        worst = min(r["sharp_shut"] / r["sharp_open"] for r in readings)
        assert worst > THRESHOLDS.min_sharpness_ratio


class TestBlurCannotImpersonateABlink:
    def test_blur_alone_would_otherwise_look_like_a_shut_eye(self, readings) -> None:
        """Documents WHY the sharpness guard exists rather than assuming it.

        On at least one real portrait a blurred frame drives the eye metric
        below the closed threshold. Without the guard that is a free blink for
        anyone willing to shake a printed photo.
        """
        ratios = [r["blurred"] / r["open"] for r in readings]
        assert min(ratios) <= THRESHOLDS.eyes_closed_ratio

    def test_but_the_sharpness_guard_rejects_every_blurred_frame(self, readings) -> None:
        worst = max(r["sharp_blurred"] / r["sharp_open"] for r in readings)
        assert worst < THRESHOLDS.min_sharpness_ratio

    def test_the_guard_separates_the_two_cases_with_margin(self, readings) -> None:
        """A blink and a blurred frame must not be near neighbours.

        If these bands ever touch, the guard is tuned on luck rather than on a
        real difference, and the threshold needs revisiting before shipping.
        """
        blink_floor = min(r["sharp_shut"] / r["sharp_open"] for r in readings)
        blur_ceiling = max(r["sharp_blurred"] / r["sharp_open"] for r in readings)
        assert blink_floor > blur_ceiling * 5
