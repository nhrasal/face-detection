"""One case per covering, plus the cases where the check must stay quiet.

The fail-open cases matter as much as the detections: a false "remove your
mask" aimed at a bare face is unactionable, and operators learn to ignore a
gate that cries wolf.
"""

from __future__ import annotations

import numpy as np

from app.engine.occlusion import OcclusionThresholds, assess_occlusion
from app.engine.types import (
    LM_EYE_LEFT,
    LM_EYE_RIGHT,
    LM_MOUTH_LEFT,
    LM_MOUTH_RIGHT,
    ReasonCode,
)
from tests.factories.images import (
    LENS_BGR,
    MASK_BGR,
    make_face,
    paint_band,
    skin_face_array,
    textured_array,
)

FACE_BBOX = (100, 80, 160, 160)


def _face():
    return make_face(bbox=FACE_BBOX)


def _centres(face) -> tuple[np.ndarray, np.ndarray, float]:
    eyes = (face.landmarks[LM_EYE_LEFT] + face.landmarks[LM_EYE_RIGHT]) / 2.0
    mouth = (face.landmarks[LM_MOUTH_LEFT] + face.landmarks[LM_MOUTH_RIGHT]) / 2.0
    return eyes, mouth, face.interocular


def _masked() -> np.ndarray:
    """Skin face with the mouth band painted surgical blue."""
    face = _face()
    image = skin_face_array(bbox=FACE_BBOX)
    _, mouth, unit = _centres(face)
    paint_band(image, mouth, unit * 1.8, unit * 0.6, MASK_BGR)
    return image


def _sunglassed() -> np.ndarray:
    """Skin face with the eye band painted a dark lens."""
    face = _face()
    image = skin_face_array(bbox=FACE_BBOX)
    eyes, _, unit = _centres(face)
    paint_band(image, eyes, unit * 2.2, unit * 0.6, LENS_BGR)
    return image


class TestUncoveredFace:
    def test_bare_skin_raises_nothing(self) -> None:
        report = assess_occlusion(skin_face_array(bbox=FACE_BBOX), _face())
        assert report.issues == ()

    def test_both_halves_are_strong_enough_to_reference_each_other(self) -> None:
        # Not 1.0: make_face sets the eyes wider apart than a real face, so a
        # band of 2.2 x interocular overhangs the patch onto the background.
        # What matters is that both sit well clear of `min_reference_skin`.
        report = assess_occlusion(skin_face_array(bbox=FACE_BBOX), _face())
        assert report.metrics["eye_skin_ratio"] > 0.7
        assert report.metrics["mouth_skin_ratio"] > 0.9


class TestLowerFaceCovering:
    def test_a_mask_over_the_mouth_is_reported(self) -> None:
        report = assess_occlusion(_masked(), _face())
        assert ReasonCode.FACE_COVERED in report.issues

    def test_a_mask_does_not_also_accuse_the_eyes(self) -> None:
        # The eyes are the reference here; reporting them too would send the
        # subject removing glasses they are not wearing.
        report = assess_occlusion(_masked(), _face())
        assert ReasonCode.EYES_COVERED not in report.issues

    def test_the_covered_half_measures_as_bare_of_skin(self) -> None:
        report = assess_occlusion(_masked(), _face())
        assert report.metrics["mouth_skin_ratio"] < 0.05
        assert report.metrics["eye_skin_ratio"] > 0.7


class TestEyeCovering:
    def test_dark_lenses_are_reported(self) -> None:
        report = assess_occlusion(_sunglassed(), _face())
        assert ReasonCode.EYES_COVERED in report.issues

    def test_dark_lenses_do_not_also_accuse_the_mouth(self) -> None:
        report = assess_occlusion(_sunglassed(), _face())
        assert ReasonCode.FACE_COVERED not in report.issues


class TestFailsOpen:
    def test_says_nothing_when_neither_half_is_skin(self) -> None:
        # A balaclava leaves no reference to judge against. Documented limit:
        # a face this hidden is the detector's problem, not the gate's.
        face = _face()
        image = skin_face_array(bbox=FACE_BBOX)
        eyes, mouth, unit = _centres(face)
        paint_band(image, eyes, unit * 2.2, unit * 0.6, LENS_BGR)
        paint_band(image, mouth, unit * 1.8, unit * 0.6, MASK_BGR)
        assert assess_occlusion(image, face).issues == ()

    def test_says_nothing_about_an_image_with_no_skin_chroma_at_all(self) -> None:
        # Random noise has no skin locus to speak of, so neither half can act
        # as a reference for the other.
        assert assess_occlusion(textured_array(640, 480, seed=1), _face()).issues == ()

    def test_records_metrics_even_when_it_raises_nothing(self) -> None:
        # Six weeks of these values is what turns the two guessed thresholds
        # into measured ones.
        report = assess_occlusion(textured_array(640, 480, seed=1), _face())
        assert set(report.metrics) == {
            "eye_skin_ratio",
            "mouth_skin_ratio",
            "eye_texture",
            "mouth_texture",
        }

    def test_degenerate_landmarks_produce_no_finding(self) -> None:
        face = make_face(bbox=(0, 0, 0, 0))
        assert assess_occlusion(skin_face_array(), face).issues == ()


class TestThresholds:
    def test_a_stricter_ratio_catches_a_partial_covering(self) -> None:
        # Half the mouth band covered: ignored at the shipped 0.35, caught once
        # the ratio is tightened. Proves the knob is the thing that decides.
        face = _face()
        image = skin_face_array(bbox=FACE_BBOX)
        _, mouth, unit = _centres(face)
        paint_band(image, mouth, unit * 0.8, unit * 0.6, MASK_BGR)
        assert assess_occlusion(image, face).issues == ()
        strict = OcclusionThresholds(max_relative_skin=0.75)
        assert ReasonCode.FACE_COVERED in assess_occlusion(image, face, strict).issues

    def test_ratio_is_measured_against_the_other_half_not_an_absolute(self) -> None:
        # The same covering on a much darker skin tone must still be caught:
        # nothing here compares against a fixed brightness.
        face = _face()
        image = skin_face_array(bbox=FACE_BBOX)
        image[image[:, :, 2] > 100] = (70, 95, 130)
        _, mouth, unit = _centres(face)
        paint_band(image, mouth, unit * 1.8, unit * 0.6, MASK_BGR)
        report = assess_occlusion(image, face)
        assert ReasonCode.FACE_COVERED in report.issues
