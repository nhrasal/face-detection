"""One case per quality metric, each isolating the condition it detects."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.engine.quality import DEFAULT_THRESHOLDS, QualityThresholds, assess_quality
from app.engine.types import (
    LM_MOUTH_LEFT,
    LM_MOUTH_RIGHT,
    BoundingBox,
    DetectedFace,
    ReasonCode,
)
from tests.factories.images import (
    MASK_BGR,
    make_face,
    paint_band,
    skin_face_array,
    structured_array,
    textured_array,
)

GOOD_BBOX = (100, 80, 160, 160)


def good_image() -> np.ndarray:
    """Sharp, well-exposed, high-contrast — passes every metric."""
    return textured_array(640, 480, seed=1)


def structured_image() -> np.ndarray:
    """Face-like structure; the honest input for sharpness behaviour."""
    return structured_array(640, 480, seed=1)


class TestPassingCase:
    def test_clean_image_and_frontal_face_passes(self) -> None:
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX))
        assert report.passed
        assert report.issues == ()
        assert 0.0 <= report.score <= 1.0

    def test_metrics_are_recorded_even_on_pass(self) -> None:
        # Persisting metrics on success is what turns nine guessed thresholds
        # into nine measured ones later; the data is unrecoverable afterwards.
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX))
        assert set(report.metrics) == {
            "face_px",
            "detection_score",
            "interocular_px",
            "sharpness",
            "blur_variance",
            "brightness",
            "contrast",
            "yaw_symmetry",
            "roll_degrees",
            "face_area_ratio",
            "contained",
            # Folded in from the occlusion check, which records on every
            # assessment for the same calibration reason.
            "eye_skin_ratio",
            "mouth_skin_ratio",
            "eye_texture",
            "mouth_texture",
        }


class TestOcclusion:
    """The gate surfaces occlusion findings; occlusion.py owns the judgement."""

    def test_a_covered_lower_face_reaches_the_quality_issues(self) -> None:
        face = make_face(bbox=GOOD_BBOX)
        image = skin_face_array(bbox=GOOD_BBOX)
        mouth = (face.landmarks[LM_MOUTH_LEFT] + face.landmarks[LM_MOUTH_RIGHT]) / 2.0
        paint_band(image, mouth, face.interocular * 1.8, face.interocular * 0.6, MASK_BGR)
        report = assess_quality(image, face)
        assert ReasonCode.FACE_COVERED in report.issues
        assert not report.passed

    def test_a_noisy_image_raises_no_occlusion_finding(self) -> None:
        # The existing passing case must stay passing: random noise has no skin
        # reference, so the check fails open rather than inventing a covering.
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX))
        assert ReasonCode.FACE_COVERED not in report.issues
        assert ReasonCode.EYES_COVERED not in report.issues


class TestFaceSize:
    def test_small_bbox_trips_face_too_small(self) -> None:
        report = assess_quality(good_image(), make_face(bbox=(10, 10, 40, 40)))
        assert ReasonCode.FACE_TOO_SMALL in report.issues
        assert not report.passed

    def test_interocular_distance_is_reported(self) -> None:
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX))
        assert report.metrics["interocular_px"] == pytest.approx(160 * 0.60, abs=1.0)


class TestDetectionConfidence:
    def test_low_score_trips_low_confidence(self) -> None:
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX, score=0.55))
        assert ReasonCode.LOW_DETECTION_CONFIDENCE in report.issues

    def test_score_at_threshold_passes(self) -> None:
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX, score=0.90))
        assert ReasonCode.LOW_DETECTION_CONFIDENCE not in report.issues


class TestBlur:
    """Sharpness is asserted RELATIVELY, never against an absolute constant.

    Absolute sharpness depends heavily on image content — white noise and
    face-like structure differ by three orders of magnitude at the same face
    size — so a synthetic image cannot justify a production threshold. These
    tests inject their own threshold to exercise the mechanism, and the shipped
    default stays permissive until real photos are measured.
    """

    STRICT = QualityThresholds(min_sharpness=0.02)

    def test_heavy_gaussian_blur_trips_blurry(self) -> None:
        blurred = cv2.GaussianBlur(structured_image(), (31, 31), 12)
        report = assess_quality(blurred, make_face(bbox=GOOD_BBOX), self.STRICT)
        assert ReasonCode.IMAGE_BLURRY in report.issues

    def test_sharp_image_passes_the_same_threshold(self) -> None:
        # Without this, the test above would also pass if everything failed.
        report = assess_quality(structured_image(), make_face(bbox=GOOD_BBOX), self.STRICT)
        assert ReasonCode.IMAGE_BLURRY not in report.issues

    @pytest.mark.parametrize("size", [90, 130, 180, 250])
    def test_sharp_and_blurred_are_separated_at_every_face_size(self, size: int) -> None:
        """The property that actually matters: blur is detectable regardless of size."""
        img = structured_image()
        face = make_face(bbox=(100, 80, size, size))
        sharp = assess_quality(img, face).metrics["sharpness"]
        blurred = assess_quality(cv2.GaussianBlur(img, (31, 31), 12), face).metrics["sharpness"]
        assert sharp > blurred * 5

    def test_normalisation_reduces_scale_dependence(self) -> None:
        """Contrast normalisation must beat raw variance across face sizes.

        This is the honest claim. It does not make the metric scale-INdependent
        — a larger crop still carries more detail per output pixel — it only
        narrows the spread (measured 5.7x raw vs 4.0x normalised).
        """
        img = structured_image()
        reports = [
            assess_quality(img, make_face(bbox=(100, 80, s, s))) for s in (90, 130, 180, 250)
        ]
        raw = [r.metrics["blur_variance"] for r in reports]
        normalised = [r.metrics["sharpness"] for r in reports]
        assert max(normalised) / min(normalised) < max(raw) / min(raw)

    def test_absolute_sharpness_is_content_dependent(self) -> None:
        """Documents why no absolute default can be trusted before calibration."""
        face = make_face(bbox=GOOD_BBOX)
        noise = assess_quality(textured_array(640, 480, seed=1), face).metrics["sharpness"]
        structured = assess_quality(structured_image(), face).metrics["sharpness"]
        assert noise > structured * 100

    def test_raw_blur_variance_is_still_recorded(self) -> None:
        # Kept alongside the normalised value because calibration wants both.
        metrics = assess_quality(structured_image(), make_face(bbox=GOOD_BBOX)).metrics
        assert metrics["blur_variance"] > 0
        assert metrics["sharpness"] > 0

    def test_sharpness_is_unreliable_below_the_contrast_floor(self) -> None:
        """Guards a coupling that is easy to break by accident.

        In a near-black frame both the Laplacian energy and the intensity
        variance collapse, so quantisation noise dominates and a dark BLURRED
        image reports higher sharpness than a clean one. Nothing escapes,
        because LOW_CONTRAST rejects every such image — but that is the only
        thing standing in the way, so it is asserted rather than assumed.
        """
        img = structured_image()
        face = make_face(bbox=(10, 10, 40, 40))
        dark_blurry = cv2.GaussianBlur((img * 0.1).astype(np.uint8), (31, 31), 12)

        clean = assess_quality(img, face).metrics["sharpness"]
        degraded = assess_quality(dark_blurry, face)

        # The metric genuinely inverts here — documenting, not endorsing.
        assert degraded.metrics["sharpness"] > clean
        # ...and the image is still rejected, by contrast rather than by blur.
        assert ReasonCode.LOW_CONTRAST in degraded.issues
        assert not degraded.passed

    def test_shipped_default_does_not_reject_a_sharp_small_face(self) -> None:
        # The regression guard for over-strict blur gating: a genuinely sharp
        # face at the minimum allowed size must not be called blurry.
        report = assess_quality(structured_image(), make_face(bbox=(100, 80, 90, 90)))
        assert ReasonCode.IMAGE_BLURRY not in report.issues


class TestExposure:
    def test_dark_image_trips_too_dark(self) -> None:
        report = assess_quality((good_image() * 0.12).astype(np.uint8), make_face(bbox=GOOD_BBOX))
        assert ReasonCode.TOO_DARK in report.issues

    def test_blown_out_image_trips_too_bright(self) -> None:
        bright = np.clip(good_image().astype(np.int32) + 190, 0, 255).astype(np.uint8)
        report = assess_quality(bright, make_face(bbox=GOOD_BBOX))
        assert ReasonCode.TOO_BRIGHT in report.issues

    def test_dark_and_bright_are_mutually_exclusive(self) -> None:
        report = assess_quality((good_image() * 0.12).astype(np.uint8), make_face(bbox=GOOD_BBOX))
        assert not (ReasonCode.TOO_DARK in report.issues and ReasonCode.TOO_BRIGHT in report.issues)

    def test_flat_grey_trips_low_contrast(self) -> None:
        flat = np.full((480, 640, 3), 128, dtype=np.uint8)
        report = assess_quality(flat, make_face(bbox=GOOD_BBOX))
        assert ReasonCode.LOW_CONTRAST in report.issues


class TestPose:
    def test_frontal_face_has_symmetric_yaw(self) -> None:
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX, yaw_offset=0.0))
        assert report.metrics["yaw_symmetry"] == pytest.approx(1.0, abs=0.02)
        assert ReasonCode.EXTREME_POSE not in report.issues

    @pytest.mark.parametrize("offset", [0.45, -0.45, 0.6])
    def test_severely_turned_head_trips_extreme_pose(self, offset: float) -> None:
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX, yaw_offset=offset))
        assert ReasonCode.EXTREME_POSE in report.issues

    @pytest.mark.parametrize("offset", [0.2, -0.2, 0.3])
    def test_moderately_turned_head_is_tolerated(self, offset: float) -> None:
        """Deliberate: SFace handles far more yaw than the first guess assumed.

        At the original 0.55 threshold, 2 of 11 verified-good official
        portraits were rejected as EXTREME_POSE despite producing correct
        identity decisions. Rejecting a usable photo costs more than embedding
        a slightly angled one.
        """
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX, yaw_offset=offset))
        assert ReasonCode.EXTREME_POSE not in report.issues

    def test_nose_outside_the_eyes_trips_extreme_pose(self) -> None:
        # Beyond the eye line the distance goes negative; the ratio must fail
        # rather than wrap around into a passing value.
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX, yaw_offset=1.5))
        assert ReasonCode.EXTREME_POSE in report.issues
        assert report.metrics["yaw_symmetry"] < 0

    def test_moderate_roll_is_tolerated(self) -> None:
        # Alignment corrects roll perfectly, so the threshold is loose by design.
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX, roll_degrees=20.0))
        assert ReasonCode.EXTREME_ROLL not in report.issues

    def test_severe_roll_trips_extreme_roll(self) -> None:
        report = assess_quality(good_image(), make_face(bbox=GOOD_BBOX, roll_degrees=70.0))
        assert ReasonCode.EXTREME_ROLL in report.issues


class TestFraming:
    def test_tiny_face_in_large_frame_trips_ratio(self) -> None:
        report = assess_quality(textured_array(2000, 2000), make_face(bbox=(10, 10, 100, 100)))
        assert ReasonCode.FACE_TOO_SMALL_IN_FRAME in report.issues

    def test_face_filling_the_frame_trips_ratio(self) -> None:
        img = textured_array(200, 200)
        report = assess_quality(img, make_face(bbox=(0, 0, 199, 199)))
        assert ReasonCode.FACE_FILLS_FRAME in report.issues

    def test_bbox_past_the_edge_trips_out_of_frame(self) -> None:
        img = textured_array(640, 480)
        face = DetectedFace(
            bbox=BoundingBox(560, 80, 200, 200),  # right edge at 760 > 640
            landmarks=make_face(bbox=(560, 80, 200, 200)).landmarks,
            score=0.99,
        )
        report = assess_quality(img, face)
        assert ReasonCode.FACE_OUT_OF_FRAME in report.issues
        assert report.metrics["contained"] == 0.0


class TestReportSemantics:
    def test_all_violated_codes_are_reported_not_just_the_first(self) -> None:
        # A user fixing one problem at a time, told only about the first, gives
        # up before getting a usable photo.
        blurry = cv2.GaussianBlur(structured_image(), (31, 31), 12)
        report = assess_quality(
            blurry,
            make_face(bbox=(10, 10, 40, 40), score=0.4),
            QualityThresholds(min_sharpness=0.02),
        )
        assert {
            ReasonCode.IMAGE_BLURRY,
            ReasonCode.FACE_TOO_SMALL,
            ReasonCode.LOW_DETECTION_CONFIDENCE,
            ReasonCode.FACE_TOO_SMALL_IN_FRAME,
        } <= set(report.issues)

    def test_passed_is_exactly_the_absence_of_issues(self) -> None:
        for face in (make_face(bbox=GOOD_BBOX), make_face(bbox=(5, 5, 30, 30), score=0.2)):
            report = assess_quality(good_image(), face)
            assert report.passed is (len(report.issues) == 0)

    def test_score_falls_as_quality_degrades(self) -> None:
        good = assess_quality(good_image(), make_face(bbox=GOOD_BBOX))
        bad = assess_quality(
            cv2.GaussianBlur((structured_image() * 0.1).astype(np.uint8), (31, 31), 12),
            make_face(bbox=(10, 10, 40, 40), score=0.3),
        )
        assert bad.score < good.score
        assert 0.0 <= bad.score <= 1.0

    def test_thresholds_are_injectable(self) -> None:
        # Calibration has to be able to sweep these without editing source.
        face = make_face(bbox=(10, 10, 40, 40))
        assert ReasonCode.FACE_TOO_SMALL in assess_quality(good_image(), face).issues
        lenient = QualityThresholds(min_face_px=10.0, min_interocular_px=5.0)
        assert ReasonCode.FACE_TOO_SMALL not in assess_quality(good_image(), face, lenient).issues

    def test_default_thresholds_are_immutable(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            DEFAULT_THRESHOLDS.min_face_px = 1.0  # type: ignore[misc]
