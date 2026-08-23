"""Full path on real images: bytes -> decode -> analyse -> decide.

The hermetic tier proves each stage in isolation with a scripted engine. This
proves they compose correctly against real weights and real photographs, which
is the only place integration mistakes actually show up.
"""

from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np
import pytest

from app.engine.decode import decode_image
from app.engine.pipeline import FaceSelectionPolicy, FaceStatus, analyse
from app.engine.quality import QualityThresholds
from app.engine.types import ReasonCode
from app.services.decision import Decision, decide_comparison
from tests.factories.images import jpeg_with_orientation

pytestmark = pytest.mark.models


def _analyse_file(engine, path, **kwargs):
    return analyse(engine, cv2.imread(str(path)), **kwargs)


class TestTerminalOutcomes:
    def test_landscape_photo_yields_no_face(self, engine, assets_dir) -> None:
        result = _analyse_file(engine, assets_dir / "no_face.jpg")
        assert result.status is FaceStatus.NO_FACE
        assert result.face_count == 0
        assert result.embedding is None

    def test_group_photo_yields_multiple_faces(self, engine, assets_dir) -> None:
        result = _analyse_file(engine, assets_dir / "group.jpg")
        assert result.status is FaceStatus.MULTIPLE_FACES
        assert result.face_count > 1
        assert result.embedding is None

    def test_group_photo_can_be_resolved_by_policy(self, engine, assets_dir) -> None:
        # What V3 live capture will need, where a bystander in frame is routine.
        # Thresholds are relaxed because faces in a 7-person crew shot are tiny;
        # what is under test is the selection policy, not the quality gate.
        result = _analyse_file(
            engine,
            assets_dir / "group.jpg",
            multi_face=FaceSelectionPolicy.LARGEST,
            thresholds=QualityThresholds(
                min_face_px=20.0, min_interocular_px=8.0, min_face_area_ratio=0.0
            ),
        )
        assert result.status is FaceStatus.OK
        assert result.face_count > 1
        assert result.embedding is not None

    def test_portrait_yields_ok_with_an_embedding(self, engine, faces_dir) -> None:
        result = _analyse_file(engine, faces_dir / "jemison_1.jpg")
        assert result.status is FaceStatus.OK
        assert result.embedding is not None
        assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-5)
        assert result.quality is not None and result.quality.passed

    def test_blurred_portrait_is_rejected_as_low_quality(self, engine, faces_dir) -> None:
        image = cv2.GaussianBlur(cv2.imread(str(faces_dir / "jemison_1.jpg")), (41, 41), 20)
        # min_sharpness ships permissive until calibration, so this injects a
        # working threshold to prove the gate itself functions.
        result = analyse(engine, image, thresholds=QualityThresholds(min_sharpness=0.02))
        assert result.status is FaceStatus.LOW_QUALITY
        assert ReasonCode.IMAGE_BLURRY in result.quality.issues
        assert result.embedding is None


class TestExifRegression:
    def test_rotated_upload_still_matches_its_upright_twin(self, engine, faces_dir) -> None:
        """The named regression for the top cause of iPhone upload failures.

        A portrait tagged "rotate 90" that is not transposed fails detection
        outright. This asserts the decode path prevents that end to end.
        """
        original = cv2.imread(str(faces_dir / "collins_1.jpg"))
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        upright = decode_image(jpeg_with_orientation(rgb, orientation=1))
        # Stored pixels rotated CCW; orientation=6 means 'rotate CW to display',
        # so a decoder honouring EXIF restores the upright original.
        rotated = decode_image(jpeg_with_orientation(np.rot90(rgb, k=1).copy(), orientation=6))

        a = analyse(engine, upright)
        b = analyse(engine, rotated)
        assert a.status is FaceStatus.OK
        assert b.status is FaceStatus.OK, "EXIF-rotated portrait failed to produce a face"

        outcome = decide_comparison(a, b, threshold=engine.info.default_threshold)
        assert outcome.decision is Decision.MATCH


class TestEndToEndDecisions:
    def test_two_photos_of_one_person_match(self, engine, faces_dir) -> None:
        outcome = decide_comparison(
            _analyse_file(engine, faces_dir / "bluford_1.jpg"),
            _analyse_file(engine, faces_dir / "bluford_2.jpg"),
            threshold=engine.info.default_threshold,
        )
        assert outcome.decision is Decision.MATCH
        assert outcome.matched
        assert outcome.similarity is not None and outcome.similarity > outcome.threshold
        assert outcome.confidence is not None and outcome.confidence > 0.5

    def test_photos_of_different_people_do_not_match(self, engine, faces_dir) -> None:
        outcome = decide_comparison(
            _analyse_file(engine, faces_dir / "bluford_1.jpg"),
            _analyse_file(engine, faces_dir / "collins_1.jpg"),
            threshold=engine.info.default_threshold,
        )
        assert outcome.decision is Decision.NO_MATCH
        assert not outcome.matched
        assert outcome.confidence is not None and outcome.confidence < 0.5

    def test_no_face_reference_short_circuits_the_comparison(
        self, engine, assets_dir, faces_dir
    ) -> None:
        outcome = decide_comparison(
            _analyse_file(engine, assets_dir / "no_face.jpg"),
            _analyse_file(engine, faces_dir / "ride_1.jpg"),
            threshold=engine.info.default_threshold,
        )
        assert outcome.decision is Decision.NO_FACE
        assert outcome.reason_code == "REFERENCE_NO_FACE"
        assert outcome.similarity is None

    def test_group_candidate_is_reported_as_multiple_faces(
        self, engine, assets_dir, faces_dir
    ) -> None:
        outcome = decide_comparison(
            _analyse_file(engine, faces_dir / "ride_1.jpg"),
            _analyse_file(engine, assets_dir / "group.jpg"),
            threshold=engine.info.default_threshold,
        )
        assert outcome.decision is Decision.MULTIPLE_FACES
        assert outcome.reason_code == "CANDIDATE_MULTIPLE_FACES"
        assert outcome.candidate_face_count > 1

    def test_confidence_rescues_a_misleading_raw_score(self, engine, faces_dir) -> None:
        """Why the API returns two numbers instead of one.

        ride_1 vs ride_2 is the weakest genuine pair in the set at ~0.47 raw
        cosine. Shown as "47% match" that reads as a failure; as confidence it
        reads as the pass it is.
        """
        outcome = decide_comparison(
            _analyse_file(engine, faces_dir / "ride_1.jpg"),
            _analyse_file(engine, faces_dir / "ride_2.jpg"),
            threshold=engine.info.default_threshold,
        )
        assert outcome.decision is Decision.MATCH
        assert outcome.similarity is not None and outcome.similarity < 0.5
        assert outcome.confidence is not None and outcome.confidence > 0.7


class TestExclusionRate:
    """The regression guard for over-strict quality gating.

    The failure mode this prevents is subtle and expensive: tighten a quality
    threshold, watch would-be MATCHes turn into LOW_QUALITY rejections, and
    conclude "the model doesn't work". It looks like a recognition problem and
    is actually a gating problem.

    Every portrait in the asset set is a verified-good official photograph that
    the recogniser handles correctly, so ANY rejection here is a false reject.
    """

    PORTRAITS: ClassVar[list[str]] = [
        "jemison_1.jpg",
        "jemison_2.jpg",
        "jemison_3.jpg",
        "ride_1.jpg",
        "ride_2.jpg",
        "bluford_1.jpg",
        "bluford_2.jpg",
        "collins_1.jpg",
        "collins_2.jpg",
        "aldrin_1.jpg",
        "aldrin_2.jpg",
    ]

    def test_no_verified_good_portrait_is_excluded(self, engine, faces_dir) -> None:
        rejected = []
        for name in self.PORTRAITS:
            result = _analyse_file(engine, faces_dir / name)
            if result.status is not FaceStatus.OK:
                codes = (
                    ",".join(i.value for i in result.quality.issues)
                    if result.quality
                    else result.status.value
                )
                rejected.append(f"{name}: {codes}")

        rate = len(rejected) / len(self.PORTRAITS)
        assert not rejected, (
            f"quality gating excludes {rate:.0%} of verified-good portraits "
            f"(budget: 0%). Loosen the offending threshold before touching the "
            f"match threshold:\n  " + "\n  ".join(rejected)
        )

    def test_every_portrait_still_produces_a_usable_embedding(self, engine, faces_dir) -> None:
        for name in self.PORTRAITS:
            result = _analyse_file(engine, faces_dir / name)
            assert result.embedding is not None, name
            assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-5), name
