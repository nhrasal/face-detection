"""Decision layer: thresholds, the review band, and failure propagation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from app.engine.pipeline import FaceStatus, ImageAnalysis
from app.engine.types import QualityReport, ReasonCode
from app.services.decision import (
    ComparisonOutcome,
    Decision,
    decide,
    decide_comparison,
)

THRESHOLD = 0.363


def ok_analysis(vector: list[float] | None = None) -> ImageAnalysis:
    embedding = np.array(vector or [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return ImageAnalysis(
        status=FaceStatus.OK,
        face_count=1,
        quality=QualityReport(passed=True, score=0.9, issues=()),
        embedding=embedding,
    )


def failed_analysis(status: FaceStatus, issues: tuple[ReasonCode, ...] = ()) -> ImageAnalysis:
    quality = (
        QualityReport(passed=False, score=0.2, issues=issues)
        if status is FaceStatus.LOW_QUALITY
        else None
    )
    count = 3 if status is FaceStatus.MULTIPLE_FACES else (0 if status is FaceStatus.NO_FACE else 1)
    return ImageAnalysis(status=status, face_count=count, quality=quality)


class TestDecideThreshold:
    @pytest.mark.parametrize(
        ("similarity", "expected"),
        [
            (1.0, Decision.MATCH),
            (0.5, Decision.MATCH),
            (0.364, Decision.MATCH),
            (0.363, Decision.MATCH),  # exactly at threshold
            (0.362, Decision.NO_MATCH),
            (0.0, Decision.NO_MATCH),
            (-1.0, Decision.NO_MATCH),
        ],
    )
    def test_threshold_boundary(self, similarity: float, expected: Decision) -> None:
        assert decide(similarity, THRESHOLD) is expected

    def test_similarity_exactly_at_threshold_is_a_match(self) -> None:
        """Fixed and tested because an off-by-epsilon here is invisible.

        A recorded similarity of 0.363 rejected against a recorded threshold of
        0.363 is the kind of thing an auditor finds, not a test suite.
        """
        assert decide(THRESHOLD, THRESHOLD) is Decision.MATCH

    def test_decision_is_monotonic_in_similarity(self) -> None:
        order = {Decision.NO_MATCH: 0, Decision.REVIEW: 1, Decision.MATCH: 2}
        ranks = [
            order[decide(s, THRESHOLD, review_margin=0.1)] for s in np.linspace(-1.0, 1.0, 201)
        ]
        assert ranks == sorted(ranks), "decision must never regress as similarity rises"


class TestReviewBand:
    def test_review_is_disabled_by_default(self) -> None:
        # Ships off: REVIEW exists in the enum from day one so enabling manual
        # review later is config, not a migration.
        assert decide(0.30, THRESHOLD) is Decision.NO_MATCH

    @pytest.mark.parametrize(
        ("similarity", "expected"),
        [
            (0.40, Decision.MATCH),
            (0.363, Decision.MATCH),
            (0.362, Decision.REVIEW),
            (0.263, Decision.REVIEW),  # exactly at the band's lower edge
            (0.262, Decision.NO_MATCH),
        ],
    )
    def test_band_boundaries(self, similarity: float, expected: Decision) -> None:
        assert decide(similarity, THRESHOLD, review_margin=0.1) is expected

    def test_wide_margin_never_swallows_the_match_region(self) -> None:
        assert decide(0.99, THRESHOLD, review_margin=5.0) is Decision.MATCH


class TestDecideComparison:
    def test_identical_embeddings_match(self) -> None:
        outcome = decide_comparison(ok_analysis(), ok_analysis(), threshold=THRESHOLD)
        assert outcome.decision is Decision.MATCH
        assert outcome.matched
        assert outcome.similarity == pytest.approx(1.0)
        assert outcome.confidence is not None and outcome.confidence > 0.99

    def test_orthogonal_embeddings_do_not_match(self) -> None:
        outcome = decide_comparison(
            ok_analysis([1.0, 0.0, 0.0, 0.0]),
            ok_analysis([0.0, 1.0, 0.0, 0.0]),
            threshold=THRESHOLD,
        )
        assert outcome.decision is Decision.NO_MATCH
        assert outcome.similarity == pytest.approx(0.0, abs=1e-6)

    def test_threshold_is_echoed_for_the_audit_record(self) -> None:
        # The stored row must carry the threshold the decision was made under,
        # not whatever config says at read time.
        outcome = decide_comparison(ok_analysis(), ok_analysis(), threshold=0.42)
        assert outcome.threshold == 0.42

    def test_face_counts_are_reported(self) -> None:
        outcome = decide_comparison(ok_analysis(), ok_analysis(), threshold=THRESHOLD)
        assert outcome.reference_face_count == 1
        assert outcome.candidate_face_count == 1

    @pytest.mark.parametrize(
        "status",
        [FaceStatus.NO_FACE, FaceStatus.MULTIPLE_FACES, FaceStatus.LOW_QUALITY],
    )
    def test_reference_failure_propagates_with_its_role(self, status: FaceStatus) -> None:
        outcome = decide_comparison(failed_analysis(status), ok_analysis(), threshold=THRESHOLD)
        assert outcome.decision.value == status.value
        assert outcome.reason_code == f"REFERENCE_{status.value}"
        assert outcome.similarity is None
        assert outcome.confidence is None

    @pytest.mark.parametrize(
        "status",
        [FaceStatus.NO_FACE, FaceStatus.MULTIPLE_FACES, FaceStatus.LOW_QUALITY],
    )
    def test_candidate_failure_propagates_with_its_role(self, status: FaceStatus) -> None:
        outcome = decide_comparison(ok_analysis(), failed_analysis(status), threshold=THRESHOLD)
        assert outcome.decision.value == status.value
        assert outcome.reason_code == f"CANDIDATE_{status.value}"

    def test_reference_is_reported_first_when_both_fail(self) -> None:
        # The reference is the side a user usually cannot re-take, so it is the
        # more useful failure to surface.
        outcome = decide_comparison(
            failed_analysis(FaceStatus.NO_FACE),
            failed_analysis(FaceStatus.LOW_QUALITY),
            threshold=THRESHOLD,
        )
        assert outcome.reason_code == "REFERENCE_NO_FACE"
        assert outcome.reference_status is FaceStatus.NO_FACE
        assert outcome.candidate_status is FaceStatus.LOW_QUALITY

    def test_quality_issues_are_carried_through(self) -> None:
        """Without these a user retries with the same bad photo."""
        issues = (ReasonCode.IMAGE_BLURRY, ReasonCode.TOO_DARK)
        outcome = decide_comparison(
            ok_analysis(),
            failed_analysis(FaceStatus.LOW_QUALITY, issues),
            threshold=THRESHOLD,
        )
        assert outcome.issues == issues

    def test_both_statuses_are_always_recorded(self) -> None:
        outcome = decide_comparison(
            ok_analysis(), failed_analysis(FaceStatus.NO_FACE), threshold=THRESHOLD
        )
        assert outcome.reference_status is FaceStatus.OK
        assert outcome.candidate_status is FaceStatus.NO_FACE

    def test_ok_status_without_an_embedding_is_a_bug_not_a_no_match(self) -> None:
        # Coercing this to NO_MATCH would hide a pipeline defect behind a
        # plausible-looking verdict.
        broken = ImageAnalysis(status=FaceStatus.OK, face_count=1, embedding=None)
        with pytest.raises(ValueError, match="no embedding"):
            decide_comparison(broken, ok_analysis(), threshold=THRESHOLD)


class TestOutcomeShape:
    def test_outcome_carries_no_embedding_field(self) -> None:
        """Structural guard: embeddings must never reach the API or the DB."""
        names = {f.name for f in dataclasses.fields(ComparisonOutcome)}
        assert not {"embedding", "feature", "vector", "template"} & names

    def test_outcome_is_immutable(self) -> None:
        outcome = decide_comparison(ok_analysis(), ok_analysis(), threshold=THRESHOLD)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            outcome.decision = Decision.NO_MATCH  # type: ignore[misc]

    def test_every_failure_status_maps_onto_a_decision(self) -> None:
        """decide_comparison does Decision(status.value) for failing images.

        Adding a FaceStatus without a matching Decision member would raise
        ValueError at runtime, on real traffic. Catch it here instead.

        OK is deliberately excluded: "a face was found" is not a verdict, and
        there is no Decision.OK by design.
        """
        for status in FaceStatus:
            if status is FaceStatus.OK:
                continue
            assert Decision(status.value) is not None

    def test_ok_is_not_a_decision(self) -> None:
        with pytest.raises(ValueError, match="not a valid Decision"):
            Decision(FaceStatus.OK.value)
