"""Cosine similarity and the confidence mapping."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.comparison import cosine_similarity, to_confidence

THRESHOLD = 0.363


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = unit(1.0, 2.0, 3.0, 4.0)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_score_minus_one(self) -> None:
        v = unit(1.0, 2.0, 3.0, 4.0)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a, b = unit(1.0, 0.0, 0.0, 0.0), unit(0.0, 1.0, 0.0, 0.0)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_is_symmetric(self) -> None:
        a, b = unit(0.2, -0.5, 0.9, 0.1), unit(-0.3, 0.7, 0.2, 0.6)
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_result_is_clamped_to_the_valid_range(self) -> None:
        # Float drift on a self-comparison would otherwise report values a
        # hair above 1.0 and trip downstream range checks.
        rng = np.random.default_rng(0)
        for _ in range(200):
            v = unit(*rng.normal(size=128))
            assert -1.0 <= cosine_similarity(v, v.copy()) <= 1.0

    def test_normalises_unnormalised_input(self) -> None:
        """Defence against an adapter that forgets to normalise.

        SFace's raw feature() has norm ~4.14; dotting two of those gives a
        similarity in the tens that sails past every threshold.
        """
        a = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)  # norm 5
        b = np.array([6.0, 8.0, 0.0, 0.0], dtype=np.float32)  # norm 10, same direction
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_returns_zero_rather_than_nan(self) -> None:
        zero = np.zeros(4, dtype=np.float32)
        assert cosine_similarity(zero, unit(1.0, 0.0, 0.0, 0.0)) == 0.0
        assert cosine_similarity(zero, zero) == 0.0

    def test_mismatched_dimensions_raise(self) -> None:
        # A 128-d SFace vector against a 512-d ArcFace vector is meaningless,
        # not merely low-scoring — thresholds do not transfer between models.
        with pytest.raises(ValueError, match="dimensions differ"):
            cosine_similarity(np.zeros(128, np.float32), np.zeros(512, np.float32))


class TestConfidence:
    def test_is_exactly_one_half_at_the_threshold(self) -> None:
        assert to_confidence(THRESHOLD, threshold=THRESHOLD) == pytest.approx(0.5)

    def test_is_monotonically_increasing(self) -> None:
        values = [to_confidence(s, threshold=THRESHOLD) for s in np.linspace(-1.0, 1.0, 200)]
        assert values == sorted(values)

    def test_stays_within_zero_and_one(self) -> None:
        for similarity in (-1.0, -0.5, 0.0, 0.363, 0.5, 1.0):
            assert 0.0 <= to_confidence(similarity, threshold=THRESHOLD) <= 1.0

    def test_extreme_inputs_do_not_overflow(self) -> None:
        # math.exp(710) raises OverflowError; the guard must hold.
        assert to_confidence(-1e6, threshold=THRESHOLD) == 0.0
        assert to_confidence(1e6, threshold=THRESHOLD) == 1.0

    def test_above_threshold_reads_above_half(self) -> None:
        assert to_confidence(0.4679, threshold=THRESHOLD) > 0.5

    def test_below_threshold_reads_below_half(self) -> None:
        assert to_confidence(0.3285, threshold=THRESHOLD) < 0.5

    def test_solves_the_misleading_percentage_problem(self) -> None:
        """The reason this function exists.

        A raw cosine of 0.4679 is the WORST genuine pair measured, yet shown as
        "47% match" it reads like a failure. As confidence it reads as a pass,
        which is what it is.
        """
        raw = 0.4679
        assert raw < 0.5  # would render as "47%" — misleading
        assert to_confidence(raw, threshold=THRESHOLD) > 0.7

    def test_slope_controls_sharpness_without_moving_the_centre(self) -> None:
        for slope in (2.0, 12.0, 40.0):
            assert to_confidence(THRESHOLD, threshold=THRESHOLD, slope=slope) == pytest.approx(0.5)
        gentle = to_confidence(0.45, threshold=THRESHOLD, slope=2.0)
        steep = to_confidence(0.45, threshold=THRESHOLD, slope=40.0)
        assert steep > gentle

    def test_follows_the_threshold(self) -> None:
        # Raising the threshold must lower the confidence of a fixed score.
        assert to_confidence(0.5, threshold=0.6) < to_confidence(0.5, threshold=0.3)
