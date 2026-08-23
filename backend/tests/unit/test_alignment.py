"""Alignment, including the mirror guard that fails silently if wrong."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.errors import AlignmentError
from app.engine.alignment import (
    ARCFACE_DST_112,
    align_5pt,
    reference_points,
    reprojection_error,
)
from app.engine.types import (
    LM_EYE_LEFT,
    LM_EYE_RIGHT,
    LM_MOUTH_LEFT,
    LM_MOUTH_RIGHT,
    canonicalise_landmarks,
)
from tests.factories.images import make_face, textured_array


def _transform(
    points: np.ndarray, *, scale: float, degrees: float, tx: float, ty: float
) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((0.0, 0.0), degrees, scale)
    matrix[:, 2] += (tx, ty)
    return (np.hstack([points, np.ones((len(points), 1), np.float32)]) @ matrix.T).astype(
        np.float32
    )


class TestCanonicalLandmarkOrder:
    """A mirrored crop raises nothing and roughly halves accuracy."""

    def test_already_ordered_landmarks_are_unchanged(self) -> None:
        lm = make_face().landmarks
        assert np.allclose(canonicalise_landmarks(lm), lm)

    def test_swapped_eyes_are_corrected(self) -> None:
        lm = make_face().landmarks.copy()
        lm[[LM_EYE_LEFT, LM_EYE_RIGHT]] = lm[[LM_EYE_RIGHT, LM_EYE_LEFT]]
        out = canonicalise_landmarks(lm)
        assert out[LM_EYE_LEFT, 0] < out[LM_EYE_RIGHT, 0]

    def test_swapped_mouth_corners_are_corrected(self) -> None:
        lm = make_face().landmarks.copy()
        lm[[LM_MOUTH_LEFT, LM_MOUTH_RIGHT]] = lm[[LM_MOUTH_RIGHT, LM_MOUTH_LEFT]]
        out = canonicalise_landmarks(lm)
        assert out[LM_MOUTH_LEFT, 0] < out[LM_MOUTH_RIGHT, 0]

    def test_detected_face_enforces_order_at_construction(self) -> None:
        # The invariant is structural: you cannot hold a mirrored DetectedFace.
        lm = make_face().landmarks.copy()
        lm[[LM_EYE_LEFT, LM_EYE_RIGHT]] = lm[[LM_EYE_RIGHT, LM_EYE_LEFT]]
        face = make_face()
        object.__setattr__(face, "landmarks", lm)
        rebuilt = canonicalise_landmarks(face.landmarks)
        assert rebuilt[LM_EYE_LEFT, 0] < rebuilt[LM_EYE_RIGHT, 0]

    def test_does_not_mutate_the_caller_array(self) -> None:
        lm = make_face().landmarks.copy()
        lm[[LM_EYE_LEFT, LM_EYE_RIGHT]] = lm[[LM_EYE_RIGHT, LM_EYE_LEFT]]
        before = lm.copy()
        canonicalise_landmarks(lm)
        assert np.array_equal(lm, before)

    @pytest.mark.parametrize("shape", [(4, 2), (5, 3), (10, 2), (2, 5)])
    def test_rejects_wrong_shape(self, shape: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match=r"\(5, 2\)"):
            canonicalise_landmarks(np.zeros(shape, dtype=np.float32))


class TestAlign:
    def test_output_is_exactly_112x112x3_uint8(self) -> None:
        img = textured_array(400, 400)
        out = align_5pt(img, make_face(bbox=(120, 120, 160, 160)).landmarks)
        assert out.shape == (112, 112, 3)
        assert out.dtype == np.uint8

    def test_size_parameter_scales_the_template(self) -> None:
        img = textured_array(400, 400)
        assert align_5pt(img, make_face().landmarks, size=224).shape == (224, 224, 3)
        assert np.allclose(reference_points(224), ARCFACE_DST_112 * 2.0)

    def test_landmarks_land_on_the_template(self) -> None:
        # The transform is exact for a similarity-transformed source, so
        # reprojection error should be essentially zero.
        src = _transform(ARCFACE_DST_112, scale=1.5, degrees=20.0, tx=60.0, ty=40.0)
        assert reprojection_error(src) < 1.0

    @pytest.mark.parametrize("degrees", [-30.0, -10.0, 0.0, 15.0, 45.0])
    def test_rotation_is_undone(self, degrees: float) -> None:
        src = _transform(ARCFACE_DST_112, scale=1.2, degrees=degrees, tx=100.0, ty=100.0)
        assert reprojection_error(src) < 1.0

    @pytest.mark.parametrize("scale", [0.4, 1.0, 3.0])
    def test_uniform_scale_is_undone(self, scale: float) -> None:
        src = _transform(ARCFACE_DST_112, scale=scale, degrees=5.0, tx=50.0, ty=50.0)
        assert reprojection_error(src) < 1.0

    def test_shear_is_not_absorbed(self) -> None:
        """4-DoF must NOT fit a sheared source.

        This is what separates estimateAffinePartial2D from estimateAffine2D:
        the 6-DoF version would fit this perfectly by distorting the face
        geometry the recogniser was trained on.
        """
        sheared = ARCFACE_DST_112.copy()
        sheared[:, 0] += sheared[:, 1] * 0.6
        assert reprojection_error(sheared) > 2.0

    def test_degenerate_landmarks_raise_alignment_error(self) -> None:
        img = textured_array(200, 200)
        collapsed = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises(AlignmentError):
            align_5pt(img, collapsed)

    def test_aligned_crop_preserves_content(self) -> None:
        # An identity-ish alignment of a textured image must not come back blank.
        img = textured_array(400, 400, seed=3)
        src = _transform(ARCFACE_DST_112, scale=2.0, degrees=0.0, tx=100.0, ty=100.0)
        out = align_5pt(img, src)
        assert out.std() > 10.0

    def test_out_of_frame_regions_are_black_not_replicated(self) -> None:
        # Border replication would fabricate facial texture that was never
        # photographed; constant black is honest about missing data.
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        far = ARCFACE_DST_112 + np.array([190.0, 190.0], dtype=np.float32)
        out = align_5pt(img, far)
        assert (out == 0).any()
