"""The phase-4 checkpoint: does this pipeline actually recognise people?

Requires real weights and real portraits, so the whole module is marked
`models` and skipped by the default hermetic run.

    ./scripts/download_models.sh
    python scripts/fetch_test_faces.py
    pytest -m models
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from app.engine.alignment import align_5pt
from app.engine.types import (
    LM_EYE_LEFT,
    LM_EYE_RIGHT,
    LM_MOUTH_LEFT,
    LM_MOUTH_RIGHT,
)

pytestmark = pytest.mark.models

# Two or three portraits each. Multiple identities matter as much as multiple
# photos: genuine pairs alone would be satisfied by a model that calls
# everything a match.
IDENTITIES = {
    "jemison": ["jemison_1.jpg", "jemison_2.jpg", "jemison_3.jpg"],
    "ride": ["ride_1.jpg", "ride_2.jpg"],
    "bluford": ["bluford_1.jpg", "bluford_2.jpg"],
    "collins": ["collins_1.jpg", "collins_2.jpg"],
    "aldrin": ["aldrin_1.jpg", "aldrin_2.jpg"],
}
ALL_PORTRAITS = [(person, name) for person, names in IDENTITIES.items() for name in names]


def _embed(engine, image) -> np.ndarray:
    faces = engine.detect(image)
    assert len(faces) == 1
    return engine.embed(engine.align(image, faces[0]))


class TestDetection:
    @pytest.mark.parametrize(("person", "filename"), ALL_PORTRAITS)
    def test_exactly_one_face_per_portrait(self, engine, faces_dir, person, filename) -> None:
        image = cv2.imread(str(faces_dir / filename))
        faces = engine.detect(image)
        assert len(faces) == 1, f"{filename}: expected 1 face, got {len(faces)}"
        assert 0.0 <= faces[0].score <= 1.0

    def test_landmarks_are_never_mirrored(self, engine, faces_dir) -> None:
        """The guard against the failure that raises nothing and halves accuracy."""
        for _person, filename in ALL_PORTRAITS:
            face = engine.detect(cv2.imread(str(faces_dir / filename)))[0]
            lm = face.landmarks
            assert lm[LM_EYE_LEFT, 0] < lm[LM_EYE_RIGHT, 0], filename
            assert lm[LM_MOUTH_LEFT, 0] < lm[LM_MOUTH_RIGHT, 0], filename

    def test_landmarks_lie_inside_the_bounding_box_region(self, engine, faces_dir) -> None:
        # Catches a coordinate-rescaling bug after the detection downscale:
        # the bbox and landmarks must be rescaled by the same factor.
        for _person, filename in ALL_PORTRAITS:
            face = engine.detect(cv2.imread(str(faces_dir / filename)))[0]
            bbox, lm = face.bbox, face.landmarks
            pad_x, pad_y = bbox.w * 0.5, bbox.h * 0.5
            assert (lm[:, 0] > bbox.x - pad_x).all() and (lm[:, 0] < bbox.right + pad_x).all()
            assert (lm[:, 1] > bbox.y - pad_y).all() and (lm[:, 1] < bbox.bottom + pad_y).all()

    def test_no_face_image_returns_empty_list(self, engine, assets_dir) -> None:
        assert engine.detect(cv2.imread(str(assets_dir / "no_face.jpg"))) == []

    def test_group_photo_returns_many_faces(self, engine, assets_dir) -> None:
        faces = engine.detect(cv2.imread(str(assets_dir / "group.jpg")))
        assert len(faces) > 1

    def test_downscaling_does_not_change_the_result(self, engine, faces_dir) -> None:
        """A large image detected at full size and at 1280px must agree.

        Guards the coordinate mapping back from the downscaled detection.
        """
        from app.engine.adapters.opencv_zoo import OpenCvZooEngine

        image = cv2.imread(str(faces_dir / "jemison_1.jpg"))
        big = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        downscaled = engine.detect(big)[0]
        full = OpenCvZooEngine(engine._model_dir, detect_max_side=10_000).detect(big)[0]

        assert abs(downscaled.bbox.x - full.bbox.x) < big.shape[1] * 0.02
        assert abs(downscaled.bbox.w - full.bbox.w) < big.shape[1] * 0.02


class TestEmbedding:
    def test_embeddings_are_l2_normalised(self, engine, faces_dir) -> None:
        """SFace's own feature() is NOT normalised; the adapter must fix that."""
        for _person, filename in ALL_PORTRAITS:
            vector = _embed(engine, cv2.imread(str(faces_dir / filename)))
            assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5), filename

    def test_embedding_dimension_matches_declared_info(self, engine, faces_dir) -> None:
        vector = _embed(engine, cv2.imread(str(faces_dir / "jemison_1.jpg")))
        assert vector.shape == (engine.info.embedding_dim,)
        assert vector.dtype == np.float32

    def test_our_cosine_agrees_with_opencv_match(self, engine, faces_dir) -> None:
        """Cross-checks normalisation against OpenCV's own FR_COSINE."""
        a = cv2.imread(str(faces_dir / "jemison_1.jpg"))
        b = cv2.imread(str(faces_dir / "ride_1.jpg"))
        fa, fb = engine.detect(a)[0], engine.detect(b)[0]
        raw_a = engine._recognizer.feature(engine.align(a, fa))
        raw_b = engine._recognizer.feature(engine.align(b, fb))

        opencv_cosine = engine._recognizer.match(raw_a, raw_b, cv2.FaceRecognizerSF_FR_COSINE)
        ours = float(_embed(engine, a) @ _embed(engine, b))
        assert ours == pytest.approx(opencv_cosine, abs=1e-5)

    def test_alignment_is_deterministic(self, engine, faces_dir) -> None:
        image = cv2.imread(str(faces_dir / "collins_1.jpg"))
        assert np.array_equal(_embed(engine, image), _embed(engine, image))

    def test_aligncrop_agrees_with_generic_align_5pt(self, engine, faces_dir) -> None:
        """The adapter delegates to alignCrop; the generic path must match.

        If these diverge, a future adapter using align_5pt would land in a
        different embedding space from this one.
        """
        image = cv2.imread(str(faces_dir / "jemison_1.jpg"))
        face = engine.detect(image)[0]
        native = engine.align(image, face)
        generic = align_5pt(image, face.landmarks, size=112)

        assert native.shape == generic.shape == (112, 112, 3)
        a = engine.embed(native)
        b = engine.embed(generic)
        assert float(a @ b) > 0.9, "alignCrop and align_5pt produce different faces"


class TestIdentity:
    """The assertion this whole phase exists to make."""

    def test_genuine_pairs_score_above_the_threshold(self, engine, faces_dir) -> None:
        threshold = engine.info.default_threshold
        failures = []
        for person, filenames in IDENTITIES.items():
            for left, right in itertools.combinations(filenames, 2):
                score = float(
                    _embed(engine, cv2.imread(str(faces_dir / left)))
                    @ _embed(engine, cv2.imread(str(faces_dir / right)))
                )
                if score < threshold:
                    failures.append(f"{person}: {left} vs {right} = {score:.4f}")
        assert not failures, "genuine pairs below threshold:\n  " + "\n  ".join(failures)

    def test_impostor_pairs_score_below_the_threshold(self, engine, faces_dir) -> None:
        threshold = engine.info.default_threshold
        vectors = {
            filename: _embed(engine, cv2.imread(str(faces_dir / filename)))
            for _person, filename in ALL_PORTRAITS
        }
        failures = []
        for (p1, f1), (p2, f2) in itertools.combinations(ALL_PORTRAITS, 2):
            if p1 == p2:
                continue
            score = float(vectors[f1] @ vectors[f2])
            if score >= threshold:
                failures.append(f"{p1}/{f1} vs {p2}/{f2} = {score:.4f}")
        assert not failures, "impostor pairs at or above threshold:\n  " + "\n  ".join(failures)

    def test_genuine_scores_separate_cleanly_from_impostor_scores(self, engine, faces_dir) -> None:
        """The distributions must not overlap.

        Stronger than either threshold test: if the worst genuine pair scores
        below the best impostor pair, NO threshold can separate them and the
        problem is upstream — alignment, landmark order, or quality gating.
        """
        vectors = {
            filename: _embed(engine, cv2.imread(str(faces_dir / filename)))
            for _person, filename in ALL_PORTRAITS
        }
        genuine, impostor = [], []
        for (p1, f1), (p2, f2) in itertools.combinations(ALL_PORTRAITS, 2):
            (genuine if p1 == p2 else impostor).append(float(vectors[f1] @ vectors[f2]))

        assert genuine and impostor
        assert min(genuine) > max(impostor), (
            f"distributions overlap: worst genuine {min(genuine):.4f} "
            f"<= best impostor {max(impostor):.4f}"
        )

    def test_an_image_matches_itself_perfectly(self, engine, faces_dir) -> None:
        vector = _embed(engine, cv2.imread(str(faces_dir / "bluford_1.jpg")))
        assert float(vector @ vector) == pytest.approx(1.0, abs=1e-5)


class TestModelInfo:
    def test_declares_permissive_licence(self, engine) -> None:
        # The default engine must stay production-cleared. If this ever reads
        # "non-commercial", the wrong adapter became the default.
        assert "non-commercial" not in engine.info.license_note.lower()

    def test_threshold_is_plausible_for_cosine_similarity(self, engine) -> None:
        # Guards against the roadmap's 0.72 being copied in. Genuine ArcFace-
        # family pairs cluster around 0.4-0.8, so a threshold up there sits in
        # the tail of the genuine distribution.
        assert 0.2 < engine.info.default_threshold < 0.6
