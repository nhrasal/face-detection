"""Pipeline orchestration, driven by a scripted engine.

Hermetic: a stub engine returns whatever the case needs, so branch coverage of
analyse() does not depend on finding real images that happen to be blurry or
crowded. The same paths are exercised against real models in the model tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engine.base import FaceEngine
from app.engine.pipeline import (
    FaceSelectionPolicy,
    FaceStatus,
    ImageAnalysis,
    analyse,
)
from app.engine.quality import QualityThresholds
from app.engine.types import DetectedFace, ModelInfo
from tests.factories.images import make_face, textured_array

STUB_INFO = ModelInfo(
    detector_name="stub",
    detector_version="0",
    recognizer_name="stub",
    recognizer_version="0",
    embedding_dim=4,
    default_threshold=0.5,
    license_note="test only",
)


class ScriptedEngine(FaceEngine):
    """Returns a fixed face list; records how often each stage ran."""

    def __init__(self, faces: list[DetectedFace]) -> None:
        self._faces = faces
        self.align_calls = 0
        self.embed_calls = 0

    @property
    def info(self) -> ModelInfo:
        return STUB_INFO

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        return list(self._faces)

    def align(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        self.align_calls += 1
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        self.embed_calls += 1
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


GOOD_BBOX = (100, 80, 160, 160)


def good_image() -> np.ndarray:
    return textured_array(640, 480, seed=1)


class TestNoFace:
    def test_empty_detection_yields_no_face(self) -> None:
        result = analyse(ScriptedEngine([]), good_image())
        assert result.status is FaceStatus.NO_FACE
        assert result.face_count == 0
        assert result.embedding is None
        assert not result.ok

    def test_no_inference_runs_when_there_is_no_face(self) -> None:
        engine = ScriptedEngine([])
        analyse(engine, good_image())
        assert engine.align_calls == 0
        assert engine.embed_calls == 0


class TestMultipleFaces:
    def test_rejects_by_default(self) -> None:
        """Auto-picking a face in a verification flow verifies the wrong person."""
        engine = ScriptedEngine([make_face(bbox=GOOD_BBOX), make_face(bbox=(400, 100, 90, 90))])
        result = analyse(engine, good_image())
        assert result.status is FaceStatus.MULTIPLE_FACES
        assert result.face_count == 2
        assert result.embedding is None
        assert engine.embed_calls == 0

    def test_largest_policy_selects_the_biggest_face(self) -> None:
        small = make_face(bbox=(400, 100, 90, 90))
        large = make_face(bbox=GOOD_BBOX)
        engine = ScriptedEngine([small, large])
        result = analyse(engine, good_image(), multi_face=FaceSelectionPolicy.LARGEST)
        assert result.status is FaceStatus.OK
        assert result.face is not None
        assert result.face.bbox.area == large.bbox.area

    def test_largest_policy_still_reports_the_true_count(self) -> None:
        # The audit record must show a face was selected from several.
        engine = ScriptedEngine([make_face(bbox=GOOD_BBOX), make_face(bbox=(400, 100, 90, 90))])
        result = analyse(engine, good_image(), multi_face=FaceSelectionPolicy.LARGEST)
        assert result.face_count == 2


class TestQualityGate:
    def test_low_quality_stops_before_embedding(self) -> None:
        # Embedding a face already known to be unusable is wasted inference.
        engine = ScriptedEngine([make_face(bbox=(10, 10, 30, 30), score=0.3)])
        result = analyse(engine, good_image())
        assert result.status is FaceStatus.LOW_QUALITY
        assert result.embedding is None
        assert engine.embed_calls == 0

    def test_low_quality_still_reports_the_face_and_its_issues(self) -> None:
        engine = ScriptedEngine([make_face(bbox=(10, 10, 30, 30), score=0.3)])
        result = analyse(engine, good_image())
        assert result.face is not None
        assert result.quality is not None
        assert result.quality.issues, "a LOW_QUALITY result must say what was wrong"

    def test_thresholds_are_injectable(self) -> None:
        engine = ScriptedEngine([make_face(bbox=(10, 10, 30, 30), score=0.95)])
        assert analyse(engine, good_image()).status is FaceStatus.LOW_QUALITY
        lenient = QualityThresholds(
            min_face_px=5.0, min_interocular_px=2.0, min_face_area_ratio=0.0
        )
        assert analyse(engine, good_image(), thresholds=lenient).status is FaceStatus.OK


class TestSuccess:
    def test_single_good_face_produces_an_embedding(self) -> None:
        engine = ScriptedEngine([make_face(bbox=GOOD_BBOX)])
        result = analyse(engine, good_image())
        assert result.status is FaceStatus.OK
        assert result.ok
        assert result.face_count == 1
        assert result.embedding is not None
        assert result.embedding.shape == (4,)
        assert result.quality is not None and result.quality.passed

    def test_alignment_and_embedding_each_run_once(self) -> None:
        engine = ScriptedEngine([make_face(bbox=GOOD_BBOX)])
        analyse(engine, good_image())
        assert engine.align_calls == 1
        assert engine.embed_calls == 1

    def test_alignment_receives_the_original_image(self) -> None:
        """Aligning from a downscaled copy measurably degrades small faces."""
        captured: list[tuple[int, int]] = []

        class Recording(ScriptedEngine):
            def align(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
                captured.append(image_bgr.shape[:2])
                return super().align(image_bgr, face)

        # textured_array, not structured_array: the latter's contrast (~11)
        # is below the quality gate, so the pipeline would stop before align.
        image = textured_array(800, 600, seed=2)
        analyse(Recording([make_face(bbox=GOOD_BBOX)]), image)
        assert captured == [(600, 800)]


class TestAnalysisShape:
    def test_is_immutable(self) -> None:
        result = analyse(ScriptedEngine([]), good_image())
        with pytest.raises((AttributeError, Exception)):
            result.status = FaceStatus.OK  # type: ignore[misc]

    def test_ok_property_agrees_with_status(self) -> None:
        for status in FaceStatus:
            assert ImageAnalysis(status=status, face_count=1).ok is (status is FaceStatus.OK)

    def test_failed_analyses_never_carry_an_embedding(self) -> None:
        cases = [
            ScriptedEngine([]),
            ScriptedEngine([make_face(bbox=GOOD_BBOX), make_face(bbox=(400, 100, 90, 90))]),
            ScriptedEngine([make_face(bbox=(10, 10, 30, 30), score=0.3)]),
        ]
        for engine in cases:
            result = analyse(engine, good_image())
            assert not result.ok
            assert result.embedding is None
