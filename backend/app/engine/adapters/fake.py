"""Deterministic test engine. Settings forbids it in production."""

from __future__ import annotations

import numpy as np

from app.engine.base import ALIGNED_SIZE, FaceEngine
from app.engine.types import BoundingBox, DetectedFace, ModelInfo

FAKE_MODEL_INFO = ModelInfo("fake", "1", "fake", "1", 4, 0.8, "test only")


class FakeEngine(FaceEngine):
    @property
    def info(self) -> ModelInfo:
        return FAKE_MODEL_INFO

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        height, width = image_bgr.shape[:2]
        side = max(32, round(min(width, height) * 0.6))
        x, y = (width - side) // 2, (height - side) // 2
        landmarks = np.array(
            [
                [x + side * 0.32, y + side * 0.38],
                [x + side * 0.68, y + side * 0.38],
                [x + side * 0.50, y + side * 0.55],
                [x + side * 0.38, y + side * 0.72],
                [x + side * 0.62, y + side * 0.72],
            ],
            dtype=np.float32,
        )
        return [DetectedFace(BoundingBox(x, y, side, side), landmarks, 0.99)]

    def align(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        import cv2

        return cv2.resize(image_bgr, (ALIGNED_SIZE, ALIGNED_SIZE))

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        means = aligned_bgr.mean(axis=(0, 1), dtype=np.float64) / 255.0
        vector = np.array([*means, float(aligned_bgr.std()) / 255.0], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else np.array([1, 0, 0, 0], dtype=np.float32)
