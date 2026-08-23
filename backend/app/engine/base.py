"""Abstract contracts every model backend implements.

The point of this layer: the pipeline, the quality checks and the business
decision must not know whether they are talking to YuNet+SFace or SCRFD+ArcFace.
Adding a third backend must not touch anything outside `adapters/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.engine.types import DetectedFace, ModelInfo

ALIGNED_SIZE = 112


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Find faces. Returns [] when there are none — never None."""


class FaceRecognizer(ABC):
    @abstractmethod
    def align(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return a (112, 112, 3) uint8 BGR crop, eyes on the ArcFace template."""

    @abstractmethod
    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Return a (D,) float32 embedding that is L2-NORMALISED.

        The contract says normalised so that cosine similarity downstream is a
        plain dot product and thresholds are comparable. Enforced by a shared
        contract test — SFace's own `feature()` returns an unnormalised vector
        (measured L2 norm ~4.14), so an adapter that forwards it raw is wrong.
        """


class FaceEngine(FaceDetector, FaceRecognizer, ABC):
    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Names, versions, embedding dimension, default threshold, licence."""

    def warmup(self) -> None:
        """Run one dummy inference through both graphs.

        The first inference is far slower than steady state (graph optimisation,
        arena allocation, lazy blob buffers). Without warmup the first real user
        absorbs that, and readiness reports healthy before the service is.
        """
        dummy = np.zeros((ALIGNED_SIZE * 4, ALIGNED_SIZE * 4, 3), dtype=np.uint8)
        self.detect(dummy)
        self.embed(np.zeros((ALIGNED_SIZE, ALIGNED_SIZE, 3), dtype=np.uint8))

    def close(self) -> None:
        """Release native resources. Must be idempotent."""
