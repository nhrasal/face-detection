"""YuNet detector + SFace recogniser, via OpenCV's own wrappers.

This is the default, shippable engine. Both models are permissively licensed.

It runs on cv2.dnn rather than ONNX Runtime, which deviates from the roadmap's
stated stack. That is deliberate: `FaceDetectorYN` and `FaceRecognizerSF` supply
correct pre/post-processing and `alignCrop` for free, and those are precisely
the two places hand-rolled pipelines go subtly wrong.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ModelsMissingError
from app.engine.base import ALIGNED_SIZE, FaceEngine
from app.engine.types import BoundingBox, DetectedFace, ModelInfo

DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"

# YuNet emits one row of 15 floats per face:
#   [x, y, w, h, then five (x, y) landmark pairs, then score]
YUNET_ROW_WIDTH = 15
_LANDMARK_SLICE = slice(4, 14)
_SCORE_INDEX = 14

MODEL_INFO = ModelInfo(
    detector_name="yunet",
    detector_version="2023mar",
    recognizer_name="sface",
    recognizer_version="2021dec",
    embedding_dim=128,
    # OpenCV's documented cosine operating point for SFace. A STARTING value
    # for calibration, not a shipped decision — and emphatically not the
    # roadmap's 0.72, which sits in the far right tail of the genuine
    # distribution and would reject most true matches.
    default_threshold=0.363,
    license_note="Apache-2.0 (OpenCV Zoo) — cleared for production use",
)


class OpenCvZooEngine(FaceEngine):
    def __init__(self, model_dir: Path, *, detect_max_side: int = 1280) -> None:
        detector_path = model_dir / DETECTOR_FILE
        recognizer_path = model_dir / RECOGNIZER_FILE
        for path in (detector_path, recognizer_path):
            if not path.is_file():
                raise ModelsMissingError(
                    f"Model weights not found: {path}",
                    detail="run ./scripts/download_models.sh",
                )

        self._model_dir = model_dir
        self._detect_max_side = detect_max_side
        # Guards the detector only. FaceDetectorYN carries mutable input-size
        # state across setInputSize/detect, so two threads sharing one instance
        # race and produce corrupted coordinates. The engine pool hands each
        # worker its own instance; this lock is defence in depth for anyone who
        # shares one anyway.
        self._lock = threading.Lock()

        self._detector = cv2.FaceDetectorYN.create(
            model=str(detector_path),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,  # quality gating applies the real bar later
            nms_threshold=0.3,
            top_k=5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(model=str(recognizer_path), config="")

    @property
    def info(self) -> ModelInfo:
        return MODEL_INFO

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        height, width = image_bgr.shape[:2]
        if height == 0 or width == 0:
            return []

        # Detect on a downscaled copy for speed, then map coordinates back and
        # align from the FULL-resolution source — aligning from the downscaled
        # image measurably degrades small faces.
        scale = min(1.0, self._detect_max_side / float(max(height, width)))
        if scale < 1.0:
            small = cv2.resize(
                image_bgr,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image_bgr

        with self._lock:
            self._detector.setInputSize((small.shape[1], small.shape[0]))
            _, raw = self._detector.detect(small)

        # detect() returns retval=1 meaning "ran successfully", NOT "found
        # something". With no faces, `raw` is None. Branching on retval instead
        # would index into None on every faceless image.
        if raw is None or len(raw) == 0:
            return []

        faces: list[DetectedFace] = []
        for row in np.asarray(raw, dtype=np.float32):
            native = row.copy()
            if scale < 1.0:
                # Rescale bbox and landmarks, but never the score.
                native[:_SCORE_INDEX] /= scale

            x, y, w, h = (float(v) for v in native[:4])
            landmarks = native[_LANDMARK_SLICE].reshape(5, 2)
            faces.append(
                DetectedFace(
                    bbox=BoundingBox(round(x), round(y), round(w), round(h)),
                    landmarks=landmarks,
                    score=float(native[_SCORE_INDEX]),
                    native=native,
                )
            )
        return faces

    def align(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Delegate to alignCrop, which uses SFace's own reference points.

        Re-implementing the warp here would be redundant and slightly worse.
        A contract test checks this agrees with our generic align_5pt path.
        """
        if face.native is None:
            from app.engine.alignment import align_5pt

            return align_5pt(image_bgr, face.landmarks, size=ALIGNED_SIZE)
        aligned: np.ndarray = self._recognizer.alignCrop(image_bgr, face.native)
        return aligned

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        feature = self._recognizer.feature(aligned_bgr)
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        # feature() is NOT normalised — measured L2 norm ~4.14. match(FR_COSINE)
        # normalises internally, which is why raw dot products of feature()
        # output land in the tens. The base contract requires unit norm.
        norm = float(np.linalg.norm(vector))
        if norm < 1e-10:
            return vector
        return (vector / norm).astype(np.float32)
