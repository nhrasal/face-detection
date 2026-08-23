"""Five-point face alignment onto the standard ArcFace template."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.errors import AlignmentError
from app.engine.types import canonicalise_landmarks

# The canonical ArcFace 112x112 destination points, in image-relative order:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def reference_points(size: int = 112) -> np.ndarray:
    """ArcFace destination points scaled to an arbitrary square output."""
    return ARCFACE_DST_112 * (size / 112.0)


def align_5pt(image_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp a face onto the ArcFace template using its 5 landmarks.

    estimateAffinePartial2D, NOT estimateAffine2D. The names differ by one word
    and both compile, but the 6-DoF version shears and non-uniformly stretches
    the face to force landmark agreement, changing the very geometry the
    recogniser was trained on. 4 DoF — rotation, uniform scale, translation —
    is what alignment means here.

    LMEDS rather than RANSAC: with exactly five correspondences, RANSAC's
    minimum-sample logic is degenerate.
    """
    src = canonicalise_landmarks(landmarks)
    dst = reference_points(size)

    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        raise AlignmentError("Could not fit a similarity transform to the landmarks.")

    return cv2.warpAffine(
        image_bgr,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def reprojection_error(landmarks: np.ndarray, size: int = 112) -> float:
    """Mean landmark distance from the template after alignment, in pixels.

    Used by the cross-adapter contract test: an adapter that delegates
    alignment to its own backend must land within a pixel or two of this path.
    """
    src = canonicalise_landmarks(landmarks)
    dst = reference_points(size)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        raise AlignmentError("Could not fit a similarity transform to the landmarks.")
    projected = (np.hstack([src, np.ones((5, 1), np.float32)]) @ matrix.T).astype(np.float32)
    return float(np.linalg.norm(projected - dst, axis=1).mean())
