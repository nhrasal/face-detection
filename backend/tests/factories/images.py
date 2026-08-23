"""Synthetic image builders for hermetic tests.

Nothing here is a face. These exist so the decode, quality and alignment tests
need no face assets, no network and no model weights — they construct exactly
the pathological input each case is about.
"""

from __future__ import annotations

import io
import zlib

import cv2
import numpy as np
from PIL import Image

from app.engine.types import BoundingBox, DetectedFace

EXIF_ORIENTATION_TAG = 274


def rgb_array(width: int = 320, height: int = 240, value: int = 128) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def textured_array(width: int = 320, height: int = 240, seed: int = 0) -> np.ndarray:
    """High-frequency noise: sharp, well-exposed, high contrast."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def encode(array: np.ndarray, fmt: str = "PNG", **kwargs: object) -> bytes:
    """Encode an RGB array to bytes in the given Pillow format."""
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def jpeg_with_orientation(array: np.ndarray, orientation: int) -> bytes:
    """JPEG carrying an EXIF orientation tag."""
    img = Image.fromarray(array)
    exif = img.getexif()
    exif[EXIF_ORIENTATION_TAG] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif, quality=95)
    return buf.getvalue()


def decompression_bomb_png(width: int = 40_000, height: int = 40_000) -> bytes:
    """A tiny PNG that declares enormous dimensions.

    Hand-built rather than rendered: a real 40000x40000 image would need 4.8 GB
    to create, which is the whole point of the attack. Only the IHDR header has
    to be honest for Pillow to read the size lazily.
    """

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            len(payload).to_bytes(4, "big")
            + tag
            + payload
            + zlib.crc32(tag + payload).to_bytes(4, "big")
        )

    ihdr = (
        width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])  # 8-bit RGBA
    )
    idat = zlib.compress(b"\x00" * 16)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def truncated_jpeg() -> bytes:
    """A valid JPEG header followed by nothing useful."""
    full = encode(textured_array(64, 64), fmt="JPEG")
    return full[: len(full) // 3]


def make_face(
    *,
    bbox: tuple[int, int, int, int] = (100, 80, 120, 120),
    score: float = 0.99,
    roll_degrees: float = 0.0,
    yaw_offset: float = 0.0,
) -> DetectedFace:
    """A DetectedFace with plausible landmarks, no image required.

    `yaw_offset` shifts the nose horizontally as a fraction of eye separation:
    0.0 is frontal, positive turns the head.
    """
    x, y, w, h = bbox
    eye_y = y + h * 0.38
    eye_dx = w * 0.30
    cx = x + w / 2.0

    eye_l = np.array([cx - eye_dx, eye_y], dtype=np.float32)
    eye_r = np.array([cx + eye_dx, eye_y], dtype=np.float32)
    nose = np.array([cx + yaw_offset * (2 * eye_dx), y + h * 0.58], dtype=np.float32)
    mouth_l = np.array([cx - eye_dx * 0.65, y + h * 0.78], dtype=np.float32)
    mouth_r = np.array([cx + eye_dx * 0.65, y + h * 0.78], dtype=np.float32)

    landmarks = np.stack([eye_l, eye_r, nose, mouth_l, mouth_r]).astype(np.float32)

    if roll_degrees:
        centre = np.array([cx, y + h / 2.0], dtype=np.float32)
        matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), -roll_degrees, 1.0)
        homogeneous = np.hstack([landmarks, np.ones((5, 1), np.float32)])
        landmarks = (homogeneous @ matrix.T).astype(np.float32)

    return DetectedFace(
        bbox=BoundingBox(x, y, w, h),
        landmarks=landmarks,
        score=score,
    )


def structured_array(width: int = 640, height: int = 480, seed: int = 1) -> np.ndarray:
    """Face-like content: smooth gradient plus large blobs and edges.

    White noise is a bad stand-in for a face when testing sharpness — its energy
    sits exactly at the frequencies resampling destroys, so it behaves nothing
    like real facial structure under rescaling.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = (120 + 60 * np.sin(xx / 40.0) * np.cos(yy / 55.0)).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for _ in range(14):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(15, 60)), int(rng.integers(15, 60)))
        colour = tuple(int(v) for v in rng.integers(20, 235, 3))
        cv2.ellipse(img, centre, axes, float(rng.integers(0, 180)), 0, 360, colour, -1)
    return img
