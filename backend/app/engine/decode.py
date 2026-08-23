"""Turn untrusted bytes into a BGR array, safely.

This is a security boundary. Everything upstream is attacker-controlled: the
declared Content-Type, the filename, the pixel dimensions, the EXIF block.
Nothing here trusts any of it.
"""

from __future__ import annotations

import io

import cv2
import filetype
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.errors import ImageTooLargeError, UnsupportedMediaError

# Multi-frame formats (GIF, TIFF) are excluded: TIFF is a decompression-bomb
# vector and neither is a photograph. SVG is excluded because it is XML, which
# means XXE and script, and is likewise not a photograph.
DEFAULT_ALLOWED_MIME = frozenset({"image/jpeg", "image/png", "image/webp"})

DEFAULT_MAX_PIXELS = 40_000_000
DEFAULT_MAX_SIDE = 8_000
DEFAULT_MAX_BYTES = 5 * 1024 * 1024

# Pillow raises above 2x this and warns above it. We do our own explicit check
# too, because we want a typed error rather than a warning.
Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_PIXELS


def sniff_mime(data: bytes) -> str | None:
    """Identify by magic bytes. The client's Content-Type is never consulted."""
    kind = filetype.guess(data[:261])
    return None if kind is None else str(kind.mime)


def decode_image(
    data: bytes,
    *,
    allowed_mime: frozenset[str] | set[str] = DEFAULT_ALLOWED_MIME,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
) -> np.ndarray:
    """Decode image bytes to a contiguous BGR uint8 array.

    Raises UnsupportedMediaError (415) or ImageTooLargeError (413).
    """
    if not data:
        raise UnsupportedMediaError("Empty upload.")

    if len(data) > max_bytes:
        raise ImageTooLargeError(
            f"Image is {len(data)} bytes; the limit is {max_bytes}.",
            detail="MAX_UPLOAD_BYTES",
        )

    mime = sniff_mime(data)
    if mime is None:
        raise UnsupportedMediaError("Could not identify the file type from its content.")
    if mime not in allowed_mime:
        raise UnsupportedMediaError(
            f"{mime} is not a supported image type.",
            detail=f"allowed: {', '.join(sorted(allowed_mime))}",
        )

    # verify() detects truncated and malformed streams, but it consumes the
    # handle — the file must be reopened afterwards to actually read pixels.
    # Pillow has its own bomb guard and raises before our explicit checks get a
    # chance. Translate it to our typed error rather than letting a PIL
    # exception escape as a 500.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except Image.DecompressionBombError as exc:
        raise ImageTooLargeError(
            "Image declares more pixels than the decoder will allocate.",
            detail="MAX_IMAGE_PIXELS",
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise UnsupportedMediaError("Image data is corrupt or truncated.") from exc

    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as exc:  # pragma: no cover - verify() caught it
        raise ImageTooLargeError(
            "Image declares more pixels than the decoder will allocate.",
            detail="MAX_IMAGE_PIXELS",
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:  # pragma: no cover - verify() caught it
        raise UnsupportedMediaError("Image data could not be opened.") from exc

    with img:
        # Image.open is lazy: size comes from the header, so both checks happen
        # BEFORE any pixels are allocated. This is what stops a 200 KB PNG that
        # expands to 40 GB of RGBA.
        width, height = img.size
        if width <= 0 or height <= 0:
            raise UnsupportedMediaError("Image has zero width or height.")
        if max(width, height) > max_side:
            raise ImageTooLargeError(
                f"Image is {width}x{height}; the longest edge may not exceed {max_side}px.",
                detail="MAX_IMAGE_SIDE",
            )
        if width * height > max_pixels:
            raise ImageTooLargeError(
                f"Image is {width * height} pixels; the limit is {max_pixels}.",
                detail="MAX_IMAGE_PIXELS",
            )

        try:
            # A portrait phone photo tagged "rotate 90" fails detection outright
            # if this is skipped — the top cause of "works on my laptop, fails
            # on iPhone uploads". GEMS' own uploader disables orientation
            # handling client-side, so the server must do it.
            oriented = ImageOps.exif_transpose(img) or img
            rgb = oriented.convert("RGB")
        except (OSError, ValueError) as exc:
            raise UnsupportedMediaError("Image data is corrupt or truncated.") from exc

    # Going through numpy drops every EXIF field, so GPS coordinates in a KYC
    # photo are stripped for free rather than as a separate step.
    arr = np.asarray(rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(bgr)
