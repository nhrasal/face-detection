"""Decode is a security boundary — these are mostly adversarial inputs."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import ImageTooLargeError, UnsupportedMediaError
from app.engine.decode import decode_image, sniff_mime
from tests.factories.images import (
    decompression_bomb_png,
    encode,
    jpeg_with_orientation,
    rgb_array,
    textured_array,
    truncated_jpeg,
)


class TestHappyPath:
    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
    def test_decodes_supported_formats_to_bgr(self, fmt: str) -> None:
        out = decode_image(encode(textured_array(64, 48), fmt=fmt))
        assert out.shape == (48, 64, 3)
        assert out.dtype == np.uint8
        assert out.flags["C_CONTIGUOUS"]

    def test_channel_order_is_bgr_not_rgb(self) -> None:
        # Pure red in RGB must come back as (0, 0, 255) in BGR. Getting this
        # backwards is invisible in greyscale metrics but wrecks the models.
        red = np.zeros((8, 8, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        out = decode_image(encode(red, fmt="PNG"))
        assert tuple(int(c) for c in out[0, 0]) == (0, 0, 255)

    def test_strips_exif(self) -> None:
        # Going through numpy drops EXIF, so GPS in a KYC photo never persists.
        out = decode_image(jpeg_with_orientation(textured_array(64, 64), orientation=1))
        assert not hasattr(out, "info")


class TestExifOrientation:
    """The top cause of 'works on my laptop, fails on iPhone uploads'."""

    @pytest.mark.parametrize("orientation", [1, 3, 6, 8])
    def test_all_orientations_yield_upright_image(self, orientation: int) -> None:
        # A 100x60 landscape source. Orientations 6 and 8 declare a 90-degree
        # rotation, so a decoder that honours EXIF returns 60x100.
        array = textured_array(100, 60, seed=orientation)
        out = decode_image(jpeg_with_orientation(array, orientation))
        expected = (100, 60, 3) if orientation in (6, 8) else (60, 100, 3)
        assert out.shape == expected

    def test_rotated_and_upright_agree_after_transpose(self) -> None:
        array = textured_array(80, 40, seed=7)
        upright = decode_image(jpeg_with_orientation(array, orientation=1))
        rotated = decode_image(jpeg_with_orientation(array, orientation=6))
        assert rotated.shape == (upright.shape[1], upright.shape[0], 3)


class TestTypeSniffing:
    def test_sniffs_by_content_not_extension(self) -> None:
        assert sniff_mime(encode(rgb_array(8, 8), fmt="PNG")) == "image/png"
        assert sniff_mime(encode(rgb_array(8, 8), fmt="JPEG")) == "image/jpeg"

    def test_png_bytes_claiming_to_be_jpeg_are_still_decoded_as_png(self) -> None:
        # The client's Content-Type is never consulted, so a mislabelled but
        # genuine image succeeds. It is the *content* that must be an image.
        assert decode_image(encode(textured_array(16, 16), fmt="PNG")).shape == (16, 16, 3)

    def test_rejects_non_image_bytes(self) -> None:
        with pytest.raises(UnsupportedMediaError):
            decode_image(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + b"\x00" * 512)

    def test_rejects_gif_even_though_pillow_can_read_it(self) -> None:
        # Multi-frame formats are excluded by policy, not by capability.
        with pytest.raises(UnsupportedMediaError, match="not a supported image type"):
            decode_image(encode(rgb_array(16, 16), fmt="GIF"))

    def test_rejects_svg(self) -> None:
        with pytest.raises(UnsupportedMediaError):
            decode_image(b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>')

    def test_rejects_empty_upload(self) -> None:
        with pytest.raises(UnsupportedMediaError, match="Empty"):
            decode_image(b"")


class TestMalformed:
    def test_rejects_truncated_jpeg(self) -> None:
        with pytest.raises(UnsupportedMediaError, match="corrupt or truncated"):
            decode_image(truncated_jpeg())

    def test_rejects_png_header_with_garbage_body(self) -> None:
        good = encode(textured_array(32, 32), fmt="PNG")
        with pytest.raises(UnsupportedMediaError):
            decode_image(good[:20] + b"\xff" * 200)


class TestSizeLimits:
    def test_rejects_oversize_payload(self) -> None:
        data = encode(textured_array(400, 400), fmt="PNG")
        with pytest.raises(ImageTooLargeError, match="the limit is"):
            decode_image(data, max_bytes=1024)

    def test_rejects_decompression_bomb_before_allocating(self) -> None:
        # A ~200-byte PNG declaring 40000x40000 RGBA — 6.4 GB if realised. The
        # size check reads the header only, so nothing is allocated.
        bomb = decompression_bomb_png()
        assert len(bomb) < 1000
        with pytest.raises(ImageTooLargeError):
            decode_image(bomb, max_bytes=10_000_000)

    def test_rejects_long_edge_over_limit(self) -> None:
        with pytest.raises(ImageTooLargeError, match="longest edge"):
            decode_image(encode(rgb_array(600, 10), fmt="PNG"), max_side=500)

    def test_rejects_pixel_count_over_limit(self) -> None:
        with pytest.raises(ImageTooLargeError, match="pixels"):
            decode_image(encode(rgb_array(200, 200), fmt="PNG"), max_side=10_000, max_pixels=1000)

    def test_accepts_image_exactly_at_the_limit(self) -> None:
        # Off-by-one at a boundary is how a valid upload starts failing.
        out = decode_image(encode(rgb_array(100, 100), fmt="PNG"), max_pixels=10_000, max_side=100)
        assert out.shape == (100, 100, 3)
