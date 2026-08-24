"""Local V1 profile-image storage.

Original bytes are never retained: every accepted image is decoded, stripped of
metadata, and re-encoded under a generated name. Paths stored in PostgreSQL are
relative keys, not host filesystem paths.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from app.core.config import Settings
from app.core.errors import ResourceNotFoundError
from app.engine.decode import decode_image


class ProfileImageStore:
    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = root.resolve()
        self.settings = settings

    def save(self, user_id: uuid.UUID, data: bytes) -> str:
        image = decode_image(
            data,
            allowed_mime=self.settings.ALLOWED_MIME,
            max_bytes=self.settings.MAX_UPLOAD_BYTES,
            max_pixels=self.settings.MAX_IMAGE_PIXELS,
            max_side=self.settings.MAX_IMAGE_SIDE,
        )
        encoded, output = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not encoded:
            raise RuntimeError("profile image re-encoding failed")
        relative = Path("profiles") / str(user_id) / f"{uuid.uuid4().hex}.jpg"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(output.tobytes())
        return relative.as_posix()

    def read(self, key: str) -> bytes:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ResourceNotFoundError("The user's reference image is unavailable.")
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = (self.root / key).resolve()
        if path.is_relative_to(self.root) and path.is_file():
            path.unlink()
