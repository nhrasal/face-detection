"""Adversarial checks for the public biometric boundary."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.core.errors import ResourceNotFoundError
from app.core.http_security import RequestSizeLimitMiddleware
from app.db.base import Base
from app.db.session import create_database_engine
from app.main import create_app
from app.services.profile_storage import ProfileImageStore
from tests.factories.images import (
    decompression_bomb_png,
    encode,
    jpeg_with_orientation,
    rgb_array,
    textured_array,
)

FORBIDDEN_BIOMETRIC_KEYS = {
    "embedding",
    "embeddings",
    "feature",
    "features",
    "vector",
    "template",
    "candidate_image",
    "reference_image",
    "image_bytes",
}


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_BIOMETRIC_KEYS & set(value)) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


@pytest.fixture
def security_client(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'security.sqlite3'}"
    engine = create_database_engine(database_url)
    Base.metadata.create_all(engine)
    engine.dispose()
    upload_dir = tmp_path / "uploads"
    settings = Settings(
        ENV="local",
        FACE_ENGINE="fake",
        INFERENCE_WORKERS=1,
        DATABASE_URL=database_url,
        UPLOAD_DIR=upload_dir,
        RATE_LIMIT_COMPARE="1000/minute",
    )
    with TestClient(create_app(settings), client=(f"security-{tmp_path.name}", 50000)) as client:
        yield client, upload_dir


def _image(seed: int = 1, *, content_type: str = "image/png") -> tuple[str, bytes, str]:
    return ("../../untrusted-name.png", encode(textured_array(320, 240, seed)), content_type)


def _create_user(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/api/v1/users",
        data={"external_id": "security-user", "name": "Security Test"},
        files={"profile_image": _image()},
    )
    assert response.status_code == 201
    return response.json()


def test_all_responses_disable_sniffing_framing_referrers_and_cache(
    security_client: tuple[TestClient, Path],
) -> None:
    client, _ = security_client
    response = client.get("/healthz")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_whole_request_is_rejected_before_multipart_parsing(tmp_path: Path) -> None:
    settings = Settings(
        ENV="local",
        FACE_ENGINE="fake",
        DATABASE_URL=f"sqlite+pysqlite:///{tmp_path / 'cap.sqlite3'}",
        UPLOAD_DIR=tmp_path / "uploads",
        MAX_REQUEST_BYTES=128,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/face/detect",
            files={"image": ("large.jpg", b"x" * 129, "image/jpeg")},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


@pytest.mark.asyncio
async def test_streamed_body_without_content_length_is_also_capped() -> None:
    chunks = iter(
        [
            {"type": "http.request", "body": b"a" * 80, "more_body": True},
            {"type": "http.request", "body": b"b" * 80, "more_body": False},
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return next(chunks)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def consuming_app(scope: dict[str, Any], receive, send) -> None:
        while (await receive()).get("more_body"):
            pass

    middleware = RequestSizeLimitMiddleware(consuming_app, max_bytes=128)
    await middleware({"type": "http", "headers": []}, receive, send)
    assert sent[0]["status"] == 413


def test_declared_mime_cannot_smuggle_disallowed_bytes(
    security_client: tuple[TestClient, Path],
) -> None:
    client, _ = security_client
    gif = encode(rgb_array(32, 32), fmt="GIF")
    response = client.post(
        "/api/v1/face/detect",
        files={"image": ("portrait.jpg", gif, "image/jpeg")},
    )
    assert response.status_code == 415
    assert "image/gif" in response.json()["error"]["message"]


def test_decompression_bomb_is_rejected_at_api_boundary(
    security_client: tuple[TestClient, Path],
) -> None:
    client, _ = security_client
    response = client.post(
        "/api/v1/face/detect",
        files={"image": ("small.png", decompression_bomb_png(), "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_profile_is_reencoded_without_exif(security_client: tuple[TestClient, Path]) -> None:
    client, upload_dir = security_client
    source = jpeg_with_orientation(textured_array(320, 240), orientation=6)
    response = client.post(
        "/api/v1/users",
        data={"external_id": "exif-user", "name": "EXIF Test"},
        files={"profile_image": ("camera.jpg", source, "image/jpeg")},
    )
    stored = upload_dir / response.json()["profile_image_url"]
    with Image.open(io.BytesIO(stored.read_bytes())) as image:
        assert not image.getexif()


def test_storage_key_cannot_escape_upload_root(tmp_path: Path) -> None:
    store = ProfileImageStore(tmp_path / "uploads", Settings(ENV="local", FACE_ENGINE="fake"))
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"secret")
    with pytest.raises(ResourceNotFoundError):
        store.read("../secret.jpg")


def test_candidate_is_never_persisted_and_no_response_leaks_biometrics(
    security_client: tuple[TestClient, Path],
) -> None:
    client, upload_dir = security_client
    user = _create_user(client)
    before = sorted(
        path.relative_to(upload_dir) for path in upload_dir.rglob("*") if path.is_file()
    )
    verified = client.post(
        f"/api/v1/users/{user['id']}/verify",
        files={"candidate_image": _image()},
    )
    history = client.get(f"/api/v1/users/{user['id']}/verifications")
    after = sorted(path.relative_to(upload_dir) for path in upload_dir.rglob("*") if path.is_file())
    assert before == after
    assert not _contains_forbidden_key(user)
    assert not _contains_forbidden_key(verified.json())
    assert not _contains_forbidden_key(history.json())
