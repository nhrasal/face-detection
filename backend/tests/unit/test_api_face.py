from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.factories.images import encode, textured_array


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(
        ENV="local",
        FACE_ENGINE="fake",
        INFERENCE_WORKERS=1,
        RATE_LIMIT_COMPARE="1000/minute",
        RATE_LIMIT_DETECT="1000/minute",
        RATE_LIMIT_DETECT_FRAME="1000/minute",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def upload(seed: int = 1) -> tuple[str, bytes, str]:
    return ("portrait.png", encode(textured_array(320, 240, seed=seed)), "image/png")


def test_readiness_reports_warmed_engine(client: TestClient) -> None:
    assert client.get("/readyz").json() == {
        "status": "ready",
        "engine": "fake",
        "warm": True,
    }


def test_detect_returns_quality_and_box_but_no_embedding(client: TestClient) -> None:
    response = client.post("/api/v1/face/detect", files={"image": upload()})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["face_detected"] is True
    assert body["face_count"] == 1
    assert body["bounding_box"]["width"] > 0
    assert not {"embedding", "feature", "vector", "template"} & set(body)


def test_detect_frame_returns_preview_geometry_without_biometrics(client: TestClient) -> None:
    response = client.post("/api/v1/face/detect/frame", files={"frame": upload()})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["face_count"] == 1
    assert body["bounding_box"]["width"] > 0
    assert not {"embedding", "feature", "vector", "template"} & set(body)


def test_frame_cap_is_tighter_than_the_photograph_cap() -> None:
    """The live endpoint trades a looser rate limit for a much smaller payload.

    The same image that /detect accepts must be refused by /detect/frame, or the
    higher call rate becomes a way to buy full-size decode work cheaply.
    """
    settings = Settings(
        ENV="local",
        FACE_ENGINE="fake",
        INFERENCE_WORKERS=1,
        MAX_FRAME_BYTES=32,
        RATE_LIMIT_DETECT="1000/minute",
        RATE_LIMIT_DETECT_FRAME="1000/minute",
    )
    with TestClient(create_app(settings)) as client:
        photo = upload()
        assert client.post("/api/v1/face/detect", files={"image": photo}).status_code == 200
        response = client.post("/api/v1/face/detect/frame", files={"frame": photo})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    assert response.json()["error"]["detail"] == "MAX_FRAME_BYTES"


def test_compare_returns_auditable_decision_without_biometrics(client: TestClient) -> None:
    response = client.post(
        "/api/v1/face/compare",
        files={"reference_image": upload(4), "candidate_image": upload(4)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "MATCH"
    assert body["matched"] is True
    assert body["similarity"] == pytest.approx(1.0)
    assert body["threshold"] == 0.8
    assert body["threshold_version"] == "sface-2021dec-default"
    assert body["detector_version"] == "fake@1"
    assert body["recognition_model_version"] == "fake@1"
    assert not {"embedding", "feature", "vector", "template"} & set(body)


def test_magic_bytes_not_declared_content_type_control_validation(client: TestClient) -> None:
    response = client.post(
        "/api/v1/face/detect",
        files={"image": ("portrait.jpg", upload()[1], "application/pdf")},
    )
    assert response.status_code == 200


def test_invalid_file_has_normalized_415_error(client: TestClient) -> None:
    response = client.post(
        "/api/v1/face/detect",
        files={"image": ("document.pdf", b"%PDF-1.7\nnot an image", "application/pdf")},
    )
    assert response.status_code == 415
    assert response.json() == {
        "success": False,
        "error": {
            "code": "UNSUPPORTED_MEDIA",
            "message": "application/pdf is not a supported image type.",
            "detail": "allowed: image/jpeg, image/png, image/webp",
        },
    }


def test_upload_is_bounded_before_decode() -> None:
    settings = Settings(ENV="local", FACE_ENGINE="fake", INFERENCE_WORKERS=1, MAX_UPLOAD_BYTES=32)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/face/detect",
            files={"image": ("large.jpg", b"x" * 33, "image/jpeg")},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_compare_requires_both_images(client: TestClient) -> None:
    response = client.post("/api/v1/face/compare", files={"reference_image": upload()})
    assert response.status_code == 422


def test_detect_rate_limit_returns_429() -> None:
    settings = Settings(
        ENV="local", FACE_ENGINE="fake", INFERENCE_WORKERS=1, RATE_LIMIT_DETECT="1/minute"
    )
    with TestClient(create_app(settings), client=("rate-limit-test", 50000)) as client:
        assert client.post("/api/v1/face/detect", files={"image": upload()}).status_code == 200
        response = client.post("/api/v1/face/detect", files={"image": upload()})
    assert response.status_code == 429


def test_unexpected_engine_error_is_normalized_without_leaking_detail() -> None:
    class BrokenPool:
        @contextmanager
        def acquire(self) -> Iterator[Any]:
            raise RuntimeError("secret model path /private/models/broken.onnx")
            yield

        def close(self) -> None:
            pass

    settings = Settings(ENV="local", FACE_ENGINE="fake", INFERENCE_WORKERS=1)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.engine_pool = BrokenPool()
        response = client.post("/api/v1/face/detect", files={"image": upload()})
    assert response.status_code == 500
    assert response.json() == {
        "success": False,
        "error": {
            "code": "PROCESSING_ERROR",
            "message": "The image could not be processed.",
            "detail": None,
        },
    }
    assert "private/models" not in response.text


# Registering a low limit on the module-level Limiter leaks into every app built
# afterwards, so — like the /detect limit test above — this stays last in the file.
def test_frame_endpoint_enforces_its_own_rate_limit() -> None:
    settings = Settings(
        ENV="local",
        FACE_ENGINE="fake",
        INFERENCE_WORKERS=1,
        RATE_LIMIT_DETECT_FRAME="3/minute",
    )
    with TestClient(create_app(settings), client=("frame-limit-test", 50001)) as client:
        for _ in range(3):
            assert (
                client.post("/api/v1/face/detect/frame", files={"frame": upload()}).status_code
                == 200
            )
        response = client.post("/api/v1/face/detect/frame", files={"frame": upload()})
    assert response.status_code == 429
