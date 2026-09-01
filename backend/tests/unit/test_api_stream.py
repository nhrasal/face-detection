from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.v1.stream import (
    CLOSE_FORBIDDEN_ORIGIN,
    CLOSE_FRAME_TOO_LARGE,
    CLOSE_TOO_MANY_SESSIONS,
)
from app.core.config import Settings
from app.main import create_app
from tests.factories.images import encode, textured_array

STREAM = "/api/v1/face/stream"


def frame(seed: int = 1) -> bytes:
    return encode(textured_array(320, 240, seed=seed), fmt="JPEG")


def build(**overrides: object) -> TestClient:
    settings = Settings(
        ENV="local",
        FACE_ENGINE="fake",
        INFERENCE_WORKERS=1,
        CORS_ORIGINS=["http://localhost:3320"],
        # The Limiter is a module-level singleton, so every app built here
        # registers limits that outlive this file. Keep them high or the suites
        # that run after this one start seeing 429s.
        RATE_LIMIT_COMPARE="1000/minute",
        RATE_LIMIT_DETECT="1000/minute",
        RATE_LIMIT_DETECT_FRAME="1000/minute",
        **overrides,  # type: ignore[arg-type]
    )
    return TestClient(create_app(settings))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with build() as test_client:
        yield test_client


def test_stream_answers_every_frame_with_a_detection(client: TestClient) -> None:
    with client.websocket_connect(STREAM) as socket:
        # Ack-paced: send one, read one, send the next. Several in a row prove the
        # socket stays open and keeps answering rather than being single-shot.
        for seed in range(3):
            socket.send_bytes(frame(seed))
            body = socket.receive_json()
            assert body["status"] == "OK"
            assert body["face_count"] == 1
            assert body["bounding_box"]["width"] > 0
            assert not {"embedding", "feature", "vector", "template"} & set(body)


def test_unreadable_frame_is_reported_without_dropping_the_stream(client: TestClient) -> None:
    with client.websocket_connect(STREAM) as socket:
        socket.send_bytes(b"%PDF-1.7\nnot an image")
        body = socket.receive_json()
        assert body["success"] is False
        assert body["error"]["code"] == "UNSUPPORTED_MEDIA"

        # A camera still warming up should not tear down the preview.
        socket.send_bytes(frame())
        assert socket.receive_json()["status"] == "OK"


def test_oversized_frame_closes_the_socket() -> None:
    with (
        build(MAX_FRAME_BYTES=32) as client,
        pytest.raises(WebSocketDisconnect) as caught,
        client.websocket_connect(STREAM) as socket,
    ):
        socket.send_bytes(frame())
        socket.receive_json()
    assert caught.value.code == CLOSE_FRAME_TOO_LARGE


def test_a_browser_on_another_origin_is_refused() -> None:
    with (
        build() as client,
        pytest.raises(WebSocketDisconnect) as caught,
        client.websocket_connect(STREAM, headers={"origin": "http://evil.example"}) as socket,
    ):
        socket.receive_json()
    assert caught.value.code == CLOSE_FORBIDDEN_ORIGIN


def test_allowed_origin_still_connects() -> None:
    with (
        build() as client,
        client.websocket_connect(STREAM, headers={"origin": "http://localhost:3320"}) as sock,
    ):
        sock.send_bytes(frame())
        assert sock.receive_json()["status"] == "OK"


def test_streams_are_capped_so_they_cannot_starve_the_verification_path() -> None:
    """The whole cost of this feature is concurrency, so the ceiling is the guard."""
    with build(MAX_STREAM_SESSIONS=1) as client:
        with client.websocket_connect(STREAM) as first:
            first.send_bytes(frame())
            assert first.receive_json()["status"] == "OK"

            with (
                pytest.raises(WebSocketDisconnect) as caught,
                client.websocket_connect(STREAM) as second,
            ):
                second.receive_json()
            assert caught.value.code == CLOSE_TOO_MANY_SESSIONS

        # The slot is returned on disconnect, not leaked for the process lifetime.
        with client.websocket_connect(STREAM) as third:
            third.send_bytes(frame())
            assert third.receive_json()["status"] == "OK"
