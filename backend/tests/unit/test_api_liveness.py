"""Liveness WebSocket contract.

Drives the socket with synthetic frames whose landmarks are controlled, so the
challenge sequence can be walked deterministically without a camera.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.engine.liveness import ChallengeKind
from app.main import create_app


def jpeg(width: int = 320, height: int = 240) -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    # Generous HTTP limits on purpose. `limiter` is a module-level singleton, so
    # @limiter.limit registers against the endpoint FUNCTION rather than the app:
    # an app built here with production defaults silently overrides the limits
    # other test modules set for the same endpoints. Liveness itself is a
    # WebSocket and is bounded by MAX_STREAM_SESSIONS, not by these.
    return TestClient(
        create_app(
            Settings(
                ENV="local",
                FACE_ENGINE="opencv_zoo",
                RATE_LIMIT_COMPARE="1000/minute",
                RATE_LIMIT_DETECT="1000/minute",
                RATE_LIMIT_DETECT_FRAME="1000/minute",
            )
        )
    )


class TestHandshake:
    def test_first_message_issues_a_challenge(self, client: TestClient) -> None:
        with client.websocket_connect("/api/v1/face/liveness") as socket:
            message = socket.receive_json()
        assert message["type"] == "challenge"
        assert message["challenge"] in {c.value for c in ChallengeKind}
        assert message["progress"] == [0, 3]
        assert message["session_id"]

    def test_the_server_owns_the_session_id(self, client: TestClient) -> None:
        """A client that could choose its own id could replay someone else's."""
        with client.websocket_connect("/api/v1/face/liveness") as a:
            first = a.receive_json()["session_id"]
        with client.websocket_connect("/api/v1/face/liveness") as b:
            second = b.receive_json()["session_id"]
        assert first != second

    def test_a_mismatched_origin_is_refused(self, client: TestClient) -> None:
        # WebSockets bypass CORS entirely, so this is the only thing stopping
        # another site from driving a victim's camera session.
        with (
            pytest.raises(Exception),  # noqa: B017 - starlette raises on close
            client.websocket_connect(
                "/api/v1/face/liveness", headers={"origin": "http://evil.example"}
            ) as socket,
        ):
            socket.receive_json()

    def test_an_allowlisted_origin_is_accepted(self, client: TestClient) -> None:
        with client.websocket_connect(
            "/api/v1/face/liveness", headers={"origin": "http://localhost:3320"}
        ) as socket:
            assert socket.receive_json()["type"] == "challenge"


class TestFrameHandling:
    def test_a_faceless_frame_does_not_advance_progress(self, client: TestClient) -> None:
        with client.websocket_connect("/api/v1/face/liveness") as socket:
            socket.receive_json()
            socket.send_bytes(jpeg())
            update = socket.receive_json()
        assert update["type"] == "progress"
        assert update["progress"] == [0, 3]

    def test_an_oversized_frame_closes_the_socket(self, client: TestClient) -> None:
        with (
            pytest.raises(Exception),  # noqa: B017
            client.websocket_connect("/api/v1/face/liveness") as socket,
        ):
            socket.receive_json()
            socket.send_bytes(b"\xff" * (2 * 1024 * 1024))
            socket.receive_json()

    def test_an_undecodable_frame_costs_an_observation_not_the_attempt(
        self, client: TestClient
    ) -> None:
        """A camera still warming up should not fail someone's liveness check."""
        with client.websocket_connect("/api/v1/face/liveness") as socket:
            socket.receive_json()
            for _ in range(3):
                socket.send_bytes(b"not-an-image")
                update = socket.receive_json()
            assert update["state"] != "FAILED"


class TestSessionIsolation:
    def test_sessions_are_discarded_on_disconnect(self, client: TestClient) -> None:
        # Single-use: a lingering PASSED session would let a second connection
        # claim a result it never earned.
        with client.websocket_connect("/api/v1/face/liveness") as socket:
            session_id = socket.receive_json()["session_id"]
        with client.websocket_connect("/api/v1/face/liveness") as socket:
            assert socket.receive_json()["session_id"] != session_id

    def test_no_embedding_ever_appears_on_the_socket(self, client: TestClient) -> None:
        with client.websocket_connect("/api/v1/face/liveness") as socket:
            first = socket.receive_json()
            socket.send_bytes(jpeg())
            second = socket.receive_json()
        for message in (first, second):
            serialised = str(message).lower()
            for banned in ("embedding", "feature", "vector", "template"):
                assert banned not in serialised
