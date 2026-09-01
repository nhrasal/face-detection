"""Presence-check WebSocket contract.

Hermetic: the fake engine detects one centred face in any frame, so the whole
path — handshake, per-frame detection payload, the hold, the blink prompt — can
be walked without ONNX weights, face assets or a camera.

The blink itself cannot be driven from here: it is measured from pixels, and
synthesising a frame that reads as a closed eye through the real detector is the
model tier's job (tests/models/test_liveness_eyes.py). What this file pins is
that the socket ASKS for a blink and does not pass without one.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.engine.liveness import DEFAULT_REQUIRED_FRAMES
from app.main import create_app
from tests.factories.images import encode, textured_array

LIVENESS = "/api/v1/face/liveness"


def frame(seed: int = 1) -> bytes:
    """A frame the fake engine reads as one sharp, well-exposed face."""
    return encode(textured_array(320, 240, seed=seed), fmt="JPEG")


def build(**overrides: object) -> TestClient:
    settings = Settings(
        ENV="local",
        FACE_ENGINE="fake",
        INFERENCE_WORKERS=1,
        CORS_ORIGINS=["http://localhost:3320"],
        # The Limiter is a module-level singleton, so every app built here
        # registers limits that outlive this file. Keep them high or the suites
        # that run after this one start seeing 429s. This endpoint is a
        # WebSocket and is bounded by MAX_STREAM_SESSIONS, not by these.
        RATE_LIMIT_COMPARE="1000/minute",
        RATE_LIMIT_DETECT="1000/minute",
        RATE_LIMIT_DETECT_FRAME="1000/minute",
        **overrides,  # type: ignore[arg-type]
    )
    return TestClient(create_app(settings))


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Entered as a context manager so lifespan runs and the engine pool exists.
    # Without it every frame falls down the "could not be analysed" path and the
    # tests below would pass while exercising nothing.
    with build() as test_client:
        yield test_client


class TestHandshake:
    def test_first_message_opens_the_session(self, client: TestClient) -> None:
        with client.websocket_connect(LIVENESS) as socket:
            message = socket.receive_json()
        assert message["type"] == "start"
        assert message["state"] == "AWAITING_FACE"
        # The blink is the final step, so the total is one more than the hold.
        assert message["progress"] == [0, DEFAULT_REQUIRED_FRAMES + 1]
        assert message["session_id"]

    def test_the_subject_is_never_asked_to_move(self, client: TestClient) -> None:
        """No challenge field on the wire — the check is passive by contract."""
        with client.websocket_connect(LIVENESS) as socket:
            message = socket.receive_json()
        assert "challenge" not in message

    def test_the_opening_message_carries_no_detection_yet(self, client: TestClient) -> None:
        with client.websocket_connect(LIVENESS) as socket:
            message = socket.receive_json()
        assert message["detection"] is None

    def test_the_server_owns_the_session_id(self, client: TestClient) -> None:
        """A client that could choose its own id could replay someone else's."""
        with client.websocket_connect(LIVENESS) as a:
            first = a.receive_json()["session_id"]
        with client.websocket_connect(LIVENESS) as b:
            second = b.receive_json()["session_id"]
        assert first != second

    def test_a_mismatched_origin_is_refused(self, client: TestClient) -> None:
        # WebSockets bypass CORS entirely, so this is the only thing stopping
        # another site from driving a victim's camera session.
        with (
            pytest.raises(Exception),  # noqa: B017 - starlette raises on close
            client.websocket_connect(LIVENESS, headers={"origin": "http://evil.example"}) as socket,
        ):
            socket.receive_json()

    def test_an_allowlisted_origin_is_accepted(self, client: TestClient) -> None:
        with client.websocket_connect(
            LIVENESS, headers={"origin": "http://localhost:3320"}
        ) as socket:
            assert socket.receive_json()["type"] == "start"


class TestRealTimeFeedback:
    def test_every_frame_reports_what_the_detector_saw(self, client: TestClient) -> None:
        """Without this the subject watches a bar that will not move.

        The panel needs the box and the reason codes to say WHICH thing to fix,
        so the detect payload rides along with the progress rather than making
        the client open a second socket to find out.
        """
        with client.websocket_connect(LIVENESS) as socket:
            socket.receive_json()
            socket.send_bytes(frame())
            update = socket.receive_json()
        detection = update["detection"]
        assert detection is not None
        assert detection["status"] == "OK"
        assert detection["face_count"] == 1
        assert detection["bounding_box"]["width"] > 0
        assert detection["quality_issues"] == []

    def test_an_unreadable_frame_reports_no_detection_rather_than_a_stale_one(
        self, client: TestClient
    ) -> None:
        # A box drawn from a previous frame, over a face that has since moved, is
        # worse than drawing nothing.
        with client.websocket_connect(LIVENESS) as socket:
            socket.receive_json()
            socket.send_bytes(b"not-an-image")
            update = socket.receive_json()
        assert update["detection"] is None

    def test_progress_advances_frame_by_frame(self, client: TestClient) -> None:
        with client.websocket_connect(LIVENESS) as socket:
            socket.receive_json()
            for expected in range(1, 4):
                socket.send_bytes(frame(expected))
                update = socket.receive_json()
                assert update["progress"] == [expected, DEFAULT_REQUIRED_FRAMES + 1]
                assert update["state"] == "HOLDING"


class TestFrameHandling:
    def test_a_sustained_hold_then_asks_for_a_blink(self, client: TestClient) -> None:
        """A held face is not enough — which is the entire point of the blink.

        A photograph held steadily to the lens reaches exactly this state and
        goes no further.
        """
        with client.websocket_connect(LIVENESS) as socket:
            socket.receive_json()
            for index in range(DEFAULT_REQUIRED_FRAMES):
                socket.send_bytes(frame(index))
                update = socket.receive_json()
            assert update["state"] == "AWAITING_BLINK"
            assert update["progress"] == [DEFAULT_REQUIRED_FRAMES, DEFAULT_REQUIRED_FRAMES + 1]

    def test_an_unblinking_face_never_passes(self, client: TestClient) -> None:
        with client.websocket_connect(LIVENESS) as socket:
            socket.receive_json()
            for index in range(DEFAULT_REQUIRED_FRAMES * 3):
                socket.send_bytes(frame(index))
                update = socket.receive_json()
        assert update["state"] != "PASSED"

    def test_a_refused_frame_does_not_advance_progress(self, client: TestClient) -> None:
        with client.websocket_connect(LIVENESS) as socket:
            socket.receive_json()
            socket.send_bytes(b"not-an-image")
            update = socket.receive_json()
        assert update["type"] == "progress"
        assert update["progress"] == [0, DEFAULT_REQUIRED_FRAMES + 1]

    def test_an_oversized_frame_closes_the_socket(self, client: TestClient) -> None:
        with (
            pytest.raises(Exception),  # noqa: B017
            client.websocket_connect(LIVENESS) as socket,
        ):
            socket.receive_json()
            socket.send_bytes(b"\xff" * (2 * 1024 * 1024))
            socket.receive_json()

    def test_an_undecodable_frame_costs_an_observation_not_the_attempt(
        self, client: TestClient
    ) -> None:
        """A camera still warming up should not cost someone their attempt."""
        with client.websocket_connect(LIVENESS) as socket:
            socket.receive_json()
            for _ in range(3):
                socket.send_bytes(b"not-an-image")
                update = socket.receive_json()
            assert update["state"] != "FAILED"

    def test_a_broken_run_starts_over(self, client: TestClient) -> None:
        # The hold has to be uninterrupted, so a sustained gap resets it rather
        # than banking the frames that came before.
        with client.websocket_connect(LIVENESS) as socket:
            socket.receive_json()
            for index in range(3):
                socket.send_bytes(frame(index))
                socket.receive_json()
            for _ in range(4):
                socket.send_bytes(b"not-an-image")
                update = socket.receive_json()
        assert update["progress"] == [0, DEFAULT_REQUIRED_FRAMES + 1]
        assert update["state"] == "AWAITING_FACE"


class TestSessionIsolation:
    def test_sessions_are_discarded_on_disconnect(self, client: TestClient) -> None:
        # Single-use: a lingering PASSED session would let a second connection
        # claim a result it never earned.
        with client.websocket_connect(LIVENESS) as socket:
            session_id = socket.receive_json()["session_id"]
        with client.websocket_connect(LIVENESS) as socket:
            assert socket.receive_json()["session_id"] != session_id

    def test_no_embedding_ever_appears_on_the_socket(self, client: TestClient) -> None:
        with client.websocket_connect(LIVENESS) as socket:
            first = socket.receive_json()
            socket.send_bytes(frame())
            second = socket.receive_json()
        for message in (first, second):
            serialised = str(message).lower()
            for banned in ("embedding", "feature", "vector", "template"):
                assert banned not in serialised
