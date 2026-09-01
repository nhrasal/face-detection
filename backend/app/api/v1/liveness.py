"""Passive presence check over a WebSocket.

Protocol, deliberately minimal:

    server -> {"type": "start", "progress": [0, 10],
               "state": "AWAITING_FACE", "session_id": "...", "detection": null}
    client -> <binary JPEG frame>
    server -> {"type": "progress", ..., "detection": {...},
               "eye": {"openness": .., "baseline": .., "ratio": ..}}  one per frame
    server -> {"type": "result", "passed": true}              then close

Every progress message carries the same detection payload the preview stream
returns, so the client can draw the box and say the ONE thing that needs fixing.
Without it a subject whose face is too dark or too far away just watches a bar
that never moves, with nothing to act on.

The subject holds still until the run is long enough, then blinks. What that
does and does not prove is spelled out in `app.engine.liveness`: it defeats a
photograph or a screen, and it does not defeat a video replay or a deepfake.

The session still lives on the server. A client cannot report its own result,
because a client that can assert "I passed" is not a check at all.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

from app.api.v1.face import _analyse_bytes_with_image, _detect_response
from app.api.v1.stream import (
    ACQUIRE_TIMEOUT_SECONDS,
    CLOSE_FORBIDDEN_ORIGIN,
    CLOSE_FRAME_TOO_LARGE,
    CLOSE_TOO_MANY_SESSIONS,
    SessionLimiter,
    _origin_allowed,
)
from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.engine.liveness import PresenceSample, sample_presence
from app.engine.pipeline import FaceSelectionPolicy
from app.services.liveness_session import LivenessSession, SessionStore

log = get_logger(__name__)


def _eye_diagnostic(
    session: LivenessSession, sample: PresenceSample | None
) -> dict[str, Any] | None:
    """The raw eye numbers, for tuning the threshold against a real camera.

    The closed threshold cannot be set from still photographs: what it has to
    clear is how far an OPEN eye wanders frame to frame, and a still cannot show
    that. Putting the live values on the wire is what lets that be measured on
    the face and camera actually in front of the service.

    Nothing here is biometric — two scalars describing local contrast, from which
    no face can be reconstructed or recognised.
    """
    if sample is None or sample.openness is None:
        return None
    baseline = session.baseline
    return {
        "openness": round(sample.openness, 4),
        "baseline": round(baseline, 4) if baseline is not None else None,
        "ratio": round(sample.openness / baseline, 4)
        if baseline is not None and baseline > 0
        else None,
    }


def _state_message(
    session: LivenessSession,
    message_type: str,
    detection: dict[str, Any] | None = None,
    sample: PresenceSample | None = None,
) -> dict[str, Any]:
    done, total = session.progress
    return {
        "type": message_type,
        "session_id": session.id,
        "state": session.state.value,
        "progress": [done, total],
        "eye": _eye_diagnostic(session, sample),
        # None whenever the frame could not be analysed at all, which the client
        # must render as "still looking" rather than as a stale verdict. This is
        # the ordinary detect schema and so carries no embedding, by construction.
        "detection": detection,
    }


def create_liveness_router(
    settings: Settings, limiter: SessionLimiter, store: SessionStore
) -> APIRouter:
    router = APIRouter(prefix="/face", tags=["face"])

    @router.websocket("/liveness")
    async def liveness_session(websocket: WebSocket) -> None:
        if not _origin_allowed(websocket, settings):
            await websocket.close(code=CLOSE_FORBIDDEN_ORIGIN, reason="Origin not allowed.")
            return
        # Shares the preview stream's ceiling: both hold an inference worker per
        # frame, so counting them separately would double the real load that
        # MAX_STREAM_SESSIONS is meant to bound.
        if not limiter.try_acquire():
            await websocket.close(
                code=CLOSE_TOO_MANY_SESSIONS, reason="Too many live camera sessions."
            )
            return

        session: LivenessSession | None = None
        await websocket.accept()
        try:
            session = store.create()
            log.info("liveness.opened", session=session.id, required=session.required_frames)
            await websocket.send_json(_state_message(session, "start"))

            while not session.finished:
                frame = await websocket.receive_bytes()
                if len(frame) > settings.MAX_FRAME_BYTES:
                    await websocket.close(
                        code=CLOSE_FRAME_TOO_LARGE, reason="Frame exceeds the size limit."
                    )
                    return
                await websocket.send_json(await _advance(websocket, session, frame))

            await websocket.send_json(
                {
                    "type": "result",
                    "session_id": session.id,
                    "passed": session.passed,
                    "failure": session.failure.value if session.failure else None,
                }
            )
        except WebSocketDisconnect:
            pass
        finally:
            limiter.release()
            if session is not None:
                # Sessions are single-use. Leaving a PASSED session retrievable
                # would let a second connection claim a result it did not earn.
                store.discard(session.id)
                log.info("liveness.closed", session=session.id, passed=session.passed)
            if websocket.application_state is WebSocketState.CONNECTED:
                await websocket.close()

    async def _advance(
        websocket: WebSocket, session: LivenessSession, frame: bytes
    ) -> dict[str, Any]:
        try:
            analysis, image = await run_in_threadpool(
                _analyse_bytes_with_image,
                websocket.app.state.engine_pool,
                frame,
                settings,
                max_bytes=settings.MAX_FRAME_BYTES,
                multi_face=FaceSelectionPolicy.LARGEST,
                timeout=ACQUIRE_TIMEOUT_SECONDS,
            )
        except AppError:
            # A frame the pipeline refused is a missing observation, not a
            # failure: a camera still warming up should not cost someone their
            # attempt.
            session.submit(None)
            return _state_message(session, "progress")
        except Exception as error:
            log.exception("liveness.frame_failed", error_type=type(error).__name__)
            session.submit(None)
            return _state_message(session, "progress")

        sample = sample_presence(analysis, image)
        session.submit(sample)
        return _state_message(
            session,
            "progress",
            _detect_response(analysis).model_dump(mode="json"),
            sample=sample,
        )

    return router
