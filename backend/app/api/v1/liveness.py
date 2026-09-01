"""Challenge-response liveness over a WebSocket.

Protocol, deliberately minimal:

    server -> {"type": "challenge", "challenge": "TURN_LEFT", "progress": [0, 3],
               "state": "AWAITING_NEUTRAL", "session_id": "..."}
    client -> <binary JPEG frame>
    server -> {"type": "progress", ...}            one per frame
    server -> {"type": "result", "passed": true}   then close

The session lives on the server. A client cannot report its own liveness result,
because a client that can assert "I passed" is not a liveness check at all.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

from app.api.v1.face import _analyse_bytes
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
from app.engine.liveness import sample_pose
from app.engine.pipeline import FaceSelectionPolicy
from app.services.liveness_session import LivenessSession, SessionStore

log = get_logger(__name__)


def _state_message(session: LivenessSession, message_type: str) -> dict[str, Any]:
    done, total = session.progress
    return {
        "type": message_type,
        "session_id": session.id,
        "state": session.state.value,
        "challenge": session.current.value if session.current else None,
        "progress": [done, total],
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
            log.info("liveness.opened", session=session.id, sequence=len(session.sequence))
            await websocket.send_json(_state_message(session, "challenge"))

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
            analysis = await run_in_threadpool(
                _analyse_bytes,
                websocket.app.state.engine_pool,
                frame,
                settings,
                max_bytes=settings.MAX_FRAME_BYTES,
                multi_face=FaceSelectionPolicy.LARGEST,
                timeout=ACQUIRE_TIMEOUT_SECONDS,
            )
        except AppError:
            # A frame the pipeline refused is a missing observation, not a
            # failed challenge: a camera still warming up should not cost
            # someone their attempt.
            session.submit(None)
            return _state_message(session, "progress")
        except Exception as error:
            log.exception("liveness.frame_failed", error_type=type(error).__name__)
            session.submit(None)
            return _state_message(session, "progress")

        session.submit(sample_pose(analysis))
        return _state_message(session, "progress")

    return router
