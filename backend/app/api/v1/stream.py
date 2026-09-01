"""Live camera detection over a WebSocket.

The client sends one JPEG frame as a binary message and gets one JSON detection
back. It is ack-paced: the next frame goes out only once the previous result has
arrived, so the stream runs at whatever rate the pipeline can actually sustain
instead of building a queue of frames whose answers describe a scene that has
already changed.

Nothing here is recorded. These are preview frames for overlay guidance; the
still the operator finally captures goes through the ordinary HTTP verification
path, where the REJECT multi-face policy and the audit row apply.

Cost, stated plainly: a stream occupies an inference worker for every frame it
sends, so concurrent viewers are bounded by MAX_STREAM_SESSIONS rather than by a
rate limit. This does not scale to many simultaneous users the way the HTTP
endpoint does — see README, "Live camera capture".
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

from app.api.v1.face import _analyse_bytes, _detect_response
from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.engine.pipeline import FaceSelectionPolicy

log = get_logger(__name__)

# Application close codes (4000-4999 is the range reserved for applications).
CLOSE_FORBIDDEN_ORIGIN = 4403
CLOSE_TOO_MANY_SESSIONS = 4429
CLOSE_FRAME_TOO_LARGE = 4413

# A frame that cannot get an inference worker promptly is already stale. Fail it
# fast and tell the client to slow down, rather than queueing ahead of a
# verification request that a person is actually waiting on.
ACQUIRE_TIMEOUT_SECONDS = 0.5


class SessionLimiter:
    """Counts live streams. Async-only, so no lock is needed to guard the count."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._active = 0

    def try_acquire(self) -> bool:
        if self._active >= self._limit:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        self._active = max(0, self._active - 1)

    @property
    def active(self) -> int:
        return self._active


def _origin_allowed(websocket: WebSocket, settings: Settings) -> bool:
    """WebSockets bypass CORS entirely, so the origin allowlist is applied here.

    A missing Origin is a non-browser client, which is not what the same-origin
    policy protects against; a *mismatched* Origin is a browser being driven by
    another site and is refused.
    """
    origin = websocket.headers.get("origin")
    return origin is None or origin in settings.CORS_ORIGINS


def create_stream_router(settings: Settings, sessions: SessionLimiter) -> APIRouter:
    router = APIRouter(prefix="/face", tags=["face"])

    @router.websocket("/stream")
    async def stream_detection(websocket: WebSocket) -> None:
        if not _origin_allowed(websocket, settings):
            await websocket.close(code=CLOSE_FORBIDDEN_ORIGIN, reason="Origin not allowed.")
            return
        if not sessions.try_acquire():
            # Accept first, then close with the application code. A browser only
            # ever sees code 1006 for a handshake that was refused outright, so
            # rejecting before accept() would leave a legitimate operator with a
            # silent fallback and no way to learn that the cameras are all busy.
            # The forbidden-origin case above is deliberately the other way round:
            # there is no legitimate operator to inform, so the handshake fails.
            await websocket.accept()
            await websocket.close(
                code=CLOSE_TOO_MANY_SESSIONS, reason="Too many live camera sessions."
            )
            return

        await websocket.accept()
        log.info("stream.opened", active=sessions.active)
        try:
            while True:
                frame = await websocket.receive_bytes()
                if len(frame) > settings.MAX_FRAME_BYTES:
                    await websocket.close(
                        code=CLOSE_FRAME_TOO_LARGE, reason="Frame exceeds the size limit."
                    )
                    return
                await websocket.send_json(await _detect(websocket, frame))
        except WebSocketDisconnect:
            pass
        finally:
            sessions.release()
            log.info("stream.closed", active=sessions.active)
            # application_state, not client_state: after this handler has already
            # closed the socket (an oversized frame) the client side can still
            # read as CONNECTED, and closing twice raises.
            if websocket.application_state is WebSocketState.CONNECTED:
                await websocket.close()

    async def _detect(websocket: WebSocket, frame: bytes) -> dict[str, object]:
        """One frame in, one JSON message out.

        A frame the pipeline refuses is reported on the socket and the stream
        stays open: a single unreadable frame from a camera that is still warming
        up should not tear down the preview.
        """
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
        except AppError as error:
            return {
                "success": False,
                "error": {"code": error.code, "message": error.message, "detail": error.detail},
            }
        except Exception as error:
            # Same containment as the HTTP handler: decoder and model failures can
            # carry paths and implementation detail, so none of it reaches the wire.
            log.exception("stream.frame_failed", error_type=type(error).__name__)
            return {
                "success": False,
                "error": {
                    "code": "PROCESSING_ERROR",
                    "message": "The frame could not be processed.",
                    "detail": None,
                },
            }
        return _detect_response(analysis).model_dump(mode="json")

    return router
