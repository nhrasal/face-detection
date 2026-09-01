"""Liveness session state.

Holds one subject's progress through a randomised challenge sequence. Kept apart
from the engine so the geometry stays pure and testable, and apart from the
transport so the same state machine serves a WebSocket today and a Redis-backed
HTTP flow later.

Sessions live in memory. That is correct for a single process and wrong for
several: the roadmap puts Redis at V3 for exactly this, and `SessionStore` is
the seam it drops into. Documented rather than discovered when a second replica
appears.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from enum import StrEnum

from app.engine.liveness import (
    DEFAULT_LIVENESS_THRESHOLDS,
    ChallengeKind,
    LivenessThresholds,
    PoseSample,
    random_sequence,
    satisfies,
)

DEFAULT_SEQUENCE_LENGTH = 3
DEFAULT_CHALLENGE_TIMEOUT = 12.0
DEFAULT_SESSION_TTL = 180.0


class SessionState(StrEnum):
    AWAITING_NEUTRAL = "AWAITING_NEUTRAL"
    AWAITING_CHALLENGE = "AWAITING_CHALLENGE"
    PASSED = "PASSED"
    FAILED = "FAILED"


class FailureReason(StrEnum):
    CHALLENGE_TIMEOUT = "CHALLENGE_TIMEOUT"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    FACE_LOST = "FACE_LOST"


@dataclass(slots=True)
class LivenessSession:
    """One run through a challenge sequence.

    Deliberately NOT frozen: this is a state machine advanced frame by frame.
    """

    id: str
    sequence: tuple[ChallengeKind, ...]
    thresholds: LivenessThresholds = DEFAULT_LIVENESS_THRESHOLDS
    challenge_timeout: float = DEFAULT_CHALLENGE_TIMEOUT
    ttl: float = DEFAULT_SESSION_TTL

    index: int = 0
    # Every challenge begins by requiring a near-frontal face, so one sustained
    # turn cannot satisfy "left then right" by drifting through the middle.
    state: SessionState = SessionState.AWAITING_NEUTRAL
    failure: FailureReason | None = None
    baseline_area: float | None = None
    # Stamped on the FIRST submitted frame, from whatever clock that call used,
    # rather than defaulting to time.monotonic() here. Defaulting would mix a
    # real clock with an injected one and make every elapsed-time comparison
    # silently wrong — negative, in fact, so the TTL branch could never fire.
    created_at: float | None = None
    challenge_started_at: float | None = None
    consecutive_missing: int = 0

    @property
    def current(self) -> ChallengeKind | None:
        if self.finished or self.index >= len(self.sequence):
            return None
        return self.sequence[self.index]

    @property
    def finished(self) -> bool:
        return self.state in (SessionState.PASSED, SessionState.FAILED)

    @property
    def passed(self) -> bool:
        return self.state is SessionState.PASSED

    @property
    def progress(self) -> tuple[int, int]:
        return (self.index, len(self.sequence))

    def _fail(self, reason: FailureReason) -> None:
        self.state = SessionState.FAILED
        self.failure = reason

    def submit(self, sample: PoseSample | None, *, now: float | None = None) -> None:
        """Advance the machine with one frame. `sample` is None when no face."""
        now = time.monotonic() if now is None else now
        if self.finished:
            return

        if self.created_at is None:
            self.created_at = now

        if now - self.created_at > self.ttl:
            self._fail(FailureReason.SESSION_EXPIRED)
            return

        if sample is None:
            # A dropped frame is normal; a face that stays gone is someone who
            # walked away, or a photo being swapped for another photo.
            self.consecutive_missing += 1
            if self.consecutive_missing >= 30:
                self._fail(FailureReason.FACE_LOST)
            return
        self.consecutive_missing = 0

        if self.state is SessionState.AWAITING_NEUTRAL:
            if abs(sample.yaw) <= self.thresholds.yaw_neutral:
                self.state = SessionState.AWAITING_CHALLENGE
                self.challenge_started_at = now
                # Face size is captured here, so MOVE_CLOSER is measured from
                # where this subject actually started rather than an absolute.
                self.baseline_area = sample.face_area
            return

        # `is None`, not `or`: a challenge_started_at of 0.0 is falsy, and `or`
        # would silently restart the timeout clock on every frame, making the
        # timeout unreachable.
        started = now if self.challenge_started_at is None else self.challenge_started_at
        if now - started > self.challenge_timeout:
            self._fail(FailureReason.CHALLENGE_TIMEOUT)
            return

        challenge = self.current
        if challenge is None:  # pragma: no cover - guarded by `finished`
            return

        if satisfies(
            challenge,
            sample,
            baseline_area=self.baseline_area,
            thresholds=self.thresholds,
        ):
            self.index += 1
            if self.index >= len(self.sequence):
                self.state = SessionState.PASSED
            else:
                self.state = SessionState.AWAITING_NEUTRAL
                self.challenge_started_at = None


class SessionStore:
    """In-memory sessions with TTL sweeping.

    The Redis seam: swap this class and nothing above it changes.
    """

    def __init__(self, ttl: float = DEFAULT_SESSION_TTL, max_sessions: int = 64) -> None:
        self._ttl = ttl
        self._max = max_sessions
        self._sessions: dict[str, LivenessSession] = {}
        # Sweeping uses the store's own insertion time: a session's created_at
        # stays unset until its first frame, and a session opened but never fed
        # still has to be reclaimed.
        self._inserted_at: dict[str, float] = {}

    def create(self, *, length: int = DEFAULT_SEQUENCE_LENGTH) -> LivenessSession:
        self._sweep()
        if len(self._sessions) >= self._max:
            raise RuntimeError("too many concurrent liveness sessions")
        session = LivenessSession(
            id=secrets.token_urlsafe(16),
            sequence=random_sequence(length),
            ttl=self._ttl,
        )
        self._sessions[session.id] = session
        self._inserted_at[session.id] = time.monotonic()
        return session

    def get(self, session_id: str) -> LivenessSession | None:
        self._sweep()
        return self._sessions.get(session_id)

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._inserted_at.pop(session_id, None)

    def _sweep(self) -> None:
        now = time.monotonic()
        expired = [k for k, at in self._inserted_at.items() if now - at > self._ttl]
        for key in expired:
            self._sessions.pop(key, None)
            self._inserted_at.pop(key, None)

    def __len__(self) -> int:
        return len(self._sessions)
