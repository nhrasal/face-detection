"""Presence session state.

Holds one subject's progress through two phases: a sustained run of usable
frames, which also builds the eye-openness baseline, and then a blink measured
against that baseline. Kept apart from the engine so the rule stays pure and
testable, and apart from the transport so the same state machine serves a
WebSocket today and a Redis-backed HTTP flow later.

What this session proves is bounded — see the module docstring of
`app.engine.liveness`. It defeats a photograph; it does not defeat a video
replay or a live deepfake.

Sessions live in memory. That is correct for a single process and wrong for
several: the roadmap puts Redis at V3 for exactly this, and `SessionStore` is
the seam it drops into. Documented rather than discovered when a second replica
appears.
"""

from __future__ import annotations

import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from statistics import median

from app.engine.liveness import (
    DEFAULT_PRESENCE_THRESHOLDS,
    DEFAULT_REQUIRED_FRAMES,
    PresenceSample,
    PresenceThresholds,
    counts_towards_hold,
)

DEFAULT_SESSION_TTL = 180.0

# How long the subject has to blink once they have been asked to. Generous: the
# prompt has to be read before it can be obeyed, and someone who blinks a beat
# too late should get another chance rather than a failure.
DEFAULT_BLINK_TIMEOUT = 15.0

# A single frame that does not count is a camera hiccup, a blink, a moment of
# motion blur — not the person leaving. The run only breaks once several in a
# row have failed, otherwise an ordinary webcam could never complete a hold.
SKIP_TOLERANCE = 3

# A face that stays gone for this many frames is someone who walked away, or a
# photo being swapped for another photo.
MAX_CONSECUTIVE_MISSING = 30

# How many recent open-eye readings the baseline is drawn from. Roughly two
# seconds at the rate the preview socket settles at, which is long enough to
# average out frame noise and short enough to follow a subject walking into
# different light.
BASELINE_WINDOW = 30

# The median of fewer readings than this is not a description of anything, so
# no dip is judged against it. The hold phase runs for ten frames before a blink
# is ever requested, so in practice the window is full long before it matters.
MIN_BASELINE_SAMPLES = 3


class SessionState(StrEnum):
    AWAITING_FACE = "AWAITING_FACE"
    # A usable face is being held; these frames also build the eye baseline.
    HOLDING = "HOLDING"
    # Baseline established, waiting to see the eyes close and reopen.
    AWAITING_BLINK = "AWAITING_BLINK"
    PASSED = "PASSED"
    FAILED = "FAILED"


class FailureReason(StrEnum):
    SESSION_EXPIRED = "SESSION_EXPIRED"
    FACE_LOST = "FACE_LOST"
    BLINK_NOT_SEEN = "BLINK_NOT_SEEN"


@dataclass(slots=True)
class LivenessSession:
    """One run through hold-then-blink.

    Deliberately NOT frozen: this is a state machine advanced frame by frame.
    """

    id: str
    required_frames: int = DEFAULT_REQUIRED_FRAMES
    thresholds: PresenceThresholds = DEFAULT_PRESENCE_THRESHOLDS
    ttl: float = DEFAULT_SESSION_TTL
    blink_timeout: float = DEFAULT_BLINK_TIMEOUT

    held: int = 0
    state: SessionState = SessionState.AWAITING_FACE
    failure: FailureReason | None = None
    # The MEDIAN of recent open-eye readings — what open looks like for this
    # face, in this light, right now. A blink is a dip below it.
    #
    # Median, not the peak it used to be. Peak-hold parks the reference at the
    # TOP of the frame-to-frame noise band instead of its centre, and that
    # breaks the check at both ends: closing at 0.75 of an inflated peak trips
    # on ordinary noise, while reopening at 0.85 of it is a bar a genuinely open
    # eye often cannot clear, so a real blink sticks in the closed state and
    # never completes. Erratic in both directions, from one wrong statistic.
    # Against a median the two ratios mean what their comments say they mean.
    baseline: float | None = None
    # The readings the median is taken over. Only frames judged open and sharp
    # enough to trust are added, so a blink never pulls the reference down
    # towards itself.
    openness_window: deque[float] = field(default_factory=lambda: deque(maxlen=BASELINE_WINDOW))
    # Peak face sharpness, kept as a peak-hold with decay. Peak is right here:
    # the guard asks "has detail collapsed relative to the best this session has
    # looked", which is a question about the best, not the typical.
    sharpness_baseline: float | None = None
    eyes_closed: bool = False
    blinked: bool = False
    # Diagnostics only — the state machine never reads these. They exist because
    # the one thing a still photograph cannot show is how this behaves on a real
    # face at a real frame rate, so the live values go on the wire to be read
    # off the running system. See `_eye_diagnostic` in the liveness route.
    eye_trusted: bool = False
    sharpness_ratio: float | None = None
    # Stamped on the FIRST submitted frame, from whatever clock that call used,
    # rather than defaulting to time.monotonic() here. Defaulting would mix a
    # real clock with an injected one and make every elapsed-time comparison
    # silently wrong — negative, in fact, so the TTL branch could never fire.
    created_at: float | None = None
    blink_armed_at: float | None = None
    consecutive_missing: int = 0
    consecutive_skipped: int = 0

    @property
    def finished(self) -> bool:
        return self.state in (SessionState.PASSED, SessionState.FAILED)

    @property
    def passed(self) -> bool:
        return self.state is SessionState.PASSED

    @property
    def progress(self) -> tuple[int, int]:
        """Completed steps out of total, where the blink is the final step.

        Counting the blink in the total is what stops the bar sitting visibly
        full while the check is still waiting for something.
        """
        total = self.required_frames + 1
        return (min(self.held, self.required_frames) + int(self.blinked), total)

    def _fail(self, reason: FailureReason) -> None:
        self.state = SessionState.FAILED
        self.failure = reason

    def _break_run(self) -> None:
        """Back to square one, baseline included.

        The baseline describes one face under one set of conditions. Whoever
        turns up after the run broke may be a different person in a different
        light, and reusing the old reference against them is how a "blink"
        appears out of nothing but a change of scene.
        """
        self.held = 0
        self.state = SessionState.AWAITING_FACE
        self.baseline = None
        self.openness_window.clear()
        self.sharpness_baseline = None
        self.eyes_closed = False
        self.blink_armed_at = None

    def _observe_eyes(self, openness: float | None, sharpness: float | None) -> None:
        """Track the baseline and watch for a dip and recovery.

        Blurred frames are discarded rather than interpreted. Blur suppresses the
        same eye detail a closed lid does, so reading one as the other is how a
        shaken photograph — or plain camera shake — manufactures a blink.
        """
        if sharpness is not None:
            self.sharpness_baseline = (
                sharpness
                if self.sharpness_baseline is None
                else max(sharpness, self.sharpness_baseline * self.thresholds.baseline_decay)
            )

        self.eye_trusted = False
        self.sharpness_ratio = (
            sharpness / self.sharpness_baseline
            if sharpness is not None and self.sharpness_baseline
            else None
        )

        if openness is None:
            return

        # Unmeasurable sharpness is treated as untrustworthy, not as a pass: if
        # the guard cannot run, the reading it guards does not get used.
        if sharpness is None or self.sharpness_baseline is None:
            return
        if sharpness < self.sharpness_baseline * self.thresholds.min_sharpness_ratio:
            return
        self.eye_trusted = True

        if self.eyes_closed:
            if self.baseline is not None and openness >= (
                self.baseline * self.thresholds.eyes_open_ratio
            ):
                # Closed, then open again: that is the blink.
                self.eyes_closed = False
                self.blinked = True
            return

        # A dip is only meaningful against a reference built from enough
        # readings to be one. Until then, frames only ever feed the window.
        if (
            self.baseline is not None
            and len(self.openness_window) >= MIN_BASELINE_SAMPLES
            and openness <= self.baseline * self.thresholds.eyes_closed_ratio
        ):
            self.eyes_closed = True
            return

        # Open frames define the reference. Recorded before the median is taken
        # so the newest reading counts towards the value it will be judged
        # against next frame, and never while the eyes are closed — the branch
        # above returns first — so a blink cannot drag the baseline to meet it.
        self.openness_window.append(openness)
        self.baseline = float(median(self.openness_window))

    def submit(self, sample: PresenceSample | None, *, now: float | None = None) -> None:
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
            self.consecutive_missing += 1
        else:
            self.consecutive_missing = 0

        if sample is None or not counts_towards_hold(sample, thresholds=self.thresholds):
            # The frame did not count: no face at all, or a face the quality gate
            # refused. Neither is a failure on its own.
            if self.consecutive_missing >= MAX_CONSECUTIVE_MISSING:
                self._fail(FailureReason.FACE_LOST)
                return
            self.consecutive_skipped += 1
            if self.consecutive_skipped >= SKIP_TOLERANCE:
                self._break_run()
            return

        self.consecutive_skipped = 0
        self.held += 1
        self._observe_eyes(sample.openness, sample.sharpness)

        if self.held < self.required_frames:
            self.state = SessionState.HOLDING
            return

        if self.blinked:
            self.state = SessionState.PASSED
            return

        # `is None`, not `or`: a blink_armed_at of 0.0 is falsy, and `or` would
        # silently restart the clock on every frame, making the timeout
        # unreachable. The same bug this file carried once before.
        if self.blink_armed_at is None:
            self.blink_armed_at = now
        elif now - self.blink_armed_at > self.blink_timeout:
            self._fail(FailureReason.BLINK_NOT_SEEN)
            return
        self.state = SessionState.AWAITING_BLINK


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

    def create(self, *, required_frames: int = DEFAULT_REQUIRED_FRAMES) -> LivenessSession:
        if required_frames < 1:
            raise ValueError("a presence hold needs at least one frame")
        self._sweep()
        if len(self._sessions) >= self._max:
            raise RuntimeError("too many concurrent liveness sessions")
        session = LivenessSession(
            id=secrets.token_urlsafe(16),
            required_frames=required_frames,
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
