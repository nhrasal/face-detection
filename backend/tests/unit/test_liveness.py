"""Eye measurement and the hold-then-blink session machine.

Hermetic: samples are constructed directly, so every branch is reachable without
a camera or a cooperative human. The measurement itself is exercised against
real portraits in tests/models/test_liveness_eyes.py, which is where a claim
about what an eye looks like belongs.

These tests pin what the check DOES prove — a held face that blinked — and
nothing more. There is deliberately no test asserting it defeats a video replay
or a deepfake, because it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engine.liveness import (
    DEFAULT_PRESENCE_THRESHOLDS,
    MIN_INTEROCULAR_PX,
    PresenceSample,
    PresenceThresholds,
    counts_towards_hold,
    eye_openness,
    face_sharpness,
    sample_presence,
)
from app.engine.pipeline import FaceStatus, ImageAnalysis
from app.services.liveness_session import (
    MAX_CONSECUTIVE_MISSING,
    SKIP_TOLERANCE,
    FailureReason,
    LivenessSession,
    SessionState,
    SessionStore,
)
from tests.factories.images import make_face, textured_array

# Openness and sharpness values are arbitrary units compared only against a
# session's own baseline, so the numbers here just need to be self-consistent.
OPEN = PresenceSample(usable=True, score=0.95, openness=1.0, sharpness=1.0)
SHUT = PresenceSample(usable=True, score=0.95, openness=0.5, sharpness=1.0)
# Same eye reading as SHUT, but the whole frame lost its detail — motion blur,
# not a blink.
BLURRED = PresenceSample(usable=True, score=0.95, openness=0.5, sharpness=0.05)


def eye(openness: float) -> PresenceSample:
    """A usable, sharp frame carrying one eye reading."""
    return PresenceSample(usable=True, score=0.95, openness=openness, sharpness=1.0)


def hold_then(session: LivenessSession, *samples: PresenceSample) -> None:
    """Complete the steady-face phase, then submit the given frames."""
    for _ in range(session.required_frames):
        session.submit(OPEN, now=0.0)
    for sample in samples:
        session.submit(sample, now=0.0)


class TestEyeOpenness:
    def gray(self, array: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    def test_is_none_when_the_face_is_too_small_to_measure(self) -> None:
        # Below this the patch is a handful of pixels and its spread is noise.
        # make_face puts the eyes 0.6 * width apart, so a 20px box is ~12px.
        face = make_face(bbox=(100, 80, 20, 20))
        assert face.interocular < MIN_INTEROCULAR_PX
        assert eye_openness(self.gray(textured_array(320, 240)), face) is None

    def test_a_flat_patch_reads_as_shut(self) -> None:
        """A uniform eye region has no pupil-against-sclera structure at all."""
        flat = np.full((240, 320, 3), 128, dtype=np.uint8)
        value = eye_openness(self.gray(flat), make_face())
        assert value is not None
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_a_detailed_patch_reads_as_open(self) -> None:
        value = eye_openness(self.gray(textured_array(320, 240)), make_face())
        assert value is not None
        assert value > 0.1

    def test_a_black_frame_is_unmeasurable_rather_than_shut(self) -> None:
        # Dividing by a near-zero mean would produce a meaningless ratio, so the
        # honest answer is "no reading", not "eyes closed".
        black = np.zeros((240, 320, 3), dtype=np.uint8)
        assert eye_openness(self.gray(black), make_face()) is None

    def test_sharpness_collapses_under_blur_but_a_flat_eye_barely_moves_it(self) -> None:
        """The whole basis of the blur guard, in miniature.

        Face-wide sharpness is what tells a blink apart from a blurred frame:
        closing the eyes touches a small part of the face, blurring touches all
        of it.
        """
        import cv2

        sharp = textured_array(320, 240)
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
        face = make_face()
        sharp_value = face_sharpness(self.gray(sharp), face)
        blurred_value = face_sharpness(self.gray(blurred), face)
        assert sharp_value is not None and blurred_value is not None
        assert blurred_value < sharp_value * DEFAULT_PRESENCE_THRESHOLDS.min_sharpness_ratio


class TestSamplePresence:
    def test_returns_none_without_a_face(self) -> None:
        analysis = ImageAnalysis(status=FaceStatus.NO_FACE, face_count=0)
        assert sample_presence(analysis, textured_array(320, 240)) is None

    def test_a_face_that_failed_quality_is_an_observation_not_a_gap(self) -> None:
        """Distinct from None: it breaks the run, but it is not a missing frame.

        Only genuinely absent faces count towards FACE_LOST, so someone sitting
        badly lit in front of the camera never gets told they walked away.
        """
        analysis = ImageAnalysis(status=FaceStatus.LOW_QUALITY, face_count=1, face=make_face())
        sample = sample_presence(analysis, textured_array(320, 240))
        assert sample is not None
        assert sample.usable is False

    def test_an_ok_analysis_carries_both_eye_measurements(self) -> None:
        analysis = ImageAnalysis(status=FaceStatus.OK, face_count=1, face=make_face())
        sample = sample_presence(analysis, textured_array(320, 240))
        assert sample is not None
        assert sample.usable is True
        assert sample.openness is not None
        assert sample.sharpness is not None


class TestCountsTowardsHold:
    def test_a_usable_confident_face_counts(self) -> None:
        assert counts_towards_hold(OPEN)

    def test_an_unusable_face_never_counts(self) -> None:
        # However confident the detector is, a frame the quality gate refused is
        # not a frame worth capturing.
        assert not counts_towards_hold(PresenceSample(usable=False, score=0.99))

    def test_a_low_confidence_detection_never_counts(self) -> None:
        assert not counts_towards_hold(PresenceSample(usable=True, score=0.2))

    def test_the_default_bar_matches_the_quality_gate(self) -> None:
        assert DEFAULT_PRESENCE_THRESHOLDS.min_detection_score == 0.75

    def test_thresholds_are_injectable(self) -> None:
        strict = PresenceThresholds(min_detection_score=0.99)
        assert counts_towards_hold(OPEN)
        assert not counts_towards_hold(OPEN, thresholds=strict)


class TestBlinkDetection:
    def make(self, frames: int = 3) -> LivenessSession:
        return LivenessSession(id="test", required_frames=frames)

    def test_a_held_face_alone_does_not_pass(self) -> None:
        """The point of the whole feature: a photograph gets this far and stops."""
        session = self.make(3)
        for _ in range(20):
            session.submit(OPEN, now=0.0)
        assert not session.passed
        assert session.state is SessionState.AWAITING_BLINK

    def test_close_then_open_passes(self) -> None:
        session = self.make(3)
        hold_then(session, SHUT, OPEN)
        assert session.passed

    def test_closing_alone_is_not_a_blink(self) -> None:
        # Eyes that shut and stay shut are a face that stopped looking, or a
        # photograph of someone mid-blink. The recovery is the evidence.
        session = self.make(3)
        hold_then(session, SHUT, SHUT, SHUT)
        assert not session.passed
        assert session.eyes_closed is True

    def test_a_shallow_dip_is_not_a_blink(self) -> None:
        # Noise wanders; an eyelid does not wander by a third.
        session = self.make(3)
        shallow = PresenceSample(usable=True, score=0.95, openness=0.9, sharpness=1.0)
        hold_then(session, shallow, OPEN, shallow, OPEN)
        assert not session.passed

    def test_recovery_needs_more_than_the_closing_threshold(self) -> None:
        """Hysteresis: without it a signal sitting on the line mints blinks."""
        session = self.make(3)
        on_the_line = PresenceSample(usable=True, score=0.95, openness=0.75, sharpness=1.0)
        hold_then(session, SHUT, on_the_line)
        assert not session.passed, "0.75 is above the close bar but below the reopen bar"
        session.submit(OPEN, now=0.0)
        assert session.passed

    def test_a_blurred_frame_cannot_fake_a_blink(self) -> None:
        """The false-accept this design exists to close.

        Blur suppresses the same eye detail a closed lid does, so without the
        sharpness guard someone could shake a printed photo and pass.
        """
        session = self.make(3)
        hold_then(session, BLURRED, OPEN, BLURRED, OPEN)
        assert not session.passed
        assert session.eyes_closed is False

    def test_a_frame_with_no_sharpness_reading_is_not_trusted(self) -> None:
        # If the guard cannot run, the reading it guards is not used.
        session = self.make(3)
        unguarded = PresenceSample(usable=True, score=0.95, openness=0.1, sharpness=None)
        hold_then(session, unguarded, OPEN)
        assert not session.passed

    def test_the_baseline_follows_worsening_light(self) -> None:
        """A rolling median, so a darkening room is not read as a blink."""
        session = self.make(3)
        for _ in range(3):
            session.submit(OPEN, now=0.0)
        # A long slow decline: every step is small relative to the one before,
        # so the decayed baseline keeps pace and never registers a dip.
        value = 1.0
        for _ in range(60):
            value *= 0.995
            session.submit(
                PresenceSample(usable=True, score=0.95, openness=value, sharpness=1.0),
                now=0.0,
            )
        assert not session.blinked

    def test_one_bright_frame_does_not_redefine_what_open_means(self) -> None:
        """The reference is the middle of the open readings, not the best of them.

        Under the old peak-hold this single 1.4 became the reference, and every
        ordinary frame afterwards sat below 0.75 of it — so the check announced
        a blink the subject never made.
        """
        session = self.make(3)
        for value in (1.0, 1.0, 1.0, 1.4, 1.0, 0.95, 1.05, 1.0):
            session.submit(eye(value), now=0.0)
        assert not session.eyes_closed
        assert not session.blinked

    def test_a_real_blink_completes_against_a_noisy_open_signal(self) -> None:
        """The other half of the same fault: blinks that stuck shut.

        A peak-inflated reference put the reopen bar above what an ordinary open
        eye reads, so a genuine blink closed and then never completed.
        """
        session = self.make(3)
        for value in (0.95, 1.05, 1.0, 1.3, 0.98, 1.02):
            session.submit(eye(value), now=0.0)
        session.submit(eye(0.5), now=0.0)
        assert session.eyes_closed, "a half-value dip is a closure by any reading"
        session.submit(eye(0.98), now=0.0)
        assert session.blinked, "an ordinary open frame must be enough to reopen"

    def test_a_broken_run_discards_the_baseline(self) -> None:
        """A new face in new light must not be judged against the old reference.

        Reusing it is how a change of scene alone produces a "blink".
        """
        session = self.make(3)
        for _ in range(3):
            session.submit(OPEN, now=0.0)
        assert session.baseline is not None
        for _ in range(SKIP_TOLERANCE):
            session.submit(None, now=0.0)
        assert session.baseline is None
        assert session.sharpness_baseline is None


class TestSessionProgression:
    def make(self, frames: int = 3) -> LivenessSession:
        return LivenessSession(id="test", required_frames=frames)

    def test_starts_by_waiting_for_a_face(self) -> None:
        session = self.make()
        assert session.state is SessionState.AWAITING_FACE
        assert session.progress == (0, 4)

    def test_the_blink_is_counted_as_the_final_step(self) -> None:
        # Otherwise the bar sits visibly full while the check is still waiting.
        session = self.make(3)
        for _ in range(3):
            session.submit(OPEN, now=0.0)
        assert session.progress == (3, 4)
        session.submit(SHUT, now=0.0)
        session.submit(OPEN, now=0.0)
        assert session.progress == (4, 4)

    def test_it_reports_holding_part_way_through(self) -> None:
        session = self.make(3)
        session.submit(OPEN, now=0.0)
        assert session.state is SessionState.HOLDING

    def test_a_brief_wobble_does_not_reset_the_run(self) -> None:
        # An ordinary webcam drops frames and blurs on motion. Resetting on the
        # first bad one would make a hold unachievable in practice.
        session = self.make(3)
        session.submit(OPEN, now=0.0)
        session.submit(PresenceSample(usable=False, score=0.9), now=1.0)
        assert session.progress == (1, 4)
        session.submit(OPEN, now=2.0)
        session.submit(OPEN, now=3.0)
        session.submit(SHUT, now=4.0)
        session.submit(OPEN, now=5.0)
        assert session.passed

    def test_a_sustained_bad_patch_resets_the_run(self) -> None:
        session = self.make(3)
        session.submit(OPEN, now=0.0)
        for _ in range(SKIP_TOLERANCE):
            session.submit(PresenceSample(usable=False, score=0.9), now=1.0)
        assert session.progress == (0, 4)
        assert session.state is SessionState.AWAITING_FACE
        assert not session.finished


class TestSessionFailure:
    def test_a_missing_blink_fails_with_its_own_reason(self) -> None:
        # Distinct from a timeout, so the screen can say what to do about it.
        session = LivenessSession(id="t", required_frames=2, blink_timeout=5.0)
        session.submit(OPEN, now=0.0)
        session.submit(OPEN, now=0.0)
        assert session.state is SessionState.AWAITING_BLINK
        session.submit(OPEN, now=99.0)
        assert session.state is SessionState.FAILED
        assert session.failure is FailureReason.BLINK_NOT_SEEN

    def test_the_blink_clock_starts_when_the_blink_is_asked_for(self) -> None:
        # Not at session creation: time spent finding the camera is not time
        # spent refusing to blink.
        session = LivenessSession(id="t", required_frames=3, blink_timeout=5.0)
        session.submit(OPEN, now=100.0)
        session.submit(OPEN, now=100.0)
        session.submit(OPEN, now=100.0)
        session.submit(SHUT, now=103.0)
        session.submit(OPEN, now=104.0)
        assert session.passed

    def test_the_blink_timeout_fires_even_when_the_clock_starts_at_zero(self) -> None:
        """Regression: `blink_armed_at or now` would treat 0.0 as unset.

        The clock would silently restart on every frame and the timeout could
        never fire. This file carried exactly that bug once before.
        """
        session = LivenessSession(id="t", required_frames=1, blink_timeout=5.0)
        session.submit(OPEN, now=0.0)
        assert session.blink_armed_at == 0.0
        session.submit(OPEN, now=99.0)
        assert session.failure is FailureReason.BLINK_NOT_SEEN

    def test_session_ttl_expiry_fails(self) -> None:
        # TTL runs from the FIRST frame, not from construction, so the clock the
        # caller passes is the only one involved.
        session = LivenessSession(id="t", required_frames=3, ttl=10.0)
        session.submit(OPEN, now=0.0)
        session.submit(OPEN, now=50.0)
        assert session.state is SessionState.FAILED
        assert session.failure is FailureReason.SESSION_EXPIRED

    def test_created_at_is_stamped_from_the_callers_clock(self) -> None:
        """Regression: created_at used to default to real time.monotonic().

        Mixed with an injected `now` that made every elapsed comparison
        negative, so the TTL branch could never fire at all.
        """
        session = LivenessSession(id="t", required_frames=3)
        assert session.created_at is None
        session.submit(OPEN, now=1234.0)
        assert session.created_at == 1234.0

    def test_a_few_missing_frames_are_tolerated(self) -> None:
        session = LivenessSession(id="t", required_frames=3)
        for _ in range(5):
            session.submit(None, now=1.0)
        assert not session.finished
        hold_then(session, SHUT, OPEN)
        assert session.passed

    def test_a_face_that_stays_gone_fails(self) -> None:
        session = LivenessSession(id="t", required_frames=3)
        for _ in range(MAX_CONSECUTIVE_MISSING):
            session.submit(None, now=1.0)
        assert session.state is SessionState.FAILED
        assert session.failure is FailureReason.FACE_LOST

    def test_an_unusable_face_is_never_face_lost(self) -> None:
        """Someone badly lit is present. Telling them they left is wrong."""
        session = LivenessSession(id="t", required_frames=3)
        for _ in range(MAX_CONSECUTIVE_MISSING * 2):
            session.submit(PresenceSample(usable=False, score=0.9), now=1.0)
        assert not session.finished

    def test_missing_frame_counter_resets_on_a_seen_face(self) -> None:
        session = LivenessSession(id="t", required_frames=3)
        for _ in range(MAX_CONSECUTIVE_MISSING - 1):
            session.submit(None, now=1.0)
        session.submit(OPEN, now=2.0)
        for _ in range(MAX_CONSECUTIVE_MISSING - 1):
            session.submit(None, now=3.0)
        assert not session.finished

    def test_a_finished_session_ignores_further_frames(self) -> None:
        session = LivenessSession(id="t", required_frames=3)
        hold_then(session, SHUT, OPEN)
        assert session.passed
        for _ in range(MAX_CONSECUTIVE_MISSING):
            session.submit(None, now=1.0)
        assert session.passed, "a passed session must not be re-openable by later frames"


class TestSessionStore:
    def test_creates_sessions_with_distinct_unguessable_ids(self) -> None:
        store = SessionStore()
        ids = {store.create().id for _ in range(20)}
        assert len(ids) == 20
        assert all(len(i) > 16 for i in ids)

    def test_get_returns_the_same_session(self) -> None:
        store = SessionStore()
        created = store.create()
        assert store.get(created.id) is created

    def test_unknown_id_returns_none(self) -> None:
        assert SessionStore().get("nope") is None

    def test_discard_removes(self) -> None:
        store = SessionStore()
        created = store.create()
        store.discard(created.id)
        assert store.get(created.id) is None

    def test_rejects_an_impossible_hold(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            SessionStore().create(required_frames=0)

    def test_concurrency_ceiling_is_enforced(self) -> None:
        store = SessionStore(max_sessions=2)
        store.create()
        store.create()
        with pytest.raises(RuntimeError, match="too many"):
            store.create()

    def test_expired_sessions_are_swept(self) -> None:
        store = SessionStore(ttl=0.0)
        created = store.create()
        assert store.get(created.id) is None
        assert len(store) == 0
