"""Liveness geometry and the session state machine.

Hermetic: pose samples are constructed directly, so every branch is reachable
without a camera or a cooperative human.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from app.engine.liveness import (
    DEFAULT_LIVENESS_THRESHOLDS,
    POSE_CHALLENGES,
    ChallengeKind,
    LivenessThresholds,
    PoseSample,
    random_sequence,
    sample_pose,
    satisfies,
    signed_yaw,
)
from app.engine.pipeline import FaceStatus, ImageAnalysis
from app.services.liveness_session import (
    FailureReason,
    LivenessSession,
    SessionState,
    SessionStore,
)
from tests.factories.images import make_face

NEUTRAL = PoseSample(yaw=0.0, face_area=5000.0, score=0.95)


def pose(yaw: float, area: float = 5000.0, score: float = 0.95) -> PoseSample:
    return PoseSample(yaw=yaw, face_area=area, score=score)


class TestSignedYaw:
    def test_frontal_face_is_near_zero(self) -> None:
        assert abs(signed_yaw(make_face(yaw_offset=0.0))) < 0.02

    def test_sign_distinguishes_direction(self) -> None:
        """The whole point: `yaw_symmetry` discards this, a challenge needs it."""
        right = signed_yaw(make_face(yaw_offset=0.4))
        left = signed_yaw(make_face(yaw_offset=-0.4))
        assert right > 0 > left
        assert right == pytest.approx(-left, abs=1e-5)

    def test_is_bounded(self) -> None:
        for offset in (-3.0, -1.0, 0.0, 1.0, 3.0):
            assert -1.0 <= signed_yaw(make_face(yaw_offset=offset)) <= 1.0

    def test_magnitude_grows_with_turn(self) -> None:
        values = [abs(signed_yaw(make_face(yaw_offset=o))) for o in (0.0, 0.1, 0.25, 0.5)]
        assert values == sorted(values)

    def test_real_portrait_range_is_separated_from_the_target(self) -> None:
        # Measured: frontal portraits fall within +-0.11, turned heads reach
        # +0.60 and -0.35. The target must sit between those bands.
        assert 0.11 < DEFAULT_LIVENESS_THRESHOLDS.yaw_target < 0.35


class TestSamplePose:
    def test_returns_none_without_a_face(self) -> None:
        assert sample_pose(ImageAnalysis(status=FaceStatus.NO_FACE, face_count=0)) is None

    def test_uses_the_face_even_when_quality_failed(self) -> None:
        """A challenge asks for a turned head, which the quality gate rejects.

        Gating liveness on quality would make the challenge unpassable by
        construction.
        """
        analysis = ImageAnalysis(
            status=FaceStatus.LOW_QUALITY, face_count=1, face=make_face(yaw_offset=0.5)
        )
        sample = sample_pose(analysis)
        assert sample is not None and sample.yaw > 0.3

    def test_accepts_a_normal_face(self) -> None:
        analysis = ImageAnalysis(status=FaceStatus.OK, face_count=1, face=make_face())
        sample = sample_pose(analysis)
        assert sample is not None and sample.face_area > 0


class TestSatisfies:
    def test_turn_left_needs_positive_yaw(self) -> None:
        assert satisfies(ChallengeKind.TURN_LEFT, pose(0.35))
        assert not satisfies(ChallengeKind.TURN_LEFT, pose(-0.35))

    def test_turn_right_needs_negative_yaw(self) -> None:
        assert satisfies(ChallengeKind.TURN_RIGHT, pose(-0.35))
        assert not satisfies(ChallengeKind.TURN_RIGHT, pose(0.35))

    def test_frontal_satisfies_neither_direction(self) -> None:
        for challenge in POSE_CHALLENGES:
            assert not satisfies(challenge, NEUTRAL)

    def test_a_low_confidence_detection_never_satisfies(self) -> None:
        # A barely-detected face is not evidence of anything.
        assert not satisfies(ChallengeKind.TURN_LEFT, pose(0.9, score=0.2))

    def test_move_closer_is_relative_to_the_baseline(self) -> None:
        # Framing varies far too much between cameras for an absolute size bar.
        assert satisfies(ChallengeKind.MOVE_CLOSER, pose(0.0, area=10000), baseline_area=5000)
        assert not satisfies(ChallengeKind.MOVE_CLOSER, pose(0.0, area=5500), baseline_area=5000)

    def test_move_closer_without_a_baseline_cannot_pass(self) -> None:
        assert not satisfies(ChallengeKind.MOVE_CLOSER, pose(0.0, area=99999), baseline_area=None)

    def test_thresholds_are_injectable(self) -> None:
        strict = LivenessThresholds(yaw_target=0.8)
        assert satisfies(ChallengeKind.TURN_LEFT, pose(0.35))
        assert not satisfies(ChallengeKind.TURN_LEFT, pose(0.35), thresholds=strict)


class TestRandomSequence:
    def test_has_the_requested_length(self) -> None:
        assert len(random_sequence(4)) == 4

    def test_never_repeats_back_to_back(self) -> None:
        """Two identical challenges in a row are satisfiable by holding one pose."""
        for _ in range(200):
            sequence = random_sequence(5)
            assert all(a is not b for a, b in pairwise(sequence))

    def test_always_ends_on_a_pose_challenge(self) -> None:
        # MOVE_CLOSER last would leave the subject too close for a good capture.
        for _ in range(200):
            assert random_sequence(3)[-1] in POSE_CHALLENGES

    def test_order_actually_varies(self) -> None:
        # A predictable sequence is exactly what a replay attack needs.
        assert len({random_sequence(3) for _ in range(200)}) > 1

    def test_rejects_an_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            random_sequence(0)


class TestSessionProgression:
    def make(self, *challenges: ChallengeKind) -> LivenessSession:
        return LivenessSession(id="test", sequence=challenges)

    def test_starts_by_requiring_a_neutral_face(self) -> None:
        session = self.make(ChallengeKind.TURN_LEFT)
        assert session.state is SessionState.AWAITING_NEUTRAL

    def test_neutral_then_challenge_passes(self) -> None:
        session = self.make(ChallengeKind.TURN_LEFT)
        session.submit(NEUTRAL, now=0.0)
        assert session.state is SessionState.AWAITING_CHALLENGE
        session.submit(pose(0.4), now=1.0)
        assert session.passed

    def test_a_sustained_turn_cannot_satisfy_two_challenges(self) -> None:
        """The reason a neutral frame is required between challenges.

        Without it, one slow sweep from left to right would satisfy "turn left"
        then "turn right" without the subject ever responding to the prompts.
        """
        session = self.make(ChallengeKind.TURN_LEFT, ChallengeKind.TURN_RIGHT)
        session.submit(NEUTRAL, now=0.0)
        session.submit(pose(0.4), now=1.0)
        assert session.progress == (1, 2)
        assert session.state is SessionState.AWAITING_NEUTRAL
        # Still turned: the next challenge does not even begin.
        session.submit(pose(-0.4), now=2.0)
        assert session.progress == (1, 2)
        assert not session.finished

    def test_returning_to_neutral_arms_the_next_challenge(self) -> None:
        session = self.make(ChallengeKind.TURN_LEFT, ChallengeKind.TURN_RIGHT)
        session.submit(NEUTRAL, now=0.0)
        session.submit(pose(0.4), now=1.0)
        session.submit(NEUTRAL, now=2.0)
        session.submit(pose(-0.4), now=3.0)
        assert session.passed
        assert session.progress == (2, 2)

    def test_wrong_direction_does_not_advance(self) -> None:
        session = self.make(ChallengeKind.TURN_LEFT)
        session.submit(NEUTRAL, now=0.0)
        session.submit(pose(-0.5), now=1.0)
        assert not session.finished
        assert session.progress == (0, 1)

    def test_baseline_is_captured_at_the_neutral_frame(self) -> None:
        session = self.make(ChallengeKind.MOVE_CLOSER)
        session.submit(pose(0.0, area=4000), now=0.0)
        assert session.baseline_area == pytest.approx(4000)
        session.submit(pose(0.0, area=6000), now=1.0)
        assert session.passed


class TestSessionFailure:
    def test_challenge_timeout_fails_the_session(self) -> None:
        session = LivenessSession(
            id="t", sequence=(ChallengeKind.TURN_LEFT,), challenge_timeout=5.0
        )
        session.submit(NEUTRAL, now=0.0)
        session.submit(pose(0.4), now=6.0)
        assert session.state is SessionState.FAILED
        assert session.failure is FailureReason.CHALLENGE_TIMEOUT

    def test_the_timeout_clock_starts_at_the_neutral_frame(self) -> None:
        # Not at session creation: time spent finding the camera is not the
        # subject failing a challenge.
        session = LivenessSession(
            id="t", sequence=(ChallengeKind.TURN_LEFT,), challenge_timeout=5.0
        )
        session.submit(NEUTRAL, now=100.0)
        session.submit(pose(0.4), now=103.0)
        assert session.passed

    def test_session_ttl_expiry_fails(self) -> None:
        # TTL runs from the FIRST frame, not from construction, so the clock the
        # caller passes is the only one involved.
        session = LivenessSession(id="t", sequence=(ChallengeKind.TURN_LEFT,), ttl=10.0)
        session.submit(NEUTRAL, now=0.0)
        session.submit(NEUTRAL, now=50.0)
        assert session.state is SessionState.FAILED
        assert session.failure is FailureReason.SESSION_EXPIRED

    def test_created_at_is_stamped_from_the_callers_clock(self) -> None:
        """Regression: created_at used to default to real time.monotonic().

        Mixed with an injected `now` that made every elapsed comparison
        negative, so the TTL branch could never fire at all.
        """
        session = LivenessSession(id="t", sequence=(ChallengeKind.TURN_LEFT,))
        assert session.created_at is None
        session.submit(NEUTRAL, now=1234.0)
        assert session.created_at == 1234.0

    def test_timeout_fires_even_when_the_clock_starts_at_zero(self) -> None:
        """Regression: `challenge_started_at or now` treated 0.0 as unset.

        The timeout clock silently restarted on every frame, so a challenge
        could never time out.
        """
        session = LivenessSession(
            id="t", sequence=(ChallengeKind.TURN_LEFT,), challenge_timeout=5.0
        )
        session.submit(NEUTRAL, now=0.0)
        assert session.challenge_started_at == 0.0
        session.submit(pose(0.4), now=99.0)
        assert session.failure is FailureReason.CHALLENGE_TIMEOUT

    def test_a_few_missing_frames_are_tolerated(self) -> None:
        session = LivenessSession(id="t", sequence=(ChallengeKind.TURN_LEFT,))
        session.submit(NEUTRAL, now=0.0)
        for _ in range(5):
            session.submit(None, now=1.0)
        assert not session.finished
        session.submit(pose(0.4), now=2.0)
        assert session.passed

    def test_a_face_that_stays_gone_fails(self) -> None:
        # Someone walked away — or a photo is being swapped for another photo.
        session = LivenessSession(id="t", sequence=(ChallengeKind.TURN_LEFT,))
        session.submit(NEUTRAL, now=0.0)
        for _ in range(30):
            session.submit(None, now=1.0)
        assert session.state is SessionState.FAILED
        assert session.failure is FailureReason.FACE_LOST

    def test_missing_frame_counter_resets_on_a_good_frame(self) -> None:
        session = LivenessSession(id="t", sequence=(ChallengeKind.TURN_LEFT,))
        session.submit(NEUTRAL, now=0.0)
        for _ in range(29):
            session.submit(None, now=1.0)
        session.submit(pose(0.0), now=2.0)
        for _ in range(29):
            session.submit(None, now=3.0)
        assert not session.finished

    def test_a_finished_session_ignores_further_frames(self) -> None:
        session = LivenessSession(id="t", sequence=(ChallengeKind.TURN_LEFT,))
        session.submit(NEUTRAL, now=0.0)
        session.submit(pose(0.4), now=1.0)
        assert session.passed
        session.submit(pose(-0.9), now=2.0)
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
