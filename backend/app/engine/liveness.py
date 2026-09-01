"""Active challenge-response liveness.

The problem this exists for: a printed photograph or a phone screen held to the
camera verifies exactly like a live person. Live capture without liveness is
arguably WORSE than an upload, because it creates the impression that someone
was seen in person while guaranteeing nothing.

The approach is deliberately geometric rather than learned. The service asks for
a randomised sequence of head movements and checks they happen, in order, within
a time budget. That is a deterministic measurement — "did signed yaw exceed this
value in the requested direction" — with no score to calibrate. A learned
passive anti-spoof model would need spoof samples to set its threshold, and
shipping one with a guessed threshold repeats the mistake this project has been
careful to avoid everywhere else.

WHAT THIS DEFEATS: a printed photo, a static image on a screen, a still held up
to the lens.

WHAT IT DOES NOT DEFEAT: a pre-recorded video replay that happens to contain the
requested motions, or a live deepfake driven in real time. Randomising the
sequence per session raises the cost of a replay attack but does not close it.
Closing it needs passive texture/depth analysis, which is complementary to this
and explicitly not included. That limitation is stated in the README rather than
left to be discovered.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import StrEnum

from app.engine.pipeline import ImageAnalysis
from app.engine.types import LM_EYE_LEFT, LM_EYE_RIGHT, LM_NOSE, DetectedFace


class ChallengeKind(StrEnum):
    """Named from the SUBJECT's point of view — what the person is told to do.

    The camera sees left and right mirrored, so the mapping onto a measured
    signed yaw lives in exactly one place (`_YAW_SIGN`) rather than being
    re-derived at each call site. Getting it backwards is the same class of
    silent bug as mirrored landmarks: nothing raises, the challenge simply
    becomes unpassable.
    """

    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    MOVE_CLOSER = "MOVE_CLOSER"


# A subject turning their head to their own left rotates their nose toward the
# camera's right, so signed yaw goes positive. Verified against the measured
# spread of real portraits rather than assumed from geometry alone.
_YAW_SIGN: dict[ChallengeKind, float] = {
    ChallengeKind.TURN_LEFT: +1.0,
    ChallengeKind.TURN_RIGHT: -1.0,
}

POSE_CHALLENGES = (ChallengeKind.TURN_LEFT, ChallengeKind.TURN_RIGHT)


@dataclass(frozen=True, slots=True)
class LivenessThresholds:
    # Frontal portraits measure within +-0.11 signed yaw; genuinely turned heads
    # reach +0.60 and -0.35. 0.30 clears the frontal band with margin while
    # staying comfortably reachable.
    yaw_target: float = 0.30
    # A face must return near-frontal between challenges, so one sustained turn
    # cannot satisfy a "left then right" sequence by drifting through it.
    yaw_neutral: float = 0.12
    # MOVE_CLOSER is relative to where the subject started, not an absolute size:
    # framing varies far too much between cameras for an absolute bar.
    closer_growth: float = 1.30
    min_detection_score: float = 0.75


DEFAULT_LIVENESS_THRESHOLDS = LivenessThresholds()


def signed_yaw(face: DetectedFace) -> float:
    """Head yaw in [-1, 1]. Positive means the nose sits toward IMAGE-right.

    The quality gate's `yaw_symmetry` deliberately discards direction, since it
    only cares how far from frontal a face is. A challenge has to know which way
    the head turned, so the signed form is computed here rather than widening
    that metric and disturbing its tested behaviour.
    """
    landmarks = face.landmarks
    to_left_eye = float(landmarks[LM_NOSE][0] - landmarks[LM_EYE_LEFT][0])
    to_right_eye = float(landmarks[LM_EYE_RIGHT][0] - landmarks[LM_NOSE][0])
    span = to_left_eye + to_right_eye
    if abs(span) < 1e-6:
        return 0.0
    return max(-1.0, min(1.0, (to_left_eye - to_right_eye) / span))


@dataclass(frozen=True, slots=True)
class PoseSample:
    """What one frame contributes to a liveness judgement."""

    yaw: float
    # Bounding-box area in PIXELS, not a fraction of the frame. Within a single
    # session the camera resolution does not change, so comparing pixel area
    # against the baseline is exactly equivalent to comparing ratios — and it
    # removes any need to thread frame dimensions through the pipeline.
    face_area: float
    score: float

    @property
    def is_neutral(self) -> bool:
        return abs(self.yaw) <= DEFAULT_LIVENESS_THRESHOLDS.yaw_neutral


def sample_pose(analysis: ImageAnalysis) -> PoseSample | None:
    """Extract pose from an analysed frame, or None if there is no usable face.

    Uses the detected face regardless of whether the QUALITY gate passed: a
    liveness challenge asks the subject to turn away from frontal, which the
    pose check is designed to reject. Gating liveness on quality would make the
    challenge unpassable by construction.
    """
    face = analysis.face
    if face is None or face.bbox.area <= 0:
        return None
    return PoseSample(
        yaw=signed_yaw(face),
        face_area=float(face.bbox.area),
        score=face.score,
    )


def satisfies(
    challenge: ChallengeKind,
    sample: PoseSample,
    *,
    baseline_area: float | None = None,
    thresholds: LivenessThresholds = DEFAULT_LIVENESS_THRESHOLDS,
) -> bool:
    """Has this frame met the challenge?

    `baseline_area` is the face size when the challenge was issued, which
    MOVE_CLOSER is measured against.
    """
    if sample.score < thresholds.min_detection_score:
        return False

    if challenge is ChallengeKind.MOVE_CLOSER:
        if baseline_area is None or baseline_area <= 0:
            return False
        return sample.face_area >= baseline_area * thresholds.closer_growth

    sign = _YAW_SIGN[challenge]
    return sample.yaw * sign >= thresholds.yaw_target


def random_sequence(length: int = 3) -> tuple[ChallengeKind, ...]:
    """A randomised challenge order, drawn per session.

    Randomising is what makes a recorded video replay costly: the attacker would
    need footage of the right movements in the right order. `secrets` rather than
    `random` because a predictable sequence is exactly what defeats the point.
    """
    if length < 1:
        raise ValueError("a liveness sequence needs at least one challenge")

    sequence: list[ChallengeKind] = []
    previous: ChallengeKind | None = None
    for index in range(length):
        # Never repeat back-to-back: two identical challenges in a row can be
        # satisfied by holding one pose across both.
        choices = [c for c in ChallengeKind if c is not previous]
        # Always finish on a pose challenge; MOVE_CLOSER as the last step leaves
        # the subject too close for a good capture frame.
        if index == length - 1:
            choices = [c for c in choices if c in POSE_CHALLENGES]
        chosen = choices[secrets.randbelow(len(choices))]
        sequence.append(chosen)
        previous = chosen
    return tuple(sequence)
