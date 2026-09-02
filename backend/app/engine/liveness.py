"""Blink-gated presence check for live capture.

WHAT THIS IS: a gate that waits until the camera has held a USABLE face — one
the detector is confident about and that clears the pipeline's own quality bar —
and has then seen that face BLINK. Only after both does it let the caller grab
the still.

WHAT IT DEFEATS: a printed photograph, a still on a phone or laptop screen, an
AI-generated portrait. None of them can close and reopen an eye.

WHAT IT DOES NOT DEFEAT: a recorded video of the enrolled person blinking, or a
live deepfake driven in real time — both blink perfectly well. Closing that gap
needs passive texture or depth analysis, which is a learned model whose
threshold has to be calibrated against real spoof samples. Out of scope here,
and stated plainly rather than left to be discovered.

HOW THE EYE IS MEASURED, and why it is done this way: YuNet gives exactly ONE
landmark per eye, so the standard Eye Aspect Ratio — which needs roughly six
contour points per eye — is not available. Instead a patch is cropped around
each eye landmark, sized from the interocular distance, and its normalised
intensity spread is measured. An open eye puts a dark pupil beside bright sclera
inside that patch and the spread is high; a closed lid is comparatively smooth
skin and the spread falls.

The absolute value of that number means nothing across people, cameras and
lighting, so it is never compared against a fixed bar. Each session builds its
own baseline from the frames it has already seen — the MEDIAN of recent open
readings, so the reference describes a typical open eye rather than the best
frame luck supplied — and looks for a transient DIP below it. That is the same
reasoning the rest of this codebase applies to thresholds: a relative
measurement that calibrates itself beats an absolute one guessed in advance.

THE FALSE-BLINK VECTOR, and the guard against it. Motion blur suppresses exactly
the high-frequency eye detail this metric reads, so a blurred frame looks like a
closed eye — which would let someone mint a "blink" by shaking a printed photo,
and would also fire on ordinary camera shake. Measured across the 11 test
portraits: blurring the whole frame drops the eye metric to 0.57-0.83 of its open
value, straddling the closed threshold. Normalising by the rest of the face does
not rescue it (0.64 worst), because blur costs the eye region proportionally more
than the flatter areas around it.

What does separate them is face-wide sharpness. Shutting the eyes leaves it at
0.66-1.01 of baseline; blurring the frame collapses it to 0.01. So eye readings
are simply DISCARDED on any frame whose face sharpness has fallen well below the
session's own baseline. A blink never trips that gate; blur always does.

CALIBRATION STATUS: the direction of the signal is verified — shut eyes produce
an unambiguous collapse. The exact dip ratio is NOT verified, because there are
no closed-eye samples in the test set; the numbers below are reasoned, not
measured, and are the first thing a labelled blink set should replace. See
README calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.engine.pipeline import ImageAnalysis
from app.engine.types import LM_EYE_LEFT, LM_EYE_RIGHT, DetectedFace

# The preview socket is ack-paced and settles somewhere between 5 and 30 FPS
# depending on the hardware, so this is roughly one to two seconds of a steady
# face. Counted in FRAMES rather than seconds because it is the RUN that carries
# the meaning, and because these same frames are what the eye baseline is built
# from — a baseline needs samples, not elapsed time.
DEFAULT_REQUIRED_FRAMES = 10

# The eye patch, as fractions of interocular distance. WIDE AND SHORT, because
# that is the shape of an eye: the aperture is roughly 0.35 x 0.12 of the
# interocular distance, and everything else in the neighbourhood — brow, socket,
# cheek — looks identical open or shut and only dilutes the measurement.
#
# Sized from measurement, not taste. A square 0.28 patch leaves the aperture at
# about 13% of its area, and across the 11 test portraits a realistic closure
# then moves the metric only to 0.63-0.95 of its open value: most faces never
# reach the closed threshold at all, and the check silently never fires. At
# 0.18 x 0.09 the same closure lands at 0.27-0.63 — every face clears it.
EYE_PATCH_WIDTH_SCALE = 0.18
EYE_PATCH_HEIGHT_SCALE = 0.09

# Below this the patch is a handful of pixels and its spread is noise. The
# quality gate's min_interocular_px is 32, so in practice this only fires for
# faces it has already rejected.
MIN_INTEROCULAR_PX = 24.0


@dataclass(frozen=True, slots=True)
class PresenceThresholds:
    # The same bar the quality gate already applies, kept injectable so live
    # capture can be made stricter than a single-photo upload without disturbing
    # the tested behaviour of `assess_quality`.
    min_detection_score: float = 0.75

    # A blink is a dip to this fraction of the session's own baseline.
    #
    # A FULL closure measures 0.27-0.63 across the test portraits, so this sits
    # above the worst of those with room to spare. The margin is deliberate: a
    # blink lasts 100-400ms and the socket samples at 5-30 FPS, so the frame
    # that lands may catch the lid halfway rather than shut, and a half-closed
    # eye dips less than a shut one.
    #
    # PROVISIONAL, and loose rather than tight on purpose — the failure this
    # setting is recovering from was a check that never fired. What it cannot be
    # set from is still-image data: the number that matters is how far an OPEN
    # eye wanders frame to frame, and that needs video of real faces. Tighten it
    # once that exists.
    eyes_closed_ratio: float = 0.75

    # Recovery is deliberately a HIGHER bar than the close, so a signal hovering
    # exactly on the threshold cannot rattle between open and closed and mint
    # blinks out of noise. Standard hysteresis.
    eyes_open_ratio: float = 0.85

    # Eye readings are discarded below this fraction of the session's peak face
    # sharpness, measured across the 11 test portraits:
    #
    #     shutting the eyes   0.822 - 1.018 of the open frame
    #     blurring the frame  0.0008 - 0.0098 of the open frame
    #
    # An 84x gap, and the bar belongs in the middle of it rather than hugging
    # one end. At 0.50 the worst blink cleared by only 1.64x — and the reference
    # is a PEAK over noisy Laplacian variance, which sits above the typical
    # frame, so that margin is spent before a subject blinks at all and the
    # frame carrying the blink is thrown away as motion blur.
    #
    # 0.25 leaves a real blink 3.3x of headroom for peak inflation while still
    # rejecting the worst blurred frame by 25x. Widened on the side that was
    # costing detections, using the side that had margin to spare.
    min_sharpness_ratio: float = 0.25

    # Decay on the SHARPNESS peak only. Without it a single crisp frame would
    # set an unreachable reference and every later frame would be discarded as
    # blurred; with it the guard follows a subject moving into worse light.
    #
    # The eye baseline does not use this: it is a rolling median of recent open
    # readings, because a peak sits at the top of the noise band rather than its
    # centre and made the check fire on noise while stalling on real blinks.
    # See LivenessSession.baseline.
    baseline_decay: float = 0.99


DEFAULT_PRESENCE_THRESHOLDS = PresenceThresholds()


@dataclass(frozen=True, slots=True)
class PresenceSample:
    """What one frame contributes to a presence judgement."""

    # The pipeline's whole verdict for the frame: a face was found, the
    # selection policy resolved it, and it passed quality. Deliberately not
    # re-derived from individual metrics here — one gate, one place.
    usable: bool
    score: float
    # None when the eyes could not be measured at all (face too small, patch off
    # the edge of the frame). Distinct from a low value, which means measured
    # and closed.
    openness: float | None = None
    # Face-wide sharpness, used only to decide whether `openness` is trustworthy
    # this frame. Never a pass/fail input in its own right — the quality gate
    # already owns that judgement.
    sharpness: float | None = None


def _eye_patch_spread(
    gray: np.ndarray, centre: np.ndarray, half_width: int, half_height: int
) -> float | None:
    """Normalised intensity spread of one eye patch, or None if unmeasurable."""
    height, width = gray.shape[:2]
    x, y = round(float(centre[0])), round(float(centre[1]))
    left, right = max(0, x - half_width), min(width, x + half_width + 1)
    top, bottom = max(0, y - half_height), min(height, y + half_height + 1)
    # A patch clipped to nothing is off the edge of the frame entirely.
    if right - left < 3 or bottom - top < 3:
        return None

    patch = gray[top:bottom, left:right]
    mean = float(patch.mean())
    # Dividing by the patch's own mean is what makes this comparable between a
    # brightly and a dimly lit face. A near-black patch has no usable ratio.
    if mean < 1.0:
        return None
    return float(patch.std()) / mean


def face_sharpness(gray: np.ndarray, face: DetectedFace) -> float | None:
    """Laplacian energy over the face box, normalised for brightness.

    Only ever compared against this session's own earlier frames. The absolute
    value is as content-dependent as the quality gate's sharpness metric — which
    is precisely why that one is documented as uncalibrated and effectively off.
    """
    height, width = gray.shape[:2]
    bbox = face.bbox
    left, right = max(0, bbox.x), min(width, bbox.right)
    top, bottom = max(0, bbox.y), min(height, bbox.bottom)
    if right - left < 3 or bottom - top < 3:
        return None

    patch = gray[top:bottom, left:right]
    mean = float(patch.mean())
    if mean < 1.0:
        return None
    # Squared because Laplacian variance scales with the square of contrast, so
    # dividing by the mean once would leave brightness in the answer.
    return float(cv2.Laplacian(patch, cv2.CV_64F).var()) / (mean * mean)


def eye_openness(gray: np.ndarray, face: DetectedFace) -> float | None:
    """How open both eyes look, in arbitrary units meaningful only within a session.

    Returns the MEAN of the two eyes: a natural blink closes both, and averaging
    is steadier than either eye alone when one is partly turned away. The
    consequence, stated so it is not discovered later, is that a deliberate
    one-eyed wink produces roughly half the dip and may not register.
    """
    if face.interocular < MIN_INTEROCULAR_PX:
        return None

    half_width = max(3, round(face.interocular * EYE_PATCH_WIDTH_SCALE))
    half_height = max(2, round(face.interocular * EYE_PATCH_HEIGHT_SCALE))
    spreads = [
        _eye_patch_spread(gray, face.landmarks[index], half_width, half_height)
        for index in (LM_EYE_LEFT, LM_EYE_RIGHT)
    ]
    measured = [value for value in spreads if value is not None]
    # Both eyes or nothing: one eye alone changes what the number means, and a
    # baseline built from two eyes cannot be compared against one.
    if len(measured) < 2:
        return None
    return sum(measured) / len(measured)


def sample_presence(analysis: ImageAnalysis, image_bgr: np.ndarray) -> PresenceSample | None:
    """Extract presence from an analysed frame, or None if no face was found.

    "No face" and "found but unusable" are different events and the state
    machine treats them differently: a frame with no face at all is a missing
    observation — a camera warming up, someone glancing away — while a face that
    fails quality is a real observation that simply does not count towards the
    run.
    """
    face = analysis.face
    if face is None:
        return None
    # One conversion feeding both measurements: this runs on every frame of a
    # live socket, so a second pass over the same pixels is not free.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return PresenceSample(
        usable=analysis.ok,
        score=face.score,
        openness=eye_openness(gray, face),
        sharpness=face_sharpness(gray, face),
    )


def counts_towards_hold(
    sample: PresenceSample,
    *,
    thresholds: PresenceThresholds = DEFAULT_PRESENCE_THRESHOLDS,
) -> bool:
    """Does this frame extend the run?"""
    return sample.usable and sample.score >= thresholds.min_detection_score
