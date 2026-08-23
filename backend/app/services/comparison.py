"""Similarity between two face embeddings.

Two numbers come out of here and they serve different audiences:

`similarity` is raw cosine — the auditable value, compared against the
threshold, reproducible by anyone re-running the models.

`confidence` is a squashed 0-1 value for display. It exists because raw cosine
is a terrible thing to show a user: 0.41 rendered as "41% match" reads as a
failure to someone who is in fact a solid match at a 0.363 threshold. Only
`confidence` may ever be rendered as a percentage.
"""

from __future__ import annotations

import math

import numpy as np

# Placeholder steepness for the confidence logistic, pending calibration.
# Chosen so the curve is informative across the observed score range rather
# than fitted to data: at threshold+0.10 it reads ~0.77, at threshold-0.03
# ~0.40. Calibration replaces this with a logistic fitted on real pairs.
DEFAULT_CONFIDENCE_SLOPE = 12.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, in [-1, 1].

    The engine contract guarantees unit-norm vectors, so this is a dot product.
    It still divides by the norms rather than assuming, because a silently
    unnormalised adapter would otherwise produce similarities in the tens that
    sail past every threshold — which is exactly how SFace's raw feature()
    behaves.
    """
    if a.shape != b.shape:
        raise ValueError(f"embedding dimensions differ: {a.shape} vs {b.shape}")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp against floating-point drift so a self-comparison never reports
    # 1.0000000000000002 and confuses a downstream range check.
    return max(-1.0, min(1.0, similarity))


def to_confidence(
    similarity: float,
    *,
    threshold: float,
    slope: float = DEFAULT_CONFIDENCE_SLOPE,
) -> float:
    """Map a raw cosine onto [0, 1], centred on the threshold.

    Exactly 0.5 at the threshold, monotonically increasing, saturating at the
    extremes. UNCALIBRATED until the calibration phase fits `slope` on real
    genuine/impostor pairs — it is a presentation aid, never a decision input.
    """
    exponent = -slope * (similarity - threshold)
    # Guard against overflow at extreme inputs; math.exp(710) raises.
    if exponent > 60.0:
        return 0.0
    if exponent < -60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))
