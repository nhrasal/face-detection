"""The business layer: turns scores into verdicts.

Deliberately imports nothing from `app.engine.adapters` and knows nothing about
models. The engine says "similarity 0.52"; this decides whether that means
MATCH. Keeping the two apart is what lets a threshold change ship without
touching inference, and a model change ship without touching policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.engine.pipeline import FaceStatus, ImageAnalysis
from app.engine.types import ReasonCode
from app.services.comparison import DEFAULT_CONFIDENCE_SLOPE, cosine_similarity, to_confidence


class Decision(StrEnum):
    """Flat outcome, matching the roadmap's result codes.

    REVIEW is present from the start even though it ships disabled
    (REVIEW_MARGIN=0). A single hard threshold makes every borderline pair a
    silent false accept or false reject; having the value in the enum and the
    database CHECK constraint from day one means enabling manual review later
    is a config change, not a migration.
    """

    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    REVIEW = "REVIEW"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_QUALITY = "LOW_QUALITY"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class ImageRole(StrEnum):
    REFERENCE = "REFERENCE"
    CANDIDATE = "CANDIDATE"


@dataclass(frozen=True, slots=True)
class ComparisonOutcome:
    """Everything the API and the audit row need, and nothing more.

    Note the absence of any embedding field: that is deliberate and enforced by
    a test.
    """

    decision: Decision
    similarity: float | None
    confidence: float | None
    threshold: float
    reference_status: FaceStatus
    candidate_status: FaceStatus
    reference_face_count: int
    candidate_face_count: int
    reason_code: str | None = None
    issues: tuple[ReasonCode, ...] = ()

    @property
    def matched(self) -> bool:
        return self.decision is Decision.MATCH


def decide(similarity: float, threshold: float, review_margin: float = 0.0) -> Decision:
    """Map a similarity onto MATCH / REVIEW / NO_MATCH.

    A similarity exactly equal to the threshold is a MATCH. That boundary is
    arbitrary but it must be fixed and tested: an off-by-epsilon here is
    invisible until an auditor asks why a recorded 0.363 was rejected against a
    recorded threshold of 0.363.
    """
    if similarity >= threshold:
        return Decision.MATCH
    if review_margin > 0.0 and similarity >= threshold - review_margin:
        return Decision.REVIEW
    return Decision.NO_MATCH


def _failure_reason(role: ImageRole, status: FaceStatus) -> str:
    return f"{role.value}_{status.value}"


def decide_comparison(
    reference: ImageAnalysis,
    candidate: ImageAnalysis,
    *,
    threshold: float,
    review_margin: float = 0.0,
    confidence_slope: float = DEFAULT_CONFIDENCE_SLOPE,
) -> ComparisonOutcome:
    """Combine two image analyses into one verdict.

    When an image fails, the flat `decision` carries its status for roadmap
    compatibility, but `reason_code` names WHICH image failed and `issues`
    carries the specific quality codes. "LOW_QUALITY" alone tells a user
    nothing and they retry with the same bad photo.

    The reference is reported first when both fail: it is the side the user
    usually cannot re-take, so it is the more important thing to surface.
    """
    for role, analysis in ((ImageRole.REFERENCE, reference), (ImageRole.CANDIDATE, candidate)):
        if analysis.ok:
            continue
        return ComparisonOutcome(
            decision=Decision(analysis.status.value),
            similarity=None,
            confidence=None,
            threshold=threshold,
            reference_status=reference.status,
            candidate_status=candidate.status,
            reference_face_count=reference.face_count,
            candidate_face_count=candidate.face_count,
            reason_code=_failure_reason(role, analysis.status),
            issues=analysis.quality.issues if analysis.quality else (),
        )

    # Both analyses are OK, so both embeddings exist. Asserting rather than
    # silently coercing: an OK status without an embedding is a pipeline bug,
    # and turning it into a NO_MATCH would hide it.
    if reference.embedding is None or candidate.embedding is None:
        raise ValueError("analysis reported OK but produced no embedding")

    similarity = cosine_similarity(reference.embedding, candidate.embedding)
    return ComparisonOutcome(
        decision=decide(similarity, threshold, review_margin),
        similarity=similarity,
        confidence=to_confidence(similarity, threshold=threshold, slope=confidence_slope),
        threshold=threshold,
        reference_status=reference.status,
        candidate_status=candidate.status,
        reference_face_count=reference.face_count,
        candidate_face_count=candidate.face_count,
    )
