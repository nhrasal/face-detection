import type { Decision, VerificationResult } from "@interfaces/face";

export const MAX_FILE_BYTES = 5 * 1024 * 1024;
export const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

export function validateCandidate(file: File): string | null {
  if (!ALLOWED_TYPES.has(file.type)) return "Choose a JPEG, PNG, or WebP photograph.";
  if (file.size > MAX_FILE_BYTES) return "The photograph must be 5 MB or smaller.";
  if (file.size === 0) return "The selected file is empty.";
  return null;
}

const issueMessages: Record<string, string> = {
  IMAGE_BLURRY: "The image is out of focus. Hold the camera steady and try again.",
  TOO_DARK: "The face is too dark. Move toward a soft, even light source.",
  TOO_BRIGHT: "The face is overexposed. Move away from direct light.",
  LOW_CONTRAST: "The face lacks contrast. Use brighter, more even lighting.",
  EXTREME_POSE: "Look straight at the camera and keep both eyes visible.",
  EXTREME_ROLL: "Keep your head upright and try again.",
  FACE_TOO_SMALL: "Move closer so your face is clear and detailed.",
  FACE_TOO_SMALL_IN_FRAME: "Move closer and center your face in the frame.",
  FACE_FILLS_FRAME: "Move slightly farther away so your full face is visible.",
  FACE_OUT_OF_FRAME: "Center your full face inside the photograph.",
  LOW_DETECTION_CONFIDENCE: "Use a clearer, front-facing photograph.",
};

const decisionCopy: Record<Decision, { title: string; detail: string; tone: string }> = {
  MATCH: { title: "Identity confirmed", detail: "The uploaded photograph is consistent with the stored profile.", tone: "success" },
  NO_MATCH: { title: "Identity not confirmed", detail: "The faces do not meet the required similarity threshold.", tone: "danger" },
  REVIEW: { title: "Manual review needed", detail: "The result is close to the threshold and needs a human decision.", tone: "warning" },
  NO_FACE: { title: "No face found", detail: "Use a clear portrait with one face looking toward the camera.", tone: "warning" },
  MULTIPLE_FACES: { title: "More than one face found", detail: "Crop or retake the photograph so only one person is visible.", tone: "warning" },
  LOW_QUALITY: { title: "Photo quality needs attention", detail: "Retake the photograph using the guidance below.", tone: "warning" },
  PROCESSING_ERROR: { title: "We could not process this photo", detail: "Try a different photograph. If the problem continues, contact support.", tone: "danger" },
};

export function resultPresentation(result: VerificationResult) {
  const guidance = result.quality_issues.map(
    (issue) => issueMessages[issue] || "Retake the photograph and try again.",
  );
  return { ...decisionCopy[result.decision], guidance: [...new Set(guidance)] };
}

export const asPercentage = (value: number | null): string =>
  value === null ? "—" : `${Math.round(value * 100)}%`;

export const similarityLabel = (value: number | null): string =>
  value === null ? "—" : value.toFixed(4);
