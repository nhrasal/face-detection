import type { LivenessFailure, LivenessProgressMessage } from "@interfaces/liveness";
import type { Guidance } from "@utils/camera";

const FAILURE_MESSAGE: Record<LivenessFailure, string> = {
  SESSION_EXPIRED: "The check timed out. Let's try again.",
  FACE_LOST: "We lost sight of your face. Let's try again.",
  BLINK_NOT_SEEN: "We didn't see you blink. Let's try again.",
};

export interface LivenessPrompt {
  message: string;
  /** True while the hold is actually accumulating, for a "going well" colour. */
  active: boolean;
}

/**
 * What to tell the person on screen.
 *
 * Only one thing is ever asked of the subject, and only once the service is
 * ready to measure it: blink. Asking earlier would waste the blink, because the
 * baseline it is compared against is built from the frames before it.
 *
 * The wording avoids implying more was established than actually was. A blink
 * rules out a photograph; it does not rule out a video or a deepfake. See the
 * backend's `app/engine/liveness.py` for the full statement.
 *
 * While the hold has not started, the live detection speaks instead: "the bar is
 * not moving" is useless on its own, and the reason it is not moving — too dark,
 * too far away, two people in shot — is the only thing the person can act on.
 * That vocabulary is `frameGuidance`'s, shared with the Camera panel so the two
 * never disagree about what a given reason code means.
 */
export function livenessPrompt(
  message: LivenessProgressMessage | null,
  guidance: Guidance | null = null,
): LivenessPrompt {
  if (!message) return { message: "Starting the camera check…", active: false };

  switch (message.state) {
    case "AWAITING_FACE":
      // Covers both "no face yet" and "the face is there but the quality gate is
      // refusing it". Defer to the live detection, which can say WHICH of those
      // it is; fall back only when no frame has been analysed yet.
      return {
        message: guidance?.message ?? "Look at the camera and keep your face in frame",
        active: false,
      };
    case "HOLDING":
      return { message: "Hold still…", active: true };
    case "AWAITING_BLINK":
      return { message: "Now blink", active: true };
    case "PASSED":
      return { message: "Got it", active: true };
    case "FAILED":
      return { message: "The camera check did not complete", active: false };
  }
}

export function describeFailure(failure: LivenessFailure | null): string {
  if (!failure) return "The camera check did not complete. Let's try again.";
  return FAILURE_MESSAGE[failure] ?? FAILURE_MESSAGE.SESSION_EXPIRED;
}

/** Fraction of the required hold completed, for a progress bar. */
export function progressFraction(progress: [number, number]): number {
  const [done, total] = progress;
  if (total <= 0) return 0;
  return Math.max(0, Math.min(1, done / total));
}
