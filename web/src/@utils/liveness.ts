import type {
  ChallengeKind,
  LivenessFailure,
  LivenessProgressMessage,
} from "@interfaces/liveness";

/**
 * What to tell the person on screen.
 *
 * Instructions are phrased about the person's OWN body ("your left"), never
 * about the image. That is deliberate: the preview is mirrored as a comfort
 * affordance, so an image-relative instruction would be backwards for the user
 * while looking correct in code. Because these describe the subject, they stay
 * right whether or not the preview is mirrored.
 */
const CHALLENGE_PROMPT: Record<ChallengeKind, string> = {
  TURN_LEFT: "Slowly turn your head to your left",
  TURN_RIGHT: "Slowly turn your head to your right",
  MOVE_CLOSER: "Move a little closer to the camera",
};

const FAILURE_MESSAGE: Record<LivenessFailure, string> = {
  CHALLENGE_TIMEOUT: "That took too long. Let's try again.",
  SESSION_EXPIRED: "The check timed out. Let's try again.",
  FACE_LOST: "We lost sight of your face. Let's try again.",
};

export interface LivenessPrompt {
  message: string;
  /** True while the person should be holding still rather than moving. */
  settle: boolean;
}

export function livenessPrompt(message: LivenessProgressMessage | null): LivenessPrompt {
  if (!message) return { message: "Starting the check…", settle: true };

  switch (message.state) {
    case "AWAITING_NEUTRAL":
      // The backend requires a near-frontal frame between challenges, so one
      // slow sweep cannot satisfy two prompts. The person needs to be told
      // that, or a passing attempt looks like a stuck screen.
      return { message: "Look straight at the camera", settle: true };
    case "AWAITING_CHALLENGE":
      return {
        message: message.challenge
          ? CHALLENGE_PROMPT[message.challenge]
          : "Follow the instruction",
        settle: false,
      };
    case "PASSED":
      return { message: "Liveness confirmed", settle: true };
    case "FAILED":
      return { message: "Liveness check failed", settle: true };
  }
}

export function describeFailure(failure: LivenessFailure | null): string {
  if (!failure) return "The liveness check did not pass. Let's try again.";
  return FAILURE_MESSAGE[failure] ?? FAILURE_MESSAGE.SESSION_EXPIRED;
}

/** Fraction complete, for a progress bar. */
export function progressFraction(progress: [number, number]): number {
  const [done, total] = progress;
  if (total <= 0) return 0;
  return Math.max(0, Math.min(1, done / total));
}
