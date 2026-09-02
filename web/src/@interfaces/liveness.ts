/** Wire types for the presence-check WebSocket (backend app/api/v1/liveness.py). */

import type { DetectResult } from "@interfaces/face";

export type LivenessState =
  | "AWAITING_FACE"
  | "HOLDING"
  | "AWAITING_BLINK"
  | "PASSED"
  | "FAILED";

export type LivenessFailure = "SESSION_EXPIRED" | "FACE_LOST" | "BLINK_NOT_SEEN";

/** Raw eye numbers, for tuning the closed threshold against a real camera. */
export interface EyeReading {
  openness: number;
  baseline: number | null;
  /** openness / baseline — what the closed threshold is compared against. */
  ratio: number | null;
  /** Whether the blur guard let this reading through at all. */
  trusted: boolean;
  /** Face sharpness over the session peak — what `trusted` was decided on. */
  sharpness_ratio: number | null;
}

export interface LivenessProgressMessage {
  type: "start" | "progress";
  session_id: string;
  state: LivenessState;
  /** [steps done, steps needed] — the blink is the final step. */
  progress: [number, number];
  /**
   * The same payload the preview stream returns, so the panel can draw the box
   * and name what to fix. Null when the frame could not be analysed at all —
   * render that as "still looking", never as a verdict.
   */
  detection: DetectResult | null;
  /** Null when this frame produced no usable eye measurement. */
  eye: EyeReading | null;
}

export interface LivenessResultMessage {
  type: "result";
  session_id: string;
  passed: boolean;
  failure: LivenessFailure | null;
}

export type LivenessMessage = LivenessProgressMessage | LivenessResultMessage;

export const isResult = (message: LivenessMessage): message is LivenessResultMessage =>
  message.type === "result";
