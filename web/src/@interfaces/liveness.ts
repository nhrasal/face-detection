/** Wire types for the liveness WebSocket (backend app/api/v1/liveness.py). */

export type ChallengeKind = "TURN_LEFT" | "TURN_RIGHT" | "MOVE_CLOSER";

export type LivenessState =
  | "AWAITING_NEUTRAL"
  | "AWAITING_CHALLENGE"
  | "PASSED"
  | "FAILED";

export type LivenessFailure =
  | "CHALLENGE_TIMEOUT"
  | "SESSION_EXPIRED"
  | "FACE_LOST";

export interface LivenessProgressMessage {
  type: "challenge" | "progress";
  session_id: string;
  state: LivenessState;
  challenge: ChallengeKind | null;
  /** [completed, total] */
  progress: [number, number];
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
