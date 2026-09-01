import { api } from "@services/api.instance";
import type { DetectResult } from "@interfaces/face";

/** Application close codes the service uses (see backend app/api/v1/stream.py). */
export const CLOSE_FORBIDDEN_ORIGIN = 4403;
export const CLOSE_TOO_MANY_SESSIONS = 4429;
export const CLOSE_FRAME_TOO_LARGE = 4413;

export interface StreamError {
  code?: string;
  message?: string;
  detail?: string | null;
}

export type StreamMessage = DetectResult | { success: false; error: StreamError };

export const isStreamError = (
  message: StreamMessage,
): message is { success: false; error: StreamError } => message.success === false;

/**
 * Resolve the WebSocket URL from the same base the REST client uses.
 *
 * Relative in development, so it rides the Vite proxy; absolute if
 * VITE_API_BASE_URL points elsewhere. The scheme follows the page, so an https
 * deployment gets wss and never trips mixed-content blocking.
 */
export function streamUrl(): string {
  const base = api.defaults.baseURL || "/api/v1";
  const url = new URL(`${base}/face/stream`, window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

/** Why the socket closed, in words an operator can act on. */
export function describeClose(code: number): string | null {
  switch (code) {
    case CLOSE_TOO_MANY_SESSIONS:
      return "Too many live cameras are open right now.";
    case CLOSE_FORBIDDEN_ORIGIN:
      return "This page is not allowed to open a live camera session.";
    case CLOSE_FRAME_TOO_LARGE:
      return "The camera frame was too large to send.";
    default:
      return null;
  }
}
