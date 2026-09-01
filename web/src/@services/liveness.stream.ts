import { api } from "@services/api.instance";
import { describeClose } from "@services/face.stream";

/**
 * Liveness rides the same WebSocket conventions as the preview stream: relative
 * in development so it uses the Vite proxy, and the scheme follows the page so
 * an https deployment gets wss rather than tripping mixed-content blocking.
 */
export function livenessUrl(): string {
  const base = api.defaults.baseURL || "/api/v1";
  const url = new URL(`${base}/face/liveness`, window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export { describeClose };
