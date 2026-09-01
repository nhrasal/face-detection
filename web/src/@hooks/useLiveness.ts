import { type RefObject, useCallback, useEffect, useState } from "react";
import type {
  LivenessFailure,
  LivenessMessage,
  LivenessProgressMessage,
} from "@interfaces/liveness";
import { isResult } from "@interfaces/liveness";
import { describeClose, livenessUrl } from "@services/liveness.stream";
import { FRAME_MAX_SIDE, FRAME_QUALITY, MIN_STREAM_INTERVAL_MS, grabFrame } from "@utils/camera";

export interface LivenessSessionState {
  status: LivenessProgressMessage | null;
  passed: boolean | null;
  failure: LivenessFailure | null;
  error: string | null;
  /** Increment to abandon the current session and open a fresh one. */
  attempt: number;
}

/**
 * Drives one liveness session over a WebSocket.
 *
 * Ack-paced like the preview stream: a frame goes out only once the previous
 * result has arrived, so the rate follows what the service can actually sustain
 * rather than queueing frames whose answers describe a pose the person has
 * already left.
 *
 * The result is never computed here. The server owns the session and decides;
 * a client that could assert "I passed" would not be a liveness check.
 */
export function useLiveness(videoRef: RefObject<HTMLVideoElement | null>, active: boolean) {
  const [status, setStatus] = useState<LivenessProgressMessage | null>(null);
  const [passed, setPassed] = useState<boolean | null>(null);
  const [failure, setFailure] = useState<LivenessFailure | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  const retry = useCallback(() => {
    setStatus(null);
    setPassed(null);
    setFailure(null);
    setError(null);
    setAttempt((value) => value + 1);
  }, []);

  useEffect(() => {
    if (!active) return;

    let stopped = false;
    let socket: WebSocket | null = null;
    let handle = 0;
    let inFlight = false;
    let lastSentAt = 0;

    const pump = () => {
      if (stopped) return;
      handle = requestAnimationFrame(pump);
      const video = videoRef.current;
      if (!video || inFlight || socket?.readyState !== WebSocket.OPEN) return;
      const now = performance.now();
      if (now - lastSentAt < MIN_STREAM_INTERVAL_MS) return;
      lastSentAt = now;
      inFlight = true;
      void grabFrame(video, FRAME_MAX_SIDE, FRAME_QUALITY)
        .then((frame) => {
          if (stopped || !frame || socket?.readyState !== WebSocket.OPEN) {
            inFlight = false;
            return;
          }
          socket.send(frame.blob);
        })
        .catch(() => {
          inFlight = false;
        });
    };

    try {
      socket = new WebSocket(livenessUrl());
    } catch {
      setError("The liveness check could not be started.");
      return;
    }

    socket.onopen = () => {
      setError(null);
      pump();
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      inFlight = false;
      if (stopped) return;
      let message: LivenessMessage;
      try {
        message = JSON.parse(event.data) as LivenessMessage;
      } catch {
        return;
      }
      if (isResult(message)) {
        setPassed(message.passed);
        setFailure(message.failure);
        return;
      }
      setStatus(message);
    };

    socket.onclose = (event: CloseEvent) => {
      if (stopped) return;
      const reason = describeClose(event.code);
      if (reason) setError(reason);
      // A close with no result and no known code is a dropped connection, not a
      // failed check — saying "you failed" would blame the person for a network
      // problem.
      else setPassed((current) => (current === null ? null : current));
    };

    socket.onerror = () => {
      if (!stopped) setError("The liveness connection was interrupted.");
    };

    return () => {
      stopped = true;
      cancelAnimationFrame(handle);
      socket?.close();
    };
  }, [videoRef, active, attempt]);

  return { status, passed, failure, error, attempt, retry };
}
