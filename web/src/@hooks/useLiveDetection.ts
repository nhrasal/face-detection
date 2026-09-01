import { type RefObject, useEffect, useRef, useState } from "react";
import { ApiError, isCanceled } from "@services/api.instance";
import { FaceService } from "@services/face.service";
import { describeClose, isStreamError, streamUrl, type StreamMessage } from "@services/face.stream";
import {
  FRAME_INTERVAL_MS,
  FRAME_MAX_SIDE,
  FRAME_QUALITY,
  MIN_STREAM_INTERVAL_MS,
  grabFrame,
} from "@utils/camera";
import type { LiveDetection } from "@interfaces/face";

/** Longest gap the HTTP fallback backs off to when the service is refusing frames. */
const MAX_BACKOFF_MS = 4_000;

/** How often the measured frame rate is recomputed. Not once per frame. */
const FPS_WINDOW_MS = 500;

export type Transport = "stream" | "polling";

export interface LiveDetectionState {
  detection: LiveDetection | null;
  error: string | null;
  transport: Transport;
  /** Measured detections per second, for showing that this really is live. */
  fps: number;
}

/**
 * Detect on the live preview, as fast as the pipeline can answer.
 *
 * The WebSocket is ack-paced: a frame goes out only once the previous result is
 * back, so the stream self-tunes to the achievable rate instead of queueing
 * frames whose answers describe a scene that has already changed. It is capped
 * at 30 FPS because nothing above that is visible to a person, and every frame
 * costs an inference on a shared worker.
 *
 * If the socket cannot be established — a proxy that does not forward upgrades,
 * a session cap already reached — this falls back to polling the HTTP endpoint
 * at 2.5 FPS. The preview degrades; it does not stop working.
 *
 * Nothing sampled here is stored. Only the still the operator captures is verified.
 */
export function useLiveDetection(
  videoRef: RefObject<HTMLVideoElement | null>,
  active: boolean,
): LiveDetectionState {
  const [detection, setDetection] = useState<LiveDetection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [transport, setTransport] = useState<Transport>("stream");
  const [fps, setFps] = useState(0);
  // Retry the socket the next time the camera is opened, rather than leaving a
  // session stuck on the fallback for the lifetime of the component.
  const wasActive = useRef(active);
  const lastFpsAt = useRef<number | null>(null);

  useEffect(() => {
    if (active && !wasActive.current) setTransport("stream");
    wasActive.current = active;
    if (!active) {
      setDetection(null);
      setError(null);
      setFps(0);
    }
  }, [active]);

  useEffect(() => {
    if (!active || transport !== "stream") return;

    let stopped = false;
    let handle: number | undefined;
    let socket: WebSocket | null = null;
    let opened = false;
    // Ack-pacing: exactly one frame is ever in flight, so a single slot is
    // enough to remember the geometry its box will come back in.
    let inFlight: { width: number; height: number } | null = null;
    let lastSentAt = 0;
    let marks: number[] = [];

    const fallback = (reason: string | null) => {
      if (stopped) return;
      if (reason) setError(reason);
      setTransport("polling");
    };

    const pump = () => {
      if (stopped) return;
      handle = requestAnimationFrame(pump);
      const video = videoRef.current;
      if (!video || inFlight || socket?.readyState !== WebSocket.OPEN) return;
      const now = performance.now();
      if (now - lastSentAt < MIN_STREAM_INTERVAL_MS) return;
      lastSentAt = now;
      inFlight = { width: 0, height: 0 };
      void grabFrame(video, FRAME_MAX_SIDE, FRAME_QUALITY)
        .then((frame) => {
          if (stopped || !frame || socket?.readyState !== WebSocket.OPEN) {
            inFlight = null;
            return;
          }
          inFlight = { width: frame.width, height: frame.height };
          socket.send(frame.blob);
        })
        .catch(() => {
          inFlight = null;
        });
    };

    try {
      socket = new WebSocket(streamUrl());
    } catch {
      fallback(null);
      return;
    }

    socket.onopen = () => {
      opened = true;
      setError(null);
      pump();
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      const frame = inFlight;
      inFlight = null;
      if (stopped) return;
      let message: StreamMessage;
      try {
        message = JSON.parse(event.data) as StreamMessage;
      } catch {
        return;
      }
      if (isStreamError(message)) {
        // One bad frame does not end the stream; the service keeps the socket open.
        setError("Live guidance could not read that frame.");
        return;
      }
      if (frame?.width) {
        setDetection({ result: message, frameWidth: frame.width, frameHeight: frame.height });
      }
      setError(null);

      const now = performance.now();
      marks.push(now);
      marks = marks.filter((mark) => now - mark <= 1_000);
      if (now - (lastFpsAt.current ?? 0) >= FPS_WINDOW_MS) {
        lastFpsAt.current = now;
        setFps(marks.length);
      }
    };

    socket.onerror = () => {
      // onerror always precedes onclose; let onclose decide what to do.
    };

    socket.onclose = (event: CloseEvent) => {
      if (stopped) return;
      const reason = describeClose(event.code);
      fallback(reason ?? (opened ? "Live stream ended — using the slower fallback." : null));
    };

    return () => {
      stopped = true;
      if (handle !== undefined) cancelAnimationFrame(handle);
      socket?.close();
    };
  }, [videoRef, active, transport]);

  useEffect(() => {
    if (!active || transport !== "polling") return;

    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | null = null;
    let backoff = FRAME_INTERVAL_MS;

    const tick = async () => {
      const video = videoRef.current;
      if (stopped || !video) return;
      try {
        const frame = await grabFrame(video, FRAME_MAX_SIDE, FRAME_QUALITY);
        if (stopped) return;
        if (frame) {
          controller = new AbortController();
          const result = await FaceService.detectFrame(frame.blob, controller.signal);
          if (stopped) return;
          setDetection({ result, frameWidth: frame.width, frameHeight: frame.height });
          setFps(Math.round(1_000 / backoff));
          backoff = FRAME_INTERVAL_MS;
        }
      } catch (caught) {
        if (stopped || isCanceled(caught)) return;
        // 429 means we are asking too often; everything else may be transient.
        // Either way, slow down rather than hammering a service under strain.
        backoff = Math.min(MAX_BACKOFF_MS, backoff * 2);
        setError(
          caught instanceof ApiError && caught.status === 429
            ? "Checking less often — the service is busy."
            : "Live guidance is unavailable. You can still capture a photo.",
        );
      } finally {
        if (!stopped) timer = setTimeout(() => void tick(), backoff);
      }
    };

    void tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller?.abort();
    };
  }, [videoRef, active, transport]);

  return { detection, error, transport, fps };
}
