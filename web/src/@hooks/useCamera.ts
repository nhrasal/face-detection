import { useCallback, useEffect, useRef, useState } from "react";

export type CameraState = "idle" | "starting" | "live" | "error";

const CONSTRAINTS: MediaStreamConstraints = {
  video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
  audio: false,
};

/** getUserMedia rejects with DOMException names; each one needs a different fix. */
function describe(caught: unknown): string {
  const name = caught instanceof DOMException ? caught.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera access was blocked. Allow it for this site in your browser, then try again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found on this device.";
    case "NotReadableError":
      return "The camera is already in use by another application.";
    default:
      return "The camera could not be started.";
  }
}

/**
 * Owns one camera stream and the video element it feeds.
 *
 * Tracks are stopped on unmount and on every stop(), because a MediaStream that
 * outlives its component leaves the camera light on and holds the device against
 * the next component that asks for it.
 */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  // getUserMedia resolves long after it is called — behind a permission prompt,
  // it can be seconds. Anything that supersedes a start (a stop, a retry, or
  // StrictMode's remount) bumps this so the late stream is stopped on arrival
  // instead of quietly leaving the camera light on.
  const generation = useRef(0);
  const [state, setState] = useState<CameraState>("idle");
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    generation.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setState("idle");
  }, []);

  const start = useCallback(async () => {
    // getUserMedia is absent outside a secure context, so this is also the
    // "opened over plain HTTP" path, not only the "ancient browser" one.
    if (!navigator.mediaDevices?.getUserMedia) {
      setState("error");
      setError("This browser cannot open a camera. A secure (https) connection is required.");
      return;
    }
    const token = ++generation.current;
    setState("starting");
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
      if (token !== generation.current || !videoRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setState("live");
    } catch (caught) {
      if (token !== generation.current) return;
      setState("error");
      setError(describe(caught));
    }
  }, []);

  useEffect(() => stop, [stop]);

  return { videoRef, state, error, start, stop };
}
