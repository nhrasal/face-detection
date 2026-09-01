import { useCallback, useEffect, useState } from "react";
import { useCamera } from "@hooks/useCamera";
import { useLiveDetection } from "@hooks/useLiveDetection";
import { useStableGuidance } from "@hooks/useStableGuidance";
import {
  AUTO_CAPTURE_HOLD_MS,
  CAPTURE_MAX_SIDE,
  CAPTURE_QUALITY,
  frameToFile,
  grabFrame,
  overlayBox,
} from "@utils/camera";

interface Props {
  onCapture: (file: File) => void;
  onCancel: () => void;
}

const DEFAULT_ASPECT = 4 / 3;

/**
 * Live preview that takes the photo itself once the frame is good.
 *
 * There is no shutter button. The service is already judging every frame and
 * saying what to fix, so a button would only ask the operator to confirm a
 * verdict they just watched arrive — and it invites the one thing the guidance
 * exists to prevent, which is capturing anyway while the frame is bad.
 */
export function CameraCapture({ onCapture, onCancel }: Props) {
  const { videoRef, state, error, start } = useCamera();
  const { detection, error: detectionError, transport, fps } = useLiveDetection(
    videoRef,
    state === "live",
  );
  const [aspect, setAspect] = useState(DEFAULT_ASPECT);
  const [capturing, setCapturing] = useState(false);
  const [captured, setCaptured] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);

  useEffect(() => void start(), [start]);

  // The box tracks every frame; the words — and the shutter that follows them —
  // only move once a verdict has held for a few frames. See useStableGuidance.
  const guidance = useStableGuidance(detection);

  const box =
    detection?.result.bounding_box &&
    overlayBox(
      detection.result.bounding_box,
      { width: detection.frameWidth, height: detection.frameHeight },
      true,
    );

  const capture = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;
    setCapturing(true);
    setCaptureError(null);
    try {
      const frame = await grabFrame(video, CAPTURE_MAX_SIDE, CAPTURE_QUALITY);
      if (!frame) {
        // `captured` stays false, so the dwell below simply arms again on the
        // next good frame rather than stranding the operator.
        setCaptureError("The camera is not ready yet. Still trying…");
        return;
      }
      setCaptured(true);
      onCapture(frameToFile(frame.blob));
    } finally {
      setCapturing(false);
    }
  }, [videoRef, onCapture]);

  // Fire once the good verdict has held for a moment. Cleared the instant the
  // frame stops being good, so a face that drifts out of position mid-dwell is
  // never photographed on the strength of a verdict that has since expired.
  useEffect(() => {
    if (state !== "live" || !guidance.ready || capturing || captured) return;
    const timer = setTimeout(() => void capture(), AUTO_CAPTURE_HOLD_MS);
    return () => clearTimeout(timer);
  }, [state, guidance.ready, capturing, captured, capture]);

  // Self-contained viewfinder styling: this panel sits inside both the light
  // candidate card and the dark registration panel, so it cannot inherit either.
  const shell = "rounded-xl bg-stone-900 p-3 text-white";

  if (state === "error") {
    return (
      <div className={shell}>
        <p className="px-1 py-3 text-sm text-orange-200" role="alert">
          {error}
        </p>
        <div className="flex gap-2">
          <button
            type="button"
            className="rounded-xl bg-lime-200 px-4 py-2.5 text-sm font-bold text-emerald-950"
            onClick={() => void start()}
          >
            Try again
          </button>
          <button
            type="button"
            className="rounded-xl border border-stone-600 px-4 py-2.5 text-sm font-bold text-white"
            onClick={onCancel}
          >
            Use a file instead
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={shell}>
      <div
        className="relative w-full overflow-hidden rounded-lg bg-black"
        style={{ aspectRatio: aspect }}
      >
        <video
          ref={videoRef}
          className="size-full -scale-x-100 object-cover"
          autoPlay
          playsInline
          muted
          onLoadedMetadata={(event) => {
            const video = event.currentTarget;
            if (video.videoWidth && video.videoHeight) {
              // Match the preview to the camera's own aspect ratio so the frame
              // we detect on and the frame on screen share one coordinate space.
              setAspect(video.videoWidth / video.videoHeight);
            }
          }}
        />
        {box && (
          <div
            className={`pointer-events-none absolute rounded-lg border-2 transition-all duration-150 ${
              guidance.ready ? "border-lime-300" : "border-orange-300"
            }`}
            style={{
              left: `${box.left}%`,
              top: `${box.top}%`,
              width: `${box.width}%`,
              height: `${box.height}%`,
            }}
            aria-hidden="true"
          />
        )}
        {state === "starting" && (
          <p className="absolute inset-0 grid place-items-center text-sm text-white/80" role="status">
            Starting the camera…
          </p>
        )}
        {fps > 0 && (
          <span
            className="absolute right-2 top-2 rounded-md bg-black/55 px-2 py-1 text-[11px] font-bold tabular-nums text-white/80"
            title={
              transport === "stream"
                ? "Live detection over a WebSocket"
                : "WebSocket unavailable — polling the HTTP endpoint"
            }
          >
            {fps} fps{transport === "polling" && " · fallback"}
          </span>
        )}
      </div>

      <p
        className={`mt-3 min-h-10 px-1 text-sm ${guidance.ready ? "text-lime-300" : "text-stone-300"}`}
        role="status"
        aria-live="polite"
      >
        {capturing ? "Taking the photo…" : guidance.message}
        {detectionError && <span className="block text-xs text-stone-400">{detectionError}</span>}
        {captureError && <span className="block text-orange-200">{captureError}</span>}
      </p>

      <div className="mt-1 flex gap-2">
        <button
          type="button"
          className="min-h-12 w-full rounded-xl border border-stone-600 px-4 py-3 text-sm font-bold text-white"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
