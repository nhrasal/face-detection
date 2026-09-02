import { useCallback, useEffect, useState } from "react";
import { useCamera } from "@hooks/useCamera";
import { useLiveDetection } from "@hooks/useLiveDetection";
import { useStableGuidance } from "@hooks/useStableGuidance";
import {
  CAPTURE_ASPECT,
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


/**
 * Live preview with a shutter under the operator's hand.
 *
 * The service judges every frame and says what to fix, and the guidance below
 * the preview reflects that in real time — but it advises rather than decides.
 * The operator chooses the moment, because they can see things the quality gate
 * cannot: whether this is the right person, whether they are ready, whether the
 * shot is worth keeping. The button stays live even while the frame is poor, so
 * a subject the detector struggles with can still be photographed; nothing is
 * verified until the still has been reviewed and Verify is pressed.
 */
export function CameraCapture({ onCapture, onCancel }: Props) {
  const { videoRef, state, error, start } = useCamera();
  const { detection, error: detectionError, transport, fps } = useLiveDetection(
    videoRef,
    state === "live",
  );
  const [capturing, setCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);

  useEffect(() => void start(), [start]);

  // The box tracks every frame; the words only move once a verdict has held for
  // a few frames. See useStableGuidance.
  const guidance = useStableGuidance(detection);

  const box =
    detection?.result.bounding_box &&
    overlayBox(
      detection.result.bounding_box,
      { width: detection.frameWidth, height: detection.frameHeight },
      false,
      CAPTURE_ASPECT,
    );

  const capture = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;
    setCapturing(true);
    setCaptureError(null);
    try {
      const frame = await grabFrame(video, CAPTURE_MAX_SIDE, CAPTURE_QUALITY, CAPTURE_ASPECT);
      if (!frame) {
        setCaptureError("The camera is not ready yet. Try again in a moment.");
        return;
      }
      onCapture(frameToFile(frame.blob));
    } finally {
      setCapturing(false);
    }
  }, [videoRef, onCapture]);

  // Self-contained viewfinder styling: this panel sits inside both the
  // candidate card and the dark registration panel, so it cannot inherit either.
  const shell = "rounded-lg border border-line bg-sunken p-3 text-ink";

  if (state === "error") {
    return (
      <div className={shell}>
        <p className="px-1 py-3 text-sm text-fail" role="alert">
          {error}
        </p>
        <div className="flex gap-2">
          <button
            type="button"
            className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-canvas transition-colors hover:bg-accent-strong"
            onClick={() => void start()}
          >
            Try again
          </button>
          <button
            type="button"
            className="rounded-md border border-line px-3 py-2 text-sm font-medium text-ink transition-colors hover:bg-line/60"
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
      <div className="relative aspect-[4/5] w-full overflow-hidden rounded bg-black">
        <video
          ref={videoRef}
          // Not mirrored. The still is encoded from the raw video, and the
          // stored reference sits right beside it — a mirrored preview would
          // be the only surface in the flow facing the other way.
          className="size-full object-cover"
          autoPlay
          playsInline
          muted
        />
        {box && (
          <div
            className={`pointer-events-none absolute rounded border-2 transition-all duration-150 ${
              guidance.ready ? "border-pass" : "border-review"
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
          <p className="absolute inset-0 grid place-items-center text-sm text-ink-soft" role="status">
            Starting the camera…
          </p>
        )}
        {fps > 0 && (
          <span
            className="absolute right-2 top-2 rounded bg-canvas/70 px-2 py-1 font-mono text-[10px] font-medium tabular-nums text-ink-soft"
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
        className={`mt-3 min-h-10 px-1 text-sm ${guidance.ready ? "text-pass" : "text-ink-soft"}`}
        role="status"
        aria-live="polite"
      >
        {capturing ? "Taking the photo…" : guidance.message}
        {detectionError && <span className="block text-xs text-ink-muted">{detectionError}</span>}
        {captureError && <span className="block text-fail">{captureError}</span>}
      </p>

      <div className="mt-1 flex gap-2">
        <button
          type="button"
          className={`flex-1 rounded-md px-4 py-2.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
            guidance.ready
              ? "bg-pass text-canvas hover:brightness-110"
              : "bg-accent text-canvas hover:bg-accent-strong"
          }`}
          disabled={state !== "live" || capturing}
          onClick={() => void capture()}
        >
          {capturing ? "Capturing…" : "Capture photo"}
        </button>
        <button
          type="button"
          className="rounded-md border border-line px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-line/60"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
