import { useEffect, useRef, useState } from "react";
import { useCamera } from "@hooks/useCamera";
import { useLiveDetection } from "@hooks/useLiveDetection";
import {
  CAPTURE_MAX_SIDE,
  CAPTURE_QUALITY,
  INITIAL_GUIDANCE,
  type Guidance,
  frameGuidance,
  frameToFile,
  grabFrame,
  initialGuidanceState,
  overlayBox,
  stabiliseGuidance,
} from "@utils/camera";

interface Props {
  onCapture: (file: File) => void;
  onCancel: () => void;
}

const DEFAULT_ASPECT = 4 / 3;

export function CameraCapture({ onCapture, onCancel }: Props) {
  const { videoRef, state, error, start } = useCamera();
  const { detection, error: detectionError, transport, fps } = useLiveDetection(
    videoRef,
    state === "live",
  );
  const [aspect, setAspect] = useState(DEFAULT_ASPECT);
  const [capturing, setCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);

  useEffect(() => void start(), [start]);

  // The box tracks every frame, but the words and the Capture button only change
  // once a new verdict has held for a few frames — see stabiliseGuidance.
  const [guidance, setGuidance] = useState<Guidance>(INITIAL_GUIDANCE);
  const stabiliser = useRef(initialGuidanceState());
  useEffect(() => {
    stabiliser.current = stabiliseGuidance(stabiliser.current, frameGuidance(detection));
    const { shown } = stabiliser.current;
    setGuidance((current) =>
      current.message === shown.message && current.ready === shown.ready ? current : shown,
    );
  }, [detection]);

  const box =
    detection?.result.bounding_box &&
    overlayBox(
      detection.result.bounding_box,
      { width: detection.frameWidth, height: detection.frameHeight },
      true,
    );

  const capture = async () => {
    const video = videoRef.current;
    if (!video) return;
    setCapturing(true);
    setCaptureError(null);
    try {
      const frame = await grabFrame(video, CAPTURE_MAX_SIDE, CAPTURE_QUALITY);
      if (!frame) {
        setCaptureError("The camera is not ready yet. Try again in a moment.");
        return;
      }
      onCapture(frameToFile(frame.blob));
    } finally {
      setCapturing(false);
    }
  };

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
        {guidance.message}
        {detectionError && <span className="block text-xs text-stone-400">{detectionError}</span>}
        {captureError && <span className="block text-orange-200">{captureError}</span>}
      </p>

      <div className="mt-1 flex gap-2">
        <button
          type="button"
          className="min-h-12 flex-1 rounded-xl bg-lime-200 px-5 py-3 font-bold text-emerald-950 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={state !== "live" || !guidance.ready || capturing}
          onClick={() => void capture()}
        >
          {capturing ? "Capturing…" : "Capture photo"}
        </button>
        <button
          type="button"
          className="rounded-xl border border-stone-600 px-4 py-3 text-sm font-bold text-white"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
