import { useEffect, useState } from "react";
import { useCamera } from "@hooks/useCamera";
import { useLiveness } from "@hooks/useLiveness";
import { useStableGuidance } from "@hooks/useStableGuidance";
import {
  CAPTURE_MAX_SIDE,
  CAPTURE_QUALITY,
  frameToFile,
  grabFrame,
  overlayBox,
} from "@utils/camera";
import { describeFailure, livenessPrompt, progressFraction } from "@utils/liveness";

interface Props {
  /** Called with the still captured the instant the blink lands. */
  onVerified: (file: File) => void;
  onCancel: () => void;
}

const DEFAULT_ASPECT = 4 / 3;

/**
 * Camera capture gated behind a hold-then-blink check.
 *
 * The subject holds still while the server counts frames that clear the quality
 * gate — those frames also build the eye baseline — and is then asked to blink.
 * The still is grabbed automatically once both are done, rather than handing
 * control back for a manual capture, so the frame that gets verified is one the
 * server actually assessed.
 *
 * The blink rules out a photograph or a still on a screen. It does NOT rule out
 * a video replay or a live deepfake, both of which blink. See the backend's
 * `app/engine/liveness.py` for the full statement of what this proves.
 */
export function LivenessCheck({ onVerified, onCancel }: Props) {
  const { videoRef, state, error: cameraError, start } = useCamera();
  const { status, detection, lowestRatio, passed, failure, error, retry } = useLiveness(
    videoRef,
    state === "live",
  );
  const [aspect, setAspect] = useState(DEFAULT_ASPECT);
  const [captured, setCaptured] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);

  useEffect(() => void start(), [start]);

  useEffect(() => {
    if (passed !== true || captured) return;
    setCaptured(true);
    const video = videoRef.current;
    if (!video) {
      setCaptureError("The camera stopped before the photo could be taken.");
      return;
    }
    void grabFrame(video, CAPTURE_MAX_SIDE, CAPTURE_QUALITY)
      .then((frame) => {
        if (!frame) {
          setCaptureError("The photo could not be captured. Try again.");
          return;
        }
        onVerified(frameToFile(frame.blob, "liveness-capture.jpg"));
      })
      .catch(() => setCaptureError("The photo could not be captured. Try again."));
  }, [passed, captured, videoRef, onVerified]);

  // Self-contained, like CameraCapture: this panel sits inside both the light
  // candidate card and the dark registration panel, so it inherits neither.
  const shell = "rounded-xl bg-stone-900 p-3 text-white";
  // The box tracks every frame; the words only change once a verdict has held
  // for a few frames, so a face on the quality threshold does not strobe.
  const guidance = useStableGuidance(detection);
  const prompt = livenessPrompt(status, guidance);
  const [done, total] = status?.progress ?? [0, 0];
  const failed = passed === false;
  const box =
    detection?.result.bounding_box &&
    overlayBox(
      detection.result.bounding_box,
      { width: detection.frameWidth, height: detection.frameHeight },
      true,
    );

  const retryButton = (label: string, onClick: () => void) => (
    <button
      type="button"
      className="rounded-xl bg-lime-200 px-4 py-2.5 text-sm font-bold text-emerald-950"
      onClick={onClick}
    >
      {label}
    </button>
  );

  if (state === "error") {
    return (
      <div className={shell}>
        <p className="px-1 py-3 text-sm text-orange-200" role="alert">
          {cameraError}
        </p>
        <div className="flex gap-2">
          {retryButton("Try again", () => void start())}
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
          playsInline
          muted
          autoPlay
          // Mirrored, as a mirror is: it is what people expect of a preview of
          // themselves. Nothing here is direction-dependent, so it is purely a
          // comfort affordance.
          className="h-full w-full -scale-x-100 object-cover"
          onLoadedMetadata={(event) => {
            const video = event.currentTarget;
            if (video.videoWidth && video.videoHeight) {
              setAspect(video.videoWidth / video.videoHeight);
            }
          }}
        />
        {box && (
          <div
            className={`pointer-events-none absolute rounded-lg border-2 transition-all duration-150 ${
              prompt.active ? "border-lime-300" : "border-orange-300"
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
        {state !== "live" && (
          <p className="absolute inset-0 grid place-items-center text-sm text-stone-300">
            Starting the camera…
          </p>
        )}
      </div>

      {failed ? (
        <div className="mt-3">
          <p className="mb-2 text-sm text-orange-200" role="alert">
            {describeFailure(failure)}
          </p>
          <div className="flex gap-2">
            {retryButton("Try again", retry)}
            <button
              type="button"
              className="rounded-xl border border-stone-600 px-4 py-2.5 text-sm font-bold text-white"
              onClick={onCancel}
            >
              Use a file instead
            </button>
          </div>
        </div>
      ) : (
        <div className="mt-3" aria-live="polite">
          <p
            className={`text-sm font-bold ${
              prompt.active ? "text-lime-200" : "text-stone-200"
            }`}
          >
            {prompt.message}
          </p>
          {total > 0 && (
            <div className="mt-2">
              <div
                className="h-1.5 w-full overflow-hidden rounded-full bg-stone-700"
                role="progressbar"
                aria-valuenow={done}
                aria-valuemin={0}
                aria-valuemax={total}
              >
                <div
                  className="h-full rounded-full bg-lime-300 transition-[width] duration-300"
                  style={{ width: `${progressFraction(status!.progress) * 100}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-stone-400">
                {done} of {total} steps
              </p>
              {/* Development only. The closed threshold cannot be set from still
                  photographs — what it has to clear is how far an open eye
                  wanders frame to frame — so the live numbers are shown here to
                  be read off a real face and a real camera. `now` should sit
                  near 1.00 with the eyes open; `min` is how deep a blink got. */}
              {import.meta.env.DEV && status?.eye && (
                <p className="mt-1 font-mono text-[11px] tabular-nums text-stone-500">
                  eye now {status.eye.ratio?.toFixed(2) ?? "—"} · min{" "}
                  {lowestRatio?.toFixed(2) ?? "—"} · blink fires below 0.75
                </p>
              )}
            </div>
          )}
          <button
            type="button"
            className="mt-3 rounded-xl border border-stone-600 px-4 py-2 text-xs font-bold text-white"
            onClick={onCancel}
          >
            Cancel
          </button>
        </div>
      )}

      {error && (
        <p className="mt-2 text-sm text-orange-200" role="alert">
          {error}
        </p>
      )}
      {captureError && (
        <p className="mt-2 text-sm text-orange-200" role="alert">
          {captureError}
        </p>
      )}
    </div>
  );
}
