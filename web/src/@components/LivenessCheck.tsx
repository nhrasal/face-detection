import { useEffect, useState } from "react";
import { useCamera } from "@hooks/useCamera";
import { useLiveness } from "@hooks/useLiveness";
import { useStableGuidance } from "@hooks/useStableGuidance";
import {
  CAPTURE_ASPECT,
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
    void grabFrame(video, CAPTURE_MAX_SIDE, CAPTURE_QUALITY, CAPTURE_ASPECT)
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
  const shell = "rounded-lg border border-line bg-sunken p-3 text-ink";
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
      false,
      CAPTURE_ASPECT,
    );

  const retryButton = (label: string, onClick: () => void) => (
    <button
      type="button"
      className="rounded-md bg-accent px-3 py-2 text-sm font-semibold text-canvas transition-colors hover:bg-accent-strong"
      onClick={onClick}
    >
      {label}
    </button>
  );

  if (state === "error") {
    return (
      <div className={shell}>
        <p className="px-1 py-3 text-sm text-fail" role="alert">
          {cameraError}
        </p>
        <div className="flex gap-2">
          {retryButton("Try again", () => void start())}
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
          playsInline
          muted
          autoPlay
          // Not mirrored, matching the camera panel: the captured still is the
          // raw view, and a preview that disagreed with it flipped the photo at
          // the moment of capture.
          className="h-full w-full object-cover"
        />
        {box && (
          <div
            className={`pointer-events-none absolute rounded border-2 transition-all duration-150 ${
              prompt.active ? "border-pass" : "border-review"
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
          <p className="absolute inset-0 grid place-items-center text-sm text-ink-soft">
            Starting the camera…
          </p>
        )}
      </div>

      {failed ? (
        <div className="mt-3">
          <p className="mb-2 text-sm text-fail" role="alert">
            {describeFailure(failure)}
          </p>
          <div className="flex gap-2">
            {retryButton("Try again", retry)}
            <button
              type="button"
              className="rounded-md border border-line px-3 py-2 text-sm font-medium text-ink transition-colors hover:bg-line/60"
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
              prompt.active ? "text-pass" : "text-ink"
            }`}
          >
            {prompt.message}
          </p>
          {total > 0 && (
            <div className="mt-2">
              <div
                className="h-1.5 w-full overflow-hidden rounded-full bg-line"
                role="progressbar"
                aria-valuenow={done}
                aria-valuemin={0}
                aria-valuemax={total}
              >
                <div
                  className="h-full rounded-full bg-pass transition-[width] duration-300"
                  style={{ width: `${progressFraction(status!.progress) * 100}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-ink-muted">
                {done} of {total} steps
              </p>
              {/* Development only. The closed threshold cannot be set from still
                  photographs — what it has to clear is how far an open eye
                  wanders frame to frame — so the live numbers are shown here to
                  be read off a real face and a real camera. `now` should sit
                  near 1.00 with the eyes open; `min` is how deep a blink got. */}
              {import.meta.env.DEV && status?.eye && (
                <p className="mt-1 font-mono text-[11px] tabular-nums text-ink-muted">
                  eye {status.eye.ratio?.toFixed(2) ?? "—"} · min{" "}
                  {lowestRatio?.toFixed(2) ?? "—"} · fires below 0.75
                  <span className={status.eye.trusted ? "" : "text-fail"}>
                    {" "}
                    · sharp {status.eye.sharpness_ratio?.toFixed(2) ?? "—"}{" "}
                    {status.eye.trusted ? "used" : "DISCARDED"}
                  </span>
                </p>
              )}
            </div>
          )}
          <button
            type="button"
            className="mt-3 rounded-md border border-line px-3 py-1.5 text-xs font-medium text-ink transition-colors hover:bg-line/60"
            onClick={onCancel}
          >
            Cancel
          </button>
        </div>
      )}

      {error && (
        <p className="mt-2 text-sm text-fail" role="alert">
          {error}
        </p>
      )}
      {captureError && (
        <p className="mt-2 text-sm text-fail" role="alert">
          {captureError}
        </p>
      )}
    </div>
  );
}
