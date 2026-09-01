import { useEffect, useState } from "react";
import { useCamera } from "@hooks/useCamera";
import { useLiveness } from "@hooks/useLiveness";
import { CAPTURE_MAX_SIDE, CAPTURE_QUALITY, frameToFile, grabFrame } from "@utils/camera";
import { describeFailure, livenessPrompt, progressFraction } from "@utils/liveness";

interface Props {
  /** Called with the still captured the instant liveness passes. */
  onVerified: (file: File) => void;
  onCancel: () => void;
}

const DEFAULT_ASPECT = 4 / 3;

/**
 * Camera capture gated behind a challenge-response liveness check.
 *
 * The still is grabbed automatically the moment the server reports PASSED,
 * rather than handing control back for a manual capture. Someone who has just
 * proved they are present should not then get a free window in which to hold up
 * a photograph for the frame that actually gets verified.
 */
export function LivenessCheck({ onVerified, onCancel }: Props) {
  const { videoRef, state, error: cameraError, start } = useCamera();
  const { status, passed, failure, error, retry } = useLiveness(videoRef, state === "live");
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
  const prompt = livenessPrompt(status);
  const [done, total] = status?.progress ?? [0, 0];
  const failed = passed === false;

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
          // Mirrored so turning feels natural, as a mirror does. Instructions
          // describe the person's own body, so they stay correct either way.
          className="h-full w-full -scale-x-100 object-cover"
          onLoadedMetadata={(event) => {
            const video = event.currentTarget;
            if (video.videoWidth && video.videoHeight) {
              setAspect(video.videoWidth / video.videoHeight);
            }
          }}
        />
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
              prompt.settle ? "text-stone-200" : "text-lime-200"
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
                {done} of {total} steps complete
              </p>
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
