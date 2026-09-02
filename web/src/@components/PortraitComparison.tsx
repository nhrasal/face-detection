import { type ChangeEvent, type DragEvent, useEffect, useRef, useState } from "react";
import { CameraCapture } from "@components/CameraCapture";
import { DetectionFrame } from "@components/DetectionFrame";
import { LivenessCheck } from "@components/LivenessCheck";
import { PhotoSourceTabs, type PhotoSource } from "@components/PhotoSourceTabs";
import { FaceService } from "@services/face.service";
import { asPercentage } from "@utils/verification";
import type { User, VerificationResult } from "@interfaces/face";

interface Props {
  user: User;
  candidate: File | null;
  previewUrl: string | null;
  verifying: boolean;
  /** Drives the verdict shown on the rail between the two faces. */
  result: VerificationResult | null;
  onSelect: (file?: File) => void;
  /** A photo from the live camera, which verifies without a further click. */
  onCapture: (file: File) => void;
  onReset: () => void;
  onVerify: () => void;
}

const portrait = "relative aspect-[4/5] overflow-hidden rounded-lg border border-line bg-canvas";
const caption = "mb-2 flex min-h-7 items-center justify-between gap-2";
const captionText = "font-mono text-[11px] uppercase tracking-wider text-ink-soft";

/**
 * The rail between the two faces. The product is a comparison, so the score
 * belongs literally between the things being compared rather than in a panel
 * further down the page.
 */
function ComparisonRail({ verifying, result }: { verifying: boolean; result: VerificationResult | null }) {
  const tone = !result ? null : result.matched ? "pass" : result.decision === "REVIEW" ? "review" : "fail";
  const line = tone === "pass" ? "bg-pass" : tone === "fail" ? "bg-fail" : tone === "review" ? "bg-review" : "bg-line-strong";
  const text = tone === "pass" ? "text-pass" : tone === "fail" ? "text-fail" : tone === "review" ? "text-review" : "text-ink-muted";

  return (
    <div className="flex shrink-0 flex-row items-center gap-3 md:h-full md:flex-col md:pt-9">
      <span className={`h-px flex-1 md:h-auto md:w-px md:flex-1 ${line}`} aria-hidden="true" />
      <div className="flex flex-col items-center gap-1 text-center">
        {verifying ? (
          <>
            <span className="size-4 animate-spin rounded-full border-2 border-line-strong border-t-accent" aria-hidden="true" />
            <span className="font-mono text-[10px] uppercase tracking-wider text-accent">Comparing</span>
          </>
        ) : result ? (
          <>
            <span className={`font-mono text-2xl font-semibold tabular-nums ${text}`}>
              {asPercentage(result.confidence)}
            </span>
            <span className={`font-mono text-[10px] uppercase tracking-wider ${text}`}>
              {result.decision.replace(/_/g, " ")}
            </span>
          </>
        ) : (
          <>
            <span className="grid size-8 place-items-center rounded-full border border-line-strong text-ink-muted" aria-hidden="true">⇄</span>
            <span className="font-mono text-[10px] uppercase tracking-wider text-ink-muted">Compare</span>
          </>
        )}
      </div>
      <span className={`h-px flex-1 md:h-auto md:w-px md:flex-1 ${line}`} aria-hidden="true" />
    </div>
  );
}

export function PortraitComparison({ user, candidate, previewUrl, verifying, result, onSelect, onCapture, onReset, onVerify }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [referenceFailed, setReferenceFailed] = useState(false);
  const [source, setSource] = useState<PhotoSource>("upload");
  const [dragging, setDragging] = useState(false);
  useEffect(() => setReferenceFailed(false), [user.id]);

  const change = (event: ChangeEvent<HTMLInputElement>) => {
    onSelect(event.target.files?.[0]);
    event.target.value = "";
  };

  const drop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setDragging(false);
    onSelect(event.dataTransfer.files?.[0]);
  };

  const candidateTone = !result ? (candidate ? "active" : "idle") : result.matched ? "pass" : "fail";
  const inViewfinder = !candidate && (source === "camera" || source === "liveness");

  return (
    <div>
      <div className="flex flex-col gap-3 md:flex-row md:items-stretch md:gap-5">
        <div className="min-w-0 flex-1">
          <p className={caption}>
            <span className={captionText}>Stored reference</span>
            <span className="rounded border border-line px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-ink-soft">
              On file
            </span>
          </p>
          <div className={portrait}>
            {referenceFailed ? (
              <p className="grid size-full place-items-center px-6 text-center text-sm text-ink-muted" role="status">
                The stored reference photo could not be loaded.
              </p>
            ) : (
              <>
                <img
                  className="absolute inset-0 size-full object-contain"
                  src={FaceService.profileImageUrl(user.id)}
                  alt={`Stored profile of ${user.name}`}
                  onError={() => setReferenceFailed(true)}
                />
                <DetectionFrame tone="idle" label="Reference" />
              </>
            )}
          </div>
          <div className="mt-2 flex items-baseline justify-between gap-3">
            <span className="truncate text-sm font-medium text-ink">{user.name}</span>
            <span className="shrink-0 font-mono text-xs text-ink-soft">{user.external_id}</span>
          </div>
        </div>

        <ComparisonRail verifying={verifying} result={result} />

        <div className="min-w-0 flex-1">
          <p className={caption}>
            <span className={captionText}>Candidate</span>
            {candidate ? (
              <button
                type="button"
                className="rounded px-1.5 py-0.5 text-xs font-medium text-accent transition-colors hover:text-accent-strong"
                onClick={onReset}
              >
                Remove
              </button>
            ) : (
              <PhotoSourceTabs source={source} onChange={setSource} />
            )}
          </p>
          <input ref={inputRef} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={change} />
          {inViewfinder ? (
            source === "liveness" ? (
              <LivenessCheck onVerified={onCapture} onCancel={() => setSource("upload")} />
            ) : (
              <CameraCapture onCapture={(file) => onSelect(file)} onCancel={() => setSource("upload")} />
            )
          ) : (
            <button
              className={`${portrait} grid w-full place-items-center transition-colors ${
                previewUrl ? "" : "border-dashed"
              } ${dragging ? "border-accent bg-accent-soft" : previewUrl ? "" : "hover:border-line-strong hover:bg-surface"} ${
                verifying ? "scanning" : ""
              }`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={drop}
              type="button"
            >
              {previewUrl ? (
                <>
                  <img className="absolute inset-0 size-full object-contain" src={previewUrl} alt="Candidate preview" />
                  <DetectionFrame
                    tone={candidateTone}
                    label={verifying ? "Scanning" : result ? (result.matched ? "Match" : "No match") : "Candidate"}
                  />
                </>
              ) : (
                <span className="flex flex-col items-center gap-1.5 px-4 text-center">
                  <span className="grid size-9 place-items-center rounded-full border border-line-strong text-base text-ink-soft" aria-hidden="true">↑</span>
                  <span className="text-sm font-medium text-ink">Add a portrait</span>
                  <span className="text-xs text-ink-muted">Drop here or browse · JPEG, PNG, WebP · max 5 MB</span>
                </span>
              )}
            </button>
          )}
          {!inViewfinder && (
            <p className="mt-2 truncate text-xs text-ink-muted">
              {candidate ? candidate.name : "No candidate photograph yet"}
            </p>
          )}
        </div>
      </div>

      {/* Still here for the upload path, and as the progress indicator while an
          auto-verify from the camera is in flight. */}
      <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
        <p className="text-xs text-ink-muted">
          {candidate ? "Ready to compare against the stored reference." : "Add a candidate photo to compare."}
        </p>
        <button
          className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-canvas transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-40"
          disabled={!candidate || verifying}
          onClick={onVerify}
        >
          {verifying ? "Comparing faces…" : "Verify identity"}
        </button>
      </div>
    </div>
  );
}
