import type { VerificationResult as Result } from "@interfaces/face";
import { asPercentage, resultPresentation, similarityLabel } from "@utils/verification";

const tones = {
  success: { chip: "border-pass-line bg-pass-soft text-pass", mark: "bg-pass", bar: "bg-pass", glyph: "✓", label: "Match" },
  warning: { chip: "border-review-line bg-review-soft text-review", mark: "bg-review", bar: "bg-review", glyph: "!", label: "Review" },
  danger: { chip: "border-fail-line bg-fail-soft text-fail", mark: "bg-fail", bar: "bg-fail", glyph: "×", label: "No match" },
} as const;

/**
 * Renders the similarity against the threshold rather than only the verdict.
 * The decision is a comparison of two numbers, and showing where the score fell
 * is what makes a borderline result reviewable instead of merely reported.
 */
function ScoreScale({ similarity, threshold }: { similarity: number; threshold: number; }) {
  const clamp = (value: number) => Math.min(100, Math.max(0, value * 100));
  return (
    <div className="mt-5">
      <div className="mb-1.5 flex items-baseline justify-between text-xs text-ink-soft">
        <span>Similarity against threshold</span>
        <span className="font-mono text-ink">{similarity.toFixed(4)}</span>
      </div>
      <div className="relative h-2 overflow-hidden rounded-full bg-sunken ring-1 ring-line">
        <div
          className={`h-full rounded-full ${similarity >= threshold ? "bg-pass" : "bg-fail"}`}
          style={{ width: `${clamp(similarity)}%` }}
        />
      </div>
      <div className="relative mt-1 h-4">
        <span
          className="absolute -translate-x-1/2 whitespace-nowrap font-mono text-[10px] text-ink-muted"
          style={{ left: `${clamp(threshold)}%` }}
        >
          ▲ {threshold.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export function VerificationResult({ result }: { result: Result }) {
  const presentation = resultPresentation(result);
  const tone = tones[presentation.tone as keyof typeof tones];
  const metrics = [
    ["Confidence", asPercentage(result.confidence)],
    ["Raw similarity", similarityLabel(result.similarity)],
    ["Threshold", result.threshold.toFixed(3)],
    ["Processing", `${result.processing_time_ms} ms`],
  ];

  return (
    <div aria-live="polite">
      <div className="flex items-start gap-3">
        <span
          className={`grid size-9 shrink-0 place-items-center rounded-full text-lg font-bold text-canvas ${tone.mark}`}
          aria-hidden="true"
        >
          {tone.glyph}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-base font-semibold tracking-tight text-ink">{presentation.title}</h3>
            <span className={`rounded border px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${tone.chip}`}>
              {result.decision.replace(/_/g, " ")}
            </span>
          </div>
          <p className="mt-1 text-sm leading-relaxed text-ink-soft">{presentation.detail}</p>
        </div>
      </div>

      {result.similarity !== null && (
        <ScoreScale similarity={result.similarity} threshold={result.threshold} />
      )}

      <dl className="mt-5 grid grid-cols-2 overflow-hidden rounded-md border border-line sm:grid-cols-4">
        {metrics.map(([label, value], index) => (
          <div
            key={label}
            className={`bg-sunken px-3 py-2.5 ${index < 2 ? "border-b border-line sm:border-b-0" : ""} ${
              index % 2 === 0 ? "border-r border-line" : ""
            } sm:border-r sm:last:border-r-0`}
          >
            <dt className="text-[10px] font-medium uppercase tracking-wider text-ink-muted">{label}</dt>
            <dd className="mt-0.5 font-mono text-sm font-semibold text-ink">{value}</dd>
          </div>
        ))}
      </dl>

      {presentation.guidance.length > 0 && (
        <div className="mt-4 rounded-md border border-line bg-sunken px-3 py-3">
          <p className="mb-1.5 text-xs font-medium text-ink">What to fix</p>
          <ul className="list-disc space-y-1 pl-4 text-sm text-ink-soft">
            {presentation.guidance.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
