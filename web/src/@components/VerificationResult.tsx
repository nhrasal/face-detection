import type { VerificationResult as Result } from "@interfaces/face";
import { asPercentage, resultPresentation, similarityLabel } from "@utils/verification";

const tones = {
  success: { border: "border-emerald-400", symbol: "bg-emerald-700" },
  warning: { border: "border-amber-400", symbol: "bg-amber-600" },
  danger: { border: "border-red-400", symbol: "bg-red-700" },
} as const;

export function VerificationResult({ result }: { result: Result }) {
  const presentation = resultPresentation(result);
  const tone = tones[presentation.tone as keyof typeof tones];
  const metrics = [
    ["Confidence", asPercentage(result.confidence)],
    ["Raw similarity", similarityLabel(result.similarity)],
    ["Decision threshold", result.threshold.toFixed(3)],
    ["Processing time", `${result.processing_time_ms} ms`],
  ];
  return (
    <section className={`mt-6 grid items-start gap-6 rounded-2xl border bg-[#fafaf6] p-7 md:grid-cols-[auto_1fr_minmax(300px,.7fr)] ${tone.border}`} aria-live="polite">
      <div className={`grid size-14 place-items-center rounded-full text-3xl text-white ${tone.symbol}`} aria-hidden="true">{result.matched ? "✓" : presentation.tone === "warning" ? "!" : "×"}</div>
      <div>
        <p className="mb-3 text-xs font-bold uppercase tracking-[.16em] text-emerald-700">Verification result</p>
        <h2 className="mb-2 font-serif text-4xl">{presentation.title}</h2>
        <p className="leading-relaxed text-stone-500">{presentation.detail}</p>
        {presentation.guidance.length > 0 && <ul className="mt-3 list-disc space-y-1 pl-5 text-stone-500">{presentation.guidance.map((item) => <li key={item}>{item}</li>)}</ul>}
      </div>
      <dl className="grid grid-cols-2 overflow-hidden rounded-xl border border-stone-300">
        {metrics.map(([label, value], index) => <div className={`p-4 ${index < 2 ? "border-b border-stone-300" : ""} ${index % 2 === 0 ? "border-r border-stone-300" : ""}`} key={label}><dt className="mb-1 text-xs uppercase tracking-wider text-stone-500">{label}</dt><dd className="m-0 text-lg font-bold">{value}</dd></div>)}
      </dl>
    </section>
  );
}
