type Tone = "idle" | "active" | "pass" | "fail";

const tones: Record<Tone, string> = {
  idle: "border-line-strong",
  active: "border-accent",
  pass: "border-pass",
  fail: "border-fail",
};

/**
 * The detector's bounding box, drawn as four corner brackets.
 *
 * This is the one mark that says what the product does, so it is a shared
 * component rather than a per-screen flourish: the same brackets sit on the
 * stored reference, on the candidate, and on the live viewfinder, and they mean
 * the same thing in all three — a face, found, here.
 */
export function DetectionFrame({ tone = "idle", label }: { tone?: Tone; label?: string }) {
  const colour = tones[tone];
  const corner = `absolute size-6 ${colour}`;
  return (
    <>
      <span className={`${corner} left-3 top-3 border-l-2 border-t-2`} aria-hidden="true" />
      <span className={`${corner} right-3 top-3 border-r-2 border-t-2`} aria-hidden="true" />
      <span className={`${corner} bottom-3 left-3 border-b-2 border-l-2`} aria-hidden="true" />
      <span className={`${corner} bottom-3 right-3 border-b-2 border-r-2`} aria-hidden="true" />
      {label && (
        <span
          className={`absolute bottom-3 left-1/2 -translate-x-1/2 rounded bg-canvas/80 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider backdrop-blur-sm ${
            tone === "pass" ? "text-pass" : tone === "fail" ? "text-fail" : tone === "active" ? "text-accent" : "text-ink-soft"
          }`}
        >
          {label}
        </span>
      )}
    </>
  );
}
