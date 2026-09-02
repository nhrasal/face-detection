import type { ReactNode } from "react";

type Tone = "error" | "warning" | "info";

const tones: Record<Tone, { shell: string; mark: string; glyph: string }> = {
  error: { shell: "border-fail-line bg-fail-soft text-fail", mark: "bg-fail", glyph: "!" },
  warning: { shell: "border-review-line bg-review-soft text-review", mark: "bg-review", glyph: "!" },
  info: { shell: "border-accent-line bg-accent-soft text-accent-strong", mark: "bg-accent", glyph: "i" },
};

interface Props {
  tone?: Tone;
  children: ReactNode;
}

/** Every inline message in the app, so a failure never looks like a new design. */
export function Alert({ tone = "error", children }: Props) {
  const style = tones[tone];
  return (
    <div className={`flex items-start gap-2.5 rounded-md border px-3 py-2.5 text-sm ${style.shell}`} role="alert">
      <span
        className={`mt-px grid size-4 shrink-0 place-items-center rounded-full text-[10px] font-bold text-canvas ${style.mark}`}
        aria-hidden="true"
      >
        {style.glyph}
      </span>
      <span className="min-w-0">{children}</span>
    </div>
  );
}
