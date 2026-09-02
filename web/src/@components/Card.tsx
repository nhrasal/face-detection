import type { ReactNode } from "react";

interface Props {
  title: string;
  /** Right-hand controls in the header — tabs, a Change button, a status chip. */
  actions?: ReactNode;
  children: ReactNode;
}

/**
 * A titled region of the working surface. Deliberately not a step: everything
 * on this screen is available whenever it makes sense, and a region that has
 * nothing to do yet is simply not rendered rather than shown disabled.
 */
export function Card({ title, actions, children }: Props) {
  return (
    <section className="rounded-lg border border-line bg-surface">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-2.5 sm:px-5">
        <h2 className="font-mono text-[11px] font-medium uppercase tracking-wider text-ink-soft">{title}</h2>
        {actions}
      </header>
      <div className="px-4 py-4 sm:px-5">{children}</div>
    </section>
  );
}
