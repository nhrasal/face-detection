export function AppHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-canvas/85 backdrop-blur">
      <div className="mx-auto flex h-14 w-full max-w-5xl items-center justify-between gap-4 px-4 sm:px-6">
        <a className="flex items-center gap-2.5 no-underline" href="/" aria-label="Face Check home">
          {/* The mark is the product: a face inside a detection frame. */}
          <span className="relative grid size-7 place-items-center rounded-md border border-accent-line bg-accent-soft" aria-hidden="true">
            <span className="absolute left-1 top-1 size-1.5 border-l border-t border-accent" />
            <span className="absolute right-1 top-1 size-1.5 border-r border-t border-accent" />
            <span className="absolute bottom-1 left-1 size-1.5 border-b border-l border-accent" />
            <span className="absolute bottom-1 right-1 size-1.5 border-b border-r border-accent" />
            <span className="size-1.5 rounded-full bg-accent" />
          </span>
          <span className="text-sm font-semibold tracking-tight text-ink">Face Check</span>
          <span className="hidden rounded border border-line px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-ink-muted sm:inline">
            v1
          </span>
        </a>
        <span className="hidden items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-ink-soft sm:flex">
          <span className="size-1.5 rounded-full bg-pass" aria-hidden="true" />
          Candidate photos not stored
        </span>
      </div>
    </header>
  );
}
