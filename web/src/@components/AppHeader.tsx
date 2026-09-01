export function AppHeader() {
  return (
    <header className="flex h-20 items-center justify-between border-b border-stone-300">
      <a className="flex items-center gap-3 font-bold text-emerald-950 no-underline" href="/" aria-label="Face Check home">
        <span className="grid size-9 place-items-center rounded-full bg-emerald-950 text-xs tracking-widest text-white" aria-hidden="true">FC</span>
        <span>Face Check</span>
      </a>
      <span className="hidden text-sm text-stone-500 sm:block">Private by design · candidate photos are not stored</span>
    </header>
  );
}
