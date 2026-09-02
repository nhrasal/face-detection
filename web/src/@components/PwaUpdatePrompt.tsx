import { useRegisterSW } from "virtual:pwa-register/react";

export function PwaUpdatePrompt() {
  const { needRefresh: [needRefresh, setNeedRefresh], updateServiceWorker } = useRegisterSW();
  if (!needRefresh) return null;
  return (
    <aside
      className="fixed inset-x-4 bottom-4 z-50 mx-auto flex max-w-md items-center justify-between gap-4 rounded-lg border border-line bg-surface p-3 text-sm shadow-lg"
      role="status"
    >
      <span className="text-ink">A new version of Face Check is ready.</span>
      <div className="flex shrink-0 gap-1">
        <button
          className="rounded-md px-2.5 py-1.5 text-xs font-medium text-ink-soft transition-colors hover:bg-line/60"
          onClick={() => setNeedRefresh(false)}
        >
          Later
        </button>
        <button
          className="rounded-md bg-accent px-2.5 py-1.5 text-xs font-semibold text-canvas transition-colors hover:bg-accent-strong"
          onClick={() => void updateServiceWorker(true)}
        >
          Update
        </button>
      </div>
    </aside>
  );
}
