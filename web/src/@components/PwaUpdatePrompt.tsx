import { useRegisterSW } from "virtual:pwa-register/react";

export function PwaUpdatePrompt() {
  const { needRefresh: [needRefresh, setNeedRefresh], updateServiceWorker } = useRegisterSW();
  if (!needRefresh) return null;
  return <aside className="fixed inset-x-4 bottom-4 z-50 mx-auto flex max-w-xl items-center justify-between gap-4 rounded-xl bg-emerald-950 p-4 text-sm text-white shadow-2xl" role="status">
    <span>A new version of Face Check is ready.</span>
    <div className="flex gap-2"><button className="rounded-lg px-3 py-2 text-emerald-100" onClick={() => setNeedRefresh(false)}>Later</button><button className="rounded-lg bg-lime-200 px-3 py-2 font-bold text-emerald-950" onClick={() => void updateServiceWorker(true)}>Update</button></div>
  </aside>;
}
