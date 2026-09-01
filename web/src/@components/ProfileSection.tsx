import type { ReactNode } from "react";

export type ProfileMode = "lookup" | "register";

interface Props {
  mode: ProfileMode;
  disabled: boolean;
  onModeChange: (mode: ProfileMode) => void;
  children: ReactNode;
}

const TABS: Array<{ value: ProfileMode; label: string }> = [
  { value: "lookup", label: "Find existing" },
  { value: "register", label: "Register new" },
];

export function ProfileSection({ mode, disabled, onModeChange, children }: Props) {
  return (
    <section
      className="grid gap-6 rounded-2xl bg-emerald-950 p-6 text-white shadow-xl shadow-emerald-950/10 md:grid-cols-[220px_1fr] md:p-7"
      aria-labelledby="profile-title"
    >
      <div>
        <span className="mb-2 block text-xs font-bold tracking-[.18em] text-emerald-300">01</span>
        <h2 id="profile-title" className="mb-4 font-serif text-3xl">
          {mode === "lookup" ? "Find the profile" : "Register a profile"}
        </h2>
        <div className="inline-flex rounded-lg bg-emerald-900 p-1" role="tablist" aria-label="Profile source">
          {TABS.map((tab) => (
            <button
              key={tab.value}
              type="button"
              role="tab"
              aria-selected={mode === tab.value}
              disabled={disabled}
              className={`rounded-md px-3 py-1.5 text-xs font-bold disabled:cursor-not-allowed disabled:opacity-50 ${
                mode === tab.value ? "bg-lime-200 text-emerald-950" : "text-emerald-100"
              }`}
              onClick={() => onModeChange(tab.value)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      <div>{children}</div>
    </section>
  );
}
