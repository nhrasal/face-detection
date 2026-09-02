interface Tab<T extends string> {
  value: T;
  label: string;
}

interface Props<T extends string> {
  tabs: ReadonlyArray<Tab<T>>;
  value: T;
  label: string;
  disabled?: boolean;
  onChange: (value: T) => void;
}

/**
 * The one segmented control in the app. Both the profile mode switch and the
 * photo source switch are the same interaction — pick one of a few peers — so
 * they share a body rather than drifting apart a class at a time.
 */
export function SegmentedTabs<T extends string>({ tabs, value, label, disabled = false, onChange }: Props<T>) {
  return (
    <div
      className="inline-flex shrink-0 rounded-md border border-line bg-sunken p-0.5"
      role="tablist"
      aria-label={label}
    >
      {tabs.map((tab) => {
        const selected = tab.value === value;
        return (
          <button
            key={tab.value}
            type="button"
            role="tab"
            aria-selected={selected}
            disabled={disabled}
            className={`rounded px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
              selected ? "bg-line-strong/50 text-ink ring-1 ring-line-strong" : "text-ink-soft hover:text-ink"
            }`}
            onClick={() => onChange(tab.value)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
