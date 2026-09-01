export type PhotoSource = "upload" | "camera";

interface Props {
  source: PhotoSource;
  onChange: (source: PhotoSource) => void;
}

const TABS: Array<{ value: PhotoSource; label: string }> = [
  { value: "upload", label: "Upload" },
  { value: "camera", label: "Camera" },
];

export function PhotoSourceTabs({ source, onChange }: Props) {
  return (
    <div className="inline-flex shrink-0 rounded-lg bg-stone-200 p-1" role="tablist" aria-label="Photo source">
      {TABS.map((tab) => (
        <button
          key={tab.value}
          type="button"
          role="tab"
          aria-selected={source === tab.value}
          className={`rounded-md px-3 py-1.5 text-xs font-bold ${
            source === tab.value ? "bg-emerald-950 text-white" : "text-emerald-950"
          }`}
          onClick={() => onChange(tab.value)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
