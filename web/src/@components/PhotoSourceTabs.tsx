import { SegmentedTabs } from "@components/SegmentedTabs";

export type PhotoSource = "upload" | "camera" | "liveness";

const TABS = [
  { value: "upload", label: "Upload" },
  { value: "camera", label: "Camera" },
  // A separate source rather than a toggle on "Camera": this flow asks for a
  // blink and captures the still itself once it lands, so it is a different
  // interaction, not the same one with a setting. Still labelled "Live check"
  // rather than "Liveness": a blink rules out a photograph, but a video replay
  // and a deepfake both blink — see backend app/engine/liveness.py.
  { value: "liveness", label: "Live check" },
] as const satisfies ReadonlyArray<{ value: PhotoSource; label: string }>;

interface Props {
  source: PhotoSource;
  onChange: (source: PhotoSource) => void;
}

export function PhotoSourceTabs({ source, onChange }: Props) {
  return <SegmentedTabs tabs={TABS} value={source} label="Photo source" onChange={onChange} />;
}
