import { SegmentedTabs } from "@components/SegmentedTabs";

export type ProfileMode = "lookup" | "register";

const TABS = [
  { value: "lookup", label: "Find existing" },
  { value: "register", label: "Register new" },
] as const satisfies ReadonlyArray<{ value: ProfileMode; label: string }>;

interface Props {
  mode: ProfileMode;
  disabled: boolean;
  onChange: (mode: ProfileMode) => void;
}

export function ProfileModeTabs({ mode, disabled, onChange }: Props) {
  return <SegmentedTabs tabs={TABS} value={mode} label="Profile source" disabled={disabled} onChange={onChange} />;
}
