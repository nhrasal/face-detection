import { FaceService } from "@services/face.service";
import type { User } from "@interfaces/face";

/**
 * The profile once it is settled. Collapsing to a single row keeps the search
 * box from occupying the top of the screen for the rest of the session, while
 * still showing which profile the comparison below is acting on.
 */
export function SelectedProfile({ user }: { user: User }) {
  return (
    <div className="flex items-center gap-3">
      <img
        className="size-9 shrink-0 rounded-md border border-line bg-sunken object-cover"
        src={FaceService.profileImageUrl(user.id)}
        alt=""
        loading="lazy"
      />
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-ink">{user.name}</p>
        <p className="truncate font-mono text-xs text-ink-soft">{user.external_id}</p>
      </div>
    </div>
  );
}

/** Sits in the card header, where the mode tabs are before a profile is chosen. */
export function SelectedProfileActions({ onChange }: { onChange: () => void }) {
  return (
    <button
      type="button"
      className="rounded-md border border-line px-2.5 py-1 text-xs font-medium text-ink-soft transition-colors hover:border-line-strong hover:text-ink"
      onClick={onChange}
    >
      Change profile
    </button>
  );
}
