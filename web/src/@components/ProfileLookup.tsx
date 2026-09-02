import { FaceService } from "@services/face.service";
import type { User } from "@interfaces/face";

interface Props {
  query: string;
  loading: boolean;
  results: User[] | null;
  onQueryChange: (value: string) => void;
  onSearch: () => void;
  onSelect: (user: User) => void;
}

export const MIN_QUERY_LENGTH = 2;

const SKELETON_ROWS = 3;

export function ProfileLookup({ query, loading, results, onQueryChange, onSearch, onSelect }: Props) {
  return (
    <div>
      <label className="mb-1.5 block text-xs font-medium text-ink-soft" htmlFor="user-search">
        External ID or name
      </label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          id="user-search"
          className="min-w-0 flex-1 rounded-md border border-line bg-sunken px-3 py-2 text-sm text-ink outline-none transition-shadow placeholder:text-ink-muted focus:border-accent focus:ring-2 focus:ring-accent/25"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && onSearch()}
          placeholder="employee-123 or Ada Lovelace"
          autoComplete="off"
          spellCheck={false}
        />
        <button
          className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-canvas transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-40"
          onClick={onSearch}
          disabled={loading}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </div>

      {/* A skeleton rather than a spinner: the rows land in the same place they
          were sketched, so the list does not jump as the response arrives. */}
      {loading && (
        <ul className="mt-3 divide-y divide-line overflow-hidden rounded-md border border-line" aria-hidden="true">
          {Array.from({ length: SKELETON_ROWS }, (_, index) => (
            <li key={index} className="flex items-center gap-3 p-2.5">
              <span className="size-9 shrink-0 animate-pulse rounded-md bg-sunken" />
              <span className="min-w-0 flex-1 space-y-1.5">
                <span className="block h-3 w-32 animate-pulse rounded bg-sunken" />
                <span className="block h-2.5 w-20 animate-pulse rounded bg-sunken" />
              </span>
            </li>
          ))}
        </ul>
      )}

      {!loading && results !== null && (
        results.length === 0 ? (
          <p className="mt-3 rounded-md border border-dashed border-line-strong px-3 py-4 text-center text-sm text-ink-soft" role="status">
            No profile matches that external ID or name.
          </p>
        ) : (
          <>
            <p className="mt-4 mb-1.5 text-xs font-medium text-ink-soft" role="status">
              {results.length} {results.length === 1 ? "profile" : "profiles"} found
            </p>
            <ul className="max-h-72 divide-y divide-line overflow-y-auto rounded-md border border-line" role="list">
              {results.map((user) => (
                <li key={user.id}>
                  <button
                    type="button"
                    className="flex w-full items-center gap-3 p-2.5 text-left transition-colors hover:bg-line/60"
                    onClick={() => onSelect(user)}
                  >
                    <img
                      className="size-9 shrink-0 rounded-md border border-line bg-sunken object-cover"
                      src={FaceService.profileImageUrl(user.id)}
                      alt=""
                      loading="lazy"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-ink">{user.name}</span>
                      <span className="block truncate font-mono text-xs text-ink-soft">{user.external_id}</span>
                    </span>
                    <span className="shrink-0 text-xs text-ink-muted" aria-hidden="true">Select →</span>
                  </button>
                </li>
              ))}
            </ul>
          </>
        )
      )}
    </div>
  );
}
