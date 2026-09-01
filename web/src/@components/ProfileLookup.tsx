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

export function ProfileLookup({ query, loading, results, onQueryChange, onSearch, onSelect }: Props) {
  return (
    <div>
      <label className="mb-2 block text-xs text-emerald-100/70" htmlFor="user-search">External ID or name</label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          id="user-search"
          className="min-w-0 flex-1 rounded-xl border border-emerald-700 bg-emerald-900 px-4 py-3 text-white outline-none placeholder:text-emerald-200/40 focus:border-lime-200 focus:ring-4 focus:ring-lime-200/10"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && onSearch()}
          placeholder="employee-123 or Ada Lovelace"
          autoComplete="off"
          spellCheck={false}
        />
        <button
          className="rounded-xl bg-lime-200 px-5 py-3 font-bold text-emerald-950 disabled:cursor-not-allowed disabled:opacity-50"
          onClick={onSearch}
          disabled={loading}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </div>

      {results !== null && (
        results.length === 0 ? (
          <p className="mt-4 text-sm text-emerald-200/60">No profile matches that external ID or name.</p>
        ) : (
          <ul className="mt-4 max-h-72 divide-y divide-emerald-800 overflow-y-auto rounded-xl border border-emerald-800" role="list">
            {results.map((user) => (
              <li key={user.id}>
                <button
                  type="button"
                  className="flex w-full items-center gap-3 p-3 text-left hover:bg-emerald-900"
                  onClick={() => onSelect(user)}
                >
                  <img
                    className="size-11 shrink-0 rounded-lg bg-emerald-900 object-cover"
                    src={FaceService.profileImageUrl(user.id)}
                    alt=""
                    loading="lazy"
                  />
                  <span className="min-w-0">
                    <span className="block truncate font-bold">{user.name}</span>
                    <span className="block truncate text-xs text-emerald-200/60">{user.external_id}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )
      )}
    </div>
  );
}
