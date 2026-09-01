interface Props {
  userId: string;
  loading: boolean;
  onUserIdChange: (value: string) => void;
  onLoad: () => void;
}

export function ProfileLookup({ userId, loading, onUserIdChange, onLoad }: Props) {
  return (
    <section className="grid gap-6 rounded-2xl bg-emerald-950 p-6 text-white shadow-xl shadow-emerald-950/10 md:grid-cols-[220px_1fr] md:p-7" aria-labelledby="lookup-title">
      <div>
        <span className="mb-2 block text-xs font-bold tracking-[.18em] text-emerald-300">01</span>
        <h2 id="lookup-title" className="font-serif text-3xl">Find the profile</h2>
      </div>
      <div>
        <label className="mb-2 block text-xs text-emerald-100/70" htmlFor="user-id">User UUID</label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            id="user-id"
            className="min-w-0 flex-1 rounded-xl border border-emerald-700 bg-emerald-900 px-4 py-3 text-white outline-none placeholder:text-emerald-200/40 focus:border-lime-200 focus:ring-4 focus:ring-lime-200/10"
            value={userId}
            onChange={(event) => onUserIdChange(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && onLoad()}
            placeholder="00000000-0000-0000-0000-000000000000"
            autoComplete="off"
            spellCheck={false}
          />
          <button className="rounded-xl bg-lime-200 px-5 py-3 font-bold text-emerald-950 disabled:cursor-not-allowed disabled:opacity-50" onClick={onLoad} disabled={loading}>
            {loading ? "Loading…" : "Load profile"}
          </button>
        </div>
      </div>
    </section>
  );
}
