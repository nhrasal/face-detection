import { type ChangeEvent, type DragEvent, useEffect, useRef, useState } from "react";
import { CameraCapture } from "@components/CameraCapture";
import { LivenessCheck } from "@components/LivenessCheck";
import { PhotoSourceTabs, type PhotoSource } from "@components/PhotoSourceTabs";
import { FaceService } from "@services/face.service";
import type { User } from "@interfaces/face";

interface Props {
  user: User;
  candidate: File | null;
  previewUrl: string | null;
  verifying: boolean;
  onSelect: (file?: File) => void;
  onReset: () => void;
  onVerify: () => void;
}

const Corner = ({ className }: { className: string }) => (
  <span className={`absolute size-7 border-lime-200 ${className}`} aria-hidden="true" />
);

export function PortraitComparison({ user, candidate, previewUrl, verifying, onSelect, onReset, onVerify }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [referenceFailed, setReferenceFailed] = useState(false);
  const [source, setSource] = useState<PhotoSource>("upload");
  useEffect(() => setReferenceFailed(false), [user.id]);
  const change = (event: ChangeEvent<HTMLInputElement>) => {
    onSelect(event.target.files?.[0]);
    event.target.value = "";
  };
  const drop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    onSelect(event.dataTransfer.files?.[0]);
  };
  const card = "rounded-2xl border border-stone-300 bg-[#fafaf6] p-6";

  return (
    <section className="mt-8 grid gap-6 md:grid-cols-2" aria-label="Face comparison">
      <article className={card}>
        <div className="flex min-h-14 items-start justify-between gap-4">
          <div><span className="mb-2 block text-xs font-bold tracking-[.18em] text-stone-400">02</span><h2 className="font-serif text-3xl">Stored reference</h2></div>
          <span className="rounded-full bg-emerald-100 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-emerald-800">Registered</span>
        </div>
        <div className="relative mt-5 aspect-[16/11] overflow-hidden rounded-xl bg-stone-200">
          {referenceFailed ? (
            <p className="grid size-full place-items-center px-6 text-center text-sm text-stone-500" role="status">
              The stored reference photo could not be loaded.
            </p>
          ) : (
            <img
              className="size-full object-cover"
              src={FaceService.profileImageUrl(user.id)}
              alt={`Stored profile of ${user.name}`}
              onError={() => setReferenceFailed(true)}
            />
          )}
          <Corner className="left-5 top-5 border-l-2 border-t-2" /><Corner className="right-5 top-5 border-r-2 border-t-2" />
          <Corner className="bottom-5 left-5 border-b-2 border-l-2" /><Corner className="bottom-5 right-5 border-b-2 border-r-2" />
        </div>
        <div className="flex items-baseline justify-between gap-3 pt-4"><strong>{user.name}</strong><span className="text-sm text-stone-500">{user.external_id}</span></div>
      </article>

      <article className={card}>
        <div className="flex min-h-14 items-start justify-between gap-4">
          <div><span className="mb-2 block text-xs font-bold tracking-[.18em] text-stone-400">03</span><h2 className="font-serif text-3xl">Candidate photo</h2></div>
          {candidate
            ? <button className="p-1 text-sm text-emerald-700 underline" onClick={onReset}>Remove</button>
            : <PhotoSourceTabs source={source} onChange={setSource} />}
        </div>
        <input ref={inputRef} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={change} />
        {!candidate && source === "liveness" ? (
          <div className="mt-5">
            <LivenessCheck onVerified={onSelect} onCancel={() => setSource("upload")} />
          </div>
        ) : !candidate && source === "camera" ? (
          <div className="mt-5">
            <CameraCapture onCapture={onSelect} onCancel={() => setSource("upload")} />
          </div>
        ) : (
          <button
            className="mt-5 grid aspect-[16/11] w-full place-items-center overflow-hidden rounded-xl border-2 border-dashed border-stone-300 bg-stone-100 text-emerald-950 hover:border-emerald-700"
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => event.preventDefault()}
            onDrop={drop}
            type="button"
          >
            {previewUrl ? <img className="size-full object-cover" src={previewUrl} alt="Candidate preview" /> : (
              <span className="flex flex-col items-center gap-2 px-4"><span className="grid size-11 place-items-center rounded-full bg-emerald-950 text-2xl text-white">↑</span><strong>Add a portrait</strong><small className="text-stone-500">Drop here or browse · JPEG, PNG, WebP · max 5 MB</small></span>
            )}
          </button>
        )}
        <button className="mt-4 min-h-12 w-full rounded-xl bg-emerald-700 px-5 py-3 font-bold text-white disabled:cursor-not-allowed disabled:opacity-50" disabled={!candidate || verifying} onClick={onVerify}>
          {verifying ? "Comparing faces…" : "Verify identity"}
        </button>
      </article>
    </section>
  );
}
