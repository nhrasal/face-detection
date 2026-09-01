import { type ChangeEvent, type FormEvent, useRef, useState } from "react";
import { CameraCapture } from "@components/CameraCapture";
import { PhotoSourceTabs, type PhotoSource } from "@components/PhotoSourceTabs";
import { usePhotoUpload } from "@hooks/usePhotoUpload";

interface Props {
  submitting: boolean;
  onSubmit: (externalId: string, name: string, profileImage: File) => void;
}

const field =
  "w-full rounded-xl border border-emerald-700 bg-emerald-900 px-4 py-3 text-white outline-none placeholder:text-emerald-200/40 focus:border-lime-200 focus:ring-4 focus:ring-lime-200/10";

export function UserRegistration({ submitting, onSubmit }: Props) {
  const [externalId, setExternalId] = useState("");
  const [name, setName] = useState("");
  const [photoError, setPhotoError] = useState<string | null>(null);
  const [missing, setMissing] = useState<string | null>(null);
  const [source, setSource] = useState<PhotoSource>("upload");
  const photo = usePhotoUpload();
  const inputRef = useRef<HTMLInputElement>(null);

  const choose = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setPhotoError(photo.select(file));
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!externalId.trim() || !name.trim()) return setMissing("External ID and name are both required.");
    if (!photo.file) return setMissing("Choose a profile photograph.");
    setMissing(null);
    onSubmit(externalId.trim(), name.trim(), photo.file);
  };

  return (
    <form onSubmit={submit} noValidate>
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label className="mb-2 block text-xs text-emerald-100/70" htmlFor="external-id">External ID</label>
          <input
            id="external-id"
            className={field}
            value={externalId}
            onChange={(event) => setExternalId(event.target.value)}
            placeholder="employee-123"
            autoComplete="off"
            spellCheck={false}
            maxLength={255}
          />
        </div>
        <div>
          <label className="mb-2 block text-xs text-emerald-100/70" htmlFor="full-name">Name</label>
          <input
            id="full-name"
            className={field}
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Example User"
            autoComplete="off"
            maxLength={255}
          />
        </div>
      </div>

      <div className="mt-4 mb-2 flex items-center justify-between gap-4">
        <p className="text-xs text-emerald-100/70">Profile photograph</p>
        <PhotoSourceTabs source={source} onChange={setSource} />
      </div>
      <input ref={inputRef} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={choose} />
      {source === "camera" ? (
        <CameraCapture
          onCapture={(file) => {
            setPhotoError(photo.select(file));
            setMissing(null);
            // Back to the thumbnail so the operator sees what was captured, and
            // so the camera stream is released rather than left running.
            setSource("upload");
          }}
          onCancel={() => setSource("upload")}
        />
      ) : (
        <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center">
          <div className="size-24 shrink-0 overflow-hidden rounded-xl border border-emerald-700 bg-emerald-900">
            {photo.previewUrl ? (
              <img className="size-full object-cover" src={photo.previewUrl} alt="Profile preview" />
            ) : (
              <span className="grid size-full place-items-center text-xs text-emerald-200/40">No photo</span>
            )}
          </div>
          <div>
            <button type="button" className="rounded-xl border border-emerald-700 px-4 py-2.5 text-sm font-bold text-emerald-100" onClick={() => inputRef.current?.click()}>
              {photo.file ? "Change photo" : "Choose photo"}
            </button>
            <p className="mt-2 text-xs text-emerald-200/50">JPEG, PNG, or WebP · max 5 MB</p>
          </div>
        </div>
      )}

      {(photoError || missing) && (
        <p className="mt-3 text-sm text-orange-300" role="alert">{photoError || missing}</p>
      )}

      <button
        type="submit"
        className="mt-5 rounded-xl bg-lime-200 px-5 py-3 font-bold text-emerald-950 disabled:cursor-not-allowed disabled:opacity-50"
        disabled={submitting}
      >
        {submitting ? "Registering…" : "Register profile"}
      </button>
    </form>
  );
}
