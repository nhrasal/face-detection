import { type ChangeEvent, type FormEvent, useRef, useState } from "react";
import { Alert } from "@components/Alert";
import { CameraCapture } from "@components/CameraCapture";
import { LivenessCheck } from "@components/LivenessCheck";
import { PhotoSourceTabs, type PhotoSource } from "@components/PhotoSourceTabs";
import { usePhotoUpload } from "@hooks/usePhotoUpload";

interface Props {
  submitting: boolean;
  onSubmit: (externalId: string, name: string, profileImage: File) => void;
}

const field =
  "w-full rounded-md border border-line bg-sunken px-3 py-2 text-sm text-ink outline-none transition-shadow placeholder:text-ink-muted focus:border-accent focus:ring-2 focus:ring-accent/25";
const label = "mb-1.5 block text-xs font-medium text-ink-soft";

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
          <label className={label} htmlFor="external-id">External ID</label>
          <input
            id="external-id"
            className={`${field} font-mono`}
            value={externalId}
            onChange={(event) => setExternalId(event.target.value)}
            placeholder="employee-123"
            autoComplete="off"
            spellCheck={false}
            maxLength={255}
          />
        </div>
        <div>
          <label className={label} htmlFor="full-name">Name</label>
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

      <div className="mt-5 mb-2 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
        <p className={`${label} mb-0`}>Profile photograph</p>
        <PhotoSourceTabs source={source} onChange={setSource} />
      </div>
      <input ref={inputRef} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={choose} />
      {source === "liveness" ? (
        <LivenessCheck
          onVerified={(file) => {
            setPhotoError(photo.select(file));
            setMissing(null);
            setSource("upload");
          }}
          onCancel={() => setSource("upload")}
        />
      ) : source === "camera" ? (
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
          <div className="size-20 shrink-0 overflow-hidden rounded-md border border-line bg-sunken">
            {photo.previewUrl ? (
              <img className="size-full object-cover" src={photo.previewUrl} alt="Profile preview" />
            ) : (
              <span className="grid size-full place-items-center text-[10px] uppercase tracking-wider text-ink-muted">
                No photo
              </span>
            )}
          </div>
          <div>
            <button
              type="button"
              className="rounded-md border border-line px-3 py-1.5 text-sm font-medium text-ink transition-colors hover:bg-line/60"
              onClick={() => inputRef.current?.click()}
            >
              {photo.file ? "Change photo" : "Choose photo"}
            </button>
            <p className="mt-1.5 text-xs text-ink-muted">JPEG, PNG, or WebP · max 5 MB</p>
          </div>
        </div>
      )}

      {(photoError || missing) && <div className="mt-4"><Alert>{photoError || missing}</Alert></div>}

      <div className="mt-5 border-t border-line pt-4">
        <button
          type="submit"
          className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-canvas transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-40"
          disabled={submitting}
        >
          {submitting ? "Registering…" : "Register profile"}
        </button>
      </div>
    </form>
  );
}
