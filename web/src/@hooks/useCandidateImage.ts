import { useEffect, useState } from "react";
import { validateCandidate } from "@utils/verification";

export function useCandidateImage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const select = (next?: File): string | null => {
    if (!next) return null;
    const error = validateCandidate(next);
    // A rejected pick must drop the previous photo too, or the operator sees an
    // error about the new file while the old one stays armed for verification.
    setFile(error ? null : next);
    return error;
  };

  const reset = () => setFile(null);
  return { file, previewUrl, select, reset };
}
