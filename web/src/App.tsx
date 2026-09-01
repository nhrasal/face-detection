import { useEffect, useRef, useState } from "react";
import { AppHeader } from "@components/AppHeader";
import { PortraitComparison } from "@components/PortraitComparison";
import { ProfileLookup } from "@components/ProfileLookup";
import { PwaUpdatePrompt } from "@components/PwaUpdatePrompt";
import { VerificationResult as ResultPanel } from "@components/VerificationResult";
import { useCandidateImage } from "@hooks/useCandidateImage";
import { ApiError, isCanceled } from "@services/api.instance";
import { FaceService } from "@services/face.service";
import type { User, VerificationResult } from "@interfaces/face";

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function App() {
  const [userId, setUserId] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingUser, setLoadingUser] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const candidate = useCandidateImage();
  const lookupRef = useRef<AbortController | null>(null);
  const verifyRef = useRef<AbortController | null>(null);

  // Abort whatever is in flight so a late response can never paint over a newer one.
  useEffect(() => () => { lookupRef.current?.abort(); verifyRef.current?.abort(); }, []);

  const loadUser = async () => {
    const normalized = userId.trim();
    if (!UUID_PATTERN.test(normalized)) return setError("Enter a valid user UUID.");
    // Switching profiles invalidates a pending verification as well as a pending lookup.
    lookupRef.current?.abort();
    verifyRef.current?.abort();
    const controller = new AbortController();
    lookupRef.current = controller;
    setLoadingUser(true); setError(null); setUser(null); candidate.reset(); setResult(null); setVerifying(false);
    try { setUser(await FaceService.getUser(normalized, controller.signal)); }
    catch (caught) {
      if (isCanceled(caught)) return;
      setError(caught instanceof ApiError ? caught.message : "Could not load this user.");
    }
    finally { if (lookupRef.current === controller) setLoadingUser(false); }
  };

  const chooseFile = (file?: File) => {
    if (!file) return;
    setResult(null);
    setError(candidate.select(file));
  };

  const verify = async () => {
    if (!user || !candidate.file) return;
    verifyRef.current?.abort();
    const controller = new AbortController();
    verifyRef.current = controller;
    setVerifying(true); setError(null); setResult(null);
    try { setResult(await FaceService.verifyUser(user.id, candidate.file, controller.signal)); }
    catch (caught) {
      if (isCanceled(caught)) return;
      setError(caught instanceof ApiError ? caught.message : "Verification failed. Try again.");
    }
    finally { if (verifyRef.current === controller) setVerifying(false); }
  };

  const resetCandidate = () => { candidate.reset(); setResult(null); setError(null); };

  return <main className="mx-auto min-h-screen w-[min(1180px,calc(100%-24px))] pb-8 text-emerald-950 sm:w-[min(1180px,calc(100%-40px))]">
    <AppHeader />
    <section className="max-w-3xl py-12 md:py-18" aria-labelledby="page-title">
      <p className="mb-3 text-xs font-bold uppercase tracking-[.16em] text-emerald-700">Photo verification</p>
      <h1 id="page-title" className="mb-6 font-serif text-5xl leading-[.92] tracking-[-.045em] sm:text-7xl md:text-8xl">Confirm a face.<br />Keep the decision explainable.</h1>
      <p className="max-w-xl text-lg leading-relaxed text-stone-500">Load a registered profile, add one recent portrait, and receive an auditable result.</p>
    </section>
    <ProfileLookup userId={userId} loading={loadingUser} onUserIdChange={setUserId} onLoad={() => void loadUser()} />
    {error && <div className="mt-5 rounded-lg border-l-4 border-orange-600 bg-orange-50 p-4 text-orange-900" role="alert">{error}</div>}
    {user && <PortraitComparison user={user} candidate={candidate.file} previewUrl={candidate.previewUrl} verifying={verifying} onSelect={chooseFile} onReset={resetCandidate} onVerify={() => void verify()} />}
    {result && <ResultPanel result={result} />}
    <footer className="mt-14 flex flex-col justify-between gap-2 border-t border-stone-300 pt-5 text-xs text-stone-500 sm:flex-row"><span>Face Check · V1 photo verification</span><span>Scores and metadata only</span></footer>
    <PwaUpdatePrompt />
  </main>;
}

export default App;
