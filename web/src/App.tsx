import { useEffect, useRef, useState } from "react";
import { AppHeader } from "@components/AppHeader";
import { PortraitComparison } from "@components/PortraitComparison";
import { MIN_QUERY_LENGTH, ProfileLookup } from "@components/ProfileLookup";
import { ProfileSection, type ProfileMode } from "@components/ProfileSection";
import { UserRegistration } from "@components/UserRegistration";
import { PwaUpdatePrompt } from "@components/PwaUpdatePrompt";
import { VerificationResult as ResultPanel } from "@components/VerificationResult";
import { usePhotoUpload } from "@hooks/usePhotoUpload";
import { ApiError, isCanceled } from "@services/api.instance";
import { FaceService } from "@services/face.service";
import type { User, VerificationResult } from "@interfaces/face";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<User[] | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingUser, setLoadingUser] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [mode, setMode] = useState<ProfileMode>("lookup");
  const [registering, setRegistering] = useState(false);
  const candidate = usePhotoUpload();
  const lookupRef = useRef<AbortController | null>(null);
  const verifyRef = useRef<AbortController | null>(null);
  const createRef = useRef<AbortController | null>(null);

  // Abort whatever is in flight so a late response can never paint over a newer one.
  useEffect(() => () => {
    lookupRef.current?.abort();
    verifyRef.current?.abort();
    createRef.current?.abort();
  }, []);

  // Every path that puts a different profile on screen invalidates the pending work
  // of the previous one, so they all clear the same state.
  const startNewProfile = () => {
    lookupRef.current?.abort();
    verifyRef.current?.abort();
    setError(null); setUser(null); setResult(null); setVerifying(false);
    candidate.reset();
  };

  const clearSearch = () => { startNewProfile(); setResults(null); };

  const searchUsers = async () => {
    const normalized = query.trim();
    if (normalized.length < MIN_QUERY_LENGTH) {
      return setError(`Enter at least ${MIN_QUERY_LENGTH} characters to search.`);
    }
    // Switching profiles invalidates a pending verification as well as a pending lookup.
    startNewProfile();
    const controller = new AbortController();
    lookupRef.current = controller;
    setLoadingUser(true);
    try { setResults(await FaceService.searchUsers(normalized, controller.signal)); }
    catch (caught) {
      if (isCanceled(caught)) return;
      setError(caught instanceof ApiError ? caught.message : "Could not search for profiles.");
    }
    finally { if (lookupRef.current === controller) setLoadingUser(false); }
  };

  const selectUser = (selected: User) => {
    startNewProfile();
    setUser(selected);
  };

  const registerUser = async (externalId: string, name: string, profileImage: File) => {
    createRef.current?.abort();
    const controller = new AbortController();
    createRef.current = controller;
    startNewProfile();
    setRegistering(true);
    try {
      const created = await FaceService.createUser(externalId, name, profileImage, controller.signal);
      // Drop straight into verification against the profile just registered.
      setUser(created);
      setQuery(created.external_id);
      setResults([created]);
      setMode("lookup");
    } catch (caught) {
      if (isCanceled(caught)) return;
      setError(caught instanceof ApiError ? caught.message : "Could not register this user.");
    } finally {
      if (createRef.current === controller) setRegistering(false);
    }
  };

  const changeMode = (next: ProfileMode) => {
    if (next === mode) return;
    createRef.current?.abort();
    clearSearch();
    setMode(next);
  };

  const chooseFile = (file?: File) => {
    if (!file) return;
    setResult(null);
    setError(candidate.select(file));
  };

  // A live capture has already been through the check the server ran on the
  // preview, so the operator has watched it confirm the frame in real time.
  // Making them press Verify afterwards asks for a second opinion on something
  // they just saw decided, so the camera paths go straight through.
  //
  // Deliberately NOT extended to the file picker: every verification writes an
  // audit row, and auto-firing on each pick would record an attempt for a file
  // chosen by mistake.
  const captureAndVerify = (file: File) => {
    setResult(null);
    const problem = candidate.select(file);
    setError(problem);
    if (!problem) void verifyPhoto(file);
  };

  // Takes the file rather than reading `candidate.file`, because an auto-verify
  // fires in the same tick as the select() that armed it — the state update has
  // not landed yet, so reading it back would verify the PREVIOUS photo.
  const verifyPhoto = async (file: File) => {
    if (!user) return;
    verifyRef.current?.abort();
    const controller = new AbortController();
    verifyRef.current = controller;
    setVerifying(true); setError(null); setResult(null);
    try { setResult(await FaceService.verifyUser(user.id, file, controller.signal)); }
    catch (caught) {
      if (isCanceled(caught)) return;
      setError(caught instanceof ApiError ? caught.message : "Verification failed. Try again.");
    }
    finally { if (verifyRef.current === controller) setVerifying(false); }
  };

  const verify = () => {
    if (candidate.file) void verifyPhoto(candidate.file);
  };

  const resetCandidate = () => { candidate.reset(); setResult(null); setError(null); };

  return <main className="mx-auto min-h-screen w-[min(1180px,calc(100%-24px))] pb-8 text-emerald-950 sm:w-[min(1180px,calc(100%-40px))]">
    <AppHeader />
    <section className="max-w-3xl py-12 md:py-18" aria-labelledby="page-title">
      <p className="mb-3 text-xs font-bold uppercase tracking-[.16em] text-emerald-700">Photo verification</p>
      <h1 id="page-title" className="mb-6 font-serif text-5xl leading-[.92] tracking-[-.045em] sm:text-7xl md:text-8xl">Confirm a face.<br />Keep the decision explainable.</h1>
      <p className="max-w-xl text-lg leading-relaxed text-stone-500">Load a registered profile, add one recent portrait, and receive an auditable result.</p>
    </section>
    <ProfileSection mode={mode} disabled={loadingUser || registering} onModeChange={changeMode}>
      {mode === "lookup"
        ? <ProfileLookup query={query} loading={loadingUser} results={results} onQueryChange={setQuery} onSearch={() => void searchUsers()} onSelect={selectUser} />
        : <UserRegistration submitting={registering} onSubmit={(externalId, name, photo) => void registerUser(externalId, name, photo)} />}
    </ProfileSection>
    {error && <div className="mt-5 rounded-lg border-l-4 border-orange-600 bg-orange-50 p-4 text-orange-900" role="alert">{error}</div>}
    {user && <PortraitComparison user={user} candidate={candidate.file} previewUrl={candidate.previewUrl} verifying={verifying} onSelect={chooseFile} onCapture={captureAndVerify} onReset={resetCandidate} onVerify={verify} />}
    {result && <ResultPanel result={result} />}
    <footer className="mt-14 flex flex-col justify-between gap-2 border-t border-stone-300 pt-5 text-xs text-stone-500 sm:flex-row"><span>Face Check · V1 photo verification</span><span>Scores and metadata only</span></footer>
    <PwaUpdatePrompt />
  </main>;
}

export default App;
