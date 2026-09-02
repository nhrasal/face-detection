import { useEffect, useRef, useState } from "react";
import { Alert } from "@components/Alert";
import { AppHeader } from "@components/AppHeader";
import { Card } from "@components/Card";
import { PortraitComparison } from "@components/PortraitComparison";
import { MIN_QUERY_LENGTH, ProfileLookup } from "@components/ProfileLookup";
import { ProfileModeTabs, type ProfileMode } from "@components/ProfileModeTabs";
import { SelectedProfile, SelectedProfileActions } from "@components/SelectedProfile";
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
  const resultRef = useRef<HTMLDivElement>(null);

  // Abort whatever is in flight so a late response can never paint over a newer one.
  useEffect(() => () => {
    lookupRef.current?.abort();
    verifyRef.current?.abort();
    createRef.current?.abort();
  }, []);

  // The result is the reason the operator is here, and on a short viewport it
  // lands below the fold. Bring it into view rather than leaving them to wonder
  // whether the verification ran at all.
  useEffect(() => {
    if (!result) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    resultRef.current?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "nearest" });
  }, [result]);

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

  // The liveness path alone verifies without a further click: its capture is the
  // completion of a challenge the server ran and passed, so the frame is already
  // decided by the time it lands and asking for Verify would only re-ask.
  //
  // Deliberately NOT used by the file picker or the plain camera: every
  // verification writes an audit row, and firing on each pick or shutter press
  // would record an attempt for a shot the operator has not reviewed.
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

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <AppHeader />
      <main className="mx-auto w-full max-w-5xl px-4 pb-16 sm:px-6">
        <div className="py-6">
          <h1 className="text-lg font-semibold tracking-tight">Verify identity</h1>
          <p className="mt-0.5 text-sm text-ink-soft">
            Compare a portrait against a registered profile. Scores and metadata only.
          </p>
        </div>

        <div className="grid gap-4">
          <Card
            title="Profile"
            actions={user
              ? <SelectedProfileActions onChange={startNewProfile} />
              : <ProfileModeTabs mode={mode} disabled={loadingUser || registering} onChange={changeMode} />}
          >
            {user ? (
              <SelectedProfile user={user} />
            ) : mode === "lookup" ? (
              <ProfileLookup
                query={query}
                loading={loadingUser}
                results={results}
                onQueryChange={setQuery}
                onSearch={() => void searchUsers()}
                onSelect={selectUser}
              />
            ) : (
              <UserRegistration
                submitting={registering}
                onSubmit={(externalId, name, photo) => void registerUser(externalId, name, photo)}
              />
            )}
          </Card>

          {error && <Alert>{error}</Alert>}

          {user && (
            <Card title="Comparison">
              <PortraitComparison
                user={user}
                candidate={candidate.file}
                previewUrl={candidate.previewUrl}
                verifying={verifying}
                result={result}
                onSelect={chooseFile}
                onCapture={captureAndVerify}
                onReset={resetCandidate}
                onVerify={verify}
              />
            </Card>
          )}

          {(result || verifying) && (
            <div ref={resultRef}>
              <Card title="Result">
                {result ? (
                  <ResultPanel result={result} />
                ) : (
                  <p className="flex items-center gap-2 text-sm text-ink-soft" role="status">
                    <span className="size-3 animate-spin rounded-full border-2 border-line-strong border-t-accent" aria-hidden="true" />
                    Comparing faces…
                  </p>
                )}
              </Card>
            </div>
          )}
        </div>

        <footer className="mt-10 flex flex-col justify-between gap-1.5 border-t border-line pt-4 text-xs text-ink-muted sm:flex-row">
          <span>Face Check · V1 photo verification</span>
          <span>Candidate photographs are not stored</span>
        </footer>
      </main>
      <PwaUpdatePrompt />
    </div>
  );
}

export default App;
