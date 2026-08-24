# Face Verification Service

Compare an uploaded photo against a user's stored profile image and decide whether
both plausibly show the same person — with a recorded, auditable decision.

This is **face verification**, not KYC. See
[`face_verification_kyc_roadmap.md`](./face_verification_kyc_roadmap.md) for the
V1→V5 product roadmap; KYC is V4 and reuses this engine.

## Status

**Phases 1–8 of 15 complete.** The core hypothesis is proven: this pipeline
recognises people. See [Does it work?](#does-it-work) for the measured numbers.

229 tests — 189 hermetic (no models or assets needed) plus 40 model-tier against real
weights and real portraits. 92% coverage, mypy strict and ruff clean.

Photo-to-photo detection and comparison plus user/profile verification are exposed over
HTTP. Every user verification writes an audit row; profile images are stripped and
re-encoded under generated storage keys.

| Phase | | Area | |
|---|---|---|---|
| 1 | Scaffold, config, logging, health | backend | done |
| 2 | Model download + checksum verification | backend | done |
| 3 | Decode, alignment, quality metrics | backend | done |
| 4 | YuNet + SFace adapter — **proves the core hypothesis** | backend | done |
| 5 | Pipeline orchestration + decision layer | backend | done |
| 6 | Postgres schema, SQLAlchemy models, Alembic, repositories | backend | done |
| 7 | HTTP layer — detect/compare, errors, rate limiting, warmup | backend | done |
| 8 | Users, reference resolver, profile upload, verify + history | backend | done |
| 9 | Security suite — size caps, MIME spoofing, embedding-leak guard | backend | next |
| 10 | React upload/verify UI with per-reason messaging | frontend | |
| 11 | Docker Compose — runs from a clean clone | ops | |
| 12 | **Threshold calibration** — V1 is not shippable until this runs | data | |
| 13 | InsightFace benchmark — optional, see open decisions | backend | |
| 14–15 | Retention/architecture docs, `.env` finalisation | docs | |

**Two phases (9–10) to a working system; four (through 12) to a shippable one.**

Phase 7 is the heaviest; 6 and 10 are moderate; 8 and 9 are small. Backend is roughly
4x the frontend, because the frontend is deliberately dependency-light — six runtime
packages, no `@gems/components`, no router, no form library.

### The one thing effort cannot unblock

**Phase 12 needs data, not code.** The match threshold, `min_sharpness`, and the pose
and detection-score gates are all still provisional. Eleven astronaut portraits proved
the pipeline works; they cannot set an operating point. That needs a labelled set —
LFW works as a sanity check, but internal GEMS employee photos would be far more
representative of the real population and carry their own approval process.

Worth starting that conversation in parallel with phases 6–10, not at phase 12.

### Open decisions

1. **Confirm the threshold correction.** The roadmap's `0.72` is replaced by SFace's
   documented `0.363` as a *starting* value. Changing this affects `ModelInfo`, the API
   contract, and the frontend's result display.
2. **Is the InsightFace adapter in scope at all?** Dropping it removes phase 13 and the
   entire non-commercial licensing question. Keeping it should be sequenced *after*
   calibration, so two calibrated engines are compared rather than one calibrated and
   one guessed.

## Quickstart

```bash
cd backend
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp ../.env.example .env
.venv/bin/uvicorn app.main:app --reload --port 8000
```

```bash
curl localhost:8000/healthz   # {"status":"ok"}
curl localhost:8000/readyz    # {"status":"ready","engine":"opencv_zoo","warm":true}
open  localhost:8000/docs
```

HTTP endpoints:

```bash
curl -F image=@portrait.jpg localhost:8000/api/v1/face/detect

curl \
  -F reference_image=@reference.jpg \
  -F candidate_image=@candidate.jpg \
  localhost:8000/api/v1/face/compare

curl \
  -F external_id=employee-123 \
  -F name='Example User' \
  -F profile_image=@profile.jpg \
  localhost:8000/api/v1/users

curl \
  -F candidate_image=@candidate.jpg \
  localhost:8000/api/v1/users/USER_UUID/verify

curl localhost:8000/api/v1/users/USER_UUID/verifications
```

Uploads are identified from magic bytes rather than filenames or declared MIME types,
bounded before decoding, and processed in a fixed warmed engine pool. Responses expose
face status, quality reasons, score, threshold, model versions, and processing time but
never embeddings. `NO_FACE`, `MULTIPLE_FACES`, and `LOW_QUALITY` are successful HTTP 200
decisions; malformed/oversized uploads and infrastructure failures use normalized errors.

User profile uploads are decoded, stripped of metadata, re-encoded as JPEG, and written
under generated UUID-based keys. Original filenames and original bytes are not retained.
Replacing a profile removes the superseded local file only after the database update
commits. Phase 8 uses local storage behind `ProfileImageStore`; a later deployment can
replace that boundary with S3/MinIO.

Checks:

```bash
.venv/bin/ruff check app/ && .venv/bin/ruff format --check app/
.venv/bin/mypy app/          # strict
.venv/bin/pytest             # hermetic tier; add -m models for the model tier
```

## Stack

Python 3.13 · FastAPI · SQLAlchemy 2 + Alembic + PostgreSQL (psycopg3) ·
OpenCV 5 (YuNet + SFace) · ONNX Runtime (opt-in adapter) · React + Vite frontend.

All dependencies are the latest stable release, resolved together and verified
`pip check`-clean on Python 3.13.5 / macOS arm64.

**Keep numpy, opencv-python-headless and onnxruntime moving together.** The whole
stack is built against NumPy 2.x. Pinning numpy backwards without also pinning the
other two produces `module compiled against API version 0x10` crashes at import.

## Models

```bash
./scripts/download_models.sh            # fetch + verify the default pair
./scripts/download_models.sh --verify   # check what is on disk, fetch nothing
./scripts/download_models.sh --record   # regenerate the checksum manifest
```

Weights are checksum-pinned in `scripts/models.sha256` and the script fails closed
on any mismatch — model weights are executable inputs that decide who gets verified
as whom. The opencv_zoo source is pinned to a **commit**, not a branch: "version
every model" means nothing if the bytes behind the name `2023mar` can change
underneath you. Weights are never committed (`models/*` is ignored).

| | bytes | licence |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 232,589 | permissive — default |
| `face_recognition_sface_2021dec.onnx` | 38,696,353 | permissive — default |
| `insightface/buffalo_l/*` | ~183 MB | **non-commercial research only** — opt-in |

### Verified behaviour of these two models

Measured directly against OpenCV 5.0 on this stack, not assumed:

- **`FaceRecognizerSF.feature()` returns an unnormalised vector** — measured L2 norm
  **4.14**, not 1.0. `match(..., FR_COSINE)` normalises internally, which is why
  dotting raw `feature()` output gives similarities in the tens. `embed()` must
  normalise explicitly. Once normalised, our cosine agrees with `match()` exactly.
- **`detect()` with no face returns `retval=1` and `faces=None`.** The return code
  means "ran successfully", not "found something" — branch on `faces is None`, never
  on `retval`, or you index into `None` on every faceless image.
- OpenCV 5 logs `Targets are not supported by the new graph engine for now` at model
  load. Cosmetic, from `setPreferableTarget`; both models load and infer correctly.
- Cold cost on this machine: ~50 ms to load each model, ~10 ms `detect()` on
  640×480, ~12 ms `feature()`. Unwarmed — first real inference is much slower, which
  is why the engine pool warms at startup.

## Does it work?

Yes — measured on 11 public-domain NASA portraits across 5 identities, using the
default YuNet + SFace engine at its documented 0.363 cosine threshold.

| | n | min | mean | max |
|---|---:|---:|---:|---:|
| genuine (same person) | 7 | **+0.4679** | +0.6812 | +0.8428 |
| impostor (different people) | 48 | −0.0030 | +0.1355 | **+0.3285** |

The two distributions do not overlap. Worst genuine (+0.4679) sits clearly above best
impostor (+0.3285), a **separation gap of +0.139**, and the threshold falls inside
that gap. This is the assertion the test suite makes, and it is stronger than testing
the threshold alone: if the distributions ever overlap, no threshold can separate them
and the fault is upstream — alignment, landmark order, or quality gating.

**The margin is thinner on the impostor side.** The best impostor pair is only 0.035
below the threshold, versus 0.105 of headroom on the genuine side. On this tiny sample
that already suggests the operating point wants to move up, and it is exactly why
phase 12 picks the threshold from a false-accept budget rather than from a document.
Eleven photos of five astronauts is a smoke test, not a calibration set.

## Quality gating is the recall risk

Thresholds in `QualityThresholds` are **provisional and deliberately permissive**.
Measured against the 11 verified-good portraits in the test set, the original guesses
excluded **27%** of them — every one of which still produced a *correct* identity
decision. `min_yaw_symmetry` went 0.55 → 0.15 and `min_detection_score` 0.90 → 0.75
on that evidence; `min_sharpness` ships effectively off.

This failure mode is expensive because it misdiagnoses: tighten a gate, watch
would-be matches become `LOW_QUALITY` rejections, and conclude the model is bad. It
presents as a recognition problem and is a gating problem. A model-tier test asserts
the exclusion rate stays at zero on known-good portraits, and phase 12's calibration
sets the real values.

**Rejecting a usable photo costs more than embedding a slightly angled one.**

## Two things that will bite

**The roadmap's `threshold: 0.72` is wrong** and must not be shipped. Both candidate
models emit raw cosine similarity of L2-normalised embeddings; genuine pairs cluster
around 0.4–0.8 and impostors near 0.0–0.15. SFace's documented operating point is
**0.363**, ArcFace's ~**0.40**. A 0.72 threshold sits in the far right tail of the
*genuine* distribution — it would reject most true matches while gaining nothing on
false accepts, and it fails as working software rather than as an error. The shipped
value comes out of phase 12, not out of a document.

**Two model licences, not one.** YuNet and SFace are permissively licensed and are
the shippable default. InsightFace `buffalo_l` weights are **non-commercial research
only** — that adapter is benchmark-only, gated at startup when `ENV=prod`, and its
weights are never baked into an image.

## Data handling

V1 stores **no candidate images and no biometric templates** — scores and metadata
only. Profile images are the sole stored image, under a generated filename and
re-encoded rather than byte-copied. `.gitignore` excludes `calibration/data/` and
`tests/assets/` from the first commit: once a face photo enters git history it is
effectively permanent.

V2 introduces stored templates and with them a deletion/revocation obligation —
design that policy when V2 starts, not after.
