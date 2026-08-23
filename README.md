# Face Verification Service

Compare an uploaded photo against a user's stored profile image and decide whether
both plausibly show the same person — with a recorded, auditable decision.

This is **face verification**, not KYC. See
[`face_verification_kyc_roadmap.md`](./face_verification_kyc_roadmap.md) for the
V1→V5 product roadmap; KYC is V4 and reuses this engine.

## Status

**Phases 1–3 of 15 complete** — backend skeleton, model fetching with checksum
verification, and the engine's pure layer: safe decoding, five-point alignment, and
quality assessment. 83 hermetic unit tests, no models or face assets required.

No model adapter yet, so nothing detects or embeds a real face until phase 4.

| Phase | | |
|---|---|---|
| 1 | Scaffold, config, logging, health | done |
| 2 | Model download + checksum verification | done |
| 3 | Decode, alignment, quality metrics | done |
| 4 | YuNet + SFace adapter — **proves the core hypothesis** | next |
| 5–11 | Pipeline, DB, HTTP, users, security, frontend, Docker | |
| 12 | **Threshold calibration** — V1 is not done until this runs | |
| 13–15 | InsightFace benchmark (optional), docs | |

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
curl localhost:8000/readyz    # {"status":"ready","engine":"opencv_zoo","warm":false}
open  localhost:8000/docs
```

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
