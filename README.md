# Face Verification Service

Compare an uploaded photo against a user's stored profile image and decide whether
both plausibly show the same person — with a recorded, auditable decision.

This is **face verification**, not KYC. See
[`face_verification_kyc_roadmap.md`](./face_verification_kyc_roadmap.md) for the
V1→V5 product roadmap; KYC is V4 and reuses this engine.

## Status

**Phase 1 of 15 complete** — backend skeleton, settings, structured logging, health
endpoints. No face pipeline yet.

| Phase | | |
|---|---|---|
| 1 | Scaffold, config, logging, health | done |
| 2 | Model download + checksum verification | next |
| 3 | Decode, alignment, quality metrics | |
| 4 | YuNet + SFace adapter — **proves the core hypothesis** | |
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
