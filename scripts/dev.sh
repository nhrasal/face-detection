#!/usr/bin/env bash
# Single entry point for local development: migrations, the API, and the web UI.
# Ctrl-C stops both processes; if either one dies the other is brought down with it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${PROJECT_DIR}/backend"
WEB_DIR="${PROJECT_DIR}/web"
VENV="${BACKEND_DIR}/.venv"

fail() { echo "error: $*" >&2; exit 1; }

[[ -x "${VENV}/bin/python" ]] || fail "backend/.venv is missing. Create it with:
  python3.13 -m venv backend/.venv && backend/.venv/bin/pip install -r backend/requirements-dev.txt"
[[ -f "${BACKEND_DIR}/.env" ]] || fail "backend/.env is missing. Create it with:
  cp backend/.env.example backend/.env"

if [[ ! -d "${WEB_DIR}/node_modules" ]]; then
  echo "==> installing web dependencies"
  (cd "${WEB_DIR}" && yarn install --frozen-lockfile)
fi

LOG_DIR="$(mktemp -d)"
pids=()
shutdown() {
  trap - INT TERM EXIT
  if [[ ${#pids[@]} -gt 0 ]]; then kill "${pids[@]}" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  rm -rf "${LOG_DIR}"
}
trap shutdown INT TERM EXIT

# Create the database named in DATABASE_URL if the server does not have it yet,
# so a clean checkout needs nothing but a running PostgreSQL.
echo "==> checking database"
if ! "${VENV}/bin/python" "${BACKEND_DIR}/scripts/ensure_database.py" 2>"${LOG_DIR}/db.log"; then
  cat "${LOG_DIR}/db.log" >&2
  fail "could not reach PostgreSQL. Check that it is running and that DATABASE_URL in backend/.env is correct."
fi

echo "==> applying database migrations"
if ! (cd "${BACKEND_DIR}" && "${VENV}/bin/alembic" upgrade head) >"${LOG_DIR}/alembic.log" 2>&1; then
  tail -n 15 "${LOG_DIR}/alembic.log" >&2
  fail "migrations failed."
fi

echo "==> api  http://127.0.0.1:8000  (OpenAPI at /docs)"
(cd "${BACKEND_DIR}" && exec "${VENV}/bin/uvicorn" app.main:app --reload --port 8000) &
pids+=($!)

echo "==> web  http://localhost:3320"
# vite directly rather than through yarn, so the pid we hold is the one to signal.
(cd "${WEB_DIR}" && exec node_modules/.bin/vite --host 0.0.0.0 --port 3320) &
pids+=($!)

while true; do
  for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null || exit 1
  done
  sleep 1
done
