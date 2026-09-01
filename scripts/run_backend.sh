#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${PROJECT_DIR}/backend"
PYTHON_BIN="${BACKEND_DIR}/.venv/bin/python"
ALEMBIC_BIN="${BACKEND_DIR}/.venv/bin/alembic"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Backend virtual environment not found."
    echo "Create it with: cd backend && python3.13 -m venv .venv"
    exit 1
fi

if [[ ! -f "${BACKEND_DIR}/.env" ]]; then
    echo "Backend environment file not found."
    echo "Create it with: cp .env.example backend/.env"
    exit 1
fi

# Same clean-clone guarantee as dev.sh: create the database named in DATABASE_URL
# if the server does not have it yet.
echo "Checking database..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/ensure_database.py"

cd "${BACKEND_DIR}"
echo "Applying database migrations..."
"${ALEMBIC_BIN}" upgrade head

echo "Starting backend at http://127.0.0.1:8000..."
exec "${PYTHON_BIN}" main.py
