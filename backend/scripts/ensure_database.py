#!/usr/bin/env python
"""Create the database named in backend/.env's DATABASE_URL if it does not exist.

Alembic can create tables but not the database that holds them, so without this a
clean checkout fails its first migration with a raw psycopg traceback. Only the
database is created here; the schema is still Alembic's job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from sqlalchemy.engine import make_url

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def database_url() -> str:
    for line in ENV_FILE.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("DATABASE_URL="):
            return stripped.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit(f"DATABASE_URL is not set in {ENV_FILE}")


def main() -> None:
    url = make_url(database_url())
    if not url.database:
        raise SystemExit("DATABASE_URL does not name a database")

    # Connect to the always-present maintenance database to inspect the server.
    with psycopg.connect(
        host=url.host or "localhost",
        port=url.port or 5432,
        user=url.username,
        password=url.password,
        dbname="postgres",
        autocommit=True,
    ) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (url.database,)
        ).fetchone()
        if exists:
            print(f"    database {url.database!r} is present")
            return
        # Identifiers cannot be parameterised; the name comes from our own .env.
        conn.execute(f'CREATE DATABASE "{url.database}"')
        print(f"    created database {url.database!r}")


if __name__ == "__main__":
    sys.exit(main())
