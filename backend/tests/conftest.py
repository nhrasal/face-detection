"""Shared test configuration.

`backend/` is on sys.path via rootdir, so `app.*` and `tests.*` both import.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings, get_settings


@pytest.fixture
def settings() -> Settings:
    """Local-mode settings with the fake engine, isolated from the environment."""
    return Settings(ENV="local", FACE_ENGINE="fake")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
