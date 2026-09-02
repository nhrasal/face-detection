"""Shared test configuration.

`backend/` is on sys.path via rootdir, so `app.*` and `tests.*` both import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings, get_settings

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ASSET_DIR = Path(__file__).resolve().parent / "assets"


@pytest.fixture
def settings() -> Settings:
    """Local-mode settings with the fake engine, isolated from the environment."""
    return Settings(ENV="local", FACE_ENGINE="fake")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def assets_dir() -> Path:
    if not (ASSET_DIR / "no_face.jpg").exists():
        pytest.skip("test assets missing — run: python scripts/fetch_test_faces.py")
    return ASSET_DIR


@pytest.fixture(scope="session")
def faces_dir(assets_dir: Path) -> Path:
    return assets_dir / "faces"


@pytest.fixture(scope="session")
def engine():
    """Real YuNet + SFace engine, built once per session.

    Loading the models costs ~100ms and warmup more, so this is deliberately
    session-scoped. Tests must not mutate it.
    """
    from app.engine.adapters.opencv_zoo import DETECTOR_FILE, OpenCvZooEngine

    if not (MODEL_DIR / DETECTOR_FILE).exists():
        pytest.skip("model weights missing — run: ./scripts/download_models.sh")
    built = OpenCvZooEngine(MODEL_DIR)
    built.warmup()
    return built
