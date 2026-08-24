"""Engine selection.

Hermetic: only the not-yet-implemented and error branches are exercised here.
Building the real opencv_zoo engine needs weights and lives in the model tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import AppError, ModelsMissingError
from app.engine.factory import _REGISTRY, build_engine


def settings_for(engine: str, **kwargs: object) -> Settings:
    return Settings(ENV="local", FACE_ENGINE=engine, **kwargs)  # type: ignore[arg-type]


class TestRegistry:
    def test_every_configurable_engine_has_a_builder(self) -> None:
        # Settings accepts these three; a value with no builder would only fail
        # at startup, in whichever environment enabled it.
        assert set(_REGISTRY) == {"opencv_zoo", "insightface", "fake"}

    def test_unknown_engine_is_rejected_by_settings(self) -> None:
        with pytest.raises(ValueError):
            settings_for("definitely-not-an-engine")


class TestNotYetImplemented:
    @pytest.mark.parametrize("name", ["insightface"])
    def test_unbuilt_engines_raise_rather_than_returning_something_broken(self, name: str) -> None:
        settings = settings_for(name, ALLOW_NONCOMMERCIAL_MODELS=True)
        with pytest.raises(AppError, match="not implemented yet"):
            build_engine(settings)

    def test_fake_engine_is_available_for_hermetic_http_tests(self) -> None:
        engine = build_engine(settings_for("fake"))
        assert engine.info.detector_name == "fake"


class TestMissingWeights:
    def test_absent_model_files_raise_with_the_fix_in_the_message(self) -> None:
        # The error a developer hits on a fresh clone, so it must say what to
        # run rather than surfacing a bare cv2 file-open failure.
        settings = settings_for("opencv_zoo", MODEL_DIR=Path("/nonexistent/models"))
        with pytest.raises(ModelsMissingError) as excinfo:
            build_engine(settings)
        assert "download_models" in (excinfo.value.detail or "")
        assert excinfo.value.http_status == 503
