"""Engine pool: backpressure, leak-safety, idempotent close.

Uses a local stub rather than a real engine so these stay hermetic — the pool's
logic has nothing to do with which model is loaded.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from app.core.errors import EngineBusyError
from app.engine.base import FaceEngine
from app.engine.pool import EnginePool
from app.engine.types import DetectedFace, ModelInfo

STUB_INFO = ModelInfo(
    detector_name="stub",
    detector_version="0",
    recognizer_name="stub",
    recognizer_version="0",
    embedding_dim=4,
    default_threshold=0.5,
    license_note="test only",
)


class StubEngine(FaceEngine):
    def __init__(self) -> None:
        self.warmed = 0
        self.closed = 0

    @property
    def info(self) -> ModelInfo:
        return STUB_INFO

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        return []

    def align(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def warmup(self) -> None:
        self.warmed += 1

    def close(self) -> None:
        self.closed += 1


class TestConstruction:
    def test_builds_the_requested_number_of_engines(self) -> None:
        built: list[StubEngine] = []

        def factory() -> FaceEngine:
            engine = StubEngine()
            built.append(engine)
            return engine

        pool = EnginePool(factory, size=3)
        assert pool.size == 3
        assert len(built) == 3

    def test_every_engine_is_warmed_before_use(self) -> None:
        # An unwarmed pool makes the first real user absorb graph optimisation
        # while readiness already reports OK.
        built: list[StubEngine] = []

        def factory() -> FaceEngine:
            engine = StubEngine()
            built.append(engine)
            return engine

        EnginePool(factory, size=2)
        assert [e.warmed for e in built] == [1, 1]

    @pytest.mark.parametrize("size", [0, -1])
    def test_rejects_non_positive_size(self, size: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            EnginePool(StubEngine, size=size)

    def test_exposes_model_info(self) -> None:
        assert EnginePool(StubEngine, size=1).info.embedding_dim == 4


class TestAcquire:
    def test_yields_an_engine(self) -> None:
        pool = EnginePool(StubEngine, size=1)
        with pool.acquire() as engine:
            assert isinstance(engine, StubEngine)

    def test_engine_is_returned_after_use(self) -> None:
        pool = EnginePool(StubEngine, size=1)
        with pool.acquire() as first:
            pass
        with pool.acquire() as second:
            assert first is second

    def test_exhausted_pool_raises_engine_busy(self) -> None:
        # Backpressure, not an OOM kill: this becomes a 503 with Retry-After.
        pool = EnginePool(StubEngine, size=1)
        # Kept nested on purpose: the second acquire must fail while the first
        # engine is still held. Flattening it into one `with` would still pass
        # but obscures which acquire is the one under test.
        with pool.acquire(), pytest.raises(EngineBusyError, match="busy"):  # noqa: SIM117
            with pool.acquire(timeout=0.01):
                pass

    def test_engine_is_returned_even_when_the_body_raises(self) -> None:
        """Otherwise a failing handler shrinks the pool one request at a time."""
        pool = EnginePool(StubEngine, size=1)
        with pytest.raises(RuntimeError, match="boom"), pool.acquire():
            raise RuntimeError("boom")
        with pool.acquire(timeout=0.01) as engine:
            assert engine is not None

    def test_concurrent_callers_get_distinct_engines(self) -> None:
        pool = EnginePool(StubEngine, size=2)
        seen: list[int] = []
        both_held = threading.Barrier(2, timeout=5)

        def worker() -> None:
            with pool.acquire(timeout=5) as engine:
                seen.append(id(engine))
                both_held.wait()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert len(seen) == 2
        assert seen[0] != seen[1], "two concurrent holders shared one engine"

    def test_serialises_when_size_is_one(self) -> None:
        pool = EnginePool(StubEngine, size=1)
        overlaps = 0
        active = 0
        lock = threading.Lock()

        def worker() -> None:
            nonlocal overlaps, active
            with pool.acquire(timeout=5):
                with lock:
                    active += 1
                    if active > 1:
                        overlaps += 1
                with lock:
                    active -= 1

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert overlaps == 0


class TestClose:
    def test_closes_every_engine(self) -> None:
        built: list[StubEngine] = []

        def factory() -> FaceEngine:
            engine = StubEngine()
            built.append(engine)
            return engine

        pool = EnginePool(factory, size=3)
        pool.close()
        assert [e.closed for e in built] == [1, 1, 1]

    def test_close_is_idempotent(self) -> None:
        built: list[StubEngine] = []

        def factory() -> FaceEngine:
            engine = StubEngine()
            built.append(engine)
            return engine

        pool = EnginePool(factory, size=2)
        pool.close()
        pool.close()
        pool.close()
        assert [e.closed for e in built] == [1, 1]

    def test_acquire_after_close_raises(self) -> None:
        pool = EnginePool(StubEngine, size=1)
        pool.close()
        with pytest.raises(EngineBusyError, match="closed"), pool.acquire():
            pass
