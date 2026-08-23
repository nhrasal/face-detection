"""A bounded pool of engine instances.

Two problems solved at once:

- `cv2.FaceDetectorYN` is stateful and not thread-safe, so concurrent requests
  cannot share one instance.
- Inference runs in a threadpool, and `threading.local()` would allocate one
  engine per worker thread with no upper bound — SFace is ~37 MB on disk but
  150-250 MB resident once buffers are allocated, and buffalo_l is ~1 GB.

A fixed pool caps memory at `size x engine` and turns overload into
backpressure (503 with Retry-After) rather than an OOM kill.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from app.core.errors import EngineBusyError
from app.core.logging import get_logger
from app.engine.base import FaceEngine

log = get_logger(__name__)


class EnginePool:
    def __init__(self, factory: Callable[[], FaceEngine], size: int = 2) -> None:
        if size < 1:
            raise ValueError("pool size must be at least 1")
        self._queue: queue.Queue[FaceEngine] = queue.Queue(maxsize=size)
        self._all: list[FaceEngine] = []
        self._closed = False
        self._close_lock = threading.Lock()

        for _ in range(size):
            engine = factory()
            # Warm before the engine is ever handed out. The first inference is
            # far slower than steady state, so an unwarmed pool makes the first
            # real user absorb that latency while readiness already reads OK.
            engine.warmup()
            self._all.append(engine)
            self._queue.put(engine)

        self.info = self._all[0].info
        log.info(
            "engine_pool.ready",
            size=size,
            detector=f"{self.info.detector_name}@{self.info.detector_version}",
            recognizer=f"{self.info.recognizer_name}@{self.info.recognizer_version}",
            embedding_dim=self.info.embedding_dim,
        )

    @property
    def size(self) -> int:
        return len(self._all)

    @contextmanager
    def acquire(self, timeout: float = 10.0) -> Iterator[FaceEngine]:
        if self._closed:
            raise EngineBusyError("Engine pool is closed.")
        try:
            engine = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise EngineBusyError("All inference workers are busy.") from None
        try:
            yield engine
        finally:
            # Returned in a finally so a raising handler cannot leak an engine
            # and shrink the pool one request at a time.
            self._queue.put(engine)

    def close(self) -> None:
        """Release every engine. Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            for engine in self._all:
                engine.close()
            log.info("engine_pool.closed", size=len(self._all))
