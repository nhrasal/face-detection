"""Structured logging.

Configured once at import of `app.main`, never per-request. `request_id` is
bound into a contextvar so every log line emitted while handling a request
carries it without being threaded through call signatures.

Discipline (see docs/DATA_RETENTION.md): never log image bytes, embeddings,
client-supplied filenames, or client IPs at INFO. Decisions, similarities,
timings and reason codes are all fine.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import orjson
import structlog


def _orjson_dumps(obj: Any, default: Any) -> str:
    return orjson.dumps(obj, default=default).decode()


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if json_output:
        processors += [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(serializer=_orjson_dumps),
        ]
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    # structlog.get_logger is untyped (returns Any); the cast is what keeps
    # `mypy --strict` meaningful at every call site downstream.
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def bind_request_id(request_id: str) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
