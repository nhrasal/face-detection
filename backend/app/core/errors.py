"""Application error hierarchy.

Each error carries the HTTP status it maps to, so the HTTP layer can translate
without a growing isinstance ladder. Note what is NOT here: NO_FACE,
MULTIPLE_FACES and LOW_QUALITY are not errors. They are correct, successful
answers to "compare these two images" and travel in a 200 response body.

Only "we could not run the pipeline" is an error.
"""

from __future__ import annotations


class AppError(Exception):
    """Base for every error this service raises deliberately."""

    code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class UnsupportedMediaError(AppError):
    """Bytes are not a supported image, or claim a type they are not."""

    code = "UNSUPPORTED_MEDIA"
    http_status = 415


class ImageTooLargeError(AppError):
    """Payload, pixel count or edge length exceeds the configured cap."""

    code = "IMAGE_TOO_LARGE"
    http_status = 413


class AlignmentError(AppError):
    """No similarity transform could be fitted to the landmarks."""

    code = "ALIGNMENT_FAILED"
    http_status = 500


class ModelsMissingError(AppError):
    """Model weights are absent or unreadable."""

    code = "MODELS_MISSING"
    http_status = 503


class EngineBusyError(AppError):
    """Every engine in the pool is checked out. Backpressure, not failure."""

    code = "ENGINE_BUSY"
    http_status = 503


class ResourceNotFoundError(AppError):
    code = "NOT_FOUND"
    http_status = 404


class ConflictError(AppError):
    code = "CONFLICT"
    http_status = 409


class InvalidRequestError(AppError):
    """A parameter arrived well-formed but is not usable as given."""

    code = "INVALID_REQUEST"
    http_status = 422
