"""ORM model registry (imported by Alembic)."""

from app.models.face_verification import FaceVerification
from app.models.user import User

__all__ = ["FaceVerification", "User"]
