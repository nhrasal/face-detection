from __future__ import annotations

import dataclasses

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.engine.pipeline import FaceStatus
from app.engine.types import ModelInfo
from app.models.face_verification import FaceVerification
from app.repositories.face_repository import FaceVerificationRepository, UserRepository
from app.services.decision import ComparisonOutcome, Decision


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


def outcome() -> ComparisonOutcome:
    return ComparisonOutcome(
        decision=Decision.MATCH,
        similarity=0.82,
        confidence=0.97,
        threshold=0.363,
        reference_status=FaceStatus.OK,
        candidate_status=FaceStatus.OK,
        reference_face_count=1,
        candidate_face_count=1,
    )


MODEL = ModelInfo("yunet", "2023mar", "sface", "2021dec", 128, 0.363, "permissive")


def test_user_repository_round_trip(session: Session) -> None:
    repo = UserRepository(session)
    user = repo.create(external_id="emp-42", name="Ada", profile_image_url="profiles/a.jpg")
    assert repo.get(user.id) is user
    assert repo.get_by_external_id("emp-42") is user


def test_verification_repository_records_audit_data_without_biometrics(session: Session) -> None:
    user = UserRepository(session).create(
        external_id="emp-42", name="Ada", profile_image_url="profiles/a.jpg"
    )
    repo = FaceVerificationRepository(session)
    row = repo.record(
        user_id=user.id,
        outcome=outcome(),
        model=MODEL,
        threshold_version="sface-default",
        processing_time_ms=24,
    )
    session.commit()

    assert row.decision == "MATCH"
    assert row.detector_version == "yunet@2023mar"
    assert repo.list_for_user(user.id) == [row]
    columns = {column.name for column in FaceVerification.__table__.columns}
    assert not {"embedding", "template", "candidate_image", "reference_image"} & columns


def test_failed_outcome_preserves_reason_and_issues(session: Session) -> None:
    user = UserRepository(session).create(
        external_id="emp-43", name="Grace", profile_image_url="profiles/g.jpg"
    )
    failed = dataclasses.replace(
        outcome(),
        decision=Decision.NO_FACE,
        similarity=None,
        confidence=None,
        candidate_status=FaceStatus.NO_FACE,
        candidate_face_count=0,
        reason_code="CANDIDATE_NO_FACE",
    )
    row = FaceVerificationRepository(session).record(
        user_id=user.id,
        outcome=failed,
        model=MODEL,
        threshold_version="sface-default",
        processing_time_ms=7,
    )
    assert row.reason_code == "CANDIDATE_NO_FACE"
    assert row.similarity_score is None


def test_session_scope_rolls_back_on_failure() -> None:
    from app.db.session import create_session_factory, session_scope

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with pytest.raises(RuntimeError), session_scope(factory) as db:
        UserRepository(db).create(
            external_id="rolled-back", name="Nope", profile_image_url="profiles/nope.jpg"
        )
        raise RuntimeError("boom")
    with Session(engine) as db:
        assert UserRepository(db).get_by_external_id("rolled-back") is None
