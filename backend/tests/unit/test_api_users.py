from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.base import Base
from app.db.session import create_database_engine
from app.main import create_app
from tests.factories.images import encode, textured_array


def image(seed: int = 1) -> tuple[str, bytes, str]:
    return ("original-name.png", encode(textured_array(320, 240, seed=seed)), "image/png")


@pytest.fixture
def phase8_client(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    database_path = tmp_path / "phase8.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"
    engine = create_database_engine(database_url)
    Base.metadata.create_all(engine)
    engine.dispose()
    upload_dir = tmp_path / "uploads"
    settings = Settings(
        ENV="local",
        FACE_ENGINE="fake",
        INFERENCE_WORKERS=1,
        DATABASE_URL=database_url,
        UPLOAD_DIR=upload_dir,
        RATE_LIMIT_COMPARE="1000/minute",
    )
    with TestClient(create_app(settings), client=(f"phase8-{tmp_path.name}", 50000)) as client:
        yield client, upload_dir


def create_user(client: TestClient, *, external_id: str = "emp-100", seed: int = 1):
    return client.post(
        "/api/v1/users",
        data={"external_id": external_id, "name": "Ada Lovelace"},
        files={"profile_image": image(seed)},
    )


def test_create_and_get_user_reencodes_profile_under_generated_name(
    phase8_client: tuple[TestClient, Path],
) -> None:
    client, upload_dir = phase8_client
    response = create_user(client)
    assert response.status_code == 201
    body = response.json()
    assert body["external_id"] == "emp-100"
    assert body["profile_image_url"].startswith(f"profiles/{body['id']}/")
    assert body["profile_image_url"].endswith(".jpg")
    assert "original-name" not in body["profile_image_url"]
    stored = upload_dir / body["profile_image_url"]
    assert stored.read_bytes().startswith(b"\xff\xd8\xff")

    fetched = client.get(f"/api/v1/users/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body

    profile = client.get(f"/api/v1/users/{body['id']}/profile-image")
    assert profile.status_code == 200
    assert profile.headers["content-type"] == "image/jpeg"
    assert profile.content.startswith(b"\xff\xd8\xff")


def test_search_matches_external_id_and_name_and_refuses_to_list_everything(
    phase8_client: tuple[TestClient, Path],
) -> None:
    client, _ = phase8_client
    for seed, (external_id, name) in enumerate(
        [("emp-001", "Ada Lovelace"), ("emp-002", "Grace Hopper")], start=1
    ):
        created = client.post(
            "/api/v1/users",
            data={"external_id": external_id, "name": name},
            files={"profile_image": image(seed)},
        )
        assert created.status_code == 201

    by_name = client.get("/api/v1/users", params={"search": "grace"})
    assert by_name.status_code == 200
    assert [item["external_id"] for item in by_name.json()["items"]] == ["emp-002"]

    by_external_id = client.get("/api/v1/users", params={"search": "emp-001"})
    assert [item["name"] for item in by_external_id.json()["items"]] == ["Ada Lovelace"]

    # A bare wildcard must be matched literally, not expanded into "everything".
    wildcard = client.get("/api/v1/users", params={"search": "%%"})
    assert wildcard.json()["items"] == []

    # No query, or one too short to be a real search, is rejected rather than
    # returning the whole table. Padding a short query with spaces does not get
    # it past the guard either.
    assert client.get("/api/v1/users").status_code == 422
    assert client.get("/api/v1/users", params={"search": "a"}).status_code == 422
    assert client.get("/api/v1/users", params={"search": "  a  "}).status_code == 422


def test_duplicate_external_id_is_a_conflict_and_leaves_one_file(
    phase8_client: tuple[TestClient, Path],
) -> None:
    client, upload_dir = phase8_client
    assert create_user(client).status_code == 201
    duplicate = create_user(client, seed=2)
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "CONFLICT"
    assert len(list(upload_dir.rglob("*.jpg"))) == 1


def test_replace_profile_removes_old_file(phase8_client: tuple[TestClient, Path]) -> None:
    client, upload_dir = phase8_client
    created = create_user(client).json()
    old_path = upload_dir / created["profile_image_url"]
    response = client.put(
        f"/api/v1/users/{created['id']}/profile-image",
        files={"profile_image": image(9)},
    )
    assert response.status_code == 200
    assert response.json()["profile_image_url"] != created["profile_image_url"]
    assert not old_path.exists()
    assert len(list(upload_dir.rglob("*.jpg"))) == 1


def test_verify_uses_stored_reference_and_persists_history(
    phase8_client: tuple[TestClient, Path],
) -> None:
    client, _ = phase8_client
    user = create_user(client, seed=5).json()
    verified = client.post(
        f"/api/v1/users/{user['id']}/verify",
        files={"candidate_image": image(5)},
    )
    assert verified.status_code == 200
    assert verified.json()["decision"] == "MATCH"
    assert verified.json()["matched"] is True

    history = client.get(f"/api/v1/users/{user['id']}/verifications")
    assert history.status_code == 200
    body = history.json()
    assert body["user_id"] == user["id"]
    assert len(body["items"]) == 1
    assert body["items"][0]["decision"] == "MATCH"
    assert body["items"][0]["detector_version"] == "fake@1"
    assert not {"embedding", "candidate_image", "reference_image"} & set(body["items"][0])


def test_unknown_user_returns_normalized_404(phase8_client: tuple[TestClient, Path]) -> None:
    client, _ = phase8_client
    response = client.get("/api/v1/users/00000000-0000-0000-0000-000000000001")
    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "NOT_FOUND",
        "message": "User not found.",
        "detail": None,
    }


def test_missing_reference_file_does_not_create_history(
    phase8_client: tuple[TestClient, Path],
) -> None:
    client, upload_dir = phase8_client
    user = create_user(client).json()
    (upload_dir / user["profile_image_url"]).unlink()
    response = client.post(
        f"/api/v1/users/{user['id']}/verify",
        files={"candidate_image": image()},
    )
    assert response.status_code == 404
    history = client.get(f"/api/v1/users/{user['id']}/verifications").json()
    assert history["items"] == []
