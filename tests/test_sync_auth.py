"""Sync auth + ownership + new-field tests."""
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User
from app.models.child import Child
from app.models.visit import Visit
from app.services import auth_service
from app.api.auth import router as auth_router
from app.api.sync import router as sync_router


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    import app.api.sync as syncmod
    monkeypatch.setattr(syncmod, "UPLOAD_DIR", tmp_path)
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    db = TestingSession()
    db.add(User(username="w", full_name="W", hashed_password=auth_service.hash_password("pw")))
    db.commit(); db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(sync_router)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    token = client.post("/api/v1/auth/login", json={"username": "w", "password": "pw"}).json()["access_token"]
    return client, token, TestingSession


def _payload():
    return {
        "local_uuid": "11111111-1111-1111-1111-111111111111",
        "child_name": "Kid", "date_of_birth": "2024-01-01", "sex": "M",
        "age_months": "29.0", "visit_date": "2026-06-01T00:00:00",
        "manual_height_cm": "75.0", "manual_weight_kg": "9.0",
        "entry_method": "manual",
    }


def _files():
    return {"image": ("img.jpg", io.BytesIO(b"fakejpeg"), "image/jpeg")}


def test_sync_requires_auth(ctx):
    client, _, _ = ctx
    r = client.post("/api/v1/sync", data=_payload(), files=_files())
    assert r.status_code == 401


def test_sync_stamps_user_and_entry_method(ctx):
    client, token, Session = ctx
    r = client.post("/api/v1/sync", data=_payload(), files=_files(),
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    db = Session()
    visit = db.query(Visit).first()
    child = db.query(Child).first()
    assert visit.user_id is not None
    assert visit.entry_method == "manual"
    assert child.user_id == visit.user_id
    db.close()


def test_sync_idempotent(ctx):
    client, token, _ = ctx
    h = {"Authorization": f"Bearer {token}"}
    r1 = client.post("/api/v1/sync", data=_payload(), files=_files(), headers=h)
    r2 = client.post("/api/v1/sync", data=_payload(), files=_files(), headers=h)
    assert r1.json()["status"] == "synced"
    assert r2.json()["status"] == "already_synced"
    assert r1.json()["server_visit_id"] == r2.json()["server_visit_id"]


def test_sync_rejects_invalid_entry_method(ctx):
    client, token, _ = ctx
    payload = _payload()
    payload["entry_method"] = "bogus"
    payload["local_uuid"] = "22222222-2222-2222-2222-222222222222"
    r = client.post("/api/v1/sync", data=payload, files=_files(),
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400


def test_sync_applies_is_archived(ctx):
    from app.models.child import Child
    client, token, Session = ctx
    payload = _payload()
    payload["is_archived"] = "true"
    payload["local_uuid"] = "33333333-3333-3333-3333-333333333333"
    r = client.post("/api/v1/sync", data=payload, files=_files(),
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    db = Session()
    child = db.query(Child).first()
    assert child.is_archived is True
    db.close()


def test_sync_manual_entry_without_image(ctx):
    from app.models.visit import Visit
    client, token, Session = ctx
    payload = _payload()
    payload["local_uuid"] = "44444444-4444-4444-4444-444444444444"
    # No files= at all → no image part.
    r = client.post("/api/v1/sync", data=payload,
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    db = Session()
    v = db.query(Visit).filter(Visit.local_uuid == payload["local_uuid"]).one()
    assert v.image_path is None
    assert v.entry_method == "manual"
    db.close()


def test_sync_rejects_empty_submission(ctx):
    client, token, _ = ctx
    # No image AND no measurements → 400.
    r = client.post("/api/v1/sync", data={
        "local_uuid": "55555555-5555-5555-5555-555555555555",
        "child_name": "X", "date_of_birth": "2024-01-01", "sex": "M",
        "age_months": "29.0", "visit_date": "2026-06-01T00:00:00",
        "entry_method": "manual",
    }, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400
