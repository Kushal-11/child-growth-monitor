"""Sync auth + ownership + new-field tests."""
import io
from datetime import date, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User
from app.models.child import Child
from app.models.measurement import MeasurementResult
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
    user = User(
        username="w",
        full_name="W",
        hashed_password=auth_service.hash_password("pw"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = auth_service.create_access_token(
        user_id=user.id, username=user.username
    )
    db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(sync_router)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
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
        "age_months": "12.0", "visit_date": "2026-06-01T00:00:00",
        "entry_method": "manual",
    }, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400


def test_sync_recomputes_tampered_poshan_classification(ctx):
    client, token, Session = ctx
    payload = _payload()
    payload.update({
        "local_uuid": "66666666-6666-6666-6666-666666666666",
        "muac_cm": "11.4",
        "muac_method": "manual",
        "bmi": "20.0",
        "bmi_status": "Normal",
        "muac_status": "Normal",
        "poshan_status": "Normal",
        "poshan_triggered_by": "[]",
        "classification_method": "client_forgery",
        "classification_rationale": "trust me",
    })
    response = client.post(
        "/api/v1/sync",
        data=payload,
        files=_files(),
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["poshan"]["final_status"] == "SAM"
    assert response.json()["poshan"]["triggered_by"] == ["muac"]
    db = Session()
    try:
        stored = (
            db.query(MeasurementResult)
            .join(Visit)
            .filter(Visit.local_uuid == payload["local_uuid"])
            .one()
        )
        assert stored.poshan_status == "SAM"
        assert stored.muac_status == "SAM"
        assert stored.poshan_triggered_by == ["muac"]
        assert stored.classification_method == "poshan_setu_v1"
        assert stored.classification_rationale != "trust me"
    finally:
        db.close()


def test_sync_forged_reference_object_source_is_not_eligible(ctx):
    client, token, Session = ctx
    payload = _payload()
    payload.pop("manual_height_cm")
    payload.update({
        "local_uuid": "77777777-7777-7777-7777-777777777777",
        "predicted_height_cm": "100.0",
        "effective_height_cm": "100.0",
        "height_source": "reference_object",
        "reference_object_detected": "true",
        "manual_weight_kg": "13.7",
        "muac_cm": "12.5",
        "muac_method": "manual",
        "poshan_status": "Normal",
    })
    response = client.post(
        "/api/v1/sync",
        data=payload,
        files=_files(),
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200, response.text
    poshan = response.json()["poshan"]
    assert poshan["bmi_status"] == "Indeterminate"
    assert poshan["muac_status"] == "Normal"
    assert poshan["final_status"] == "Indeterminate"
    db = Session()
    try:
        stored = (
            db.query(MeasurementResult)
            .join(Visit)
            .filter(Visit.local_uuid == payload["local_uuid"])
            .one()
        )
        assert stored.height_source == "unavailable"
        assert stored.reference_object_detected == "false"
        assert stored.bmi_status == "Indeterminate"
        assert stored.poshan_status == "Indeterminate"
    finally:
        db.close()


def test_sync_rejects_future_visit_and_dob_after_visit(ctx):
    client, token, _ = ctx
    headers = {"Authorization": f"Bearer {token}"}

    future = _payload()
    future["local_uuid"] = "88888888-8888-8888-8888-888888888888"
    future["visit_date"] = (
        date.today() + timedelta(days=1)
    ).isoformat() + "T00:00:00"
    response = client.post(
        "/api/v1/sync", data=future, files=_files(), headers=headers
    )
    assert response.status_code == 400
    assert "future" in response.json()["detail"]

    reversed_dates = _payload()
    reversed_dates["local_uuid"] = "99999999-9999-9999-9999-999999999999"
    reversed_dates["date_of_birth"] = "2026-06-02"
    response = client.post(
        "/api/v1/sync", data=reversed_dates, files=_files(), headers=headers
    )
    assert response.status_code == 400
    assert "after visit_date" in response.json()["detail"]
