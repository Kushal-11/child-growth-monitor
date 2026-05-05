"""Tests for POST /api/v1/sync — idempotent ingestion of mobile assessments."""
import io
import uuid

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _payload():
    return {
        "local_uuid": str(uuid.uuid4()),
        "child_name": "Test Child",
        "date_of_birth": "2024-01-01",
        "sex": "M",
        "age_months": "16.0",
        "visit_date": "2026-05-05T10:00:00",
        "predicted_height_cm": "78.0",
        "predicted_weight_kg": "9.5",
        "haz_zscore": "-1.0",
        "whz_zscore": "-0.5",
        "haz_status": "Normal",
        "whz_status": "Normal",
        "muac_cm": "14.0",
        "muac_status": "Normal",
        "muac_method": "estimated_from_whz",
        "ml_wasting_status": "Normal",
        "ml_estimated_weight_kg": "9.4",
        "confidence_score": "0.85",
        "body_build": "average",
        "side_view_used": "false",
        "sam_probability": "0.02",
        "mam_probability": "0.10",
        "normal_probability": "0.85",
        "risk_probability": "0.02",
        "overweight_probability": "0.01",
    }


def _file():
    return ("front.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")


def test_sync_happy_path_returns_synced():
    body = _payload()
    response = client.post(
        "/api/v1/sync",
        data=body,
        files={"image": _file()},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "synced"
    assert isinstance(data["server_visit_id"], int)


def test_sync_same_local_uuid_twice_is_idempotent():
    body = _payload()
    first = client.post("/api/v1/sync", data=body, files={"image": _file()})
    assert first.status_code == 200
    first_id = first.json()["server_visit_id"]

    second = client.post("/api/v1/sync", data=body, files={"image": _file()})
    assert second.status_code == 200
    assert second.json()["status"] == "already_synced"
    assert second.json()["server_visit_id"] == first_id


def test_sync_missing_required_field_returns_422():
    body = _payload()
    del body["local_uuid"]
    response = client.post("/api/v1/sync", data=body, files={"image": _file()})
    assert response.status_code == 422
