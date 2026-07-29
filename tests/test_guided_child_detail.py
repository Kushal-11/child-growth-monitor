"""Guided-visit fields in the owner-scoped child timeline contract."""

from datetime import date, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.auth import router as auth_router
from app.api.routes import router as api_router
from app.models.camera_result import CameraResult
from app.models.capture_asset import CaptureAsset
from app.models.child import Child
from app.models.database import Base, get_db
from app.models.measurement import MeasurementResult
from app.models.user import User
from app.models.visit import Visit
from app.services import auth_service


@pytest.fixture()
def guided_detail_context():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(bind=engine)

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    db = testing_session()
    owner = User(
        username="owner",
        full_name="Owner",
        hashed_password=auth_service.hash_password("pw"),
    )
    other = User(
        username="other",
        full_name="Other",
        hashed_password=auth_service.hash_password("pw"),
    )
    db.add_all([owner, other])
    db.flush()
    child = Child(
        name="Child 001",
        date_of_birth=date(2024, 1, 1),
        sex="F",
        user_id=owner.id,
    )
    other_child = Child(
        name="Other child",
        date_of_birth=date(2024, 1, 1),
        sex="M",
        user_id=other.id,
    )
    db.add_all([child, other_child])
    db.flush()
    visit = Visit(
        child_id=child.id,
        user_id=owner.id,
        local_uuid="10000000-0000-0000-0000-000000000001",
        visit_date=datetime(2026, 7, 29),
        age_months=30,
        entry_method="guided_capture",
        capture_state="measured_report",
        media_deleted_at=None,
    )
    db.add(visit)
    db.flush()
    db.add_all(
        [
            CaptureAsset(
                asset_uuid="20000000-0000-0000-0000-000000000001",
                visit_id=visit.id,
                role="front",
                captured_at=datetime(2026, 7, 29, 10),
                quality_verdict="accepted",
                sync_state="synced",
                server_acknowledged_at=datetime(2026, 7, 29, 11),
            ),
            CaptureAsset(
                asset_uuid="20000000-0000-0000-0000-000000000002",
                visit_id=visit.id,
                role="side",
                captured_at=datetime(2026, 7, 29, 10, 1),
                quality_verdict="accepted",
                sync_state="pending",
            ),
            CameraResult(
                result_uuid="30000000-0000-0000-0000-000000000001",
                visit_id=visit.id,
                version=1,
                estimated_height_cm=88,
                method="camera_screening_v1",
                model_version="camera-v1",
                manifest_checksum="a" * 64,
                training_data_label="research_only",
                non_clinical=True,
            ),
            CameraResult(
                result_uuid="30000000-0000-0000-0000-000000000002",
                visit_id=visit.id,
                version=2,
                supersedes_result_uuid=(
                    "30000000-0000-0000-0000-000000000001"
                ),
                estimated_height_cm=87,
                estimated_weight_kg=11,
                estimated_stunting_status="Moderate Stunting",
                method="camera_screening_v1",
                model_version="camera-v2",
                manifest_checksum="b" * 64,
                training_data_label="research_only",
                non_clinical=True,
            ),
            MeasurementResult(
                visit_id=visit.id,
                manual_height_cm=83.5,
                height_method="manual",
                haz_zscore=-2.1,
                haz_status="Moderate Stunting",
                measurement_mode="standing_height",
                oedema="not_checked",
            ),
        ]
    )
    db.commit()
    child_id = child.id
    other_child_id = other_child.id
    db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(api_router)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    def headers(username: str):
        token = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": "pw"},
        ).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    yield client, headers, child_id, other_child_id
    engine.dispose()


def test_guided_visit_fields_are_typed_and_complete(guided_detail_context):
    client, headers, child_id, _ = guided_detail_context

    response = client.get(
        f"/api/v1/children/{child_id}",
        headers=headers("owner"),
    )

    assert response.status_code == 200
    visit = response.json()["visits"][0]
    assert visit["local_uuid"] == "10000000-0000-0000-0000-000000000001"
    assert visit["entry_method"] == "guided_capture"
    assert visit["capture_state"] == "measured_report"
    assert visit["has_measured_report"] is True
    assert visit["camera_result_summary"] == {
        "result_uuid": "30000000-0000-0000-0000-000000000002",
        "version": 2,
        "estimated_height_cm": 87.0,
        "estimated_weight_kg": 11.0,
        "estimated_stunting_status": "Moderate Stunting",
        "estimated_wasting_status": None,
        "experimental_overall_category": None,
        "method": "camera_screening_v1",
        "model_version": "camera-v2",
        "non_clinical": True,
    }
    assert visit["required_asset_acknowledgement"] == {
        "front": "acknowledged",
        "side": "pending",
    }
    assert visit["required_assets_acknowledged"] is False
    assert visit["media_deleted_at"] is None
    assert visit["measurement"]["manual_height_cm"] == 83.5


def test_guided_child_detail_remains_owner_scoped(guided_detail_context):
    client, headers, child_id, other_child_id = guided_detail_context

    assert (
        client.get(
            f"/api/v1/children/{child_id}",
            headers=headers("other"),
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/v1/children/{other_child_id}",
            headers=headers("owner"),
        ).status_code
        == 404
    )
