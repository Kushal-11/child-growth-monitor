"""Shared fixtures for guided-sync API tests."""

import base64
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.auth import router as auth_router
from app.api.guided_sync import (
    get_guided_sync_service,
    router as guided_sync_router,
)
from app.models.child import Child
from app.models.database import Base, get_db
from app.models.user import User
from app.services import auth_service
from app.services.guided_sync_service import GuidedSyncService
from app.services.who_data_service import WHODataService


@dataclass
class GuidedSyncContext:
    client: TestClient
    session_factory: sessionmaker
    service: GuidedSyncService
    owner_id: int
    other_id: int
    child_id: int
    other_child_id: int
    owner_headers: dict[str, str]
    other_headers: dict[str, str]
    engine: object

    def close(self) -> None:
        self.client.close()
        self.engine.dispose()


_WHO: WHODataService | None = None


def _who() -> WHODataService:
    global _WHO
    if _WHO is None:
        _WHO = WHODataService()
        _WHO.load_all()
    return _WHO


def build_context(media_root: Path) -> GuidedSyncContext:
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
        username="sync-owner",
        full_name="Sync Owner",
        hashed_password=auth_service.hash_password("pw"),
    )
    other = User(
        username="sync-other",
        full_name="Sync Other",
        hashed_password=auth_service.hash_password("pw"),
    )
    db.add_all([owner, other])
    db.flush()
    child = Child(
        name="Child 001",
        date_of_birth=date(2024, 1, 29),
        sex="F",
        user_id=owner.id,
    )
    other_child = Child(
        name="Child 002",
        date_of_birth=date(2024, 1, 29),
        sex="M",
        user_id=other.id,
    )
    db.add_all([child, other_child])
    db.commit()
    values = owner.id, other.id, child.id, other_child.id
    db.close()

    service = GuidedSyncService(media_root=media_root, who_data=_who())
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(guided_sync_router)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_guided_sync_service] = lambda: service
    client = TestClient(app)

    def headers(username: str) -> dict[str, str]:
        token = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": "pw"},
        ).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    return GuidedSyncContext(
        client=client,
        session_factory=testing_session,
        service=service,
        owner_id=values[0],
        other_id=values[1],
        child_id=values[2],
        other_child_id=values[3],
        owner_headers=headers("sync-owner"),
        other_headers=headers("sync-other"),
        engine=engine,
    )


def visit_payload(
    child_id: int,
    visit_uuid: str,
    *,
    state: str = "draft_capture",
) -> dict:
    return {
        "local_uuid": visit_uuid,
        "child_id": child_id,
        "visit_date": "2026-07-29T00:00:00",
        "age_months": 30.0,
        "capture_state": state,
        "device_metadata": {"platform": "android"},
        "consent_version": "guided_capture_consent_v1",
        "consent_timestamp": "2026-07-29T09:00:00Z",
        "consent_operator_identifier": "operator-7",
        "capture_started_at": "2026-07-29T09:00:00Z",
        "capture_completed_at": None,
    }


def asset_payload(
    visit_uuid: str,
    asset_uuid: str,
    role: str,
    *,
    content: bytes | None = None,
) -> dict:
    raw = content or f"{role}-asset-bytes".encode()
    return {
        "asset_uuid": asset_uuid,
        "visit_uuid": visit_uuid,
        "role": role,
        "captured_at": "2026-07-29T09:05:00Z",
        "selected_rank": 0,
        "quality": {
            "pose": 0.91,
            "coverage": 0.92,
            "orientation": 0.93,
            "sharpness": 0.94,
            "lighting": 0.95,
            "overall": 0.93,
            "threshold_version": "guided_capture_quality_v1",
        },
        "image_width": 1080,
        "image_height": 1920,
        "exif_orientation": 1,
        "display_orientation": 0,
        "device_camera_metadata": {"lens": "back"},
        "content_type": "image/jpeg",
        "content_checksum": hashlib.sha256(raw).hexdigest(),
        "content_base64": base64.b64encode(raw).decode(),
    }


def camera_payload(visit_uuid: str, result_uuid: str) -> dict:
    return {
        "result_uuid": result_uuid,
        "visit_uuid": visit_uuid,
        "version": 1,
        "supersedes_result_uuid": None,
        "estimated_height_cm": 88.0,
        "estimated_weight_kg": 11.0,
        "height_source": "who_height_for_age_median_v1",
        "weight_source": "ml_weight_estimator_v1",
        "estimated_haz": -1.2,
        "estimated_whz": -0.8,
        "estimated_stunting_status": "Normal",
        "estimated_wasting_status": "Normal",
        "experimental_overall_category": None,
        "component_probabilities": {},
        "body_proportion_features": {"shoulder_hip_ratio": 0.82},
        "capture_quality_summary": {
            "overall": 0.93,
            "used_views": ["front", "side"],
        },
        "method": "camera_screening_v1",
        "model_version": "camera-v1",
        "manifest_checksum": "a" * 64,
        "training_data_label": "research_only",
        "non_clinical": True,
        "created_at": "2026-07-29T09:10:00Z",
    }


def revision_payload(
    visit_uuid: str,
    revision_uuid: str,
    *,
    revision_number: int = 1,
    height_cm: float | None = 83.58,
) -> dict:
    return {
        "revision_uuid": revision_uuid,
        "visit_uuid": visit_uuid,
        "revision_number": revision_number,
        "before": {},
        "after": {
            "height_cm": height_cm,
            "weight_kg": None,
            "muac_cm": None,
            "measurement_mode": "standing_height",
            "oedema": "not_checked",
            "measured_at": "2026-07-29T10:00:00Z",
            "notes": "Height board only",
            "haz_status": "Normal",
            "poshan_status": "Normal",
        },
        "editor_user_id": 999,
        "created_at": "2026-07-29T10:00:00Z",
        "reason": "same-day follow-up",
    }


def put_visit(ctx: GuidedSyncContext, visit_uuid: str):
    return ctx.client.put(
        f"/api/v1/sync/guided/visits/{visit_uuid}",
        json=visit_payload(ctx.child_id, visit_uuid),
        headers=ctx.owner_headers,
    )


def put_required_assets(ctx: GuidedSyncContext, visit_uuid: str):
    responses = []
    for suffix, role in (("1", "front"), ("2", "side")):
        asset_uuid = f"20000000-0000-0000-0000-00000000000{suffix}"
        responses.append(
            ctx.client.put(
                (
                    f"/api/v1/sync/guided/visits/{visit_uuid}"
                    f"/assets/{asset_uuid}"
                ),
                json=asset_payload(visit_uuid, asset_uuid, role),
                headers=ctx.owner_headers,
            )
        )
    return responses
