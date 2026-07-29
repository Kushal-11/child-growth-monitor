"""Transactional guided-visit orchestration tests."""

from datetime import date, datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker

from app.models import (
    CameraResult,
    CaptureAsset,
    Child,
    MeasuredDetailRevision,
    MeasurementResult,
    User,
    Visit,
)
from app.models.database import Base
from app.schemas.guided_capture import (
    CameraResultSubmission,
    CaptureAssetSubmission,
    MeasuredDetailsSubmission,
)
from app.services.guided_capture_contract import CaptureState
from app.services.guided_visit_service import GuidedVisitService
from app.services.who_data_service import WHODataService


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(scope="module")
def service():
    who = WHODataService()
    who.load_all()
    return GuidedVisitService(who)


def _owner_and_child(db):
    owner = User(
        username=f"owner-{uuid4()}",
        full_name="Owner",
        hashed_password="hash",
    )
    child = Child(
        name="Child 001",
        date_of_birth=date(2024, 1, 29),
        sex="F",
        owner=owner,
    )
    other = User(
        username=f"other-{uuid4()}",
        full_name="Other",
        hashed_password="hash",
    )
    db.add_all([owner, child, other])
    db.commit()
    return owner, child, other


def _create_visit(db, service):
    owner, child, other = _owner_and_child(db)
    visit_uuid = uuid4()
    visit = service.create_draft_visit(
        db,
        owner_user_id=owner.id,
        child_id=child.id,
        local_uuid=visit_uuid,
        visit_date=date(2026, 7, 29),
        device_metadata={"platform": "android"},
        consent_version="guided_capture_consent_v1",
        consent_timestamp=datetime.now(timezone.utc),
        consent_operator_identifier=str(owner.id),
    )
    return visit, owner, child, other, visit_uuid


def _asset(visit_uuid, role, *, verdict="accepted"):
    return CaptureAssetSubmission(
        asset_uuid=uuid4(),
        visit_uuid=visit_uuid,
        role=role,
        captured_at=datetime.now(timezone.utc),
        local_path=f"/retained/{role}.jpg",
        quality_verdict=verdict,
        overall_score=0.9,
    )


def _camera_result(visit_uuid, *, result_uuid=None, version=1):
    return CameraResultSubmission(
        result_uuid=result_uuid or uuid4(),
        visit_uuid=visit_uuid,
        version=version,
        estimated_height_cm=88,
        estimated_weight_kg=12,
        height_source="who_statistical",
        weight_source="ml_estimated",
        component_probabilities={"SAM": 0.1, "MAM": 0.2, "Normal": 0.7},
        method="camera_screening_v1",
        model_version="model-v1",
        manifest_checksum="a" * 64,
        training_data_label="research_only",
        non_clinical=True,
        created_at=datetime.now(timezone.utc),
    )


def _reach_estimated_report(db, service):
    visit, owner, child, other, visit_uuid = _create_visit(db, service)
    service.append_capture_asset(db, owner.id, _asset(visit_uuid, "front"))
    service.append_capture_asset(db, owner.id, _asset(visit_uuid, "side"))
    service.transition_visit(
        db,
        owner.id,
        visit_uuid,
        CaptureState.PROCESSING,
    )
    result = service.append_camera_result(
        db,
        owner.id,
        _camera_result(visit_uuid),
    )
    db.refresh(visit)
    assert visit.capture_state == CaptureState.ESTIMATED_REPORT.value
    return visit, owner, child, other, visit_uuid, result


def test_create_draft_is_owner_scoped_and_persists_before_capture(db, service):
    visit, owner, _child, other, visit_uuid = _create_visit(db, service)

    assert visit.capture_state == CaptureState.DRAFT_CAPTURE.value
    assert visit.local_uuid == str(visit_uuid)
    assert visit.user_id == owner.id
    assert visit.device_metadata_json == {"platform": "android"}
    with pytest.raises(LookupError):
        service.transition_visit(
            db,
            other.id,
            visit_uuid,
            CaptureState.INCOMPLETE_CAPTURE,
        )


def test_processing_requires_accepted_front_and_side(db, service):
    visit, owner, _child, _other, visit_uuid = _create_visit(db, service)
    service.append_capture_asset(db, owner.id, _asset(visit_uuid, "front"))

    with pytest.raises(ValueError, match="accepted front and side"):
        service.transition_visit(
            db,
            owner.id,
            visit_uuid,
            CaptureState.PROCESSING,
        )
    db.refresh(visit)
    assert visit.capture_state == CaptureState.DRAFT_CAPTURE.value

    service.append_capture_asset(
        db,
        owner.id,
        _asset(visit_uuid, "side", verdict="rejected"),
    )
    with pytest.raises(ValueError, match="accepted front and side"):
        service.transition_visit(
            db,
            owner.id,
            visit_uuid,
            CaptureState.PROCESSING,
        )


def test_camera_result_is_immutable_and_moves_processing_to_estimated(db, service):
    visit, owner, _child, _other, visit_uuid, result = _reach_estimated_report(
        db, service
    )
    original_count = db.scalar(select(func.count(CameraResult.id)))
    changed = _camera_result(
        visit_uuid,
        result_uuid=result.result_uuid,
        version=result.version,
    ).model_copy(update={"estimated_height_cm": 99})

    with pytest.raises(ValueError, match="immutable"):
        service.append_camera_result(db, owner.id, changed)

    assert db.scalar(select(func.count(CameraResult.id))) == original_count
    assert db.get(CameraResult, result.id).estimated_height_cm == 88
    assert visit.capture_state == CaptureState.ESTIMATED_REPORT.value


def test_camera_result_cannot_be_appended_before_processing(db, service):
    _visit, owner, _child, _other, visit_uuid = _create_visit(db, service)

    with pytest.raises(ValueError, match="processing"):
        service.append_camera_result(
            db,
            owner.id,
            _camera_result(visit_uuid),
        )


def test_partial_measured_save_preserves_camera_result_and_appends_revision(
    db,
    service,
):
    visit, owner, _child, _other, visit_uuid, camera = _reach_estimated_report(
        db, service
    )
    details = MeasuredDetailsSubmission(
        measurement_mode="standing_height",
        oedema="not_checked",
        height_cm=83.58,
        measured_at=datetime.now(timezone.utc),
        notes="Height board only",
    )
    report = service.save_measured_details(
        db,
        owner_user_id=owner.id,
        visit_uuid=visit_uuid,
        measurement_date=date(2026, 7, 29),
        details=details,
        editor_user_id=owner.id,
    )

    assert report.manual_height_cm == 83.58
    assert report.manual_weight_kg is None
    assert report.haz_zscore == pytest.approx(-2.01, abs=0.01)
    assert report.whz_zscore is None
    assert report.whz_status is None
    assert report.muac_status is None
    assert report.who_acute_status == "UNKNOWN"
    assert report.poshan_status == "Indeterminate"
    assert db.get(CameraResult, camera.id) is not None
    assert db.scalar(
        select(func.count(MeasuredDetailRevision.id)).where(
            MeasuredDetailRevision.visit_id == visit.id
        )
    ) == 1
    db.refresh(visit)
    assert visit.capture_state == CaptureState.MEASURED_REPORT.value


def test_oedema_triggers_who_sam_without_changing_poshan(db, service):
    visit, owner, _child, _other, visit_uuid, _camera = _reach_estimated_report(
        db, service
    )
    details = MeasuredDetailsSubmission(
        measurement_mode="standing_height",
        oedema="yes",
        measured_at=datetime.now(timezone.utc),
    )
    report = service.save_measured_details(
        db,
        owner_user_id=owner.id,
        visit_uuid=visit_uuid,
        measurement_date=date(2026, 7, 29),
        details=details,
        editor_user_id=owner.id,
    )

    assert report.who_acute_status == "SAM"
    assert report.who_acute_triggered_by == '["oedema"]'
    assert report.poshan_status == "Indeterminate"
    assert report.classification_method == "poshan_setu_v1"


def test_invalid_measured_save_rolls_back_current_report_and_history(db, service):
    visit, owner, _child, _other, visit_uuid, _camera = _reach_estimated_report(
        db, service
    )
    valid = MeasuredDetailsSubmission(
        measurement_mode="standing_height",
        oedema="no",
        height_cm=88,
        weight_kg=12,
        measured_at=datetime.now(timezone.utc),
    )
    original = service.save_measured_details(
        db,
        owner_user_id=owner.id,
        visit_uuid=visit_uuid,
        measurement_date=date(2026, 7, 29),
        details=valid,
        editor_user_id=owner.id,
    )
    original_snapshot = (
        original.manual_height_cm,
        original.manual_weight_kg,
        original.haz_zscore,
        original.whz_zscore,
    )
    revision_count = db.scalar(
        select(func.count(MeasuredDetailRevision.id)).where(
            MeasuredDetailRevision.visit_id == visit.id
        )
    )

    with pytest.raises(ValueError, match="measurement date"):
        service.save_measured_details(
            db,
            owner_user_id=owner.id,
            visit_uuid=visit_uuid,
            measurement_date=date(2026, 7, 28),
            details=valid.model_copy(update={"height_cm": 70}),
            editor_user_id=owner.id,
        )

    db.expire_all()
    stored = db.scalar(
        select(MeasurementResult).where(MeasurementResult.visit_id == visit.id)
    )
    assert (
        stored.manual_height_cm,
        stored.manual_weight_kg,
        stored.haz_zscore,
        stored.whz_zscore,
    ) == original_snapshot
    assert db.scalar(
        select(func.count(MeasuredDetailRevision.id)).where(
            MeasuredDetailRevision.visit_id == visit.id
        )
    ) == revision_count


def test_non_finite_measured_values_fail_before_database_mutation():
    with pytest.raises(ValidationError):
        MeasuredDetailsSubmission(
            measurement_mode="standing_height",
            oedema="no",
            height_cm=float("nan"),
            measured_at=datetime.now(timezone.utc),
        )


def test_media_deletion_requires_each_asset_acknowledgement(db, service):
    visit, owner, _child, _other, visit_uuid, _camera = _reach_estimated_report(
        db, service
    )

    with pytest.raises(ValueError, match="every asset UUID"):
        service.delete_visit_media(db, owner.id, visit_uuid)

    assets = list(visit.capture_assets)
    assets[0].server_acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    with pytest.raises(ValueError, match="every asset UUID"):
        service.delete_visit_media(db, owner.id, visit_uuid)

    assets[1].server_acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    updated = service.delete_visit_media(db, owner.id, visit_uuid)

    assert updated.media_deleted_at is not None
    assert all(asset.local_path is None for asset in updated.capture_assets)
