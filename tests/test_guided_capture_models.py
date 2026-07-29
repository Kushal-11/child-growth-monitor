"""Persistence constraints for guided visit aggregates."""

from datetime import date, datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
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


def _visit(db):
    user = User(
        username=f"worker-{uuid4()}",
        full_name="Field Worker",
        hashed_password="hash",
    )
    child = Child(
        name="Child 001",
        date_of_birth=date(2024, 1, 1),
        sex="F",
        owner=user,
    )
    visit = Visit(
        child=child,
        user_id=None,
        local_uuid=str(uuid4()),
        visit_date=datetime(2026, 7, 29),
        age_months=30,
        capture_state="draft_capture",
        capture_started_at=datetime.now(timezone.utc),
    )
    db.add_all([user, child, visit])
    db.commit()
    visit.user_id = user.id
    db.commit()
    return visit, user


def test_visit_owns_ordered_assets_results_and_revisions(db):
    visit, user = _visit(db)
    later = datetime(2026, 7, 29, 10, 1)
    earlier = datetime(2026, 7, 29, 10, 0)
    visit.capture_assets.extend(
        [
            CaptureAsset(
                asset_uuid=str(uuid4()),
                role="side",
                captured_at=later,
                local_path="/retained/side.jpg",
            ),
            CaptureAsset(
                asset_uuid=str(uuid4()),
                role="front",
                captured_at=earlier,
                local_path="/retained/front.jpg",
            ),
        ]
    )
    visit.camera_results.extend(
        [
            CameraResult(
                result_uuid=str(uuid4()),
                version=2,
                supersedes_result_uuid=str(uuid4()),
                method="camera_screening_v1",
                model_version="model-v1",
                manifest_checksum="a" * 64,
                training_data_label="research_only",
            ),
            CameraResult(
                result_uuid=str(uuid4()),
                version=1,
                method="camera_screening_v1",
                model_version="model-v1",
                manifest_checksum="a" * 64,
                training_data_label="research_only",
            ),
        ]
    )
    visit.measured_revisions.extend(
        [
            MeasuredDetailRevision(
                revision_uuid=str(uuid4()),
                revision_number=2,
                before_json="{}",
                after_json='{"weight_kg": 12}',
                editor_user_id=user.id,
            ),
            MeasuredDetailRevision(
                revision_uuid=str(uuid4()),
                revision_number=1,
                before_json="{}",
                after_json='{"height_cm": 88}',
                editor_user_id=user.id,
            ),
        ]
    )
    db.commit()
    db.expire_all()
    stored = db.get(Visit, visit.id)

    assert [asset.role for asset in stored.capture_assets] == ["front", "side"]
    assert [result.version for result in stored.camera_results] == [1, 2]
    assert [revision.revision_number for revision in stored.measured_revisions] == [
        1,
        2,
    ]
    assert all(result.non_clinical is True for result in stored.camera_results)


@pytest.mark.parametrize(
    ("factory", "attribute"),
    [
        (
            lambda visit_id, entity_uuid: CaptureAsset(
                visit_id=visit_id,
                asset_uuid=entity_uuid,
                role="front",
                captured_at=datetime.utcnow(),
            ),
            "asset_uuid",
        ),
        (
            lambda visit_id, entity_uuid: CameraResult(
                visit_id=visit_id,
                result_uuid=entity_uuid,
                version=1,
                method="camera_screening_v1",
                model_version="v1",
                manifest_checksum="b" * 64,
                training_data_label="research_only",
            ),
            "result_uuid",
        ),
        (
            lambda visit_id, entity_uuid: MeasuredDetailRevision(
                visit_id=visit_id,
                revision_uuid=entity_uuid,
                revision_number=1,
                before_json="{}",
                after_json="{}",
            ),
            "revision_uuid",
        ),
    ],
)
def test_entity_uuids_are_globally_unique(db, factory, attribute):
    first_visit, _ = _visit(db)
    second_visit, _ = _visit(db)
    entity_uuid = str(uuid4())
    db.add(factory(first_visit.id, entity_uuid))
    db.commit()

    db.add(factory(second_visit.id, entity_uuid))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    assert attribute


def test_database_rejects_clinical_camera_result(db):
    visit, _ = _visit(db)
    db.add(
        CameraResult(
            visit_id=visit.id,
            result_uuid=str(uuid4()),
            version=1,
            method="camera_screening_v1",
            model_version="v1",
            manifest_checksum="c" * 64,
            training_data_label="research_only",
            non_clinical=False,
        )
    )

    with pytest.raises(IntegrityError):
        db.commit()


def test_deleting_media_metadata_preserves_visit_and_measurement(db):
    visit, _ = _visit(db)
    measurement = MeasurementResult(visit_id=visit.id, manual_height_cm=88)
    asset = CaptureAsset(
        visit_id=visit.id,
        asset_uuid=str(uuid4()),
        role="front",
        captured_at=datetime.utcnow(),
    )
    db.add_all([measurement, asset])
    db.commit()
    asset_id = asset.id
    visit_id = visit.id

    db.delete(asset)
    db.commit()

    assert db.get(CaptureAsset, asset_id) is None
    assert db.get(Visit, visit_id) is not None
    assert db.scalar(
        select(MeasurementResult).where(MeasurementResult.visit_id == visit_id)
    ) is not None


def test_deleting_visit_cascades_guided_children(db):
    visit, _ = _visit(db)
    asset = CaptureAsset(
        asset_uuid=str(uuid4()),
        role="front",
        captured_at=datetime.utcnow(),
    )
    result = CameraResult(
        result_uuid=str(uuid4()),
        version=1,
        method="camera_screening_v1",
        model_version="v1",
        manifest_checksum="d" * 64,
        training_data_label="research_only",
    )
    revision = MeasuredDetailRevision(
        revision_uuid=str(uuid4()),
        revision_number=1,
        before_json="{}",
        after_json="{}",
    )
    visit.capture_assets.append(asset)
    visit.camera_results.append(result)
    visit.measured_revisions.append(revision)
    db.commit()
    ids = (asset.id, result.id, revision.id)

    db.delete(visit)
    db.commit()

    assert db.get(CaptureAsset, ids[0]) is None
    assert db.get(CameraResult, ids[1]) is None
    assert db.get(MeasuredDetailRevision, ids[2]) is None
