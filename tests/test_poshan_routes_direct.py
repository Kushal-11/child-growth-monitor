"""Direct coroutine tests for routes while Starlette TestClient is unavailable."""
import asyncio
from datetime import date, timedelta
import inspect
import io

from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401
from app.api.routes import assess_child
from app.api.sync import sync_assessment
from app.models.database import Base
from app.models.measurement import MeasurementResult
from app.models.user import User
from app.models.visit import Visit
from app.services.measurement_service import PoseRuntimeUnavailableError


def _call_kwargs(callable_, **overrides):
    """Resolve FastAPI Form/File defaults for a direct endpoint call."""
    values = {}
    for name, parameter in inspect.signature(callable_).parameters.items():
        if name in overrides:
            values[name] = overrides[name]
            continue
        default = parameter.default
        if default is inspect.Parameter.empty:
            raise AssertionError(f"missing direct-call argument: {name}")
        values[name] = getattr(default, "default", default)
    return values


def _session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    user = User(username="direct", full_name="Direct", hashed_password="unused")
    db.add(user)
    db.commit()
    db.refresh(user)
    return db, user


def _sync_call(db, user, **overrides):
    values = {
        "local_uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "child_name": "Direct Child",
        "date_of_birth": "2024-01-01",
        "sex": "M",
        "age_months": 29.0,
        "visit_date": "2026-06-01T00:00:00",
        "manual_height_cm": 100.0,
        "manual_weight_kg": 13.7,
        "muac_cm": 12.5,
        "muac_method": "manual",
        "entry_method": "manual",
        "db": db,
        "current": user,
    }
    values.update(overrides)
    return asyncio.run(sync_assessment(**_call_kwargs(sync_assessment, **values)))


def test_sync_direct_recomputes_tampered_classification():
    db, user = _session()
    try:
        result = _sync_call(
            db,
            user,
            local_uuid="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            muac_cm=11.4,
            bmi=20.0,
            bmi_status="Normal",
            muac_status="Normal",
            poshan_status="Normal",
            poshan_triggered_by="[]",
            classification_method="client_forgery",
            classification_rationale="trust me",
        )
        stored = (
            db.query(MeasurementResult)
            .join(Visit)
            .filter(Visit.local_uuid == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
            .one()
        )
    finally:
        db.close()

    assert result["poshan"]["final_status"] == "SAM"
    assert result["poshan"]["triggered_by"] == ["muac"]
    assert stored.poshan_status == "SAM"
    assert stored.classification_method == "poshan_setu_v1"
    assert stored.classification_rationale != "trust me"


def test_sync_direct_rejects_forged_reference_object_provenance():
    db, user = _session()
    try:
        result = _sync_call(
            db,
            user,
            local_uuid="cccccccc-cccc-cccc-cccc-cccccccccccc",
            manual_height_cm=None,
            predicted_height_cm=100.0,
            effective_height_cm=100.0,
            height_source="reference_object",
            reference_object_detected="true",
        )
        stored = (
            db.query(MeasurementResult)
            .join(Visit)
            .filter(Visit.local_uuid == "cccccccc-cccc-cccc-cccc-cccccccccccc")
            .one()
        )
    finally:
        db.close()

    assert result["poshan"]["bmi_status"] == "Indeterminate"
    assert result["poshan"]["final_status"] == "Indeterminate"
    assert stored.height_source == "unavailable"
    assert stored.reference_object_detected == "false"
    assert stored.haz_zscore is None
    assert stored.whz_zscore is None


def test_sync_direct_rejects_future_visit_and_reversed_dates():
    db, user = _session()
    try:
        future_date = (date.today() + timedelta(days=1)).isoformat()
        try:
            _sync_call(
                db,
                user,
                local_uuid="dddddddd-dddd-dddd-dddd-dddddddddddd",
                visit_date=f"{future_date}T00:00:00",
            )
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "future" in exc.detail
        else:
            raise AssertionError("future visit_date was accepted")

        try:
            _sync_call(
                db,
                user,
                local_uuid="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                date_of_birth="2026-06-02",
            )
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "after visit_date" in exc.detail
        else:
            raise AssertionError("DOB after visit was accepted")
    finally:
        db.close()


def test_assess_direct_maps_pose_runtime_failure_to_503(tmp_path, monkeypatch):
    from app.api import routes as routes_module

    class UnavailableService:
        @staticmethod
        def _compute_age_months(_dob, _today):
            return 24.0

        @staticmethod
        def _validate_inputs(**_kwargs):
            return None

        @staticmethod
        def assess(**_kwargs):
            raise PoseRuntimeUnavailableError("model asset is missing")

    monkeypatch.setattr(routes_module, "UPLOAD_DIR", tmp_path)
    upload = UploadFile(filename="front.jpg", file=io.BytesIO(b"image"))
    kwargs = _call_kwargs(
        assess_child,
        image=upload,
        child_name="Runtime Child",
        date_of_birth="2024-01-01",
        sex="M",
        db=object(),
        svc=UnavailableService(),
    )

    try:
        asyncio.run(assess_child(**kwargs))
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "Pose measurement runtime is unavailable" in exc.detail
    else:
        raise AssertionError("pose runtime failure was not mapped to HTTP 503")
    assert not list(tmp_path.iterdir())


def test_assess_direct_accepts_manual_measurements_without_image():
    calls = []

    class ManualService:
        @staticmethod
        def _compute_age_months(_dob, _today):
            return 24.0

        @staticmethod
        def _validate_inputs(**_kwargs):
            return None

        @staticmethod
        def assess(**kwargs):
            calls.append(kwargs)
            return {"ok": True}

    result = asyncio.run(
        assess_child(
            **_call_kwargs(
                assess_child,
                image=None,
                child_name="Manual Route Child",
                date_of_birth="2024-01-01",
                assessment_date="2026-01-01",
                sex="F",
                height_cm=90.0,
                weight_kg=11.0,
                db=object(),
                svc=ManualService(),
            )
        )
    )
    assert result == {"ok": True}
    assert calls[0]["image_path"] is None
    assert calls[0]["assessment_date"] == date(2026, 1, 1)


def test_sync_recomputes_who_and_discards_untrusted_mobile_ml(monkeypatch):
    from app.api import sync as sync_module

    class Nutrition:
        @staticmethod
        def compute_haz(_sex, age_months, _height):
            assert age_months == 29
            return -1.234

        @staticmethod
        def classify_haz(_z):
            return "Normal"

        @staticmethod
        def compute_whz(_sex, _age, _height, _weight):
            return -2.345

        @staticmethod
        def classify_whz(_z):
            return "Moderate Acute Malnutrition (MAM)"

    monkeypatch.setattr(
        sync_module, "_get_sync_nutrition_service", lambda: Nutrition()
    )
    db, user = _session()
    try:
        _sync_call(
            db,
            user,
            local_uuid="ffffffff-ffff-ffff-ffff-ffffffffffff",
            haz_zscore=9.0,
            whz_zscore=9.0,
            haz_status="forged",
            whz_status="forged",
            ml_wasting_status="SAM",
            ml_model_version="forged-mobile-model",
            ml_training_data="unknown",
            ml_non_clinical="false",
            sam_probability=1.0,
        )
        stored = (
            db.query(MeasurementResult)
            .join(Visit)
            .filter(Visit.local_uuid == "ffffffff-ffff-ffff-ffff-ffffffffffff")
            .one()
        )
    finally:
        db.close()

    assert stored.haz_zscore == -1.234
    assert stored.whz_zscore == -2.345
    assert stored.haz_status == "Normal"
    assert stored.whz_status == "Moderate Acute Malnutrition (MAM)"
    assert stored.ml_wasting_status is None
    assert stored.ml_model_version is None
    assert stored.ml_training_data == "client_sync_unverified_discarded"
    assert stored.ml_non_clinical is True
    assert stored.sam_probability is None


def test_sync_uuid_collision_does_not_disclose_other_owners_visit_id():
    db, first_user = _session()
    second_user = User(
        username="second",
        full_name="Second",
        hashed_password="unused",
    )
    db.add(second_user)
    db.commit()
    db.refresh(second_user)
    local_uuid = "12121212-1212-1212-1212-121212121212"
    try:
        first = _sync_call(
            db, first_user, local_uuid=local_uuid
        )
        try:
            _sync_call(db, second_user, local_uuid=local_uuid)
        except HTTPException as exc:
            assert exc.status_code == 409
            assert str(first["server_visit_id"]) not in str(exc.detail)
        else:
            raise AssertionError("cross-owner local_uuid collision was accepted")
    finally:
        db.close()
