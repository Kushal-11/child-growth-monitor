"""Regression tests for authoritative manual-height resolution."""
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import get_assessment_service
from app.models.database import Base, get_db
from app.services.assessment_service import AssessmentService
from app.services.measurement_service import BodySegments, MeasurementOutput, SideViewSegments
from app.services.muac_service import MUACResult, MUACService
from main import app


MANUAL_HEIGHT = 80.0
IMAGE_HEIGHT = 95.0


def _service_with_spies(monkeypatch):
    service = AssessmentService.__new__(AssessmentService)
    segments = BodySegments(
        total_height_px=800.0,
        shoulder_width_px=160.0,
        hip_width_px=140.0,
        upper_arm_length_px=120.0,
        torso_length_px=240.0,
    )
    service.measurement_svc = Mock()
    service.measurement_svc.process_image_with_estimation.return_value = MeasurementOutput(
        predicted_height_cm=IMAGE_HEIGHT,
        body_segments=segments,
        confidence_score=0.9,
        estimation_method="who_statistical",
    )
    service.measurement_svc.process_side_image.return_value = SideViewSegments(
        total_height_px=800.0,
        chest_depth_px=60.0,
        abd_depth_px=55.0,
        chest_confidence=0.9,
        abd_confidence=0.9,
    )

    service.ml_svc = Mock()
    service.ml_svc.predict.return_value = SimpleNamespace(
        estimated_weight_kg=10.0,
        sam_probability=0.1,
        mam_probability=0.1,
        normal_probability=0.8,
        risk_probability=0.0,
        overweight_probability=0.0,
        wasting_status="Normal",
        wasting_method="ml_classifier",
    )
    service.who_data = Mock()
    service.who_data.get_median_weight_for_height.return_value = 10.0
    service.nutrition_svc = Mock()
    service.nutrition_svc.compute_haz.return_value = -1.0
    service.nutrition_svc.compute_whz.return_value = -0.5
    service.nutrition_svc.classify_haz.return_value = "Normal"
    service.nutrition_svc.classify_whz.return_value = "Normal"

    muac_calls = []

    def fake_muac(**kwargs):
        muac_calls.append(kwargs)
        return MUACResult(12.5, "Normal", "estimated_from_landmarks", True)

    monkeypatch.setattr(MUACService, "estimate", staticmethod(fake_muac))
    return service, muac_calls


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def _assess(service, db_session):
    return service.assess(
        db=db_session,
        image_path="front.jpg",
        child_name="Manual wins",
        dob=date(2023, 7, 1),
        sex="F",
        height_cm=MANUAL_HEIGHT,
        side_image=b"side",
    )


def _assert_manual_height_reaches_all_consumers(service, muac_calls):
    service.measurement_svc.process_side_image.assert_called_once_with(
        b"side", MANUAL_HEIGHT
    )
    assert service.ml_svc.predict.call_args.args[3] == MANUAL_HEIGHT
    assert all(
        call.args[1] == MANUAL_HEIGHT
        for call in service.who_data.get_median_weight_for_height.call_args_list
    )
    assert service.nutrition_svc.compute_haz.call_args.args[2] == MANUAL_HEIGHT
    assert service.nutrition_svc.compute_whz.call_args.args[2] == MANUAL_HEIGHT
    assert muac_calls[0]["height_cm"] == MANUAL_HEIGHT
    # Landmark scaling also uses 80 / 800, not the disagreeing image height.
    assert muac_calls[0]["upper_arm_length_cm"] == pytest.approx(12.0)


def test_manual_height_controls_every_downstream_calculation(
    monkeypatch, db_session
):
    service, muac_calls = _service_with_spies(monkeypatch)
    result = _assess(service, db_session)

    assert result.measurement.effective_height_cm == MANUAL_HEIGHT
    assert result.measurement.height_method == "manual"
    assert result.measurement.predicted_height_cm == IMAGE_HEIGHT
    assert "80.0 cm (manual input)" in result.summary
    _assert_manual_height_reaches_all_consumers(service, muac_calls)


def test_api_returns_effective_manual_height_and_uses_it_downstream(
    monkeypatch, db_session
):
    service, muac_calls = _service_with_spies(monkeypatch)
    app.dependency_overrides[get_assessment_service] = lambda: service
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        response = TestClient(app).post(
            "/api/v1/assess",
            data={
                "child_name": "API manual wins",
                "date_of_birth": "2023-07-01",
                "sex": "F",
                "height_cm": str(MANUAL_HEIGHT),
            },
            files={"image": ("front.jpg", b"image", "image/jpeg"),
                   "image_side": ("side.jpg", b"side", "image/jpeg")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    measurement = response.json()["measurement"]
    assert measurement["effective_height_cm"] == MANUAL_HEIGHT
    assert measurement["height_method"] == "manual"
    assert measurement["predicted_height_cm"] == IMAGE_HEIGHT
    _assert_manual_height_reaches_all_consumers(service, muac_calls)
