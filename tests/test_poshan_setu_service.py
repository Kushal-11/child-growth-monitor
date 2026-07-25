"""Canonical Poshan Setu v1 and assessment-provenance tests."""
from datetime import date, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - register all SQLAlchemy models
from app.models.database import Base
from app.models.measurement import MeasurementResult
from app.services.assessment_service import AssessmentService
from app.services.measurement_service import (
    BodySegments,
    MeasurementOutput,
    SideViewSegments,
)
from app.services.poshan_setu_service import classify_poshan_setu


CASES_PATH = (
    Path(__file__).resolve().parents[1] / "shared" / "poshan_setu_v1_cases.json"
)
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_boundary_cases(case):
    has_bmi_values = (
        case["weight_kg"] is not None and case["height_cm"] is not None
    )
    result = classify_poshan_setu(
        sex=case["sex"],
        age_months=case["age_months"],
        weight_kg=case["weight_kg"],
        height_cm=case["height_cm"],
        weight_source="manual" if has_bmi_values else "unavailable",
        height_source="manual" if has_bmi_values else "unavailable",
        muac_cm=case["muac_cm"],
        muac_method="manual" if case["muac_cm"] is not None else "unavailable",
    )

    assert result.bmi_status == case["expected_bmi_status"]
    assert result.muac_status == case["expected_muac_status"]
    assert result.final_status == case["expected_final_status"]
    assert result.classification_method == "poshan_setu_v1"
    assert result.final_status in result.rationale


def test_estimated_values_cannot_certify_normal():
    result = classify_poshan_setu(
        sex="F",
        age_months=36.0,
        weight_kg=14.0,
        height_cm=100.0,
        weight_source="ml_estimated",
        height_source="who_statistical",
        muac_cm=14.0,
        muac_method="whz_derived",
    )

    assert result.bmi is None
    assert result.bmi_status == "Indeterminate"
    assert result.muac_status == "Indeterminate"
    assert result.final_status == "Indeterminate"
    assert result.triggered_by == ()
    assert result.complete is False
    assert "ml_estimated" in result.rationale
    assert "whz_derived" in result.rationale


def test_known_mam_is_retained_when_other_component_missing():
    result = classify_poshan_setu(
        sex="M",
        age_months=36.0,
        weight_kg=13.2,
        height_cm=100.0,
        weight_source="manual",
        height_source="manual",
        muac_cm=None,
        muac_method="unavailable",
    )

    assert result.bmi_status == "MAM"
    assert result.final_status == "Indeterminate"
    assert result.triggered_by == ("bmi",)
    assert "both components are required" in result.rationale


def test_complete_normal_is_jointly_triggered_and_component_conflict_uses_maximum():
    normal = classify_poshan_setu(
        sex="M",
        age_months=36.0,
        weight_kg=13.7,
        height_cm=100.0,
        weight_source="manual",
        height_source="reference_object",
        muac_cm=12.5,
        muac_method="tape",
    )
    conflict = classify_poshan_setu(
        sex="F",
        age_months=36.0,
        weight_kg=13.5,
        height_cm=100.0,
        weight_source="manual",
        height_source="manual",
        muac_cm=12.0,
        muac_method="manual",
    )

    assert normal.final_status == "Normal"
    assert normal.triggered_by == ("bmi", "muac")
    assert normal.complete is True
    assert "severity maximum used" in normal.rationale
    assert conflict.bmi_status == "Normal"
    assert conflict.muac_status == "MAM"
    assert conflict.final_status == "MAM"
    assert conflict.triggered_by == ("muac",)


def test_sam_survives_missing_component_but_is_marked_incomplete():
    result = classify_poshan_setu(
        sex="F",
        age_months=3.0,
        weight_kg=12.7,
        height_cm=100.0,
        weight_source="manual",
        height_source="manual",
        muac_cm=10.0,
        muac_method="manual",
    )

    assert result.bmi_status == "SAM"
    assert result.muac_status == "Indeterminate"
    assert result.final_status == "SAM"
    assert result.triggered_by == ("bmi",)
    assert result.complete is False
    assert "outside age 6 to <60 months" in result.rationale


class _FakeMeasurementService:
    def process_image_with_estimation(self, **_kwargs):
        return MeasurementOutput(
            predicted_height_cm=80.0,
            reference_object_detected=True,
            scale_factor=0.1,
            confidence_score=0.9,
            annotated_image_filename="annotated.jpg",
            body_segments=BodySegments(
                total_height_px=100.0,
                upper_arm_length_px=15.0,
                shoulder_width_px=20.0,
            ),
            estimation_method="reference_object",
            body_build={"body_build": "average", "weight_adjustment": 1.0},
        )

    def process_side_image(self, _image, _height):
        return SideViewSegments(
            chest_depth_px=10.0,
            abd_depth_px=9.0,
            total_height_px=100.0,
            chest_confidence=0.9,
            abd_confidence=0.9,
        )


class _FakeNutritionService:
    def compute_haz(self, *_args):
        return -1.0

    def classify_haz(self, _value):
        return "Normal"

    def compute_whz(self, *_args):
        return -0.5

    def classify_whz(self, _value):
        return "Normal"


class _FakeMLService:
    def predict(self, *_args):
        return SimpleNamespace(
            estimated_weight_kg=13.6,
            sam_probability=0.01,
            mam_probability=0.02,
            normal_probability=0.95,
            risk_probability=0.01,
            overweight_probability=0.01,
            wasting_status="Normal",
            wasting_method="tflite_classifier",
            model_version="poshan-2026-07",
            training_data="synthetic_v1",
            non_clinical=True,
        )


class _FakeWHOData:
    def get_median_weight_for_height(self, *_args, **_kwargs):
        return 13.7


def _assessment_service():
    service = object.__new__(AssessmentService)
    service.measurement_svc = _FakeMeasurementService()
    service.nutrition_svc = _FakeNutritionService()
    service.ml_svc = _FakeMLService()
    service.who_data = _FakeWHOData()
    return service


def test_assessment_manual_height_priority_and_persists_full_provenance():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    dob = date.today() - timedelta(days=round(36 * 30.4375))

    try:
        result = _assessment_service().assess(
            db=db,
            image_path="front.jpg",
            child_name="Parity Child",
            dob=dob,
            sex="M",
            weight_kg=13.7,
            height_cm=100.0,
            muac_cm=12.5,
            side_image=b"side",
        )
        stored = db.query(MeasurementResult).one()
    finally:
        db.close()

    assert result.measurement.predicted_height_cm == 80.0
    assert result.measurement.effective_height_cm == 100.0
    assert result.measurement.height_source == "manual"
    assert result.measurement.effective_weight_kg == 13.7
    assert result.measurement.weight_source == "manual"
    assert result.poshan.final_status == "Normal"
    assert result.poshan.triggered_by == ["bmi", "muac"]
    assert result.poshan.classification_method == "poshan_setu_v1"
    assert result.ml_prediction.model_version == "poshan-2026-07"
    assert result.ml_prediction.non_clinical is True

    assert stored.predicted_height_cm == 80.0
    assert stored.effective_height_cm == 100.0
    assert stored.height_source == "manual"
    assert stored.effective_weight_kg == 13.7
    assert stored.weight_source == "manual"
    assert stored.bmi == pytest.approx(13.7)
    assert stored.bmi_status == "Normal"
    assert stored.muac_status == "Normal"
    assert stored.poshan_status == "Normal"
    assert stored.poshan_triggered_by == ["bmi", "muac"]
    assert stored.classification_method == "poshan_setu_v1"
    assert "final Normal" in stored.classification_rationale
    assert stored.body_build == "average"
    assert stored.side_view_used is True
    assert stored.chest_depth_cm == 10.0
    assert stored.abd_depth_cm == 9.0
    assert stored.ml_estimated_weight_kg == 13.6
    assert stored.ml_wasting_status == "Normal"
    assert stored.ml_model_version == "poshan-2026-07"
    assert stored.ml_training_data == "synthetic_v1"
    assert stored.ml_non_clinical is True
    assert stored.sam_probability == 0.01
    assert stored.mam_probability == 0.02
    assert stored.normal_probability == 0.95
    assert stored.risk_probability == 0.01
    assert stored.overweight_probability == 0.01
    assert stored.muac_cm == 12.5
    assert stored.muac_method == "manual"


def test_assessment_keeps_unvalidated_ml_weight_as_screening_only():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    service = _assessment_service()
    service.who_data = SimpleNamespace(
        get_median_weight_for_height=lambda *_args, **_kwargs: None
    )
    dob = date.today() - timedelta(days=round(36 * 30.4375))

    try:
        result = service.assess(
            db=db,
            image_path="front.jpg",
            child_name="Screening Child",
            dob=dob,
            sex="M",
            height_cm=100.0,
            muac_cm=12.5,
        )
        stored = db.query(MeasurementResult).one()
    finally:
        db.close()

    assert result.ml_prediction.estimated_weight_kg == 13.6
    assert result.measurement.effective_weight_kg is None
    assert result.measurement.weight_source == "unavailable"
    assert result.poshan.bmi_status == "Indeterminate"
    assert result.poshan.muac_status == "Normal"
    assert result.poshan.final_status == "Indeterminate"
    assert stored.ml_estimated_weight_kg == 13.6
    assert stored.effective_weight_kg is None
    assert stored.poshan_status == "Indeterminate"


def test_assessment_classifies_unrounded_manual_muac_at_exact_boundary():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    dob = date.today() - timedelta(days=round(36 * 30.4375))

    try:
        result = _assessment_service().assess(
            db=db,
            image_path="front.jpg",
            child_name="MUAC Boundary Child",
            dob=dob,
            sex="M",
            weight_kg=14.0,
            height_cm=100.0,
            muac_cm=11.499,
        )
        stored = db.query(MeasurementResult).one()
    finally:
        db.close()

    assert result.muac.muac_cm == 11.499
    assert result.poshan.muac_status == "SAM"
    assert result.poshan.final_status == "SAM"
    assert stored.muac_cm == 11.499
    assert stored.muac_status == "SAM"
