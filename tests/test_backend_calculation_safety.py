"""Focused direct tests for authoritative calculation/provenance rules."""
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401
from app.models.database import Base
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from app.services.age_service import age_months_at, completed_months
from app.services.assessment_service import AssessmentService
from app.services.measurement_service import MeasurementOutput


def _session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


class _Who:
    @staticmethod
    def get_median_weight_for_height(*_args, **_kwargs):
        return 12.0


class _Nutrition:
    def __init__(self):
        self.haz_ages = []

    def compute_haz(self, _sex, age_months, _height):
        self.haz_ages.append(age_months)
        return -1.0

    @staticmethod
    def classify_haz(_z):
        return "Normal"

    @staticmethod
    def compute_whz(*_args):
        return -2.5

    @staticmethod
    def classify_whz(_z):
        return "Moderate Acute Malnutrition (MAM)"


class _NoImageAllowed:
    @staticmethod
    def process_image_with_estimation(**_kwargs):
        raise AssertionError("manual-only assessment loaded the pose runtime")


class _StatisticalImage:
    @staticmethod
    def process_image_with_estimation(**_kwargs):
        return MeasurementOutput(
            predicted_height_cm=90.0,
            estimation_method="who_statistical",
            confidence_score=0.8,
        )


def test_calendar_age_uses_month_end_anniversaries():
    assert completed_months(date(2024, 1, 31), date(2024, 2, 28)) == 0
    assert completed_months(date(2024, 1, 31), date(2024, 2, 29)) == 1
    assert completed_months(date(2023, 1, 31), date(2023, 2, 28)) == 1
    assert age_months_at(date(2023, 1, 31), date(2023, 2, 28)) == 1.0


def test_manual_only_assessment_bypasses_pose_and_uses_visit_date():
    db = _session()
    service = AssessmentService(_Who())
    service.measurement_svc = _NoImageAllowed()
    nutrition = _Nutrition()
    service.nutrition_svc = nutrition
    try:
        result = service.assess(
            db=db,
            image_path=None,
            child_name="Manual Child",
            dob=date(2024, 1, 31),
            assessment_date=date(2026, 2, 28),
            sex="M",
            height_cm=100.0,
            weight_kg=12.0,
            muac_cm=11.4,
        )
        visit = db.query(Visit).one()
        stored = db.query(MeasurementResult).one()
    finally:
        db.close()

    assert result.poshan.final_status == "SAM"
    assert result.measurement.height_source == "manual"
    assert result.measurement.weight_source == "manual"
    assert result.muac.muac_cm == 11.4
    assert visit.image_path is None
    assert visit.visit_date.date() == date(2026, 2, 28)
    assert visit.age_months == age_months_at(
        date(2024, 1, 31), date(2026, 2, 28)
    )
    assert nutrition.haz_ages == [25]
    assert stored.haz_zscore == -1.0
    assert stored.whz_zscore == -2.5


def test_statistical_height_and_weight_do_not_create_who_normal_or_muac():
    db = _session()
    service = AssessmentService(_Who())
    service.measurement_svc = _StatisticalImage()

    class NoClinicalNutrition:
        def __getattr__(self, name):
            raise AssertionError(f"statistical values called {name}")

    service.nutrition_svc = NoClinicalNutrition()
    try:
        result = service.assess(
            db=db,
            image_path="screening.jpg",
            child_name="Screening Child",
            dob=date(2024, 1, 1),
            assessment_date=date(2026, 1, 1),
            sex="F",
        )
        stored = db.query(MeasurementResult).one()
    finally:
        db.close()

    assert result.measurement.height_source == "who_statistical"
    assert result.measurement.weight_source == "who_statistical"
    assert result.nutrition.haz_zscore is None
    assert result.nutrition.whz_zscore is None
    assert result.nutrition.haz_status is None
    assert result.nutrition.whz_status is None
    assert result.muac.muac_cm is None
    assert result.muac.muac_method == "unavailable"
    assert result.poshan.final_status == "Indeterminate"
    assert stored.haz_zscore is None
    assert stored.whz_zscore is None
    assert stored.muac_cm is None
