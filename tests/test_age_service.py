"""Tests for clinical age calculation and request validation."""
from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas.assessment import AssessmentRequest
from app.services.age_service import AgeService


age_service = AgeService()


def test_future_date_of_birth_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be in the future"):
        age_service.validate_clinical_age(date(2026, 7, 29), date(2026, 7, 28))


def test_date_of_birth_today_is_age_zero() -> None:
    age = age_service.validate_clinical_age(date(2026, 7, 28), date(2026, 7, 28))
    assert age.days == age.months == age.completed_months == age.completed_years == 0


def test_leap_day_birthday_uses_calendar_years() -> None:
    age = age_service.calculate_age(date(2020, 2, 29), date(2023, 2, 28))
    assert age.completed_years == 3


def test_completed_years_just_before_and_on_birthday() -> None:
    dob = date(2020, 7, 28)
    assert age_service.calculate_age(dob, date(2024, 7, 27)).completed_years == 3
    assert age_service.calculate_age(dob, date(2024, 7, 28)).completed_years == 4


@pytest.mark.parametrize("months", [0, 24, 59])
def test_supported_month_boundaries(months: int) -> None:
    dob = date(2021, 1, 15)
    as_of = date(dob.year + months // 12, dob.month + months % 12, dob.day)
    assert age_service.validate_clinical_age(dob, as_of).completed_months == months


def test_60_month_boundary_is_rejected_without_clamping() -> None:
    with pytest.raises(ValueError, match="0 through 59 months"):
        age_service.validate_clinical_age(date(2021, 1, 15), date(2026, 1, 15))


def test_assessment_schema_rejects_future_date() -> None:
    with pytest.raises(ValidationError, match="cannot be in the future"):
        AssessmentRequest(child_name="Child", date_of_birth=date(2999, 1, 1), sex="F")


def test_schema_uses_request_scoped_as_of_date() -> None:
    dob = date(2021, 7, 29)
    request = AssessmentRequest.model_validate(
        {"child_name": "Child", "date_of_birth": dob, "sex": "F"},
        context={"age_service": age_service, "as_of": date(2026, 7, 28)},
    )
    assert request.date_of_birth == dob

    with pytest.raises(ValidationError, match="0 through 59 months"):
        AssessmentRequest.model_validate(
            {"child_name": "Child", "date_of_birth": dob, "sex": "F"},
            context={"age_service": age_service, "as_of": date(2026, 7, 29)},
        )
