"""Tests for clinical age calculation and request validation."""
from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas.assessment import AssessmentRequest
from app.services.age_service import calculate_age, validate_clinical_age


def test_future_date_of_birth_is_rejected():
    with pytest.raises(ValueError, match="cannot be in the future"):
        validate_clinical_age(date(2026, 7, 29), date(2026, 7, 28))


def test_date_of_birth_today_is_age_zero():
    age = validate_clinical_age(date(2026, 7, 28), date(2026, 7, 28))
    assert age.days == age.months == age.completed_months == age.completed_years == 0


def test_leap_day_birthday_uses_calendar_years():
    assert calculate_age(date(2020, 2, 29), date(2023, 2, 28)).completed_years == 3


def test_completed_years_just_before_and_on_birthday():
    dob = date(2020, 7, 28)
    assert calculate_age(dob, date(2024, 7, 27)).completed_years == 3
    assert calculate_age(dob, date(2024, 7, 28)).completed_years == 4


@pytest.mark.parametrize("months", [0, 24, 59])
def test_supported_month_boundaries(months):
    dob = date(2021, 1, 15)
    as_of = date(dob.year + months // 12, dob.month + months % 12, dob.day)
    assert validate_clinical_age(dob, as_of).completed_months == months


def test_60_month_boundary_is_rejected_without_clamping():
    with pytest.raises(ValueError, match="0 through 59 months"):
        validate_clinical_age(date(2021, 1, 15), date(2026, 1, 15))


def test_assessment_schema_rejects_future_date():
    with pytest.raises(ValidationError, match="cannot be in the future"):
        AssessmentRequest(child_name="Child", date_of_birth=date(2999, 1, 1), sex="F")
