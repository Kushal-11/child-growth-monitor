"""Tests for the pure helpers in scripts/batch_assess.py."""
from datetime import date

from scripts.batch_assess import (
    _collapse_wasting,
    _combine_status,
    _compute_age_months,
    _muac_status,
)


def test_age_uses_measurement_date_not_today():
    dob = date(2024, 1, 1)
    at = date(2026, 1, 1)
    assert abs(_compute_age_months(dob, at) - 24.0) < 0.2


def test_age_defaults_to_today_when_no_measurement_date():
    # Backward compatible: at=None still works (uses today's date)
    assert _compute_age_months(date(2024, 1, 1), None) > 24.0


def test_muac_thresholds_are_who_fixed():
    assert _muac_status(11.4) == "SAM"
    assert _muac_status(11.5) == "MAM"
    assert _muac_status(12.4) == "MAM"
    assert _muac_status(12.5) == "Normal"
    assert _muac_status(None) is None


def test_collapse_wasting():
    assert _collapse_wasting("SAM") == "SAM"
    assert _collapse_wasting("MAM") == "MAM"
    assert _collapse_wasting("Normal") == "Normal"
    assert _collapse_wasting("Risk_Overweight") == "Normal"
    assert _collapse_wasting("Overweight") == "Normal"
    assert _collapse_wasting(None) is None


def test_combine_status_or_rule():
    # Worst of the two arms wins
    assert _combine_status("Normal", "SAM", False) == "SAM"
    assert _combine_status("SAM", "Normal", False) == "SAM"
    assert _combine_status("MAM", "Normal", False) == "MAM"
    assert _combine_status("Normal", "Normal", False) == "Normal"
    # Oedema is an independent SAM trigger, regardless of the other arms
    assert _combine_status("Normal", "Normal", True) == "SAM"
    assert _combine_status(None, None, True) == "SAM"
    # Nothing known -> None, never a fabricated 'Normal'
    assert _combine_status(None, None, False) is None
    # One arm known
    assert _combine_status(None, "MAM", False) == "MAM"
