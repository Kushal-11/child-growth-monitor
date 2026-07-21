"""Tests for scripts/validate_ground_truth.py."""
from scripts.validate_ground_truth import validate_rows


def _good_row(**over) -> dict:
    row = {
        "child_id": "001", "sex": "M",
        "date_of_birth": "2023-04-12", "measurement_date": "2026-07-15",
        "actual_height_cm": "82.5", "actual_weight_kg": "10.4",
        "muac_cm": "13.2", "oedema": "no", "notes": "",
    }
    row.update(over)
    return row


def test_valid_row_passes():
    errors, warnings = validate_rows([_good_row()])
    assert errors == []
    assert warnings == []


def test_height_out_of_range_rejected():
    errors, _ = validate_rows([_good_row(actual_height_cm="8.5")])
    assert len(errors) == 1 and "height" in errors[0]


def test_weight_out_of_range_rejected():
    errors, _ = validate_rows([_good_row(actual_weight_kg="110")])
    assert len(errors) == 1 and "weight" in errors[0]


def test_muac_out_of_range_rejected():
    errors, _ = validate_rows([_good_row(muac_cm="25.0")])
    assert len(errors) == 1 and "muac" in errors[0].lower()


def test_measurement_before_birth_rejected():
    errors, _ = validate_rows([_good_row(measurement_date="2022-01-01")])
    assert any("before date_of_birth" in e for e in errors)


def test_future_measurement_rejected():
    errors, _ = validate_rows([_good_row(measurement_date="2099-01-01")])
    assert any("future" in e for e in errors)


def test_age_over_60_months_rejected():
    errors, _ = validate_rows([_good_row(date_of_birth="2018-01-01")])
    assert any("age" in e for e in errors)


def test_bad_sex_rejected():
    errors, _ = validate_rows([_good_row(sex="X")])
    assert any("sex" in e for e in errors)


def test_bad_oedema_rejected():
    errors, _ = validate_rows([_good_row(oedema="maybe")])
    assert any("oedema" in e for e in errors)


def test_duplicate_child_id_rejected():
    errors, _ = validate_rows([_good_row(), _good_row()])
    assert any("duplicate" in e for e in errors)


def test_missing_required_field_rejected():
    errors, _ = validate_rows([_good_row(date_of_birth="")])
    assert any("date_of_birth" in e for e in errors)


def test_missing_optional_measurements_warn_not_error():
    errors, warnings = validate_rows(
        [_good_row(actual_height_cm="", actual_weight_kg="", muac_cm="", oedema="")]
    )
    assert errors == []
    assert len(warnings) == 3  # height, weight, muac missing
