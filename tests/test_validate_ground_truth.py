"""Tests for scripts/validate_ground_truth.py."""
from scripts.validate_ground_truth import (
    ALL_COLS, check_header, load_csv, read_header, validate_rows,
)


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


# ---------------------------------------------------------------------------
# I-1: a header typo (e.g. 'muac' for 'muac_cm') must hard-fail, not read as
# "blank on every row" (a warning only). Without this check, a child with a
# SAM-level MUAC and a normal WHZ silently loses the MUAC arm of the WHO
# OR-rule and flips from SAM to Normal with zero errors printed.
# ---------------------------------------------------------------------------

def test_check_header_flags_missing_and_unknown_columns():
    header = ["child_id", "sex", "date_of_birth", "measurement_date",
              "actual_height_cm", "actual_weight_kg", "muac", "oedema", "notes"]
    errors = check_header(header)
    assert any("muac_cm" in e and "missing" in e for e in errors)
    assert any("'muac'" in e for e in errors)


def test_check_header_suggests_likely_intended_column():
    header = ["child_id", "sex", "date_of_birth", "measurement_date",
              "actual_height_cm", "actual_weight_kg", "muac", "oedema", "notes"]
    errors = check_header(header)
    assert any("did you mean 'muac_cm'" in e for e in errors)


def test_check_header_passes_for_exact_all_cols():
    assert check_header(ALL_COLS) == []


def test_check_header_flags_duplicate_column():
    """Probe from review: check_header compares column SETS, so a duplicate
    column (e.g. 'muac_cm' twice, from an Excel copy-column) is invisible to
    a set difference. csv.DictReader keeps the LAST duplicate's value, so a
    blank second 'muac_cm' silently blanks MUAC on every row - the same
    silent-data-loss failure mode the header check exists to catch."""
    header = ALL_COLS + ["muac_cm"]
    errors = check_header(header)
    assert any("muac_cm" in e and "duplicate" in e.lower() for e in errors)


def test_validate_rows_without_fieldnames_skips_header_check():
    """Backward compatible: omitting `fieldnames` (existing callers/tests
    that only ever had rows) must not suddenly start hard-failing."""
    errors, _ = validate_rows([_good_row()])
    assert errors == []


def test_muac_header_typo_silently_flips_sam_to_normal_without_header_check(tmp_path):
    """Reproduces the exact probe: header says 'muac' instead of 'muac_cm'
    for a child with MUAC 10.9 cm (SAM) and a normal WHZ. Row-level checks
    alone (no fieldnames passed) must NOT catch this - that is exactly the
    bug. Only wiring the header into validate_rows catches it."""
    bad_csv = tmp_path / "ground_truth.csv"
    bad_csv.write_text(
        "child_id,sex,date_of_birth,measurement_date,"
        "actual_height_cm,actual_weight_kg,muac,oedema,notes\n"
        "001,M,2023-04-12,2026-07-15,90.0,12.0,10.9,no,\n"
    )
    rows = load_csv(bad_csv)
    fieldnames = read_header(bad_csv)

    errors_without_header, _ = validate_rows(rows)
    assert errors_without_header == [], (
        "sanity check: row-level validation alone must not catch this typo"
    )

    errors, _ = validate_rows(rows, fieldnames=fieldnames)
    assert errors != [], "the header-shape check must now hard-fail this CSV"
    assert any("muac_cm" in e for e in errors)


def test_duplicate_header_column_rejected_end_to_end(tmp_path):
    """A CSV with 'muac_cm' appearing twice (Excel copy-column) must hard-fail
    through validate_rows(rows, fieldnames=...), not just check_header in
    isolation. csv.DictReader silently keeps the last duplicate's value, so
    a blank second occurrence would otherwise blank MUAC on every row with
    zero errors printed."""
    dup_csv = tmp_path / "ground_truth.csv"
    dup_csv.write_text(
        "child_id,sex,date_of_birth,measurement_date,"
        "actual_height_cm,actual_weight_kg,muac_cm,oedema,notes,muac_cm\n"
        "001,M,2023-04-12,2026-07-15,90.0,12.0,10.9,no,,\n"
    )
    rows = load_csv(dup_csv)
    fieldnames = read_header(dup_csv)

    errors, _ = validate_rows(rows, fieldnames=fieldnames)
    assert errors != [], "duplicate header column must hard-fail"
    assert any("muac_cm" in e and "duplicate" in e.lower() for e in errors)
