"""Tests for the pure helpers in scripts/batch_assess.py."""
import csv
import io
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.batch_assess import (
    TEMPLATE_CSV,
    _collapse_wasting,
    _combine_status,
    _compute_age_months,
    _load_master_ground_truth,
    _muac_status,
    _parse_dob,
    _process_child_image,
    _run_per_child,
)


class _FakeMeasurement:
    """Stand-in for the MeasurementService result object."""

    def __init__(self, predicted_height_cm=None, body_segments=None, confidence_score=None):
        self.predicted_height_cm = predicted_height_cm
        self.body_segments = body_segments
        self.confidence_score = confidence_score
        self.estimation_method = "fake"
        self.annotated_image_filename = None


class _FakeMLPrediction:
    """Stand-in for the MLService prediction result object."""

    def __init__(self, wasting_status, estimated_weight_kg=12.0):
        self.wasting_status = wasting_status
        self.estimated_weight_kg = estimated_weight_kg
        self.sam_probability = 0.9
        self.mam_probability = 0.05


class _FakeMeasService:
    """Stand-in for MeasurementService — no MediaPipe/pose model involved."""

    def __init__(self, predicted_height_cm=None, body_segments=None):
        self._predicted_height_cm = predicted_height_cm
        self._body_segments = body_segments

    def process_image_with_estimation(self, image_path, age_months, sex, who_data):
        return _FakeMeasurement(self._predicted_height_cm, self._body_segments)

    def process_side_image(self, *args, **kwargs):
        return None


class _FakeMLService:
    """Stand-in for MLService — no TensorFlow model involved."""

    is_available = False

    def __init__(self, wasting_status=None):
        self._wasting_status = wasting_status

    def predict(self, *args, **kwargs):
        if self._wasting_status is None:
            return None
        return _FakeMLPrediction(self._wasting_status)

    def extract_features(self, *args, **kwargs):
        return None


class _FakeNutritionService:
    """Stand-in for NutritionService — no WHO LMS table lookups involved."""

    def compute_haz(self, *args, **kwargs):
        return None

    def compute_whz(self, *args, **kwargs):
        return None


class _FakeWHOData:
    pass


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


def test_unparseable_dob_sets_error_and_suppresses_verdict():
    """An unparseable date_of_birth must not silently fabricate an age and
    still emit a usable app verdict — the row must be marked as an error
    instead (no silent failures rule)."""
    row = _process_child_image(
        fname="child1.jpg",
        img_path=Path("child1.jpg"),
        gt={"date_of_birth": "not-a-date", "sex": "M"},
        meas_svc=_FakeMeasService(predicted_height_cm=85.0, body_segments={"stub": True}),
        ml_svc=_FakeMLService(wasting_status="SAM"),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    assert row["error"], "an unparseable date_of_birth must populate the error field"
    assert "date_of_birth" in row["error"]
    assert row["pred_status_final"] is None, (
        "pred_status_final must never be derived from a fabricated age"
    )


def test_valid_dob_with_missing_height_weight_processes_normally():
    """Regression guard: a child with a valid DOB but no height/weight must
    still process without an error (only unparseable DOB should error).
    measurement_date is pinned explicitly (rather than left to the
    today's-date fallback) so this assertion is stable regardless of which
    real day the suite runs on - the fixed dob would otherwise cross the
    60-month ceiling in 2028 and start failing for an unrelated reason."""
    row = _process_child_image(
        fname="child_ok.jpg",
        img_path=Path("child_ok.jpg"),
        gt={
            "date_of_birth": "2023-01-01",
            "measurement_date": "2024-06-01",
            "sex": "M",
        },
        meas_svc=_FakeMeasService(),
        ml_svc=_FakeMLService(),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    assert row["error"] == ""
    assert row["age_months"] is not None


def test_invalid_measurement_date_warns_but_falls_back_to_today(capsys):
    """A typo'd measurement_date (e.g. '2026-13-45') must not be silently
    indistinguishable from 'not supplied' — it should print a clear warning
    naming the child and the bad value, but still fall back to today's date
    rather than failing the row. date_of_birth is pinned relative to
    today (an offset, not a fixed calendar date) so the fallback-to-today
    age this test exercises stays ~24 months regardless of which real day
    it runs on - a fixed calendar dob would eventually cross the 60-month
    ceiling and fail for a reason unrelated to what this test checks."""
    dob = date.today() - timedelta(days=730)
    row = _process_child_image(
        fname="child2.jpg",
        img_path=Path("child2.jpg"),
        gt={
            "date_of_birth": dob.isoformat(),
            "measurement_date": "2026-13-45",
            "sex": "F",
            "child_name": "Child Two",
        },
        meas_svc=_FakeMeasService(),
        ml_svc=_FakeMLService(),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    captured = capsys.readouterr()
    assert "Child Two" in captured.out
    assert "2026-13-45" in captured.out
    assert row["error"] == ""


def test_negative_age_from_swapped_dates_sets_error_and_suppresses_verdict():
    """A parseable date_of_birth combined with a measurement_date before it
    (e.g. the two columns got swapped) yields a negative age. WHO growth
    standards are only defined for 0-60 months, so this must be treated the
    same as an unparseable DOB: error set, pred_status_final forced None."""
    row = _process_child_image(
        fname="child_swapped.jpg",
        img_path=Path("child_swapped.jpg"),
        gt={
            "date_of_birth": "2024-01-01",
            "measurement_date": "2023-01-01",
            "sex": "M",
        },
        meas_svc=_FakeMeasService(predicted_height_cm=85.0, body_segments={"stub": True}),
        ml_svc=_FakeMLService(wasting_status="SAM"),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    assert abs(row["age_months"] - (-12.0)) < 0.5
    assert row["error"], "a negative computed age must populate the error field"
    assert "-12" in row["error"] or "-12.0" in row["error"]
    assert row["pred_status_final"] is None


def test_future_dob_sets_error_and_suppresses_verdict():
    """A date_of_birth far in the future (relative to the measurement date,
    or to today when measurement_date is blank) produces a negative age and
    must be rejected the same way, rather than emitting a clean verdict."""
    row = _process_child_image(
        fname="child_future.jpg",
        img_path=Path("child_future.jpg"),
        gt={
            "date_of_birth": "2099-01-01",
            "sex": "M",
        },
        meas_svc=_FakeMeasService(predicted_height_cm=85.0, body_segments={"stub": True}),
        ml_svc=_FakeMLService(wasting_status="SAM"),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    assert row["age_months"] < 0
    assert row["error"], "a future date_of_birth must populate the error field"
    assert row["pred_status_final"] is None


def test_age_over_60_months_sets_error_and_suppresses_verdict():
    """WHO growth standards are only defined for 0-60 months; an age far
    beyond that (e.g. a one-digit year typo in measurement_date) must be
    rejected rather than driving a clean-looking verdict."""
    row = _process_child_image(
        fname="child_ancient.jpg",
        img_path=Path("child_ancient.jpg"),
        gt={
            "date_of_birth": "2024-01-01",
            "measurement_date": "2074-01-01",
            "sex": "M",
        },
        meas_svc=_FakeMeasService(predicted_height_cm=85.0, body_segments={"stub": True}),
        ml_svc=_FakeMLService(wasting_status="SAM"),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    assert abs(row["age_months"] - 600.0) < 0.5
    assert row["error"], "an age over 60 months must populate the error field"
    assert "600" in row["error"]
    assert row["pred_status_final"] is None


def test_age_boundary_error_message_shows_precise_value_not_60_0():
    """A child measured 1827 days after birth (60.0246 months - just past
    the WHO 60-month ceiling) must still be rejected (the 0.0-60.0 inclusive
    boundary itself does not change), but the error message must show that
    precisely (e.g. '60.02'), not round to the self-contradictory
    '60.0 out of range 0-60', which reads like a bug to a field worker."""
    dob = date(2020, 1, 1)
    mdate = dob + timedelta(days=1827)
    row = _process_child_image(
        fname="child_boundary.jpg",
        img_path=Path("child_boundary.jpg"),
        gt={
            "date_of_birth": dob.isoformat(),
            "measurement_date": mdate.isoformat(),
            "sex": "M",
        },
        meas_svc=_FakeMeasService(predicted_height_cm=85.0, body_segments={"stub": True}),
        ml_svc=_FakeMLService(wasting_status="SAM"),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    expected_age = _compute_age_months(dob, mdate)
    assert expected_age > 60.0, "test fixture must actually be past the ceiling"
    assert row["error"], "an age just past 60 months must still be rejected"
    assert f"{expected_age:.2f}" in row["error"]
    assert "60.0 out of range" not in row["error"], (
        "rounding to '60.0' is self-contradictory next to 'out of range 0-60'"
    )
    assert row["pred_status_final"] is None


def test_no_date_of_birth_supplied_is_distinct_from_malformed():
    """The --images-only mode (no ground-truth CSV) leaves date_of_birth
    blank. That is a supported, working mode - not a malformed value - so
    the error message must say plainly that nothing was supplied, distinct
    from the 'unparseable' message used for a garbled value."""
    row = _process_child_image(
        fname="child_no_gt.jpg",
        img_path=Path("child_no_gt.jpg"),
        gt={},
        meas_svc=_FakeMeasService(predicted_height_cm=85.0, body_segments={"stub": True}),
        ml_svc=_FakeMLService(wasting_status="SAM"),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    assert row["error"], "a blank date_of_birth must still suppress the verdict"
    assert "no date_of_birth supplied" in row["error"]
    assert "unparseable" not in row["error"]
    assert row["pred_status_final"] is None


def test_ragged_master_csv_row_does_not_abort_batch():
    """A master ground-truth CSV row with more fields than its header makes
    csv.DictReader stash the overflow under a None key with a list value.
    That must not raise and kill the whole batch - only (at most) the
    affected child's row may be degraded; every other child must still be
    processed and appear in the results. measurement_date is pinned
    explicitly for child_ok so this assertion is stable regardless of which
    real day the suite runs on (a fixed dob + today's-date fallback would
    otherwise cross the 60-month ceiling in 2028)."""
    master_csv_text = (
        "child_id,sex,date_of_birth,measurement_date\n"
        "child_bad,M,2023-01-01,2024-01-01,unexpected,extra,columns\n"
        "child_ok,F,2023-06-01,2023-08-01\n"
    )
    master_gt: dict = {}
    for row in csv.DictReader(io.StringIO(master_csv_text)):
        cid = (row.get("child_id") or "").strip()
        if cid:
            master_gt[cid] = row

    entries = [
        ("child_bad", Path("child_bad_front.jpg"), None, {}),
        ("child_ok", Path("child_ok_front.jpg"), None, {}),
    ]

    results = _run_per_child(
        entries,
        meas_svc=_FakeMeasService(),
        ml_svc=_FakeMLService(),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        verbose=False,
        master_gt=master_gt,
    )

    assert len(results) == 2, "the ragged row must not abort processing of other children"
    child_ok_row = results[1]
    assert child_ok_row["error"] == ""


def test_ragged_master_row_marks_child_error_instead_of_silent_shift():
    """Even when the validate_ground_truth gate is bypassed (defense in
    depth), a ragged master row (one extra field shifts every column after
    it) must never be silently sanitized into a clean-looking verdict.
    Verified reproduction: 'child_x,M,2023-01-01,,2024-06-01,90,12,13,no,'
    has one extra empty field, so muac_cm picks up the weight column ('12'),
    producing actual_muac_status='MAM' / actual_combined_status='MAM' - a
    wrong nutritional status that must be flagged with a non-empty error,
    not presented as valid."""
    header = (
        "child_id,sex,date_of_birth,measurement_date,"
        "actual_height_cm,actual_weight_kg,muac_cm,oedema,notes"
    )
    ragged_row = "child_x,M,2023-01-01,,2024-06-01,90,12,13,no,"
    master_csv_text = header + "\n" + ragged_row + "\n"

    master_gt: dict = {}
    for row in csv.DictReader(io.StringIO(master_csv_text)):
        cid = (row.get("child_id") or "").strip()
        if cid:
            master_gt[cid] = row

    entries = [("child_x", Path("child_x_front.jpg"), None, {})]

    results = _run_per_child(
        entries,
        meas_svc=_FakeMeasService(),
        ml_svc=_FakeMLService(),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        verbose=False,
        master_gt=master_gt,
    )

    assert len(results) == 1
    row = results[0]
    # Reproduce the exact wrong-status bug from the shifted columns.
    assert row["actual_muac_status"] == "MAM"
    assert row["actual_combined_status"] == "MAM"
    # ... but it must now be flagged, never silently presented as valid.
    assert row["error"], (
        "a ragged master row must never be silently sanitized into a "
        "clean-looking row - the child's error field must be non-empty"
    )


def test_master_gt_validation_gate_blocks_bad_master_csv(tmp_path, capsys):
    """A master ground-truth CSV that scripts.validate_ground_truth.validate_rows
    flags as broken (here: the same ragged row above, which produces 4
    plausibility errors) must abort the whole batch with a non-zero exit
    before a single child is assessed - a ground-truth file with impossible
    values must never silently produce a study. The operator must be
    pointed at scripts/validate_ground_truth.py to fix the file."""
    bad_csv = tmp_path / "master_ground_truth.csv"
    bad_csv.write_text(
        "child_id,sex,date_of_birth,measurement_date,"
        "actual_height_cm,actual_weight_kg,muac_cm,oedema,notes\n"
        "child_x,M,2023-01-01,,2024-06-01,90,12,13,no,\n"
    )

    with pytest.raises(SystemExit) as exc_info:
        _load_master_ground_truth(bad_csv)

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    # Stream target updated for the stdout/stderr consistency minor fix:
    # fatal/error output now goes to stderr (matching main()'s convention),
    # not stdout. Same assertion, corrected stream - see final-review-fixes
    # report for why this one existing assertion needed updating.
    assert "validate_ground_truth.py" in captured.err


def test_master_gt_validation_gate_allows_valid_csv(tmp_path):
    """A well-formed master CSV (no plausibility errors) must pass the gate
    and build the child_id -> row lookup normally; optional-field warnings
    must not block."""
    good_csv = tmp_path / "master_ground_truth.csv"
    good_csv.write_text(
        "child_id,sex,date_of_birth,measurement_date,"
        "actual_height_cm,actual_weight_kg,muac_cm,oedema,notes\n"
        "child_a,M,2023-01-01,2024-06-01,85.0,11.2,14.5,no,\n"
        "child_b,F,2023-07-01,2024-06-01,,,13.2,,weight not available\n"
    )

    master_gt = _load_master_ground_truth(good_csv)

    assert set(master_gt.keys()) == {"child_a", "child_b"}
    assert master_gt["child_a"]["sex"] == "M"


def test_master_gt_validation_gate_blocks_header_typo(tmp_path, capsys):
    """I-1: a header typo ('muac' instead of 'muac_cm') must be caught by
    the real safety gate batch_assess.py runs before any child is assessed,
    not just by validate_ground_truth.py in isolation. A child with MUAC
    10.9 (SAM) and a normal WHZ must never sail through un-flagged."""
    bad_csv = tmp_path / "master_ground_truth.csv"
    bad_csv.write_text(
        "child_id,sex,date_of_birth,measurement_date,"
        "actual_height_cm,actual_weight_kg,muac,oedema,notes\n"
        "child_x,M,2023-01-01,2024-06-01,90.0,12.0,10.9,no,\n"
    )

    with pytest.raises(SystemExit) as exc_info:
        _load_master_ground_truth(bad_csv)

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "muac_cm" in captured.err


def test_template_csv_includes_measurement_date_muac_and_oedema():
    """The --template CSV must include measurement_date, muac_cm, and
    oedema, otherwise anyone starting from it can never fill in the columns
    needed for the age fix or the WHO OR-rule. image_file stays the join
    key for the flat layout (not child_id, which is per-child-only)."""
    header = TEMPLATE_CSV.strip().splitlines()[0]
    columns = [c.strip() for c in header.split(",")]
    assert "measurement_date" in columns
    assert "muac_cm" in columns
    assert "oedema" in columns
    assert "image_file" in columns
    assert "child_id" not in columns


# ---------------------------------------------------------------------------
# I-3: no child names in these files, and the assessment stage must key the
# child strictly on the folder id / image filename, never on an
# operator-supplied name.
# ---------------------------------------------------------------------------

def test_template_csv_has_no_child_name_column_or_pii():
    header = TEMPLATE_CSV.strip().splitlines()[0]
    columns = [c.strip() for c in header.split(",")]
    assert "child_name" not in columns
    assert "Child 1" not in TEMPLATE_CSV
    assert "Child 2" not in TEMPLATE_CSV


def test_row_child_name_is_image_filename_not_operator_name_flat_layout():
    """Even if a ground-truth row carries a 'child_name' column (against
    the rules), the row's own persisted child_name field must be the image
    filename, never the operator-supplied name — this is the field
    analyze_results.coverage() joins against the intake manifest's
    child_id, and a personal name there both breaks that join and violates
    the no-PII-in-CSVs rule."""
    row = _process_child_image(
        fname="child1.jpg",
        img_path=Path("child1.jpg"),
        gt={"child_name": "Child 1", "date_of_birth": "2023-01-01",
            "measurement_date": "2024-06-01", "sex": "M"},
        meas_svc=_FakeMeasService(),
        ml_svc=_FakeMLService(),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id=None,
        verbose=False,
    )
    assert row["child_name"] == "child1.jpg"


def test_row_child_name_is_folder_id_not_operator_name_per_child_layout():
    """Per-child layout: identity must be the folder id, even when the
    master CSV row was mistakenly given a 'child_name' column with a
    personal name — child_id must win."""
    row = _process_child_image(
        fname="front.jpg",
        img_path=Path("front.jpg"),
        gt={"child_name": "Child 1", "date_of_birth": "2023-01-01",
            "measurement_date": "2024-06-01", "sex": "M"},
        meas_svc=_FakeMeasService(),
        ml_svc=_FakeMLService(),
        nutr_svc=_FakeNutritionService(),
        who_data=_FakeWHOData(),
        side_image_bytes=None,
        child_id="001",
        verbose=False,
    )
    assert row["child_name"] == "001"


def test_i3_child_name_column_in_master_csv_does_not_break_coverage_join():
    """End-to-end regression for the exact I-3 probe: an operator who keeps
    a 'child_name' column with literal personal names in their master
    ground-truth CSV must not have that leak into the join key
    analyze_results.coverage() uses — the folder id must win, so coverage
    is not voided (previously: assessed 0, unknown_ids ['Child 1', 'Child 2'])."""
    from scripts.analyze_results import coverage

    master_gt = {
        "001": {"child_name": "Child 1", "sex": "M",
                "date_of_birth": "2023-01-01", "measurement_date": "2024-06-01"},
        "002": {"child_name": "Child 2", "sex": "F",
                "date_of_birth": "2023-06-01", "measurement_date": "2024-06-01"},
    }
    entries = [
        ("001", Path("001_front.jpg"), None, {}),
        ("002", Path("002_front.jpg"), None, {}),
    ]
    results = _run_per_child(
        entries,
        meas_svc=_FakeMeasService(), ml_svc=_FakeMLService(),
        nutr_svc=_FakeNutritionService(), who_data=_FakeWHOData(),
        verbose=False, master_gt=master_gt,
    )
    child_names = [r["child_name"] for r in results]
    assert child_names == ["001", "002"]
    assert "Child 1" not in child_names and "Child 2" not in child_names

    intake = [{"child_id": "001"}, {"child_id": "002"}]
    cov = coverage(intake, [], results)
    assert cov["unknown_ids"] == []
    assert cov["assessed"] == 2
    assert cov["missing_data"] == 0
