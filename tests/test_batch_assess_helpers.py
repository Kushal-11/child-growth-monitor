"""Tests for the pure helpers in scripts/batch_assess.py."""
import csv
import io
from datetime import date
from pathlib import Path

from scripts.batch_assess import (
    TEMPLATE_CSV,
    _collapse_wasting,
    _combine_status,
    _compute_age_months,
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
    still process without an error (only unparseable DOB should error)."""
    row = _process_child_image(
        fname="child_ok.jpg",
        img_path=Path("child_ok.jpg"),
        gt={"date_of_birth": "2023-01-01", "sex": "M"},
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
    rather than failing the row."""
    row = _process_child_image(
        fname="child2.jpg",
        img_path=Path("child2.jpg"),
        gt={
            "date_of_birth": "2023-01-01",
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
    processed and appear in the results."""
    master_csv_text = (
        "child_id,sex,date_of_birth\n"
        "child_bad,M,2023-01-01,unexpected,extra,columns\n"
        "child_ok,F,2023-06-01\n"
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
