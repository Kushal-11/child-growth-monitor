"""End-to-end smoke: real MediaPipe on one sample child, then full chain
with a stubbed cleaner so the chain runs even where the model is absent."""
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.intake_check import run_intake
from scripts.validate_ground_truth import validate_rows
from scripts.analyze_results import analyze, coverage, render_report

SAMPLE_DIR = Path("sample images")
POSE_MODEL = Path("data/pose_landmarker_heavy.task")
RUN_REAL_POSE_ENV = "CGM_RUN_REAL_MEDIAPIPE_TEST"
REAL_POSE_TIMEOUT_SECONDS = 45

GT_HEADER = ("child_id,sex,date_of_birth,measurement_date,"
             "actual_height_cm,actual_weight_kg,muac_cm,oedema,notes\n")


def _build_raw(tmp_path: Path) -> Path:
    """Copy the first sample child into a field_data-shaped tree."""
    sample_children = [d for d in SAMPLE_DIR.iterdir() if d.is_dir()] \
        if SAMPLE_DIR.is_dir() else []
    if not sample_children:
        pytest.skip("no sample images available")
    raw = tmp_path / "field_data" / "raw"
    dest = raw / "001"
    dest.mkdir(parents=True)
    for f in sample_children[0].iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    return raw


def test_intake_and_ground_truth_chain(tmp_path):
    raw = _build_raw(tmp_path)
    gt = tmp_path / "field_data" / "ground_truth.csv"
    gt.write_text(GT_HEADER + "001,M,2023-04-12,2026-07-15,82.5,10.4,13.2,no,\n")

    rows = list(csv.DictReader(gt.open()))
    errors, _ = validate_rows(rows)
    assert errors == []

    manifest = tmp_path / "field_data" / "reports" / "intake_manifest.csv"
    intake_rows = run_intake(raw, gt, manifest)
    assert len(intake_rows) == 1
    assert manifest.exists()


@pytest.mark.skipif(
    os.environ.get(RUN_REAL_POSE_ENV) != "1",
    reason=(
        "opt-in real MediaPipe test; run "
        "CGM_RUN_REAL_MEDIAPIPE_TEST=1 pytest "
        "tests/test_field_pipeline_smoke.py::test_clean_media_with_real_model"
    ),
)
def test_clean_media_with_real_model(tmp_path):
    """Run real MediaPipe out-of-process with a hard timeout."""
    if not POSE_MODEL.exists():
        pytest.skip("pose model not downloaded")
    raw = _build_raw(tmp_path)
    report = tmp_path / "field_data" / "reports" / "qc_report.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-real-clean",
            str(raw),
            str(tmp_path / "field_data" / "cleaned"),
            str(report),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=REAL_POSE_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert report.exists()


def test_analysis_chain_from_synthetic_results():
    results = [{
        "child_name": "001", "age_months": "39.0", "sex": "M",
        "actual_height_cm": "82.5", "pred_height_cm": "84.0",
        "actual_weight_kg": "10.4", "pred_weight_ml_kg": "10.9",
        "actual_combined_status": "Normal", "pred_status_final": "Normal",
        "error": "",
    }]
    text = render_report(
        analyze(results),
        coverage([{"child_id": "001"}],
                 [{"child_id": "001", "verdict": "ok"}], results),
    )
    assert "## Status agreement" in text


def _run_real_clean(raw: Path, cleaned: Path, report: Path) -> int:
    from scripts.clean_media import run_clean

    rows = run_clean(raw, cleaned, report, False)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one child QC row, got {len(rows)}")
    if rows[0]["verdict"] not in ("ok", "usable_no_side", "failed"):
        raise RuntimeError(f"Unexpected QC verdict: {rows[0]['verdict']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--run-real-clean":
        raise SystemExit(
            _run_real_clean(
                Path(sys.argv[2]),
                Path(sys.argv[3]),
                Path(sys.argv[4]),
            )
        )
    raise SystemExit("Use pytest, or --run-real-clean RAW CLEANED REPORT")
