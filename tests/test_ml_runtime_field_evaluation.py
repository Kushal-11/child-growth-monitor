"""Focused safety/reproducibility tests for the promoted ML runtime."""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from app.services import measurement_service as measurement_module
from app.services.measurement_service import MeasurementService
from ml.inference import (
    WastingFeatures,
    WastingPredictor,
    validate_raw_outputs,
)
from ml.models import FEATURE_NAMES
from scripts import promote_ml_runtime as promotion_module
from scripts.evaluate_field_data import (
    REQUIRED_RESULT_COLUMNS,
    evaluate_rows,
    main as field_evaluation_main,
    manual_poshan_status,
    render_markdown,
)
from scripts.validate_ground_truth import ALL_COLS, validate_rows


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_MODELS = REPO_ROOT / "data" / "models"
FLUTTER_MODELS = REPO_ROOT / "flutter_app" / "assets" / "models"
GOLDEN_CASES = REPO_ROOT / "ml" / "runtime_golden_cases.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_main_import_does_not_import_mediapipe(tmp_path):
    """A hostile MediaPipe stub proves API startup never touches the package."""

    (tmp_path / "mediapipe.py").write_text(
        "raise RuntimeError('mediapipe imported during API startup')\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(REPO_ROOT)))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import main; "
            "assert 'mediapipe' not in sys.modules; "
            "print('startup-ok')",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "startup-ok" in completed.stdout


def test_missing_pose_runtime_fails_at_actual_processing(tmp_path, monkeypatch):
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), 255 * __import__("numpy").ones((20, 20, 3), dtype="uint8"))
    monkeypatch.setattr(
        measurement_module,
        "_load_pose_runtime",
        lambda: (_ for _ in ()).throw(
            measurement_module.PoseRuntimeUnavailableError("pose unavailable")
        ),
    )
    with pytest.raises(
        measurement_module.PoseRuntimeUnavailableError, match="pose unavailable"
    ):
        MeasurementService().process_image(str(image_path))


def test_no_pose_never_returns_who_median(tmp_path, monkeypatch):
    image_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(image_path), 255 * __import__("numpy").ones((20, 20, 3), dtype="uint8"))
    service = MeasurementService()
    monkeypatch.setattr(
        service,
        "_detect_pose",
        lambda *_: {
            "head_y": None,
            "heel_y": None,
            "confidence": 0.0,
            "landmarks_px": [],
            "raw_landmarks": None,
            "posture_valid": False,
            "posture_issues": ["No pose detected"],
        },
    )
    monkeypatch.setattr(service, "_detect_parlegi", lambda *_: (None, False, None))
    monkeypatch.setattr(service, "_draw_annotations", lambda *_: None)
    who = SimpleNamespace(
        get_median_height_for_age=lambda *_: pytest.fail(
            "WHO median must not be consulted without a pose"
        )
    )

    output = service.process_image_with_estimation(
        str(image_path), age_months=24, sex="M", who_data=who
    )
    assert output.predicted_height_cm is None
    assert output.body_segments is None
    assert output.estimation_method == "none"
    assert output.pose_accepted is False
    assert output.pose_failure_reason == "no_pose_detected"
    assert output.estimation_provenance["who_population_statistic_used"] is False


def test_low_confidence_pose_is_rejected_before_measurement(tmp_path, monkeypatch):
    image_path = tmp_path / "low-confidence.jpg"
    cv2.imwrite(str(image_path), 255 * __import__("numpy").ones((20, 20, 3), dtype="uint8"))
    service = MeasurementService()
    monkeypatch.setattr(
        service,
        "_detect_pose",
        lambda *_: {
            "head_y": 1.0,
            "heel_y": 19.0,
            "confidence": 0.49,
            "landmarks_px": [],
            "raw_landmarks": object(),
            "posture_valid": True,
            "posture_issues": [],
        },
    )
    monkeypatch.setattr(service, "_detect_parlegi", lambda *_: (None, False, None))
    monkeypatch.setattr(service, "_draw_annotations", lambda *_: None)
    monkeypatch.setattr(
        service,
        "_measure_body_segments",
        lambda *_: pytest.fail("low-confidence landmarks must not be measured"),
    )

    output = service.process_image_with_estimation(
        str(image_path), age_months=24, sex="M", who_data=object()
    )
    assert output.predicted_height_cm is None
    assert output.body_segments is None
    assert output.pose_detected is True
    assert output.pose_accepted is False
    assert "below_threshold" in output.pose_failure_reason


def test_backend_and_flutter_runtime_bundles_are_byte_identical():
    backend_manifest_path = BACKEND_MODELS / "model_manifest.json"
    flutter_manifest_path = FLUTTER_MODELS / "model_manifest.json"
    assert backend_manifest_path.read_bytes() == flutter_manifest_path.read_bytes()
    manifest = json.loads(backend_manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_version"] == "cgm-wasting-14f-synth-v1"
    assert manifest["feature_schema_version"] == 1
    assert manifest["feature_count"] == 14
    assert manifest["feature_names"] == FEATURE_NAMES
    assert manifest["training_data"] == "synthetic"
    assert manifest["evaluation"]["non_clinical"] is True
    assert manifest["evaluation"]["engine"] == "tensorflow_lite"
    assert manifest["evaluation"]["evaluation_contract_version"] == 2
    assert manifest["evaluation"]["sam_sample_count"] > 0
    assert manifest["evaluation"]["invalid_prediction_count"] == 0
    assert manifest["evaluation"]["sam_recall"] >= 0.80
    assert len(manifest["evaluation"]["dataset"]["sha256"]) == 64

    for filename in (
        "weight_estimator.tflite",
        "wasting_classifier.tflite",
        "feature_scaler.json",
        "label_encoder.json",
    ):
        backend = BACKEND_MODELS / filename
        flutter = FLUTTER_MODELS / filename
        assert backend.read_bytes() == flutter.read_bytes()
        assert _sha256(backend) == manifest["artifacts"][filename]["sha256"]
        assert backend.stat().st_size == manifest["artifacts"][filename]["size_bytes"]
        assert (
            manifest["evaluation"]["evaluated_artifacts"][filename]
            == manifest["artifacts"][filename]
        )


def test_runtime_postprocessing_matches_cross_language_golden_cases():
    fixture = json.loads(GOLDEN_CASES.read_text(encoding="utf-8"))
    assert fixture["contract_version"] == 1
    labels = ["MAM", "Normal", "Overweight", "Risk_Overweight", "SAM"]
    for case in fixture["postprocessing_cases"]:
        if case["valid"]:
            weight, probabilities, status = validate_raw_outputs(
                case["estimated_weight_kg"],
                case["probabilities"],
                labels,
            )
            assert weight == case["estimated_weight_kg"], case["name"]
            assert list(probabilities.values()) == case["probabilities"], case["name"]
            assert status == case["expected_status"], case["name"]
        else:
            with pytest.raises(ValueError):
                validate_raw_outputs(
                    case["estimated_weight_kg"],
                    case["probabilities"],
                    labels,
                )

    for invalid in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            validate_raw_outputs(invalid, [0.1, 0.2, 0.3, 0.15, 0.25], labels)
        with pytest.raises(ValueError, match="non-finite"):
            validate_raw_outputs(
                9.0,
                [0.1, 0.2, invalid, 0.35, 0.35],
                labels,
            )


def test_exact_tflite_runtime_matches_golden_inference_outputs():
    fixture = json.loads(GOLDEN_CASES.read_text(encoding="utf-8"))
    predictor = WastingPredictor()
    assert predictor.is_available, predictor.load_error

    for case in fixture["inference_cases"]:
        prediction = predictor.predict(WastingFeatures(**case["features"]))
        assert prediction is not None, case["name"]
        expected = case["expected"]
        actual_probabilities = [
            prediction.mam_probability,
            prediction.normal_probability,
            prediction.overweight_probability,
            prediction.risk_probability,
            prediction.sam_probability,
        ]
        assert prediction.estimated_weight_kg == pytest.approx(
            expected["estimated_weight_kg"], abs=1e-6
        )
        assert actual_probabilities == pytest.approx(
            expected["probabilities"], abs=1e-6
        )
        assert prediction.wasting_status == expected["wasting_status"]


def _promotion_report_from_manifest() -> dict:
    manifest = json.loads(
        (BACKEND_MODELS / "model_manifest.json").read_text(encoding="utf-8")
    )
    evaluation = manifest["evaluation"]
    return {
        "evaluation_contract_version": evaluation["evaluation_contract_version"],
        "engine": evaluation["engine"],
        "dataset": copy.deepcopy(evaluation["dataset"]),
        "split": copy.deepcopy(evaluation["split"]),
        "runtime_artifacts": copy.deepcopy(
            evaluation["evaluated_artifacts"]
        ),
        "sample_count": evaluation["sample_count"],
        "sam_sample_count": evaluation["sam_sample_count"],
        "mam_sample_count": evaluation["mam_sample_count"],
        "invalid_prediction_count": evaluation["invalid_prediction_count"],
        "weight_mae_kg": evaluation["weight_mae_kg"],
        "classification_accuracy": evaluation["classification_accuracy"],
        "sam_recall": evaluation["sam_recall"],
        "mam_recall": evaluation["mam_recall"],
        "mam_precision": evaluation["mam_precision"],
        "sam_recall_floor": evaluation["sam_recall_floor"],
        "sam_recall_floor_met": evaluation["sam_recall_floor_met"],
        "non_clinical": evaluation["non_clinical"],
    }


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("sam_sample_count", 0, "no evaluable SAM"),
        ("sam_recall", float("nan"), "must be finite"),
        ("invalid_prediction_count", 1, "invalid model predictions"),
    ],
)
def test_promotion_rejects_unsafe_or_nonfinite_reports(
    tmp_path,
    monkeypatch,
    field,
    bad_value,
    message,
):
    dataset = tmp_path / "synthetic_dataset.csv"
    dataset.write_text("synthetic-test-data\n", encoding="utf-8")
    monkeypatch.setattr(promotion_module, "SYNTHETIC_DATASET", dataset)
    report = _promotion_report_from_manifest()
    report["dataset"]["sha256"] = _sha256(dataset)
    report["dataset"]["size_bytes"] = dataset.stat().st_size
    report[field] = bad_value
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        promotion_module._load_metrics(
            report_path,
            report["runtime_artifacts"],
        )


def test_promotion_rejects_report_for_different_runtime_artifact(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "synthetic_dataset.csv"
    dataset.write_text("synthetic-test-data\n", encoding="utf-8")
    monkeypatch.setattr(promotion_module, "SYNTHETIC_DATASET", dataset)
    report = _promotion_report_from_manifest()
    report["dataset"]["sha256"] = _sha256(dataset)
    report["dataset"]["size_bytes"] = dataset.stat().st_size
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    actual_artifacts = copy.deepcopy(report["runtime_artifacts"])
    actual_artifacts["label_encoder.json"]["sha256"] = "0" * 64

    with pytest.raises(SystemExit, match="artifact hashes"):
        promotion_module._load_metrics(report_path, actual_artifacts)


def test_promoted_backend_runtime_files_are_not_gitignored():
    for filename in (
        "weight_estimator.tflite",
        "wasting_classifier.tflite",
        "feature_scaler.json",
        "label_encoder.json",
        "model_manifest.json",
    ):
        completed = subprocess.run(
            ["git", "check-ignore", str(BACKEND_MODELS / filename)],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 1, filename


def _field_row(**changes):
    row = {
        "age_months": "24",
        "sex": "M",
        "actual_height_cm": "85",
        "actual_weight_kg": "9",
        "muac_cm": "11",
        "actual_whz_status": "SAM",
        "pred_weight_ml_kg": "10",
        "ml_wasting_status": "SAM",
        "effective_height_source": "manual_measured",
        "ml_model_version": "cgm-wasting-14f-synth-v1",
        "ml_training_data": "synthetic",
        "ml_non_clinical": "true",
        "ml_runtime_manifest_sha256": "a" * 64,
        "pred_status_final": "SAM",
        "error": "",
    }
    row.update(changes)
    return row


def test_field_evaluation_uses_manual_measurements_and_aggregate_metrics():
    result = evaluate_rows(
        [
            _field_row(),
            _field_row(
                sex="F",
                actual_height_cm="90",
                actual_weight_kg="12",
                muac_cm="12",
                actual_whz_status="MAM",
                pred_weight_ml_kg="11",
                ml_wasting_status="MAM",
                pred_status_final="MAM",
            ),
        ]
    )
    assert result["weight_mae_kg"]["value"] == 1.0
    assert result["weight_mae_kg"]["sample_count"] == 2
    assert result["weight_mae_kg"]["manual_height_assisted_count"] == 2
    assert result["classification_accuracy"]["value"] == 1.0
    assert result["classification_accuracy"]["manual_height_assisted_count"] == 2
    assert result["sam_recall"]["value"] == 1.0
    assert result["mam_recall"]["value"] == 1.0
    assert result["mam_precision"]["value"] == 1.0
    assert result["screening_vs_manual_poshan_agreement"]["value"] == 1.0
    report = render_markdown(result)
    assert "001" not in report and "002" not in report
    assert "85" not in report and "11.0" not in report


def test_poshan_gold_is_indeterminate_when_non_sam_component_is_missing():
    row = _field_row(muac_cm="", actual_height_cm="90", actual_weight_kg="12")
    assert manual_poshan_status(row) == "Indeterminate"
    assert manual_poshan_status(_field_row(muac_cm="", actual_weight_kg="8")) == "SAM"


def test_field_evaluator_fails_on_header_only_csv_without_writing_reports(tmp_path):
    results = tmp_path / "batch_results.csv"
    columns = sorted(REQUIRED_RESULT_COLUMNS)
    results.write_text(",".join(columns) + "\n", encoding="utf-8")
    json_output = tmp_path / "evaluation.json"
    report_output = tmp_path / "evaluation.md"
    assert field_evaluation_main(
        [
            "--results",
            str(results),
            "--json-output",
            str(json_output),
            "--report-output",
            str(report_output),
        ]
    ) == 1
    assert not json_output.exists()
    assert not report_output.exists()


def test_field_sam_recall_counts_errored_manual_sam_as_miss():
    result = evaluate_rows(
        [
            _field_row(),
            _field_row(
                error="pose failed",
                pred_weight_ml_kg="",
                ml_wasting_status="",
                pred_status_final="",
                ml_model_version="",
                ml_training_data="",
                ml_non_clinical="",
                ml_runtime_manifest_sha256="",
            ),
        ]
    )
    assert result["sam_recall"]["numerator"] == 1
    assert result["sam_recall"]["denominator"] == 2
    assert result["sam_recall"]["value"] == 0.5
    assert result["sam_recall"]["missed_due_to_error_or_missing_prediction"] == 1


def test_field_evaluation_rejects_mixed_model_provenance():
    with pytest.raises(ValueError, match="mixed ML model versions"):
        evaluate_rows(
            [
                _field_row(),
                _field_row(ml_model_version="different-runtime"),
            ]
        )


def test_field_evaluation_blocks_when_sam_floor_is_not_evaluable(tmp_path):
    results = tmp_path / "batch_results.csv"
    row = _field_row(
        actual_weight_kg="14",
        muac_cm="13",
        actual_whz_status="Normal",
        ml_wasting_status="Normal",
        pred_status_final="Normal",
    )
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted(REQUIRED_RESULT_COLUMNS)
        )
        writer.writeheader()
        writer.writerow(row)
    assert field_evaluation_main(
        [
            "--results",
            str(results),
            "--json-output",
            str(tmp_path / "evaluation.json"),
            "--report-output",
            str(tmp_path / "evaluation.md"),
        ]
    ) == 3


def test_ground_truth_header_only_nonfinite_and_exact_60_months_are_rejected():
    errors, _ = validate_rows([], fieldnames=ALL_COLS)
    assert any("no child data rows" in error for error in errors)
    base = {
        "child_id": "001",
        "sex": "M",
        "date_of_birth": "2021-01-01",
        "measurement_date": "2025-12-31",
        "actual_height_cm": "30",
        "actual_weight_kg": "0.5",
        "muac_cm": "5",
        "oedema": "no",
        "notes": "",
    }
    errors, _ = validate_rows([base])
    assert errors == []
    for field in ("actual_height_cm", "actual_weight_kg", "muac_cm"):
        errors, _ = validate_rows([{**base, field: "nan"}])
        assert errors

    exact_60 = {
        **base,
        "date_of_birth": "2020-01-01",
        # 1826.25-day semantics are tested more directly in batch helpers;
        # this clearly exceeds/exactly reaches the exclusive under-five cap.
        "measurement_date": "2025-01-01",
    }
    errors, _ = validate_rows([exact_60])
    assert any("age" in error for error in errors)
