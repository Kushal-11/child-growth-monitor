"""Promote one deterministic ML runtime bundle to backend and Flutter.

The training directory contains Keras and pickle artifacts that are convenient
for experimentation but unsuitable as the only production source: they are
large, Python-specific, and historically gitignored.  This command converts
the trusted local scaler/label encoder to JSON, copies the promoted TFLite
models byte-for-byte to Flutter, and writes identical version/hash manifests
for both runtimes.

Prepare candidate metadata, evaluate the exact candidate bytes, then promote:

    PYTHONPATH=. .venv/bin/python scripts/promote_ml_runtime.py --prepare-only
    PYTHONPATH=. .venv/bin/python ml/evaluate.py \
        --json-output data/models/synthetic_evaluation.json
    PYTHONPATH=. .venv/bin/python scripts/promote_ml_runtime.py

Pickle inputs must come from this repository's own ``ml/train.py``.  Never run
the command against untrusted pickle files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle  # noqa: S403 - trusted artifacts produced by this repository
import shutil
from pathlib import Path
from typing import Any

from ml.models import FEATURE_NAMES


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "data" / "models"
FLUTTER_DIR = REPO_ROOT / "flutter_app" / "assets" / "models"

MODEL_FILENAMES = ("weight_estimator.tflite", "wasting_classifier.tflite")
SCALER_FILENAME = "feature_scaler.json"
LABELS_FILENAME = "label_encoder.json"
RUNTIME_FILENAMES = (*MODEL_FILENAMES, SCALER_FILENAME, LABELS_FILENAME)
MANIFEST_FILENAME = "model_manifest.json"
DEFAULT_MODEL_VERSION = "cgm-wasting-14f-synth-v1"
SAM_RECALL_FLOOR = 0.80
EVALUATION_CONTRACT_VERSION = 2
SYNTHETIC_DATASET = REPO_ROOT / "data" / "training_data" / "synthetic_dataset.csv"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _load_trusted_pickle(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(
            f"Missing training artifact: {path}. Run ml/train.py before promotion."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - explicitly trusted repo artifact


def _load_metrics(
    path: Path,
    artifact_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"Missing synthetic evaluation report: {path}. Run "
            "`PYTHONPATH=. .venv/bin/python ml/evaluate.py "
            "--json-output data/models/synthetic_evaluation.json` first."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "evaluation_contract_version",
        "engine",
        "dataset",
        "split",
        "runtime_artifacts",
        "sample_count",
        "sam_sample_count",
        "mam_sample_count",
        "invalid_prediction_count",
        "weight_mae_kg",
        "classification_accuracy",
        "sam_recall",
        "mam_recall",
        "mam_precision",
        "sam_recall_floor",
        "sam_recall_floor_met",
        "non_clinical",
    }
    missing = sorted(required.difference(report))
    if missing:
        raise SystemExit(f"Evaluation report is missing fields: {', '.join(missing)}")
    if report.get("non_clinical") is not True:
        raise SystemExit("Evaluation report must explicitly set non_clinical=true.")
    if report.get("evaluation_contract_version") != EVALUATION_CONTRACT_VERSION:
        raise SystemExit(
            "Evaluation report contract is stale; rerun the current ml/evaluate.py."
        )
    if report.get("engine") != "tensorflow_lite":
        raise SystemExit(
            "Evaluation report must come from the exact TensorFlow Lite runtime."
        )
    if report.get("runtime_artifacts") != artifact_records:
        raise SystemExit(
            "Evaluation report artifact hashes do not match the runtime bundle. "
            "Rerun ml/evaluate.py."
        )

    dataset = report.get("dataset")
    expected_dataset = _artifact_record(SYNTHETIC_DATASET)
    if (
        not isinstance(dataset, dict)
        or dataset.get("sha256") != expected_dataset["sha256"]
        or dataset.get("size_bytes") != expected_dataset["size_bytes"]
    ):
        raise SystemExit(
            "Evaluation report dataset hash does not match synthetic_dataset.csv."
        )

    if (
        isinstance(report["sample_count"], bool)
        or not isinstance(report["sample_count"], int)
        or report["sample_count"] <= 0
    ):
        raise SystemExit("Evaluation report has no validation samples.")
    if (
        isinstance(report["sam_sample_count"], bool)
        or not isinstance(report["sam_sample_count"], int)
        or report["sam_sample_count"] <= 0
    ):
        raise SystemExit(
            "Refusing promotion: validation has no evaluable SAM samples."
        )
    if (
        isinstance(report["invalid_prediction_count"], bool)
        or not isinstance(report["invalid_prediction_count"], int)
        or report["invalid_prediction_count"] != 0
    ):
        raise SystemExit(
            "Refusing promotion: evaluation contained invalid model predictions."
        )
    metric_names = (
        "weight_mae_kg",
        "classification_accuracy",
        "sam_recall",
        "mam_recall",
        "mam_precision",
    )
    for metric_name in metric_names:
        metric = report[metric_name]
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise SystemExit(
                f"Evaluation report metric {metric_name} must be finite."
            )
    if report.get("sam_recall_floor_met") is not True:
        raise SystemExit("Evaluation report does not declare a passing SAM floor.")
    declared_floor = report["sam_recall_floor"]
    if (
        isinstance(declared_floor, bool)
        or not isinstance(declared_floor, (int, float))
        or not math.isfinite(float(declared_floor))
        or float(declared_floor) != SAM_RECALL_FLOOR
    ):
        raise SystemExit(
            f"Evaluation report must use the {SAM_RECALL_FLOOR:.2f} SAM floor."
        )
    if float(report["sam_recall"]) < SAM_RECALL_FLOOR:
        raise SystemExit(
            f"Refusing promotion: SAM recall {report['sam_recall']:.3f} is below "
            f"the {SAM_RECALL_FLOOR:.2f} safety floor."
        )
    return report


def prepare_runtime_metadata(model_version: str) -> None:
    """Export trusted training metadata to the JSON files runtimes consume."""

    scaler = _load_trusted_pickle(BACKEND_DIR / "feature_scaler.pkl")
    encoder = _load_trusted_pickle(BACKEND_DIR / "label_encoder.pkl")

    means = [float(v) for v in scaler.mean_]
    scales = [float(v) for v in scaler.scale_]
    labels = [str(v) for v in encoder.classes_]
    if len(means) != len(FEATURE_NAMES) or len(scales) != len(FEATURE_NAMES):
        raise SystemExit(
            "Scaler feature count drift: expected "
            f"{len(FEATURE_NAMES)}, got mean={len(means)} scale={len(scales)}."
        )
    if labels != ["MAM", "Normal", "Overweight", "Risk_Overweight", "SAM"]:
        raise SystemExit(f"Unexpected classifier label order: {labels}")

    scaler_json = {
        "feature_names": FEATURE_NAMES,
        "mean": means,
        "scale": scales,
        "feature_count": len(FEATURE_NAMES),
        "feature_schema_version": 1,
    }
    labels_json = {
        "classes": labels,
        "model_version": model_version,
    }
    backend_scaler = BACKEND_DIR / SCALER_FILENAME
    backend_labels = BACKEND_DIR / LABELS_FILENAME
    _write_json(backend_scaler, scaler_json)
    _write_json(backend_labels, labels_json)


def promote(model_version: str, metrics_path: Path) -> dict[str, Any]:
    prepare_runtime_metadata(model_version)
    for filename in MODEL_FILENAMES:
        source = BACKEND_DIR / filename
        if not source.is_file():
            raise SystemExit(f"Missing promoted TFLite model: {source}")
    artifact_names = RUNTIME_FILENAMES
    artifacts = {
        filename: _artifact_record(BACKEND_DIR / filename)
        for filename in artifact_names
    }
    metrics = _load_metrics(metrics_path, artifacts)
    labels = json.loads(
        (BACKEND_DIR / LABELS_FILENAME).read_text(encoding="utf-8")
    )["classes"]

    FLUTTER_DIR.mkdir(parents=True, exist_ok=True)
    for filename in RUNTIME_FILENAMES:
        shutil.copyfile(BACKEND_DIR / filename, FLUTTER_DIR / filename)

    manifest = {
        "model_version": model_version,
        "feature_schema_version": 1,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "labels": labels,
        "artifacts": artifacts,
        "training_data": "synthetic",
        "evaluation": {
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "engine": "tensorflow_lite",
            "dataset": metrics["dataset"],
            "split": metrics["split"],
            "evaluated_artifacts": metrics["runtime_artifacts"],
            "sample_count": int(metrics["sample_count"]),
            "sam_sample_count": int(metrics["sam_sample_count"]),
            "mam_sample_count": int(metrics["mam_sample_count"]),
            "invalid_prediction_count": int(
                metrics["invalid_prediction_count"]
            ),
            "weight_mae_kg": float(metrics["weight_mae_kg"]),
            "classification_accuracy": float(metrics["classification_accuracy"]),
            "sam_recall": float(metrics["sam_recall"]),
            "mam_recall": float(metrics["mam_recall"]),
            "mam_precision": float(metrics["mam_precision"]),
            "sam_recall_floor": SAM_RECALL_FLOOR,
            "sam_recall_floor_met": True,
            "non_clinical": True,
            "clinical_validity": "not_established",
        },
    }
    backend_manifest = BACKEND_DIR / MANIFEST_FILENAME
    _write_json(backend_manifest, manifest)
    shutil.copyfile(backend_manifest, FLUTTER_DIR / MANIFEST_FILENAME)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=BACKEND_DIR / "synthetic_evaluation.json",
        help="Machine-readable output from ml/evaluate.py.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Export scaler/label JSON before evaluating candidate TFLites. "
            "Does not copy or write a manifest."
        ),
    )
    args = parser.parse_args(argv)
    if args.prepare_only:
        prepare_runtime_metadata(args.model_version)
        print(
            f"Prepared runtime metadata for {args.model_version}; "
            "run ml/evaluate.py next."
        )
        return 0
    manifest = promote(args.model_version, args.metrics)
    print(
        f"Promoted {manifest['model_version']} to backend and Flutter "
        f"({manifest['feature_count']} features; synthetic/non-clinical)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
