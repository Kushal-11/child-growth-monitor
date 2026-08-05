"""Evaluate the exact TFLite assets shipped in the Flutter application.

This is the release baseline. It verifies the asset manifest hashes, reuses the
locked child-level test split, and evaluates interpreter outputs rather than
the pre-conversion Keras models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.models import FEATURE_NAMES, WASTING_LABELS
from ml.splits import load_split_manifest, rows_for_split

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "data" / "training_data" / "synthetic_dataset.csv"
DEFAULT_SPLITS = BASE_DIR / "data" / "training_data" / "synthetic_split_manifest.json"
ASSET_DIR = BASE_DIR / "flutter_app" / "assets" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_shipped_assets(asset_dir: Path = ASSET_DIR) -> dict:
    manifest = json.loads((asset_dir / "model_manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("non_clinical") is not True:
        raise ValueError("unsupported or unsafe camera model manifest")
    for name, expected in manifest["files"].items():
        path = asset_dir / name
        if path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"asset size mismatch: {name}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"asset checksum mismatch: {name}")
    return manifest


def _interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def _run_model(model_path: Path, inputs: np.ndarray) -> np.ndarray:
    interpreter = _interpreter(model_path)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"]) != (1, len(FEATURE_NAMES)):
        raise ValueError(f"unexpected input shape in {model_path.name}")
    outputs = []
    for row in inputs:
        interpreter.set_tensor(
            input_detail["index"], row.reshape(1, -1).astype("float32")
        )
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_detail["index"])[0].copy())
    return np.asarray(outputs)


def expected_calibration_error(
    probs: np.ndarray, truth: np.ndarray, bins: int = 15
) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == truth
    result = 0.0
    for lower, upper in zip(
        np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]
    ):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(result)


def evaluate(dataset: Path = DEFAULT_DATASET, splits: Path = DEFAULT_SPLITS) -> dict:
    manifest = verify_shipped_assets()
    df = pd.read_csv(dataset)
    split_manifest = load_split_manifest(df, splits)
    test = rows_for_split(df, split_manifest, "test")
    scaler = json.loads((ASSET_DIR / "feature_scaler.json").read_text())
    raw = test[FEATURE_NAMES].to_numpy(dtype="float32")
    scaled = (raw - np.asarray(scaler["mean"], dtype="float32")) / np.asarray(
        scaler["scale"], dtype="float32"
    )
    weight = _run_model(ASSET_DIR / "weight_estimator.tflite", scaled).reshape(-1)
    probs = _run_model(ASSET_DIR / "wasting_classifier.tflite", scaled)
    true_labels = test["label"].astype(str).to_numpy()
    true_class = np.asarray([WASTING_LABELS.index(label) for label in true_labels])
    pred_class = probs.argmax(axis=1)
    pred_labels = np.asarray(WASTING_LABELS)[pred_class]
    abs_weight_error = np.abs(weight - test["weight_kg"].to_numpy(dtype="float32"))

    def recall(label: str) -> float:
        selected = true_labels == label
        return (
            float((pred_labels[selected] == label).mean())
            if selected.any()
            else float("nan")
        )

    def precision(label: str) -> float:
        selected = pred_labels == label
        return (
            float((true_labels[selected] == label).mean())
            if selected.any()
            else float("nan")
        )

    return {
        "model_version": manifest["model_version"],
        "training_data_label": manifest["training_data_label"],
        "dataset_fingerprint": split_manifest["dataset_fingerprint"],
        "test_children": int(test["child_id"].nunique()),
        "five_class_accuracy": float((pred_class == true_class).mean()),
        "sam_recall": recall("SAM"),
        "sam_precision": precision("SAM"),
        "mam_recall": recall("MAM"),
        "mam_precision": precision("MAM"),
        "weight_mae_kg": float(abs_weight_error.mean()),
        "weight_median_absolute_error_kg": float(np.median(abs_weight_error)),
        "weight_p95_absolute_error_kg": float(np.quantile(abs_weight_error, 0.95)),
        "expected_calibration_error": expected_calibration_error(probs, true_class),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    args = parser.parse_args()
    metrics = evaluate(args.dataset, args.split_manifest)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print("Exact shipped TFLite baseline (experimental, non-diagnostic)")
        for name, value in metrics.items():
            print(f"  {name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
