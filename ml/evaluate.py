"""
Evaluate trained wasting detection models.

Prints:
  - Per-class precision / recall / F1 (weight estimator and classifier)
  - Confusion matrix
  - Weight estimation MAE by wasting category
  - SAM recall (most safety-critical metric — false negatives are dangerous)
  - Per-age-bin (0-5 / 6-23 / 24-59 mo) SAM recall and MAM precision
  - Per-sex (M / F) SAM recall and MAM precision
  - Calibration: expected calibration error (ECE)
  - SAM-threshold sweep showing the SAM-recall vs MAM-precision tradeoff

The evaluator loads the exact TFLite, scaler JSON, and label JSON files used
by the backend and Flutter. It never substitutes a Keras model when deciding
whether the runtime bundle passes the promotion gate.

Run:
    python ml/evaluate.py
    python ml/evaluate.py --json-output data/models/synthetic_evaluation.json

This evaluates synthetic data only.  The results are useful for regression
tracking and are not evidence of clinical validity.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
DATA_CSV   = DATA_DIR / "training_data" / "synthetic_dataset.csv"

from ml.models import FEATURE_NAMES, WASTING_LABELS
from ml.inference import validate_raw_outputs

AGE_BINS = [(0, 6, "0-5mo"), (6, 24, "6-23mo"), (24, 60, "24-59mo")]
EVALUATION_CONTRACT_VERSION = 2
RUNTIME_ARTIFACT_FILENAMES = (
    "weight_estimator.tflite",
    "wasting_classifier.tflite",
    "feature_scaler.json",
    "label_encoder.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _interpreter_class():
    """Use the same lightweight-runtime preference as production inference."""

    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
            return Interpreter
        except ImportError:
            import tensorflow as tf
            return tf.lite.Interpreter


def _run_tflite(path: Path, inputs: np.ndarray, output_width: int) -> np.ndarray:
    """Run the exact promoted TFLite bytes over a deterministic batch."""

    Interpreter = _interpreter_class()
    interpreter = Interpreter(model_path=str(path))
    input_detail = interpreter.get_input_details()[0]
    if input_detail["dtype"] != np.float32:
        raise SystemExit(f"{path.name} input must be float32")
    shape_signature = tuple(int(value) for value in input_detail["shape_signature"])
    if shape_signature != (-1, len(FEATURE_NAMES)):
        raise SystemExit(
            f"{path.name} input signature must be "
            f"[-1,{len(FEATURE_NAMES)}], got {shape_signature}"
        )
    interpreter.resize_tensor_input(
        input_detail["index"],
        [len(inputs), len(FEATURE_NAMES)],
        strict=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if output_detail["dtype"] != np.float32:
        raise SystemExit(f"{path.name} output must be float32")
    interpreter.set_tensor(input_detail["index"], inputs.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    expected_shape = (len(inputs), output_width)
    if output.shape != expected_shape:
        raise SystemExit(
            f"{path.name} output must be {expected_shape}, got {output.shape}"
        )
    return output


def _load_runtime_metadata() -> tuple[np.ndarray, np.ndarray, list[str]]:
    scaler = json.loads(
        (MODELS_DIR / "feature_scaler.json").read_text(encoding="utf-8")
    )
    labels_data = json.loads(
        (MODELS_DIR / "label_encoder.json").read_text(encoding="utf-8")
    )
    if scaler.get("feature_names") != FEATURE_NAMES:
        raise SystemExit("Runtime scaler feature order differs from ml.models")
    means = np.asarray(scaler.get("mean"), dtype=np.float32)
    scales = np.asarray(scaler.get("scale"), dtype=np.float32)
    if means.shape != (len(FEATURE_NAMES),) or scales.shape != (
        len(FEATURE_NAMES),
    ):
        raise SystemExit("Runtime scaler must contain exactly 14 means/scales")
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise SystemExit("Runtime scaler contains non-finite values")
    if (scales <= 0).any():
        raise SystemExit("Runtime scaler contains non-positive scales")
    labels = [str(label) for label in labels_data.get("classes", [])]
    if labels != sorted(WASTING_LABELS):
        raise SystemExit("Runtime label encoder order differs from training contract")
    return means, scales, labels


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error — Guo et al. 2017."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _bin_metrics(true_labels, pred_labels):
    """Return SAM/MAM recall and precision for a subset."""
    sam_mask = true_labels == "SAM"
    mam_mask = true_labels == "MAM"
    pred_sam_mask = pred_labels == "SAM"
    pred_mam_mask = pred_labels == "MAM"

    sam_recall = (
        (pred_labels[sam_mask] == "SAM").mean() if sam_mask.any() else float("nan")
    )
    mam_recall = (
        (pred_labels[mam_mask] == "MAM").mean() if mam_mask.any() else float("nan")
    )
    mam_precision = (
        (true_labels[pred_mam_mask] == "MAM").mean()
        if pred_mam_mask.any() else float("nan")
    )
    sam_precision = (
        (true_labels[pred_sam_mask] == "SAM").mean()
        if pred_sam_mask.any() else float("nan")
    )

    # Clinical "wasted recall" — predicting SAM or MAM when truth is SAM or MAM
    wasted_mask = sam_mask | mam_mask
    wasted_pred = pred_sam_mask | pred_mam_mask
    wasted_recall = (
        (wasted_pred[wasted_mask]).mean() if wasted_mask.any() else float("nan")
    )

    return {
        "n": len(true_labels),
        "n_sam": int(sam_mask.sum()),
        "n_mam": int(mam_mask.sum()),
        "sam_recall":    sam_recall,
        "sam_precision": sam_precision,
        "mam_recall":    mam_recall,
        "mam_precision": mam_precision,
        "wasted_recall": wasted_recall,
    }


def _print_metric_row(label: str, m: dict):
    def f(v):
        return "  N/A " if np.isnan(v) else f"{v:.3f}"
    print(
        f"  {label:14s}  n={m['n']:>5}  sam={m['n_sam']:>3}  mam={m['n_mam']:>4}  "
        f"SAM_rec={f(m['sam_recall'])}  SAM_prec={f(m['sam_precision'])}  "
        f"MAM_rec={f(m['mam_recall'])}  MAM_prec={f(m['mam_precision'])}  "
        f"wasted_rec={f(m['wasted_recall'])}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Also write deterministic machine-readable synthetic metrics.",
    )
    args = parser.parse_args(argv)

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        mean_absolute_error,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print(
        "SYNTHETIC HOLDOUT EVALUATION — non-clinical; clinical validity "
        "has not been established.\n"
    )
    df = pd.read_csv(DATA_CSV)
    X  = df[FEATURE_NAMES].values.astype("float32")
    y_weight = df["weight_kg"].values.astype("float32")
    y_labels = df["label"].astype(str).values

    # Reproduce ml/train.py's 70/15/15 split exactly. The last 15% is the
    # untouched validation set; the middle 15% was used only for calibration.
    all_indices = np.arange(len(df), dtype=np.int64)
    train_indices, temporary_indices = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=42,
        stratify=y_labels,
    )
    _, validation_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=42,
        stratify=y_labels[temporary_indices],
    )
    X_val = X[validation_indices]
    yw_val = y_weight[validation_indices]
    val_labels = y_labels[validation_indices]
    age_val = df["age_months"].values.astype("float32")[validation_indices]
    sex_val = df["sex"].values[validation_indices]

    means, scales, labels = _load_runtime_metadata()
    expected_scaler = StandardScaler().fit(X[train_indices])
    if not np.allclose(means, expected_scaler.mean_, rtol=0.0, atol=1e-5):
        raise SystemExit(
            "Runtime scaler mean does not match ml/train.py's training split"
        )
    if not np.allclose(scales, expected_scaler.scale_, rtol=0.0, atol=1e-5):
        raise SystemExit(
            "Runtime scaler scale does not match ml/train.py's training split"
        )
    X_val_s = ((X_val - means) / scales).astype("float32")
    if not np.isfinite(X_val_s).all():
        raise SystemExit("Scaled validation features contain non-finite values")

    # ── Weight estimator ──────────────────────────────────────────────────────
    pred_weight = _run_tflite(
        MODELS_DIR / "weight_estimator.tflite",
        X_val_s,
        output_width=1,
    ).flatten()

    # ── Classifier ────────────────────────────────────────────────────────────
    probs = _run_tflite(
        MODELS_DIR / "wasting_classifier.tflite",
        X_val_s,
        output_width=len(labels),
    )
    pred_labels_list: list[str] = []
    for weight, row_probabilities in zip(pred_weight, probs):
        try:
            _, _, top_class = validate_raw_outputs(
                float(weight),
                row_probabilities.tolist(),
                labels,
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid TFLite output in validation batch: {exc}") from exc
        pred_labels_list.append(top_class)
    pred_labels = np.asarray(pred_labels_list)

    mae_overall = mean_absolute_error(yw_val, pred_weight)
    print(f"Weight estimator MAE (overall): {mae_overall:.3f} kg\n")

    for lbl in sorted(set(val_labels)):
        mask = val_labels == lbl
        mae_lbl = mean_absolute_error(yw_val[mask], pred_weight[mask])
        print(f"  MAE for {lbl:18s}: {mae_lbl:.3f} kg")

    # ── Classifier — overall ──────────────────────────────────────────────────
    headline = _bin_metrics(val_labels, pred_labels)
    classification_accuracy = float((pred_labels == val_labels).mean())

    print("\n--- Classification Report ---")
    print(
        classification_report(
            val_labels,
            pred_labels,
            labels=labels,
            target_names=labels,
            zero_division=0,
        )
    )

    cm = confusion_matrix(val_labels, pred_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm_df.to_string())

    # ── Calibration ───────────────────────────────────────────────────────────
    label_to_index = {label: index for index, label in enumerate(labels)}
    true_indices = np.asarray([label_to_index[label] for label in val_labels])
    ece = _ece(probs, true_indices)
    print(f"\nExpected Calibration Error (15 bins): {ece:.4f}")

    # ── Per-age-bin metrics ───────────────────────────────────────────────────
    print("\n--- Per age bin ---")
    for lo, hi, name in AGE_BINS:
        mask = (age_val >= lo) & (age_val < hi)
        if mask.any():
            _print_metric_row(name, _bin_metrics(val_labels[mask], pred_labels[mask]))

    # ── Per-sex metrics ───────────────────────────────────────────────────────
    print("\n--- Per sex ---")
    for s in ("M", "F"):
        mask = sex_val == s
        if mask.any():
            _print_metric_row(s, _bin_metrics(val_labels[mask], pred_labels[mask]))

    # ── SAM-threshold sweep ───────────────────────────────────────────────────
    sam_idx = labels.index("SAM")
    sam_probs = probs[:, sam_idx]
    sam_truth = (val_labels == "SAM")
    print("\n--- SAM-threshold sweep (raise threshold = higher precision, lower recall) ---")
    print("  thr   SAM_recall  SAM_prec  predicted_SAM  predicted_MAM  MAM_prec")
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        new_pred = pred_labels.copy()
        # Re-route: if SAM prob below threshold, fall back to second-best class
        switch_to_other = (sam_probs < thr) & (pred_labels == "SAM")
        if switch_to_other.any():
            other_probs = probs[switch_to_other].copy()
            other_probs[:, sam_idx] = -1
            new_pred[switch_to_other] = np.asarray(labels)[
                other_probs.argmax(axis=1)
            ]
        sam_pred_mask = new_pred == "SAM"
        sam_rec = (
            (new_pred[sam_truth] == "SAM").mean() if sam_truth.any() else float("nan")
        )
        sam_prec = (
            (sam_truth[sam_pred_mask]).mean() if sam_pred_mask.any() else float("nan")
        )
        mam_pred_mask = new_pred == "MAM"
        mam_prec = (
            (val_labels[mam_pred_mask] == "MAM").mean()
            if mam_pred_mask.any() else float("nan")
        )
        print(f"  {thr:.2f}    {sam_rec:.3f}      {sam_prec:.3f}    "
              f"{int(sam_pred_mask.sum()):>5}          {int(mam_pred_mask.sum()):>5}        {mam_prec:.3f}")

    # ── Headline SAM recall (existing behaviour) ──────────────────────────────
    sam_mask = val_labels == "SAM"
    floor_met = False
    if sam_mask.any():
        sam_recall = (pred_labels[sam_mask] == "SAM").mean()
        floor_met = bool(sam_recall >= 0.80)
        print(f"\n*** SAM recall: {sam_recall:.3f} (target ≥ 0.80) ***")
        if sam_recall < 0.80:
            print("    WARNING: SAM recall is below 0.80 — consider adjusting"
                  " class weights or resampling in train.py")
    else:
        sam_recall = float("nan")
        print("No SAM samples in validation set.")

    print(
        f"Classification accuracy: {classification_accuracy:.3f}\n"
        f"MAM recall: {headline['mam_recall']:.3f}\n"
        f"MAM precision: {headline['mam_precision']:.3f}"
    )

    metrics_for_gate = {
        "weight_mae_kg": float(mae_overall),
        "classification_accuracy": classification_accuracy,
        "sam_recall": float(sam_recall),
        "mam_recall": float(headline["mam_recall"]),
        "mam_precision": float(headline["mam_precision"]),
    }
    nonfinite_metrics = [
        name for name, value in metrics_for_gate.items() if not math.isfinite(value)
    ]
    sam_sample_count = int(sam_mask.sum())
    if sam_sample_count <= 0:
        print("Safety gate failed: validation set contains no evaluable SAM samples.")
        floor_met = False
    if nonfinite_metrics:
        print(
            "Safety gate failed: non-finite metrics: "
            + ", ".join(nonfinite_metrics)
        )
        floor_met = False

    if args.json_output is not None and floor_met:
        validation_selection = ",".join(
            str(int(index)) for index in sorted(validation_indices)
        ).encode("ascii")
        artifact_records = {
            filename: _artifact_record(MODELS_DIR / filename)
            for filename in RUNTIME_ARTIFACT_FILENAMES
        }
        report = {
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "engine": "tensorflow_lite",
            "dataset": {
                "name": "synthetic_dataset",
                "path": str(DATA_CSV.relative_to(BASE_DIR)),
                "sha256": _sha256(DATA_CSV),
                "size_bytes": DATA_CSV.stat().st_size,
                "row_count": int(len(df)),
            },
            "split": {
                "method": "train_calibration_validation",
                "train_fraction": 0.70,
                "calibration_fraction": 0.15,
                "validation_fraction": 0.15,
                "random_state": 42,
                "stratified": True,
                "validation_selection_sha256": hashlib.sha256(
                    validation_selection
                ).hexdigest(),
            },
            "runtime_artifacts": artifact_records,
            "sample_count": int(len(val_labels)),
            "sam_sample_count": sam_sample_count,
            "mam_sample_count": int((val_labels == "MAM").sum()),
            "invalid_prediction_count": 0,
            **metrics_for_gate,
            "sam_recall_floor": 0.80,
            "sam_recall_floor_met": floor_met,
            "non_clinical": True,
            "clinical_validity": "not_established",
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nMetrics JSON written to {args.json_output}")
    elif args.json_output is not None:
        print("\nMetrics JSON was not written because the safety gate failed.")

    return 0 if floor_met else 2


if __name__ == "__main__":
    sys.exit(main())
