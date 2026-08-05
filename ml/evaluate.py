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

Pickle is used here only to load *our own* scaler/label-encoder produced by
train.py in the same repo. Inputs are never untrusted.

Run:  python ml/evaluate.py
"""

import pickle  # noqa: S403  trusted artifacts produced by train.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
DATA_CSV = DATA_DIR / "training_data" / "synthetic_dataset.csv"
SPLIT_MANIFEST = DATA_DIR / "training_data" / "synthetic_split_manifest.json"

from ml.models import FEATURE_NAMES, WASTING_LABELS
from ml.splits import load_split_manifest, rows_for_split

AGE_BINS = [(0, 6, "0-5mo"), (6, 24, "6-23mo"), (24, 60, "24-59mo")]


def _load_artifacts():
    import tensorflow as tf

    we_model = tf.keras.models.load_model(MODELS_DIR / "weight_estimator.keras")
    wc_model = tf.keras.models.load_model(MODELS_DIR / "wasting_classifier.keras")
    with open(MODELS_DIR / "feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)  # noqa: S301
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)  # noqa: S301
    return we_model, wc_model, scaler, le


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
        if pred_mam_mask.any()
        else float("nan")
    )
    sam_precision = (
        (true_labels[pred_sam_mask] == "SAM").mean()
        if pred_sam_mask.any()
        else float("nan")
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
        "sam_recall": sam_recall,
        "sam_precision": sam_precision,
        "mam_recall": mam_recall,
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


def main():
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        mean_absolute_error,
    )

    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_NAMES].values.astype("float32")
    y_weight = df["weight_kg"].values.astype("float32")

    we_model, wc_model, scaler, le = _load_artifacts()
    y_class = le.transform(df["label"]).astype("int32")

    manifest = load_split_manifest(df, SPLIT_MANIFEST)
    test_rows = rows_for_split(df, manifest, "test")
    test_idx = test_rows.index.to_numpy()
    X_val = X[test_idx]
    yw_val = y_weight[test_idx]
    yc_val = y_class[test_idx]
    age_val = df.loc[test_idx, "age_months"].to_numpy(dtype="float32")
    sex_val = df.loc[test_idx, "sex"].to_numpy()
    X_val_s = scaler.transform(X_val).astype("float32")

    # ── Weight estimator ──────────────────────────────────────────────────────
    pred_weight = we_model.predict(X_val_s, verbose=0).flatten()
    mae_overall = mean_absolute_error(yw_val, pred_weight)
    print(f"Weight estimator MAE (overall): {mae_overall:.3f} kg\n")

    val_labels = le.inverse_transform(yc_val)
    for lbl in sorted(set(val_labels)):
        mask = val_labels == lbl
        mae_lbl = mean_absolute_error(yw_val[mask], pred_weight[mask])
        print(f"  MAE for {lbl:18s}: {mae_lbl:.3f} kg")

    # ── Classifier — overall ──────────────────────────────────────────────────
    probs = wc_model.predict(X_val_s, verbose=0)
    pred_cls = probs.argmax(axis=1)
    pred_labels = le.inverse_transform(pred_cls)

    print("\n--- Classification Report ---")
    print(
        classification_report(
            val_labels, pred_labels, target_names=sorted(set(val_labels))
        )
    )

    cm = confusion_matrix(val_labels, pred_labels, labels=sorted(WASTING_LABELS))
    cm_df = pd.DataFrame(
        cm, index=sorted(WASTING_LABELS), columns=sorted(WASTING_LABELS)
    )
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm_df.to_string())

    # ── Calibration ───────────────────────────────────────────────────────────
    ece = _ece(probs, yc_val)
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
    sam_idx = list(le.classes_).index("SAM")
    sam_probs = probs[:, sam_idx]
    sam_truth = val_labels == "SAM"
    print(
        "\n--- SAM-threshold sweep (raise threshold = higher precision, lower recall) ---"
    )
    print("  thr   SAM_recall  SAM_prec  predicted_SAM  predicted_MAM  MAM_prec")
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        new_pred = pred_labels.copy()
        # Re-route: if SAM prob below threshold, fall back to second-best class
        switch_to_other = (sam_probs < thr) & (pred_labels == "SAM")
        if switch_to_other.any():
            other_probs = probs[switch_to_other].copy()
            other_probs[:, sam_idx] = -1
            new_pred[switch_to_other] = le.inverse_transform(other_probs.argmax(axis=1))
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
            if mam_pred_mask.any()
            else float("nan")
        )
        print(
            f"  {thr:.2f}    {sam_rec:.3f}      {sam_prec:.3f}    "
            f"{int(sam_pred_mask.sum()):>5}          {int(mam_pred_mask.sum()):>5}        {mam_prec:.3f}"
        )

    # ── Headline SAM recall (existing behaviour) ──────────────────────────────
    sam_mask = val_labels == "SAM"
    if sam_mask.any():
        sam_recall = (pred_labels[sam_mask] == "SAM").mean()
        print(f"\n*** SAM recall: {sam_recall:.3f} (target ≥ 0.80) ***")
        if sam_recall < 0.80:
            print(
                "    WARNING: SAM recall is below 0.80 — consider adjusting"
                " class weights or resampling in train.py"
            )
    else:
        print("No SAM samples in validation set.")


if __name__ == "__main__":
    sys.exit(main())
