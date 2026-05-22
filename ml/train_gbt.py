"""
Train a LightGBM 5-class classifier on the same WHO-derived synthetic dataset
that train.py uses for the MLP.

Why an alternative classifier?
------------------------------
On small/medium tabular datasets gradient-boosted trees consistently
outperform MLPs (Grinsztajn et al. 2022, "Why do tree-based models still
outperform deep learning on tabular data?"). LightGBM also produces
better-calibrated probabilities than a softmax MLP and inference is small.

Outputs
-------
data/models/wasting_classifier_gbt.txt          LightGBM native model
data/models/wasting_classifier_gbt_meta.json    feature names / class order

Run:  python ml/train_gbt.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
DATA_CSV   = DATA_DIR / "training_data" / "synthetic_dataset.csv"

from ml.models import FEATURE_NAMES, WASTING_LABELS
from ml.calibration import (
    ConformalCalibrator,
    apply_temperature,
    fit_temperature,
    save_calibration,
)


def main():
    import lightgbm as lgb
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_NAMES].values.astype("float32")

    le = LabelEncoder()
    le.fit(WASTING_LABELS)
    y = le.transform(df["label"]).astype("int32")

    # ── Train / calibration / val split ───────────────────────────────────────
    # 70 / 15 / 15 — calibration set must be disjoint from training and val
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_cal, X_va, y_cal, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    print(f"Train: {len(X_tr)}  Calib: {len(X_cal)}  Val: {len(X_va)}")

    # ── Class-balanced sample weights with SAM bump ───────────────────────────
    from sklearn.utils.class_weight import compute_class_weight
    class_w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    sam_idx = list(le.classes_).index("SAM")
    # LightGBM needs a stronger SAM bump than the MLP — without it, trees
    # collapse onto the majority "Normal" class for boundary samples.
    class_w[sam_idx] *= 2.0
    sample_w = class_w[y_tr]

    # ── Train LightGBM ────────────────────────────────────────────────────────
    train_set = lgb.Dataset(X_tr, label=y_tr, weight=sample_w,
                             feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(X_va, label=y_va, feature_name=FEATURE_NAMES,
                           reference=train_set)

    # Smaller trees + tighter regularization → ~50× smaller model for mobile.
    # Fewer leaves and shallower depth keep this around ~250 KB on disk.
    params = {
        "objective":       "multiclass",
        "num_class":       len(WASTING_LABELS),
        "metric":          "multi_logloss",
        "learning_rate":   0.05,
        "num_leaves":      15,
        "max_depth":       6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq":    5,
        "min_data_in_leaf": 100,
        "lambda_l2":       1.0,
        "verbose":         -1,
        "seed":            42,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "wasting_classifier_gbt.txt"
    booster.save_model(str(model_path))
    print(f"\nLightGBM model → {model_path.name}  "
          f"({model_path.stat().st_size // 1024} KB)")

    # ── Predict on calibration & val ──────────────────────────────────────────
    probs_cal = booster.predict(X_cal)   # already softmaxed (multiclass)
    probs_va  = booster.predict(X_va)

    # ── Temperature scaling ───────────────────────────────────────────────────
    # LightGBM gives probabilities, not logits — convert to log-probs for
    # temperature scaling, then re-apply softmax.
    eps = 1e-9
    log_probs_cal = np.log(np.clip(probs_cal, eps, 1.0))
    T = fit_temperature(log_probs_cal, y_cal)
    print(f"Fitted temperature: T = {T:.3f}")

    log_probs_va = np.log(np.clip(probs_va, eps, 1.0))
    probs_va_cal = apply_temperature(log_probs_va, T)

    # ── Conformal calibration on the calibration set ──────────────────────────
    log_probs_cal_t = np.log(np.clip(probs_cal, eps, 1.0))
    probs_cal_after = apply_temperature(log_probs_cal_t, T)

    conformal = ConformalCalibrator.fit(
        probs_cal_after, y_cal, class_names=list(le.classes_), alpha=0.10,
    )
    print("Conformal thresholds (1 - p_true) at α=0.10:")
    for name, thr in conformal.thresholds.items():
        print(f"  {name:18s}: {thr:.3f}")

    # ── Validation metrics (top-1 with calibrated probs) ──────────────────────
    pred_va = probs_va_cal.argmax(axis=1)
    print("\n--- Validation classification report (calibrated argmax) ---")
    print(classification_report(
        le.inverse_transform(y_va),
        le.inverse_transform(pred_va),
        target_names=list(le.classes_),
    ))

    # SAM headline
    sam_recall = (
        (le.inverse_transform(pred_va)[le.inverse_transform(y_va) == "SAM"] == "SAM").mean()
    )
    print(f"SAM recall: {sam_recall:.3f} (target ≥ 0.80)")

    # Conformal coverage on val
    pred_sets = conformal.predict_set(probs_va_cal)
    set_sizes = np.array([len(s) for s in pred_sets])
    covered = sum(
        1 for i, s in enumerate(pred_sets) if le.classes_[y_va[i]] in s
    )
    print(f"\nConformal — average set size: {set_sizes.mean():.2f}")
    print(f"            empirical coverage: {covered/len(pred_sets):.3f} (target ≥ {1-conformal.alpha:.2f})")
    print(f"            singleton sets: {(set_sizes == 1).mean():.3f}")
    print(f"            abstain (size != 1): {(set_sizes != 1).mean():.3f}")

    # ── Persist meta + calibration ────────────────────────────────────────────
    meta = {
        "feature_names": FEATURE_NAMES,
        "classes":       list(le.classes_),
        "labels_sorted": list(le.classes_),
    }
    meta_path = MODELS_DIR / "wasting_classifier_gbt_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nMeta → {meta_path.name}")

    cal_path = MODELS_DIR / "wasting_classifier_gbt_calibration.json"
    save_calibration(cal_path, T, conformal, classifier_kind="lightgbm")
    print(f"Calibration → {cal_path.name}")


if __name__ == "__main__":
    sys.exit(main())
