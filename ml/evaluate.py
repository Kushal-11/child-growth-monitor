"""
Evaluate trained wasting detection models.

Prints:
  - Per-class precision / recall / F1 (weight estimator and classifier)
  - Confusion matrix
  - Weight estimation MAE by wasting category
  - SAM recall (most safety-critical metric — false negatives are dangerous)

Run:  python ml/evaluate.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
DATA_CSV   = DATA_DIR / "training_data" / "synthetic_dataset.csv"

from ml.models import FEATURE_NAMES, WASTING_LABELS


def _load_artifacts():
    import tensorflow as tf
    we_model = tf.keras.models.load_model(MODELS_DIR / "weight_estimator.keras")
    wc_model = tf.keras.models.load_model(MODELS_DIR / "wasting_classifier.keras")
    with open(MODELS_DIR / "feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return we_model, wc_model, scaler, le


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error

    df = pd.read_csv(DATA_CSV)
    X  = df[FEATURE_NAMES].values.astype("float32")
    y_weight = df["weight_kg"].values.astype("float32")

    we_model, wc_model, scaler, le = _load_artifacts()
    y_class = le.transform(df["label"]).astype("int32")

    _, X_val, _, yw_val, _, yc_val = train_test_split(
        X, y_weight, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    X_val_s = scaler.transform(X_val).astype("float32")

    # --- Weight estimator ---
    pred_weight = we_model.predict(X_val_s, verbose=0).flatten()
    mae_overall  = mean_absolute_error(yw_val, pred_weight)
    print(f"Weight estimator MAE (overall): {mae_overall:.3f} kg\n")

    # MAE by label
    val_labels = le.inverse_transform(yc_val)
    for lbl in sorted(set(val_labels)):
        mask = val_labels == lbl
        mae_lbl = mean_absolute_error(yw_val[mask], pred_weight[mask])
        print(f"  MAE for {lbl:18s}: {mae_lbl:.3f} kg")

    # --- Wasting classifier ---
    probs   = wc_model.predict(X_val_s, verbose=0)
    pred_cls = probs.argmax(axis=1)
    true_labels = le.inverse_transform(yc_val)
    pred_labels = le.inverse_transform(pred_cls)

    print("\n--- Classification Report ---")
    print(classification_report(true_labels, pred_labels, target_names=sorted(set(true_labels))))

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=sorted(WASTING_LABELS))
    cm_df = pd.DataFrame(cm, index=sorted(WASTING_LABELS), columns=sorted(WASTING_LABELS))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm_df.to_string())

    # Highlight SAM recall (most safety-critical)
    sam_mask = true_labels == "SAM"
    if sam_mask.any():
        sam_recall = (pred_labels[sam_mask] == "SAM").mean()
        print(f"\n*** SAM recall: {sam_recall:.3f} (target ≥ 0.80) ***")
        if sam_recall < 0.80:
            print("    WARNING: SAM recall is below 0.80 — consider adjusting"
                  " class weights or resampling in train.py")
    else:
        print("No SAM samples in validation set.")


if __name__ == "__main__":
    sys.exit(main())
