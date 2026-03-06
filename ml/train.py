"""
Train wasting-detection ML models on the synthetic WHO-derived dataset.

Training data: data/training_data/synthetic_dataset.csv
Trained models saved to:
  data/models/weight_estimator.keras
  data/models/wasting_classifier.keras
  data/models/weight_estimator.tflite      (for Android/iOS)
  data/models/wasting_classifier.tflite    (for Android/iOS)
  data/models/feature_scaler.pkl           (StandardScaler for inference)
  data/models/label_encoder.pkl            (LabelEncoder for class names)

Run:  python ml/train.py
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

from ml.models import FEATURE_NAMES, WASTING_LABELS, build_weight_estimator, build_wasting_classifier


def load_data():
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURE_NAMES].values.astype("float32")
    y_weight = df["weight_kg"].values.astype("float32")

    # Encode labels to integers (sorted alphabetically = sklearn default)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(WASTING_LABELS)
    y_class = le.transform(df["label"]).astype("int32")

    return X, y_weight, y_class, le, df


def scale_features(X_train, X_val):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    return X_train_s.astype("float32"), X_val_s.astype("float32"), scaler


def compute_class_weights(y_class):
    """
    Compute class weights to handle imbalance.
    SAM is a dangerous false-negative — overweight it by a factor of 2 on top
    of the standard inverse-frequency weighting.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_class)
    weights = compute_class_weight("balanced", classes=classes, y=y_class)
    cw_dict = dict(zip(classes.tolist(), weights.tolist()))

    # Extra penalty for missing SAM (false negatives are clinically dangerous)
    from ml.models import WASTING_LABELS
    sam_idx = sorted(WASTING_LABELS).index("SAM")
    if sam_idx in cw_dict:
        cw_dict[sam_idx] *= 2.0

    return cw_dict


def export_tflite(model, out_path: Path):
    """Convert Keras model to TFLite with default optimizations (quantization)."""
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    size_kb = out_path.stat().st_size // 1024
    print(f"  TFLite → {out_path.name}  ({size_kb} KB)")


def train_weight_estimator(X_train, y_train, X_val, y_val):
    import tensorflow as tf

    model = build_weight_estimator()
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                          monitor="val_mae"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        callbacks=cb,
        verbose=0,
    )
    val_mae = min(history.history["val_mae"])
    print(f"  Weight estimator  best val MAE: {val_mae:.3f} kg")
    return model


def train_wasting_classifier(X_train, y_train, X_val, y_val, cw_dict):
    import tensorflow as tf

    model = build_wasting_classifier()
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                          monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        class_weight=cw_dict,
        callbacks=cb,
        verbose=0,
    )
    val_acc = max(history.history["val_accuracy"])
    print(f"  Wasting classifier best val accuracy: {val_acc:.3f}")
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading synthetic dataset...")
    X, y_weight, y_class, le, df = load_data()
    print(f"  {len(X)} samples, {X.shape[1]} features")

    # Train/val split (80/20, stratified on class label)
    from sklearn.model_selection import train_test_split
    X_train, X_val, yw_train, yw_val, yc_train, yc_val = train_test_split(
        X, y_weight, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    X_train_s, X_val_s, scaler = scale_features(X_train, X_val)
    cw_dict = compute_class_weights(yc_train)

    print("\nTraining weight estimator...")
    we_model = train_weight_estimator(X_train_s, yw_train, X_val_s, yw_val)

    print("Training wasting classifier...")
    wc_model = train_wasting_classifier(X_train_s, yc_train, X_val_s, yc_val, cw_dict)

    # Save Keras models
    we_path = MODELS_DIR / "weight_estimator.keras"
    wc_path = MODELS_DIR / "wasting_classifier.keras"
    we_model.save(we_path)
    wc_model.save(wc_path)
    print(f"\nSaved Keras models → {MODELS_DIR}")

    # Save preprocessing artifacts
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    le_path     = MODELS_DIR / "label_encoder.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved scaler → {scaler_path.name}")
    print(f"Saved label encoder → {le_path.name}")

    # Export TFLite
    print("\nExporting TFLite models...")
    export_tflite(we_model, MODELS_DIR / "weight_estimator.tflite")
    export_tflite(wc_model, MODELS_DIR / "wasting_classifier.tflite")

    print("\nDone. Run `python ml/evaluate.py` to see per-class metrics.")


if __name__ == "__main__":
    sys.exit(main())
