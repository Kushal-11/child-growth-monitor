"""
ML inference module.

Loads trained models at startup and exposes a single predict() function used
by app/services/ml_service.py.

Falls back gracefully to None if models are not yet trained (e.g. first run
before python ml/train.py has been executed).
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"

from ml.models import FEATURE_NAMES, WASTING_LABELS


@dataclass
class WastingFeatures:
    """
    14 features extracted from child pose measurement(s).
    All measurements in cm (converted from pixels using height as scale reference).

    Features 0-9: derived from a frontal photo (always required).
    Features 10-13: AP depth from a side-view photo.
      - When a side photo is provided, these are real measurements.
      - When omitted, to_array() imputes them from lateral widths using
        Snyder 1975 AP/lateral ratios (chest ≈ 0.45×shoulder, abd ≈ 0.50×hip).
    """
    age_months: float
    sex_binary: int            # 1 = Male, 0 = Female
    height_cm: float
    shoulder_width_cm: float
    hip_width_cm: float
    torso_length_cm: float
    upper_arm_length_cm: float
    shoulder_height_ratio: float
    hip_height_ratio: float
    body_build_score: int      # -1 = slender, 0 = average, 1 = stocky
    # Side-view depth features (None → imputed in to_array)
    chest_depth_cm: Optional[float] = None
    abd_depth_cm: Optional[float] = None
    chest_depth_ratio: Optional[float] = None
    abd_depth_ratio: Optional[float] = None

    def to_array(self) -> np.ndarray:
        # Impute AP depth from lateral widths when side view is unavailable
        cd  = self.chest_depth_cm    if self.chest_depth_cm    is not None else self.shoulder_width_cm * 0.45
        ad  = self.abd_depth_cm      if self.abd_depth_cm      is not None else self.hip_width_cm * 0.50
        cdr = self.chest_depth_ratio if self.chest_depth_ratio is not None else cd / self.height_cm
        adr = self.abd_depth_ratio   if self.abd_depth_ratio   is not None else ad / self.height_cm
        return np.array([
            self.age_months,
            self.sex_binary,
            self.height_cm,
            self.shoulder_width_cm,
            self.hip_width_cm,
            self.torso_length_cm,
            self.upper_arm_length_cm,
            self.shoulder_height_ratio,
            self.hip_height_ratio,
            self.body_build_score,
            cd, ad, cdr, adr,
        ], dtype="float32")


@dataclass
class WastingPrediction:
    """Output from the ML wasting detection pipeline."""
    # Weight estimator output
    estimated_weight_kg: Optional[float]

    # Direct classifier output
    sam_probability:   float
    mam_probability:   float
    normal_probability: float
    risk_probability:  float
    overweight_probability: float

    # Final classification (from classifier)
    wasting_status: str   # SAM / MAM / Normal / Risk_Overweight / Overweight

    # Source metadata
    wasting_method: str   # "ml_classifier" when weight is ML-estimated


class WastingPredictor:
    """
    Loads trained Keras models + scaler and runs inference.
    Thread-safe for FastAPI (models loaded once at startup).
    """
    def __init__(self):
        self._we_model  = None   # weight estimator
        self._wc_model  = None   # wasting classifier
        self._scaler    = None
        self._le        = None
        self._available = False
        self._load()

    def _load(self):
        we_path     = MODELS_DIR / "weight_estimator.keras"
        wc_path     = MODELS_DIR / "wasting_classifier.keras"
        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        le_path     = MODELS_DIR / "label_encoder.pkl"

        if not all(p.exists() for p in [we_path, wc_path, scaler_path, le_path]):
            return   # models not yet trained; service will be unavailable

        try:
            import tensorflow as tf
            self._we_model = tf.keras.models.load_model(we_path)
            self._wc_model = tf.keras.models.load_model(wc_path)
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            with open(le_path, "rb") as f:
                self._le = pickle.load(f)
            self._available = True
        except Exception as e:
            print(f"[WastingPredictor] Could not load models: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def predict(self, features: WastingFeatures) -> Optional[WastingPrediction]:
        """Run both models and return a WastingPrediction, or None on error."""
        if not self._available:
            return None
        try:
            x = features.to_array().reshape(1, -1)
            x_s = self._scaler.transform(x).astype("float32")

            # Weight prediction
            est_weight = float(self._we_model.predict(x_s, verbose=0)[0, 0])
            est_weight = max(est_weight, 1.0)   # sanity floor

            # Class probabilities
            probs = self._wc_model.predict(x_s, verbose=0)[0]
            labels = list(self._le.classes_)
            prob_dict = dict(zip(labels, probs.tolist()))

            top_class = labels[int(probs.argmax())]

            return WastingPrediction(
                estimated_weight_kg    = round(est_weight, 2),
                sam_probability        = round(prob_dict.get("SAM",            0.0), 4),
                mam_probability        = round(prob_dict.get("MAM",            0.0), 4),
                normal_probability     = round(prob_dict.get("Normal",         0.0), 4),
                risk_probability       = round(prob_dict.get("Risk_Overweight",0.0), 4),
                overweight_probability = round(prob_dict.get("Overweight",     0.0), 4),
                wasting_status         = top_class,
                wasting_method         = "ml_classifier",
            )
        except Exception as e:
            print(f"[WastingPredictor] Prediction error: {e}")
            return None


# Module-level singleton (loaded once on import)
_predictor: Optional[WastingPredictor] = None


def get_predictor() -> WastingPredictor:
    global _predictor
    if _predictor is None:
        _predictor = WastingPredictor()
    return _predictor
