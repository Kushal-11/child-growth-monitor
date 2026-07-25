"""
ML inference module.

Loads the tracked TFLite runtime bundle on first prediction and exposes a
single predict() function used by app/services/ml_service.py.  Backend and
Flutter consume byte-identical models/scaler metadata.

Falls back gracefully to None if models are not yet trained (e.g. first run
before python ml/train.py has been executed).
"""
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from config import (
    ML_CLASSIFIER_TFLITE,
    ML_LABEL_ENCODER_PATH,
    ML_MODEL_MANIFEST_PATH,
    ML_SCALER_PATH,
    ML_WEIGHT_ESTIMATOR_TFLITE,
)
from ml.models import FEATURE_NAMES, WASTING_LABELS

RUNTIME_PATHS = (
    ML_WEIGHT_ESTIMATOR_TFLITE,
    ML_CLASSIFIER_TFLITE,
    ML_SCALER_PATH,
    ML_LABEL_ENCODER_PATH,
    ML_MODEL_MANIFEST_PATH,
)

MIN_PLAUSIBLE_WEIGHT_KG = 0.5
MAX_PLAUSIBLE_WEIGHT_KG = 40.0
PROBABILITY_SUM_TOLERANCE = 1e-3


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_artifacts_present() -> bool:
    """Cheap availability check that does not import TensorFlow."""

    return all(path.is_file() for path in RUNTIME_PATHS)


def validate_raw_outputs(
    estimated_weight_kg: float,
    probabilities: Sequence[float],
    labels: Sequence[str],
) -> tuple[float, dict[str, float], str]:
    """Validate unmodified TFLite outputs and select the highest class.

    This is deliberately shared by normal inference and golden parity tests.
    Invalid model output is rejected instead of being clamped, normalized, or
    rounded into a value that appears clinically usable.
    """

    weight = float(estimated_weight_kg)
    if not math.isfinite(weight):
        raise ValueError("Weight model output is non-finite")
    if not MIN_PLAUSIBLE_WEIGHT_KG <= weight <= MAX_PLAUSIBLE_WEIGHT_KG:
        raise ValueError(
            "Weight model output is outside the plausible "
            f"{MIN_PLAUSIBLE_WEIGHT_KG:g}-{MAX_PLAUSIBLE_WEIGHT_KG:g} kg range"
        )

    if len(probabilities) != len(labels) or not labels:
        raise ValueError("Classifier output length differs from label count")
    values = [float(value) for value in probabilities]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Classifier output contains non-finite probabilities")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Classifier output contains probabilities outside [0, 1]")
    probability_sum = sum(values)
    if abs(probability_sum - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            "Classifier probabilities do not sum to 1 within "
            f"{PROBABILITY_SUM_TOLERANCE:g}"
        )

    top_index = max(range(len(values)), key=values.__getitem__)
    probability_by_label = dict(zip(labels, values))
    return weight, probability_by_label, str(labels[top_index])


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
    model_version: str = "unknown"
    training_data: str = "synthetic"
    non_clinical: bool = True


class WastingPredictor:
    """
    Loads the versioned TFLite/JSON runtime bundle and runs inference.

    The interpreters are protected by a lock because a TFLite Interpreter
    instance cannot safely be invoked concurrently.
    """
    def __init__(self):
        self._weight_interpreter = None
        self._classifier_interpreter = None
        self._weight_input = None
        self._weight_output = None
        self._classifier_input = None
        self._classifier_output = None
        self._mean = None
        self._scale = None
        self._labels: list[str] = []
        self._model_version = "unknown"
        self._training_data = "unknown"
        self._non_clinical = True
        self._available = False
        self._load_error: Optional[str] = None
        self._invoke_lock = threading.Lock()
        self._load()

    @staticmethod
    def _interpreter_class():
        """Prefer a small standalone runtime, with TensorFlow as fallback."""

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

    @staticmethod
    def _validate_tensor(detail: dict, expected_shape: tuple[int, ...], name: str) -> None:
        shape = tuple(int(v) for v in detail["shape"])
        if shape != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}, got {shape}")
        if detail["dtype"] != np.float32:
            raise ValueError(f"{name} dtype must be float32, got {detail['dtype']}")

    def _load(self):
        if not runtime_artifacts_present():
            missing = [str(path) for path in RUNTIME_PATHS if not path.is_file()]
            self._load_error = f"Missing ML runtime artifacts: {', '.join(missing)}"
            return

        try:
            manifest = json.loads(ML_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
            if manifest.get("feature_schema_version") != 1:
                raise ValueError(
                    "Unsupported feature_schema_version "
                    f"{manifest.get('feature_schema_version')!r}"
                )
            if manifest.get("feature_count") != len(FEATURE_NAMES):
                raise ValueError(
                    f"Manifest feature_count must be {len(FEATURE_NAMES)}"
                )
            if manifest.get("feature_names") != FEATURE_NAMES:
                raise ValueError("Manifest feature order does not match ml.models")

            artifact_paths = {
                "weight_estimator.tflite": ML_WEIGHT_ESTIMATOR_TFLITE,
                "wasting_classifier.tflite": ML_CLASSIFIER_TFLITE,
                "feature_scaler.json": ML_SCALER_PATH,
                "label_encoder.json": ML_LABEL_ENCODER_PATH,
            }
            for filename, path in artifact_paths.items():
                artifact_record = manifest.get("artifacts", {}).get(filename, {})
                expected_hash = artifact_record.get("sha256")
                expected_size = artifact_record.get("size_bytes")
                if (
                    not isinstance(expected_hash, str)
                    or len(expected_hash) != 64
                    or _sha256(path) != expected_hash
                ):
                    raise ValueError(f"SHA-256 mismatch for {filename}")
                if expected_size != path.stat().st_size:
                    raise ValueError(f"Size mismatch for {filename}")

            evaluation = manifest.get("evaluation", {})
            if (
                evaluation.get("evaluation_contract_version") != 2
                or evaluation.get("engine") != "tensorflow_lite"
                or evaluation.get("sam_recall_floor_met") is not True
                or evaluation.get("non_clinical") is not True
                or isinstance(evaluation.get("sam_sample_count"), bool)
                or not isinstance(evaluation.get("sam_sample_count"), int)
                or evaluation["sam_sample_count"] <= 0
                or isinstance(evaluation.get("invalid_prediction_count"), bool)
                or not isinstance(
                    evaluation.get("invalid_prediction_count"), int
                )
                or evaluation["invalid_prediction_count"] != 0
            ):
                raise ValueError("Manifest does not contain a passing TFLite safety gate")
            for metric_name in (
                "weight_mae_kg",
                "classification_accuracy",
                "sam_recall",
                "mam_recall",
                "mam_precision",
            ):
                metric = evaluation.get(metric_name)
                if (
                    isinstance(metric, bool)
                    or not isinstance(metric, (int, float))
                    or not math.isfinite(metric)
                ):
                    raise ValueError(
                        f"Manifest evaluation metric {metric_name} is not finite"
                    )
            declared_floor = evaluation.get("sam_recall_floor")
            if (
                isinstance(declared_floor, bool)
                or not isinstance(declared_floor, (int, float))
                or not math.isfinite(declared_floor)
                or float(declared_floor) != 0.80
            ):
                raise ValueError("Manifest does not declare the required SAM floor")
            if float(evaluation["sam_recall"]) < float(declared_floor):
                raise ValueError("Manifest SAM recall is below its declared floor")
            evaluated_artifacts = evaluation.get("evaluated_artifacts", {})
            for filename, artifact_record in manifest["artifacts"].items():
                if evaluated_artifacts.get(filename) != artifact_record:
                    raise ValueError(
                        f"Evaluation report is not bound to {filename}"
                    )

            scaler = json.loads(ML_SCALER_PATH.read_text(encoding="utf-8"))
            if scaler.get("feature_names") != FEATURE_NAMES:
                raise ValueError("Scaler feature order does not match ml.models")
            self._mean = np.asarray(scaler["mean"], dtype=np.float32)
            self._scale = np.asarray(scaler["scale"], dtype=np.float32)
            if self._mean.shape != (len(FEATURE_NAMES),):
                raise ValueError("Scaler mean must contain exactly 14 values")
            if self._scale.shape != (len(FEATURE_NAMES),):
                raise ValueError("Scaler scale must contain exactly 14 values")
            if not np.isfinite(self._mean).all() or not np.isfinite(self._scale).all():
                raise ValueError("Scaler contains non-finite values")
            if (self._scale <= 0).any():
                raise ValueError("Scaler scale values must be positive")

            label_data = json.loads(
                ML_LABEL_ENCODER_PATH.read_text(encoding="utf-8")
            )
            self._labels = [str(value) for value in label_data.get("classes", [])]
            if self._labels != list(manifest.get("labels", [])):
                raise ValueError("Label encoder and manifest label order differ")
            if self._labels != sorted(WASTING_LABELS):
                raise ValueError("Runtime label order does not match training contract")
            if label_data.get("model_version") != manifest.get("model_version"):
                raise ValueError("Label encoder and manifest model versions differ")

            Interpreter = self._interpreter_class()
            self._weight_interpreter = Interpreter(
                model_path=str(ML_WEIGHT_ESTIMATOR_TFLITE)
            )
            self._classifier_interpreter = Interpreter(
                model_path=str(ML_CLASSIFIER_TFLITE)
            )
            self._weight_interpreter.allocate_tensors()
            self._classifier_interpreter.allocate_tensors()
            self._weight_input = self._weight_interpreter.get_input_details()[0]
            self._weight_output = self._weight_interpreter.get_output_details()[0]
            self._classifier_input = self._classifier_interpreter.get_input_details()[0]
            self._classifier_output = self._classifier_interpreter.get_output_details()[0]
            self._validate_tensor(
                self._weight_input, (1, len(FEATURE_NAMES)), "weight input"
            )
            self._validate_tensor(self._weight_output, (1, 1), "weight output")
            self._validate_tensor(
                self._classifier_input,
                (1, len(FEATURE_NAMES)),
                "classifier input",
            )
            self._validate_tensor(
                self._classifier_output,
                (1, len(self._labels)),
                "classifier output",
            )

            self._model_version = str(manifest["model_version"])
            self._training_data = str(manifest.get("training_data", "unknown"))
            self._non_clinical = bool(
                manifest.get("evaluation", {}).get("non_clinical", True)
            )
            self._available = True
        except Exception as e:
            self._load_error = str(e)
            print(f"[WastingPredictor] Could not load runtime bundle: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def predict(self, features: WastingFeatures) -> Optional[WastingPrediction]:
        """Run both models and return a WastingPrediction, or None on error."""
        if not self._available:
            return None
        try:
            x = features.to_array().reshape(1, -1)
            if not np.isfinite(x).all():
                raise ValueError("Feature vector contains non-finite values")
            x_s = ((x - self._mean) / self._scale).astype(np.float32)
            if not np.isfinite(x_s).all():
                raise ValueError("Scaled feature vector contains non-finite values")

            with self._invoke_lock:
                self._weight_interpreter.set_tensor(
                    self._weight_input["index"], x_s
                )
                self._weight_interpreter.invoke()
                est_weight = float(
                    self._weight_interpreter.get_tensor(
                        self._weight_output["index"]
                    )[0, 0]
                )

                self._classifier_interpreter.set_tensor(
                    self._classifier_input["index"], x_s
                )
                self._classifier_interpreter.invoke()
                probs = self._classifier_interpreter.get_tensor(
                    self._classifier_output["index"]
                )[0].astype(np.float64)

            est_weight, prob_dict, top_class = validate_raw_outputs(
                est_weight,
                probs.tolist(),
                self._labels,
            )

            return WastingPrediction(
                estimated_weight_kg    = est_weight,
                sam_probability        = prob_dict.get("SAM",             0.0),
                mam_probability        = prob_dict.get("MAM",             0.0),
                normal_probability     = prob_dict.get("Normal",          0.0),
                risk_probability       = prob_dict.get("Risk_Overweight", 0.0),
                overweight_probability = prob_dict.get("Overweight",      0.0),
                wasting_status         = top_class,
                wasting_method         = "tflite_classifier",
                model_version          = self._model_version,
                training_data          = self._training_data,
                non_clinical           = self._non_clinical,
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
