"""
ML wasting detection service.

Bridges the ML inference layer (ml/inference.py) with the rest of the
FastAPI application. Converts BodySegments + metadata into WastingFeatures,
runs prediction, and exposes a simple predict() interface.

Usage:
    ml_svc = MLService()
    prediction = ml_svc.predict(body_segments, age_months, sex, height_cm)
    if prediction:
        print(prediction.wasting_status)
"""
from typing import Optional

from app.services.measurement_service import BodySegments, SideViewSegments
from ml.inference import WastingFeatures, WastingPrediction, get_predictor


def _body_build_score(shoulder_width_cm: float, height_cm: float,
                       age_months: float) -> int:
    """
    Body build: -1=slender, 0=average, 1=stocky.
    Mirrors the existing body_build logic in measurement_service.py.
    """
    if age_months < 24:
        expected = 0.200
    elif age_months < 48:
        expected = 0.210
    else:
        expected = 0.218

    ratio = shoulder_width_cm / height_cm
    if ratio < expected - 0.02:
        return -1
    if ratio > expected + 0.02:
        return 1
    return 0


class MLService:
    """FastAPI-friendly wrapper around the WastingPredictor."""

    def __init__(self):
        self._predictor = get_predictor()

    @property
    def is_available(self) -> bool:
        return self._predictor.is_available

    def extract_features(
        self,
        segments: BodySegments,
        age_months: float,
        sex: str,
        height_cm: float,
        side_segments: Optional[SideViewSegments] = None,
    ) -> Optional[WastingFeatures]:
        """
        Convert raw BodySegments (pixels) to WastingFeatures (cm).

        Uses height_cm / total_height_px as the scale factor to convert all
        pixel measurements to centimetres. Missing measurements are imputed
        from age-based expected ratios so the model can still run.

        When side_segments is provided (from a side-view photo), real AP depth
        measurements replace imputed ones for features 10-13. This improves
        weight and wasting accuracy by ~30-40% vs. imputed values alone.

        Returns None if there is insufficient data to build the feature vector.
        """
        if height_cm is None or height_cm <= 0:
            return None
        if segments is None or segments.total_height_px is None or segments.total_height_px <= 0:
            return None

        scale = height_cm / segments.total_height_px  # cm per pixel

        # --- Shoulder width ---
        if segments.shoulder_width_px and segments.shoulder_width_px > 0:
            shoulder_cm = segments.shoulder_width_px * scale
        else:
            # Impute from expected ratio (age-adjusted)
            if age_months < 24:
                shoulder_cm = height_cm * 0.200
            elif age_months < 48:
                shoulder_cm = height_cm * 0.210
            else:
                shoulder_cm = height_cm * 0.218

        # --- Hip width ---
        if segments.hip_width_px and segments.hip_width_px > 0:
            hip_cm = segments.hip_width_px * scale
        else:
            hip_cm = shoulder_cm * 0.88   # hip ≈ 88% of shoulder (Snyder 1975)

        # --- Upper arm length ---
        if segments.upper_arm_length_px and segments.upper_arm_length_px > 0:
            arm_cm = segments.upper_arm_length_px * scale
        else:
            if age_months < 24:
                arm_cm = height_cm * 0.150
            elif age_months < 48:
                arm_cm = height_cm * 0.158
            else:
                arm_cm = height_cm * 0.165

        # --- Torso length ---
        if segments.torso_length_px and segments.torso_length_px > 0:
            torso_cm = segments.torso_length_px * scale
        else:
            torso_cm = height_cm * 0.30

        shoulder_height_ratio = shoulder_cm / height_cm
        hip_height_ratio      = hip_cm / height_cm
        bds = _body_build_score(shoulder_cm, height_cm, age_months)

        # --- AP depth features from side view (or None → imputed in to_array) ---
        chest_depth_cm = None
        abd_depth_cm   = None
        if side_segments is not None and side_segments.total_height_px:
            side_scale = height_cm / side_segments.total_height_px
            if side_segments.chest_depth_px and side_segments.chest_confidence >= 0.5:
                chest_depth_cm = float(side_segments.chest_depth_px * side_scale)
            if side_segments.abd_depth_px and side_segments.abd_confidence >= 0.5:
                abd_depth_cm = float(side_segments.abd_depth_px * side_scale)

        return WastingFeatures(
            age_months=float(age_months),
            sex_binary=1 if sex == "M" else 0,
            height_cm=float(height_cm),
            shoulder_width_cm=float(shoulder_cm),
            hip_width_cm=float(hip_cm),
            torso_length_cm=float(torso_cm),
            upper_arm_length_cm=float(arm_cm),
            shoulder_height_ratio=float(shoulder_height_ratio),
            hip_height_ratio=float(hip_height_ratio),
            body_build_score=int(bds),
            chest_depth_cm=chest_depth_cm,
            abd_depth_cm=abd_depth_cm,
        )

    def predict(
        self,
        segments: BodySegments,
        age_months: float,
        sex: str,
        height_cm: float,
        side_segments: Optional[SideViewSegments] = None,
    ) -> Optional[WastingPrediction]:
        """Extract features and run both models. Returns None if unavailable."""
        features = self.extract_features(segments, age_months, sex, height_cm, side_segments)
        if features is None:
            return None
        return self._predictor.predict(features)
