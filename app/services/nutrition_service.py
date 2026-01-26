"""
Nutrition assessment service.

Computes Z-scores using the WHO LMS method and classifies nutritional status.

LMS Method:
  Z = ((measurement / M) ** L - 1) / (L * S)    when L != 0
  Z = ln(measurement / M) / S                     when L == 0

For Height-for-Age (HAZ), since the CSV data does not provide L/M/S,
we use linear interpolation between the provided Z-score boundary values.
"""
import math
from typing import List, Optional, Tuple

from config import ZSCORE_CLASSIFICATIONS
from app.services.who_data_service import WHODataService


class NutritionService:
    def __init__(self, who_data: WHODataService):
        self.who_data = who_data

    def compute_haz(
        self, sex: str, age_months: int, height_cm: float
    ) -> Optional[float]:
        """Compute Height-for-Age Z-score using boundary interpolation."""
        boundaries = self.who_data.get_haz_boundaries(sex, age_months)
        if boundaries is None:
            return None

        z_points: List[Tuple[float, float]] = [
            (-3, boundaries["z_minus_3"]),
            (-2, boundaries["z_minus_2"]),
            (-1, boundaries["z_minus_1"]),
            (0, boundaries["z_0"]),
            (1, boundaries["z_plus_1"]),
            (2, boundaries["z_plus_2"]),
            (3, boundaries["z_plus_3"]),
        ]
        return self._interpolate_zscore(height_cm, z_points)

    def compute_whz(
        self, sex: str, age_months: float, height_cm: float, weight_kg: float
    ) -> Optional[float]:
        """Compute Weight-for-Height Z-score using LMS method."""
        lms = self.who_data.get_wfh_lms(sex, height_cm, age_months)
        if lms is None:
            return None
        L, M, S = lms
        return self._lms_zscore(weight_kg, L, M, S)

    @staticmethod
    def _lms_zscore(measurement: float, L: float, M: float, S: float) -> float:
        """Compute Z-score from measurement using LMS parameters."""
        if M <= 0 or S <= 0:
            return 0.0
        if abs(L) < 1e-6:
            return math.log(measurement / M) / S
        return (((measurement / M) ** L) - 1) / (L * S)

    @staticmethod
    def _interpolate_zscore(
        value: float, z_points: List[Tuple[float, float]]
    ) -> float:
        """Linearly interpolate Z-score from boundary values.

        z_points is a sorted list of (z_score, reference_value) tuples.
        """
        # Below the lowest boundary: extrapolate
        if value <= z_points[0][1]:
            z0, v0 = z_points[0]
            z1, v1 = z_points[1]
            if v1 == v0:
                return z0
            return z0 - (v0 - value) / (v1 - v0)

        # Above the highest boundary: extrapolate
        if value >= z_points[-1][1]:
            z0, v0 = z_points[-2]
            z1, v1 = z_points[-1]
            if v1 == v0:
                return z1
            return z1 + (value - v1) / (v1 - v0)

        # Find the two bounding points and interpolate
        for i in range(len(z_points) - 1):
            z_low, v_low = z_points[i]
            z_high, v_high = z_points[i + 1]
            if v_low <= value <= v_high:
                if v_high == v_low:
                    return z_low
                fraction = (value - v_low) / (v_high - v_low)
                return z_low + fraction * (z_high - z_low)

        return 0.0

    def classify_haz(self, z: float) -> str:
        """Classify HAZ Z-score into nutritional status."""
        return self._classify(z, ZSCORE_CLASSIFICATIONS["haz"])

    def classify_whz(self, z: float) -> str:
        """Classify WHZ Z-score into nutritional status."""
        return self._classify(z, ZSCORE_CLASSIFICATIONS["whz"])

    @staticmethod
    def _classify(z: float, thresholds: dict) -> str:
        for (low, high), label in thresholds.items():
            if low <= z < high:
                return label
        return "Unknown"
