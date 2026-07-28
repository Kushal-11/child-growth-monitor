"""
Nutrition assessment service.

Computes Z-scores using the WHO LMS method and classifies nutritional status.

LMS Method:
  Z = ((measurement / M) ** L - 1) / (L * S)    when L != 0
  Z = ln(measurement / M) / S                     when L == 0

Height-for-Age (HAZ) and Weight-for-Height (WHZ) both use authoritative
WHO Excel LMS parameters.
"""
import math
from typing import Optional

from config import WastingStatus, ZSCORE_CLASSIFICATIONS
from app.services.who_data_service import WHODataService


class NutritionService:
    def __init__(self, who_data: WHODataService):
        self.who_data = who_data

    def compute_haz(
        self, sex: str, age_months: int, height_cm: float
    ) -> Optional[float]:
        """Compute Height-for-Age Z-score using WHO LMS parameters."""
        lms = self.who_data.get_haz_lms(sex, age_months)
        if lms is None or height_cm <= 0:
            return None
        return self._lms_zscore(height_cm, *lms)

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
        if measurement <= 0 or M <= 0 or S <= 0:
            raise ValueError("LMS inputs require positive measurement, M, and S")
        if abs(L) < 1e-6:
            return math.log(measurement / M) / S
        return (((measurement / M) ** L) - 1) / (L * S)

    def classify_haz(self, z: float) -> str:
        """Classify HAZ Z-score into nutritional status."""
        return self._classify(z, ZSCORE_CLASSIFICATIONS["haz"])

    def classify_whz(self, z: float) -> WastingStatus:
        """Classify WHZ Z-score into nutritional status."""
        return self._classify(z, ZSCORE_CLASSIFICATIONS["whz"], WastingStatus.UNKNOWN)

    @staticmethod
    def _classify(z: float, thresholds: dict, unknown="Unknown"):
        for (low, high), label in thresholds.items():
            if low <= z < high:
                return label
        return unknown
