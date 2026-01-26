"""Tests for nutrition Z-score computation and classification."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.services.nutrition_service import NutritionService
from app.services.who_data_service import WHODataService


@pytest.fixture(scope="module")
def nutrition_svc():
    who_data = WHODataService()
    who_data.load_all()
    return NutritionService(who_data)


class TestHAZComputation:
    def test_normal_female(self, nutrition_svc):
        """A female at 12 months with median height should have HAZ ≈ 0."""
        # Median height for F at 12 months is ~74.0 cm
        z = nutrition_svc.compute_haz("F", 12, 74.0)
        assert z is not None
        assert -0.5 < z < 0.5

    def test_stunted_male(self, nutrition_svc):
        """A short height should produce negative Z-score."""
        # At 24 months, male z_minus_2 height is about 81.7 cm
        z = nutrition_svc.compute_haz("M", 24, 79.0)
        assert z is not None
        assert z < -2

    def test_out_of_range(self, nutrition_svc):
        """Age out of range should return None."""
        z = nutrition_svc.compute_haz("M", 100, 100.0)
        assert z is None


class TestWHZComputation:
    def test_normal_weight(self, nutrition_svc):
        """Median weight for height should give WHZ ≈ 0."""
        # For a 30-month male at 85 cm, the median weight is roughly 11.8 kg
        z = nutrition_svc.compute_whz("M", 30, 85.0, 11.8)
        assert z is not None
        assert -1 < z < 1

    def test_low_weight(self, nutrition_svc):
        """Very low weight should produce negative Z-score."""
        z = nutrition_svc.compute_whz("F", 30, 85.0, 7.0)
        assert z is not None
        assert z < -2


class TestClassification:
    def test_haz_normal(self, nutrition_svc):
        assert nutrition_svc.classify_haz(0.0) == "Normal"

    def test_haz_stunted(self, nutrition_svc):
        assert nutrition_svc.classify_haz(-2.5) == "Stunted"

    def test_haz_severely_stunted(self, nutrition_svc):
        assert nutrition_svc.classify_haz(-3.5) == "Severely Stunted"

    def test_whz_normal(self, nutrition_svc):
        assert nutrition_svc.classify_whz(0.0) == "Normal"

    def test_whz_mam(self, nutrition_svc):
        assert nutrition_svc.classify_whz(-2.5) == "Moderate Acute Malnutrition (MAM)"

    def test_whz_sam(self, nutrition_svc):
        assert nutrition_svc.classify_whz(-3.5) == "Severe Acute Malnutrition (SAM)"

    def test_whz_overweight(self, nutrition_svc):
        assert nutrition_svc.classify_whz(2.5) == "Overweight"


class TestLMSFormula:
    def test_lms_at_median(self):
        """When measurement equals M, Z-score should be 0."""
        z = NutritionService._lms_zscore(10.0, L=-0.35, M=10.0, S=0.09)
        assert z == pytest.approx(0.0, abs=0.01)

    def test_lms_above_median(self):
        """Measurement above M should give positive Z-score."""
        z = NutritionService._lms_zscore(12.0, L=-0.35, M=10.0, S=0.09)
        assert z > 0

    def test_lms_below_median(self):
        """Measurement below M should give negative Z-score."""
        z = NutritionService._lms_zscore(8.0, L=-0.35, M=10.0, S=0.09)
        assert z < 0
