"""Tests for the measurement service hybrid approach."""
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import MagicMock
from app.services.measurement_service import (
    MeasurementService,
    MeasurementOutput,
    BodySegments,
)
from app.services.who_data_service import WHODataService
from config import get_anthropometric_ratios, ANTHROPOMETRIC_RATIOS


@pytest.fixture(scope="module")
def measurement_svc():
    return MeasurementService()


@pytest.fixture(scope="module")
def who_data():
    svc = WHODataService()
    svc.load_all()
    return svc


class TestAnthropometricRatios:
    """Tests for age-based anthropometric ratios."""
    
    def test_infant_ratios(self):
        """Infants have larger head ratio."""
        ratios = get_anthropometric_ratios(6)  # 6 months
        assert ratios["head_ratio"] == 0.28
        assert ratios["torso_ratio"] == 0.32
        assert ratios["leg_ratio"] == 0.40
    
    def test_toddler_ratios(self):
        """Toddlers have smaller head ratio than infants."""
        ratios = get_anthropometric_ratios(18)  # 18 months
        assert ratios["head_ratio"] == 0.25
        assert ratios["head_ratio"] < get_anthropometric_ratios(6)["head_ratio"]
    
    def test_preschool_ratios(self):
        """Preschool children have ratios approaching adult proportions."""
        ratios = get_anthropometric_ratios(48)  # 4 years
        assert ratios["head_ratio"] == 0.20
        assert ratios["leg_ratio"] == 0.50
    
    def test_ratios_sum_approximately_to_one(self):
        """Head + torso + legs should approximately equal total height."""
        for age_group, ratios in ANTHROPOMETRIC_RATIOS.items():
            total = ratios["head_ratio"] + ratios["torso_ratio"] + ratios["leg_ratio"]
            assert 0.95 <= total <= 1.05, f"Ratios for {age_group} don't sum to ~1.0"


class TestBodySegments:
    """Tests for the BodySegments dataclass."""
    
    def test_empty_segments(self):
        segments = BodySegments()
        assert segments.head_height_px is None
        assert segments.total_height_px is None
        assert segments.head_confidence == 0.0
    
    def test_segments_with_values(self):
        segments = BodySegments(
            head_height_px=100,
            torso_length_px=150,
            leg_length_px=250,
            total_height_px=500,
            head_confidence=0.9,
            torso_confidence=0.8,
            leg_confidence=0.7
        )
        assert segments.total_height_px == 500
        assert segments.head_confidence == 0.9


class TestAnthropometricEstimation:
    """Tests for _estimate_height_from_anthropometric_ratios."""
    
    def test_estimate_from_head(self, measurement_svc):
        """Test height estimation from head segment only."""
        # For a 24-month child with head_ratio = 0.25
        # If head is 100px, total should be 100/0.25 = 400px
        segments = BodySegments(
            head_height_px=100,
            head_confidence=0.9
        )
        result = measurement_svc._estimate_height_from_anthropometric_ratios(
            segments, age_months=18
        )
        assert result["combined_height_px"] is not None
        assert result["combined_height_px"] == pytest.approx(400, abs=10)
    
    def test_estimate_from_multiple_segments(self, measurement_svc):
        """Test estimation combining multiple segments."""
        # Using ratios for 24-48 months: head=0.22, torso=0.30, legs=0.48
        # If total height is 500px:
        # - head: 500 * 0.22 = 110
        # - torso: 500 * 0.30 = 150
        # - legs: 500 * 0.48 = 240
        segments = BodySegments(
            head_height_px=110,
            torso_length_px=150,
            leg_length_px=240,
            head_confidence=0.8,
            torso_confidence=0.9,
            leg_confidence=0.7
        )
        result = measurement_svc._estimate_height_from_anthropometric_ratios(
            segments, age_months=36
        )
        assert result["combined_height_px"] is not None
        # All segments point to ~500px total height
        assert 480 < result["combined_height_px"] < 520
        assert result["confidence"] > 0.5
    
    def test_no_segments(self, measurement_svc):
        """Test with no segment data."""
        segments = BodySegments()
        result = measurement_svc._estimate_height_from_anthropometric_ratios(
            segments, age_months=24
        )
        assert result["combined_height_px"] is None
        assert result["confidence"] == 0.0


class TestWHOStatisticalEstimation:
    """Tests for _estimate_height_from_who_statistics."""
    
    def test_estimate_returns_median(self, measurement_svc, who_data):
        """Primary estimate should be the WHO median for age."""
        segments = BodySegments()
        result = measurement_svc._estimate_height_from_who_statistics(
            segments, age_months=24, sex="M", who_data=who_data
        )
        assert result["height_cm"] is not None
        # 24-month male median should be ~87 cm
        assert 85 < result["height_cm"] < 90
        assert result["method"] == "who_statistical"
    
    def test_female_vs_male(self, measurement_svc, who_data):
        """Female median should be slightly less than male at same age."""
        segments = BodySegments()
        male = measurement_svc._estimate_height_from_who_statistics(
            segments, age_months=24, sex="M", who_data=who_data
        )
        female = measurement_svc._estimate_height_from_who_statistics(
            segments, age_months=24, sex="F", who_data=who_data
        )
        assert male["height_cm"] > female["height_cm"]
    
    def test_includes_range(self, measurement_svc, who_data):
        """Result should include height range."""
        segments = BodySegments()
        result = measurement_svc._estimate_height_from_who_statistics(
            segments, age_months=24, sex="M", who_data=who_data
        )
        assert result["height_range"] is not None
        min_h, max_h = result["height_range"]
        assert min_h < result["height_cm"] < max_h


class TestValidation:
    """Tests for _validate_height_estimate."""
    
    def test_valid_height(self, measurement_svc, who_data):
        """Height near median should be valid."""
        # Get median for 24-month male
        median = who_data.get_median_height_for_age("M", 24)
        result = measurement_svc._validate_height_estimate(
            median, age_months=24, sex="M", who_data=who_data
        )
        assert result["is_valid"] is True
        assert result["is_plausible"] is True
        assert abs(result["z_score_approx"]) < 0.5
        assert result["confidence"] > 0.8
    
    def test_outlier_height(self, measurement_svc, who_data):
        """Height >3 SD should be flagged as invalid."""
        # Very short for a 24-month male
        result = measurement_svc._validate_height_estimate(
            60.0, age_months=24, sex="M", who_data=who_data
        )
        assert result["is_valid"] is False
        assert abs(result["z_score_approx"]) > 3
        assert result["confidence"] < 0.5
    
    def test_extreme_height(self, measurement_svc, who_data):
        """Extremely unrealistic height should be implausible."""
        result = measurement_svc._validate_height_estimate(
            30.0, age_months=24, sex="M", who_data=who_data
        )
        assert result["is_plausible"] is False
        assert len(result["warnings"]) > 0


class TestBodyBuild:
    """Tests for _estimate_body_build."""
    
    def test_average_build(self, measurement_svc):
        """Test average body build classification."""
        # For a 36-month child, expected shoulder ratio ~0.21
        # Total height 500px, shoulder width ~105px
        segments = BodySegments(
            shoulder_width_px=105,
            total_height_px=500,
            torso_confidence=0.8
        )
        result = measurement_svc._estimate_body_build(segments, age_months=36)
        assert result["body_build"] == "average"
        assert result["weight_adjustment"] == 1.0
    
    def test_slender_build(self, measurement_svc):
        """Test slender body build (narrow shoulders)."""
        segments = BodySegments(
            shoulder_width_px=80,  # Narrow
            total_height_px=500,
            torso_confidence=0.8
        )
        result = measurement_svc._estimate_body_build(segments, age_months=36)
        assert result["body_build"] == "slender"
        assert result["weight_adjustment"] < 1.0
    
    def test_stocky_build(self, measurement_svc):
        """Test stocky body build (wide shoulders)."""
        segments = BodySegments(
            shoulder_width_px=140,  # Wide
            total_height_px=500,
            torso_confidence=0.8
        )
        result = measurement_svc._estimate_body_build(segments, age_months=36)
        assert result["body_build"] == "stocky"
        assert result["weight_adjustment"] > 1.0
    
    def test_unknown_build(self, measurement_svc):
        """Test unknown build when data missing."""
        segments = BodySegments()
        result = measurement_svc._estimate_body_build(segments, age_months=36)
        assert result["body_build"] == "unknown"
        assert result["weight_adjustment"] == 1.0


class TestMeasurementOutput:
    """Tests for the MeasurementOutput dataclass."""
    
    def test_default_values(self):
        output = MeasurementOutput()
        assert output.predicted_height_cm is None
        assert output.estimation_method == "none"
        assert output.weight_adjustment == 1.0
    
    def test_with_estimates(self):
        output = MeasurementOutput(
            predicted_height_cm=87.5,
            estimation_method="who_statistical",
            confidence_score=0.85,
            height_estimates={"who_statistical": {"height_cm": 87.5}}
        )
        assert output.predicted_height_cm == 87.5
        assert output.estimation_method == "who_statistical"
