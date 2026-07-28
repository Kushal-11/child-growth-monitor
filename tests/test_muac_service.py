"""Safety and provenance tests for MUAC estimation."""

import pytest

from app.services.muac_service import MUACService
from config import WastingStatus


def landmark(muac_target: float, visibility: float = 0.95):
    """Construct proportional inputs yielding approximately target MUAC."""
    age, height = 24.0, 90.0
    median = MUACService._median_for_age(age, "M")
    # Vary shoulder ratio only; its exponent is 0.5.
    shoulder = height * 0.212 * (muac_target / median) ** 2
    return MUACService.estimate(
        age, "M", whz=None, upper_arm_length_cm=height * 0.160,
        shoulder_width_cm=shoulder, height_cm=height,
        landmark_visibility=visibility,
    )


@pytest.mark.parametrize("target,threshold", [(11.5, 11.5), (12.5, 12.5)])
def test_landmark_uncertainty_crossing_threshold_requires_confirmation(target, threshold):
    result = landmark(target)
    assert result.uncertainty_lower_cm < threshold <= result.uncertainty_upper_cm
    assert result.requires_confirmation
    assert "refer" in result.referral_guidance.lower()


def test_low_landmark_visibility_is_low_confidence_and_not_autonomous():
    result = landmark(13.5, visibility=0.25)
    assert result.confidence == 0.25
    assert result.requires_confirmation
    assert result.muac_status is None
    combined = MUACService.combine_with_whz_status(
        result.muac_status,
        WastingStatus.NORMAL,
        muac_method=result.muac_method,
        is_direct_measurement=result.is_direct_measurement,
    )
    assert combined.status == WastingStatus.NORMAL
    assert "muac" not in combined.triggered_by


def test_missing_whz_has_no_invented_muac():
    result = MUACService.estimate(24, "F", None)
    assert result.muac_cm is None
    assert result.confidence is None
    assert not result.is_direct_measurement
    assert result.requires_confirmation


def test_age_outside_supported_range_is_not_classified():
    result = MUACService.estimate(60, "M", -3, manual_muac_cm=11.0)
    assert not result.age_in_range
    assert result.muac_status is None
    assert result.is_direct_measurement


def test_direct_and_whz_derived_muac_have_distinct_provenance():
    direct = MUACService.estimate(24, "M", -3, manual_muac_cm=11.4)
    derived = MUACService.estimate(24, "M", -3)
    assert direct.is_direct_measurement and direct.confidence == 1.0
    assert not derived.is_direct_measurement
    assert derived.muac_method == "estimated_from_whz"
    assert derived.muac_status is None
    # Derived MUAC must not double count the WHZ arm.
    combined = MUACService.combine_with_whz_status(
        derived.muac_status,
        WastingStatus.SAM,
        muac_method=derived.muac_method,
        is_direct_measurement=derived.is_direct_measurement,
    )
    assert combined.triggered_by == ["whz"]


def test_noncanonical_status_is_rejected_even_when_muac_cannot_trigger():
    with pytest.raises(ValueError, match="canonical wasting status"):
        MUACService.combine_with_whz_status(
            "Requires Confirmation",
            WastingStatus.NORMAL,
            muac_method="landmark_estimated",
            is_direct_measurement=False,
        )
