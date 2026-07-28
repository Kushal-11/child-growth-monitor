"""Regression tests for the canonical final wasting classification."""
import pytest

from config import WastingStatus, canonicalize_wasting_status
from app.services.muac_service import MUACService


def combine(muac, whz):
    """Combine a tape MUAC arm with WHZ using explicit provenance."""
    return MUACService.combine_with_whz_status(
        muac,
        whz,
        muac_method="manual" if muac is not None else None,
        is_direct_measurement=muac is not None,
    )


@pytest.mark.parametrize("legacy, expected", [
    ("Severe Acute Malnutrition (SAM)", WastingStatus.SAM),
    ("Moderate Acute Malnutrition (MAM)", WastingStatus.MAM),
    ("Possible Risk of Overweight", WastingStatus.RISK_OVERWEIGHT),
    ("Normal", WastingStatus.NORMAL),
    ("Overweight", WastingStatus.OVERWEIGHT),
    ("Obese", WastingStatus.OBESE),
])
def test_full_descriptive_legacy_whz_values_migrate(legacy, expected):
    assert canonicalize_wasting_status(legacy) is expected


def test_combiner_accepts_only_canonical_values():
    with pytest.raises(ValueError, match="canonical"):
        MUACService.combine_with_whz_status(
            WastingStatus.NORMAL,
            "Severe Acute Malnutrition (SAM)",
            muac_method="manual",
            is_direct_measurement=True,
        )


@pytest.mark.parametrize("muac, whz, trigger", [
    (WastingStatus.SAM, WastingStatus.NORMAL, "muac"),
    (WastingStatus.NORMAL, WastingStatus.SAM, "whz"),
    (WastingStatus.SAM, WastingStatus.MAM, "muac"),
    (WastingStatus.MAM, WastingStatus.SAM, "whz"),
    (WastingStatus.SAM, None, "muac"),
    (None, WastingStatus.SAM, "whz"),
])
def test_either_sam_arm_always_produces_final_sam(muac, whz, trigger):
    result = combine(muac, whz)
    assert result.status is WastingStatus.SAM
    assert trigger in result.triggered_by


def test_mam_arm_wins_disagreement_with_normal():
    result = combine(WastingStatus.MAM, WastingStatus.NORMAL)
    assert result.status is WastingStatus.MAM
    assert result.triggered_by == ["muac"]


@pytest.mark.parametrize("muac, whz, expected", [
    (None, WastingStatus.MAM, WastingStatus.MAM),
    (WastingStatus.SAM, None, WastingStatus.SAM),
    (None, None, WastingStatus.UNKNOWN),
])
def test_missing_arms(muac, whz, expected):
    assert combine(muac, whz).status is expected
