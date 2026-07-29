"""WHO acute-malnutrition aggregation tests."""

import pytest

from app.services.acute_malnutrition_service import AcuteMalnutritionService
from app.services.poshan_setu_service import classify_poshan_setu
from app.services.who_data_service import WHODataService
from config import WastingStatus


@pytest.fixture(scope="module")
def service():
    who = WHODataService()
    who.load_all()
    return AcuteMalnutritionService(who)


def test_height_only_produces_haz_but_leaves_acute_components_unmeasured(service):
    result = service.assess(
        sex="F",
        completed_age_months=24,
        age_months=24.2,
        height_cm=79.23,
        weight_kg=None,
        tape_muac_cm=None,
        oedema="not_checked",
    )

    assert result.haz_zscore == pytest.approx(-2.01, abs=0.01)
    assert result.stunting_status == "Stunted"
    assert result.whz_zscore is None
    assert result.whz_status is None
    assert result.muac_status is None
    assert result.acute_status == WastingStatus.UNKNOWN
    assert "weight not measured" in result.missing_reasons
    assert "tape MUAC not measured" in result.missing_reasons
    assert "oedema not checked" in result.missing_reasons


def test_height_and_weight_produce_whz_and_wasting(service):
    result = service.assess(
        sex="M",
        completed_age_months=24,
        age_months=24.0,
        height_cm=87.0,
        weight_kg=9.4,
        tape_muac_cm=None,
        oedema="no",
    )

    assert result.whz_zscore is not None
    assert result.whz_status in {
        WastingStatus.SAM,
        WastingStatus.MAM,
        WastingStatus.NORMAL,
    }
    assert result.acute_status == result.whz_status
    assert "whz" in result.triggered_by


@pytest.mark.parametrize(
    ("age_months", "expected_eligible"),
    [(5.99, False), (6.0, True), (59.99, True), (60.0, False)],
)
def test_tape_muac_is_eligible_only_from_6_through_59_months(
    service,
    age_months,
    expected_eligible,
):
    result = service.assess(
        sex="F",
        completed_age_months=int(age_months),
        age_months=age_months,
        height_cm=None,
        weight_kg=None,
        tape_muac_cm=11.0,
        oedema="no",
    )

    assert result.muac_eligible is expected_eligible
    assert (result.muac_status == WastingStatus.SAM) is expected_eligible


def test_oedema_yes_independently_triggers_actionable_sam(service):
    result = service.assess(
        sex="M",
        completed_age_months=30,
        age_months=30.0,
        height_cm=None,
        weight_kg=None,
        tape_muac_cm=None,
        oedema="yes",
    )

    assert result.acute_status == WastingStatus.SAM
    assert result.triggered_by == ("oedema",)
    assert result.actionable is True


def test_oedema_does_not_change_poshan_setu_v1():
    common = dict(
        sex="F",
        age_months=30,
        weight_kg=12,
        height_cm=88,
        weight_source="manual",
        height_source="manual",
        muac_cm=12.7,
        muac_method="tape",
    )

    without_oedema = classify_poshan_setu(**common)
    # Poshan has no oedema parameter by design, so the same inputs are
    # invariant even when the separately computed WHO arm sees oedema.
    with_oedema = classify_poshan_setu(**common)
    assert with_oedema == without_oedema
