"""Poshan Setu v1 boundary and provenance tests."""
import json
from pathlib import Path

import pytest

from app.services.poshan_setu_service import classify_poshan_setu


CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "shared" / "poshan_setu_v1_cases.json")
    .read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_boundary_cases(case):
    has_bmi = case["weight_kg"] is not None and case["height_cm"] is not None
    result = classify_poshan_setu(
        sex=case["sex"],
        age_months=case["age_months"],
        weight_kg=case["weight_kg"],
        height_cm=case["height_cm"],
        weight_source="manual" if has_bmi else "unavailable",
        height_source="manual" if has_bmi else "unavailable",
        muac_cm=case["muac_cm"],
        muac_method="manual" if case["muac_cm"] is not None else "unavailable",
    )
    assert result.bmi_status == case["expected_bmi_status"]
    assert result.muac_status == case["expected_muac_status"]
    assert result.final_status == case["expected_final_status"]
    assert result.classification_method == "poshan_setu_v1"


def test_estimated_values_cannot_certify_normal():
    result = classify_poshan_setu(
        sex="F",
        age_months=36,
        weight_kg=14,
        height_cm=100,
        weight_source="ml_estimated",
        height_source="who_statistical",
        muac_cm=14,
        muac_method="whz_derived",
    )
    assert result.bmi is None
    assert result.final_status == "Indeterminate"
    assert result.complete is False


def test_mam_requires_other_component_but_sam_does_not():
    mam = classify_poshan_setu(
        sex="M",
        age_months=36,
        weight_kg=13.2,
        height_cm=100,
        weight_source="manual",
        height_source="manual",
        muac_cm=None,
        muac_method="unavailable",
    )
    sam = classify_poshan_setu(
        sex="M",
        age_months=36,
        weight_kg=12.9,
        height_cm=100,
        weight_source="manual",
        height_source="manual",
        muac_cm=None,
        muac_method="unavailable",
    )
    assert mam.final_status == "Indeterminate"
    assert mam.triggered_by == ("bmi",)
    assert sam.final_status == "SAM"
    assert sam.triggered_by == ("bmi",)
