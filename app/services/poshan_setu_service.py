"""Canonical, provenance-aware Poshan Setu v1 classification."""
from dataclasses import dataclass
import math
from typing import Optional

from config import (
    POSHAN_BMI_THRESHOLDS,
    POSHAN_MUAC_MAX_AGE_MONTHS,
    POSHAN_MUAC_MIN_AGE_MONTHS,
    POSHAN_MUAC_NORMAL_MIN_CM,
    POSHAN_MUAC_SAM_MAX_CM,
)


CLASSIFICATION_METHOD = "poshan_setu_v1"
INDETERMINATE = "Indeterminate"
ELIGIBLE_BMI_SOURCES = frozenset({"manual", "reference_object"})
ELIGIBLE_MUAC_METHODS = frozenset({"manual", "tape"})
CANONICAL_SOURCES = frozenset(
    {
        "manual",
        "reference_object",
        "ml_estimated",
        "who_statistical",
        "whz_derived",
        "landmark_estimated",
        "unavailable",
    }
)
_SOURCE_ALIASES = {
    "none": "unavailable",
    "unknown": "unavailable",
    "who_median_estimated": "who_statistical",
    "estimated_from_whz": "whz_derived",
    "anthropometric": "landmark_estimated",
}
_SEVERITY = {"Normal": 0, "MAM": 1, "SAM": 2}


@dataclass(frozen=True)
class PoshanSetuResult:
    bmi: Optional[float]
    bmi_status: str
    muac_status: str
    final_status: str
    triggered_by: tuple[str, ...]
    classification_method: str
    rationale: str
    complete: bool


def normalize_source(source: Optional[str]) -> str:
    if source is None:
        return "unavailable"
    value = str(source).strip().lower()
    value = _SOURCE_ALIASES.get(value, value)
    return value if value in CANONICAL_SOURCES else "unavailable"


def normalize_muac_method(method: Optional[str]) -> str:
    if method is None:
        return "unavailable"
    value = str(method).strip().lower()
    if value == "tape":
        return "manual"
    return normalize_source(value)


def _is_finite_number(value: Optional[float]) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def classify_poshan_setu(
    *,
    sex: str,
    age_months: float,
    weight_kg: Optional[float],
    height_cm: Optional[float],
    weight_source: Optional[str],
    height_source: Optional[str],
    muac_cm: Optional[float],
    muac_method: Optional[str],
) -> PoshanSetuResult:
    """Classify only measurements eligible under the Poshan Setu v1 contract."""
    normalized_sex = str(sex).strip().upper()
    height_source_n = normalize_source(height_source)
    weight_source_n = normalize_source(weight_source)
    raw_muac_method = str(muac_method).strip().lower() if muac_method else ""
    muac_method_n = normalize_muac_method(muac_method)

    bmi = None
    bmi_status = INDETERMINATE
    bmi_reason = "BMI unavailable"
    bmi_values_valid = (
        _is_finite_number(height_cm)
        and float(height_cm) > 0
        and _is_finite_number(weight_kg)
        and float(weight_kg) > 0
    )
    bmi_sources_eligible = (
        height_source_n in ELIGIBLE_BMI_SOURCES
        and weight_source_n in ELIGIBLE_BMI_SOURCES
    )
    if normalized_sex not in ("M", "F"):
        bmi_reason = "BMI ineligible because sex is not M or F"
    elif not bmi_values_valid:
        bmi_reason = "BMI ineligible because height or weight is unavailable"
    elif not bmi_sources_eligible:
        bmi_reason = (
            "BMI ineligible because height/weight sources are "
            f"{height_source_n}/{weight_source_n}"
        )
    else:
        bmi = float(weight_kg) / ((float(height_cm) / 100.0) ** 2)
        sam_threshold, normal_threshold = POSHAN_BMI_THRESHOLDS[normalized_sex]
        bmi_status = (
            "SAM"
            if bmi < sam_threshold
            else "MAM"
            if bmi < normal_threshold
            else "Normal"
        )
        bmi_reason = f"eligible BMI is {bmi:.3f} ({bmi_status})"

    muac_status = INDETERMINATE
    muac_reason = "MUAC unavailable"
    age_eligible = (
        _is_finite_number(age_months)
        and POSHAN_MUAC_MIN_AGE_MONTHS
        <= float(age_months)
        < POSHAN_MUAC_MAX_AGE_MONTHS
    )
    muac_valid = _is_finite_number(muac_cm) and float(muac_cm) > 0
    muac_source_eligible = (
        raw_muac_method in ELIGIBLE_MUAC_METHODS or muac_method_n == "manual"
    )
    if not age_eligible:
        muac_reason = "MUAC ineligible outside age 6 to <60 months"
    elif not muac_valid:
        muac_reason = "MUAC ineligible because a tape measurement is unavailable"
    elif not muac_source_eligible:
        muac_reason = f"MUAC ineligible because source is {muac_method_n}"
    else:
        muac_status = (
            "SAM"
            if float(muac_cm) < POSHAN_MUAC_SAM_MAX_CM
            else "MAM"
            if float(muac_cm) < POSHAN_MUAC_NORMAL_MIN_CM
            else "Normal"
        )
        muac_reason = f"eligible MUAC is {float(muac_cm):.3f} cm ({muac_status})"

    complete = bmi_status != INDETERMINATE and muac_status != INDETERMINATE
    sam_triggers = tuple(
        name
        for name, status in (("bmi", bmi_status), ("muac", muac_status))
        if status == "SAM"
    )
    if sam_triggers:
        final_status = "SAM"
        triggered_by = sam_triggers
        final_reason = "at least one eligible component is SAM"
    elif complete:
        final_status = max(
            (bmi_status, muac_status), key=lambda status: _SEVERITY[status]
        )
        triggered_by = tuple(
            name
            for name, status in (("bmi", bmi_status), ("muac", muac_status))
            if status == final_status
        )
        final_reason = "both eligible components are available; severity maximum used"
    else:
        final_status = INDETERMINATE
        triggered_by = tuple(
            name
            for name, status in (("bmi", bmi_status), ("muac", muac_status))
            if status == "MAM"
        )
        final_reason = (
            "both components are required unless an eligible SAM signal is present"
        )

    return PoshanSetuResult(
        bmi=bmi,
        bmi_status=bmi_status,
        muac_status=muac_status,
        final_status=final_status,
        triggered_by=triggered_by,
        classification_method=CLASSIFICATION_METHOD,
        rationale=(
            f"Poshan Setu v1: {bmi_reason}; {muac_reason}; "
            f"final {final_status} because {final_reason}."
        ),
        complete=complete,
    )
