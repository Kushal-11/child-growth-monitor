"""Canonical Poshan Setu v1 classification.

The classifier in this module is intentionally pure: it has no database,
network, WHO-table, or ML dependencies.  It classifies only measurements whose
provenance is eligible under ``docs/POSHAN_SETU_V1.md``.
"""
from dataclasses import dataclass
import math
from typing import Optional


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
    """Structured result returned by the canonical classifier."""

    bmi: Optional[float]
    bmi_status: str
    muac_status: str
    final_status: str
    triggered_by: tuple[str, ...]
    classification_method: str
    rationale: str
    complete: bool


def normalize_source(source: Optional[str]) -> str:
    """Return a canonical measurement-source value."""
    if source is None:
        return "unavailable"
    value = str(source).strip().lower()
    value = _SOURCE_ALIASES.get(value, value)
    return value if value in CANONICAL_SOURCES else "unavailable"


def normalize_muac_method(method: Optional[str]) -> str:
    """Normalize legacy MUAC method names to canonical provenance values."""
    if method is None:
        return "unavailable"
    value = str(method).strip().lower()
    if value == "tape":
        return "manual"
    return normalize_source(value)


def is_finite_number(value: Optional[float]) -> bool:
    """Whether ``value`` can safely participate in a numeric calculation."""
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
    """Classify eligible BMI and MUAC measurements using Poshan Setu v1.

    A component is ``Indeterminate`` when its raw value, provenance, or
    eligibility is missing.  A single eligible SAM signal is sufficient for a
    final SAM result; otherwise both components must be eligible before a
    definitive MAM or Normal result can be issued.
    """
    normalized_sex = str(sex).strip().upper()
    height_source_n = normalize_source(height_source)
    weight_source_n = normalize_source(weight_source)
    raw_muac_method = (
        str(muac_method).strip().lower() if muac_method is not None else ""
    )
    muac_method_n = normalize_muac_method(muac_method)

    bmi = None
    bmi_status = INDETERMINATE
    bmi_reason = "BMI unavailable"
    reliable_bmi_sources = (
        height_source_n in ELIGIBLE_BMI_SOURCES
        and weight_source_n in ELIGIBLE_BMI_SOURCES
    )
    valid_bmi_values = (
        is_finite_number(height_cm)
        and float(height_cm) > 0
        and is_finite_number(weight_kg)
        and float(weight_kg) > 0
    )

    if normalized_sex not in ("M", "F"):
        bmi_reason = "BMI ineligible because sex is not M or F"
    elif not valid_bmi_values:
        bmi_reason = "BMI ineligible because height or weight is unavailable"
    elif not reliable_bmi_sources:
        bmi_reason = (
            "BMI ineligible because height/weight sources are "
            f"{height_source_n}/{weight_source_n}"
        )
    else:
        bmi = float(weight_kg) / ((float(height_cm) / 100.0) ** 2)
        sam_threshold, normal_threshold = (
            (13.0, 13.7) if normalized_sex == "M" else (12.8, 13.5)
        )
        if bmi < sam_threshold:
            bmi_status = "SAM"
        elif bmi < normal_threshold:
            bmi_status = "MAM"
        else:
            bmi_status = "Normal"
        bmi_reason = f"eligible BMI is {bmi:.3f} ({bmi_status})"

    muac_status = INDETERMINATE
    muac_reason = "MUAC unavailable"
    valid_age = is_finite_number(age_months) and 6.0 <= float(age_months) < 60.0
    valid_muac = is_finite_number(muac_cm) and float(muac_cm) > 0
    # ``tape`` is accepted as an input alias even though it is stored as manual.
    eligible_muac_method = (
        raw_muac_method in ELIGIBLE_MUAC_METHODS or muac_method_n == "manual"
    )

    if not valid_age:
        muac_reason = "MUAC ineligible outside age 6 to <60 months"
    elif not valid_muac:
        muac_reason = "MUAC ineligible because a tape measurement is unavailable"
    elif not eligible_muac_method:
        muac_reason = f"MUAC ineligible because source is {muac_method_n}"
    else:
        if float(muac_cm) < 11.5:
            muac_status = "SAM"
        elif float(muac_cm) < 12.5:
            muac_status = "MAM"
        else:
            muac_status = "Normal"
        muac_reason = f"eligible MUAC is {float(muac_cm):.3f} cm ({muac_status})"

    complete = bmi_status != INDETERMINATE and muac_status != INDETERMINATE

    sam_triggers = tuple(
        component
        for component, status in (("bmi", bmi_status), ("muac", muac_status))
        if status == "SAM"
    )
    if sam_triggers:
        final_status = "SAM"
        triggered_by = sam_triggers
        final_reason = "at least one eligible component is SAM"
    elif complete:
        final_status = max(
            (bmi_status, muac_status),
            key=lambda status: _SEVERITY[status],
        )
        triggered_by = tuple(
            component
            for component, status in (("bmi", bmi_status), ("muac", muac_status))
            if status == final_status
        )
        final_reason = "both eligible components are available; severity maximum used"
    else:
        final_status = INDETERMINATE
        triggered_by = tuple(
            component
            for component, status in (("bmi", bmi_status), ("muac", muac_status))
            if status == "MAM"
        )
        final_reason = (
            "both components are required unless an eligible SAM signal is present"
        )

    rationale = (
        f"Poshan Setu v1: {bmi_reason}; {muac_reason}; "
        f"final {final_status} because {final_reason}."
    )
    return PoshanSetuResult(
        bmi=bmi,
        bmi_status=bmi_status,
        muac_status=muac_status,
        final_status=final_status,
        triggered_by=triggered_by,
        classification_method=CLASSIFICATION_METHOD,
        rationale=rationale,
        complete=complete,
    )
