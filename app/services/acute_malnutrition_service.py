"""Separately named WHO measured-report aggregation."""

from dataclasses import dataclass
from typing import Optional

from app.services.nutrition_service import NutritionService
from app.services.who_data_service import WHODataService
from config import (
    POSHAN_MUAC_MAX_AGE_MONTHS,
    POSHAN_MUAC_MIN_AGE_MONTHS,
    POSHAN_MUAC_NORMAL_MIN_CM,
    POSHAN_MUAC_SAM_MAX_CM,
    WastingStatus,
)


@dataclass(frozen=True)
class AcuteMalnutritionResult:
    haz_zscore: Optional[float]
    stunting_status: Optional[str]
    whz_zscore: Optional[float]
    whz_status: Optional[WastingStatus]
    muac_status: Optional[WastingStatus]
    muac_eligible: bool
    oedema: str
    acute_status: WastingStatus
    triggered_by: tuple[str, ...]
    missing_reasons: tuple[str, ...]
    actionable: bool
    rationale: str


class AcuteMalnutritionService:
    """Compute independent WHO HAZ and acute-malnutrition components."""

    def __init__(self, who_data: WHODataService):
        self._nutrition = NutritionService(who_data)

    def assess(
        self,
        *,
        sex: str,
        completed_age_months: int,
        age_months: float,
        height_cm: Optional[float],
        weight_kg: Optional[float],
        tape_muac_cm: Optional[float],
        oedema: str,
    ) -> AcuteMalnutritionResult:
        haz_zscore = None
        stunting_status = None
        whz_zscore = None
        whz_status = None
        muac_status = None
        missing_reasons = []

        if height_cm is None:
            missing_reasons.append("height or length not measured")
        else:
            haz_zscore = self._nutrition.compute_haz(
                sex,
                completed_age_months,
                height_cm,
            )
            if haz_zscore is not None:
                stunting_status = self._nutrition.classify_haz(haz_zscore)

        if weight_kg is None:
            missing_reasons.append("weight not measured")
        elif height_cm is not None:
            whz_zscore = self._nutrition.compute_whz(
                sex,
                age_months,
                height_cm,
                weight_kg,
            )
            if whz_zscore is not None:
                whz_status = self._nutrition.classify_whz(whz_zscore)
            else:
                missing_reasons.append(
                    "WHZ unavailable for the measured length or height"
                )

        muac_eligible = (
            POSHAN_MUAC_MIN_AGE_MONTHS
            <= age_months
            < POSHAN_MUAC_MAX_AGE_MONTHS
        )
        if tape_muac_cm is None:
            missing_reasons.append("tape MUAC not measured")
        elif not muac_eligible:
            missing_reasons.append("tape MUAC ineligible outside 6-59 months")
        elif tape_muac_cm < POSHAN_MUAC_SAM_MAX_CM:
            muac_status = WastingStatus.SAM
        elif tape_muac_cm < POSHAN_MUAC_NORMAL_MIN_CM:
            muac_status = WastingStatus.MAM
        else:
            muac_status = WastingStatus.NORMAL

        if oedema == "not_checked":
            missing_reasons.append("oedema not checked")

        component_statuses: list[tuple[str, WastingStatus]] = []
        if whz_status is not None:
            acute_whz = (
                whz_status
                if whz_status in (WastingStatus.SAM, WastingStatus.MAM)
                else WastingStatus.NORMAL
            )
            component_statuses.append(("whz", acute_whz))
        if muac_status is not None:
            component_statuses.append(("muac", muac_status))
        if oedema == "yes":
            component_statuses.append(("oedema", WastingStatus.SAM))

        sam_triggers = tuple(
            name
            for name, status in component_statuses
            if status == WastingStatus.SAM
        )
        mam_triggers = tuple(
            name
            for name, status in component_statuses
            if status == WastingStatus.MAM
        )
        if sam_triggers:
            acute_status = WastingStatus.SAM
            triggered_by = sam_triggers
        elif mam_triggers:
            acute_status = WastingStatus.MAM
            triggered_by = mam_triggers
        elif component_statuses:
            acute_status = WastingStatus.NORMAL
            triggered_by = tuple(name for name, _status in component_statuses)
        else:
            acute_status = WastingStatus.UNKNOWN
            triggered_by = ()

        return AcuteMalnutritionResult(
            haz_zscore=haz_zscore,
            stunting_status=stunting_status,
            whz_zscore=whz_zscore,
            whz_status=whz_status,
            muac_status=muac_status,
            muac_eligible=muac_eligible,
            oedema=oedema,
            acute_status=acute_status,
            triggered_by=triggered_by,
            missing_reasons=tuple(missing_reasons),
            actionable=acute_status in (WastingStatus.SAM, WastingStatus.MAM),
            rationale=(
                "WHO acute malnutrition uses eligible measured WHZ, tape MUAC, "
                f"and oedema; status {acute_status.value}; "
                f"triggers {list(triggered_by)}; "
                f"missing {missing_reasons}."
            ),
        )
