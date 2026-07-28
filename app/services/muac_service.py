"""
MUAC (Mid-Upper Arm Circumference) estimation service.

MUAC is the WHO-recommended field screening tool for acute malnutrition.
Thresholds (WHO, 6–59 months, absolute — NOT age-adjusted):
  < 11.5 cm  → SAM (Severe Acute Malnutrition)
  11.5–12.5  → At Risk (MAM / Moderate Acute Malnutrition)
  ≥ 12.5 cm  → Normal

Three estimation pathways, in priority order:

  1. **Manual** — tape measurement provided, used directly.
  2. **Landmark-based** — an experimental screening estimate. It cannot make
     an autonomous clinical call until paired tape-measure validation shows
     SAM recall >= 0.80.
  3. **WHZ-based** — fallback when landmarks aren't available. Uses the
     formula MUAC = median(age,sex) × (1 + 0.087 × clamp(WHZ, ±3)).

WHZ-derived MUAC is explanatory only. It is not a second diagnostic arm and
is therefore ignored by the final combiner (otherwise the same WHZ evidence
would be counted twice).

Reference for MUAC medians:
    WHO Child Growth Standards (2006): Arm circumference-for-age.
    https://www.who.int/tools/child-growth-standards
"""

from dataclasses import dataclass
from typing import Optional

from config import WastingStatus


# ── WHO MUAC-for-age medians (cm) ──────────────────────────────────────────
# Source: WHO Child Growth Standards 2006 — Arm circumference-for-age tables
# These are the L=0 medians (M column) at representative ages.
# Values for boys and girls are slightly different; both are embedded here.

_MUAC_BOYS: list[tuple[float, float]] = [
    (3,  12.5),
    (6,  14.0),
    (9,  14.8),
    (12, 15.2),
    (18, 15.5),
    (24, 15.7),
    (30, 15.8),
    (36, 15.9),
    (42, 16.0),
    (48, 16.1),
    (54, 16.1),
    (60, 16.2),
]

_MUAC_GIRLS: list[tuple[float, float]] = [
    (3,  12.3),
    (6,  13.8),
    (9,  14.6),
    (12, 14.9),
    (18, 15.2),
    (24, 15.4),
    (30, 15.5),
    (36, 15.6),
    (42, 15.7),
    (48, 15.7),
    (54, 15.8),
    (60, 15.8),
]


@dataclass
class MUACResult:
    muac_cm: Optional[float]        # rounded to 1 decimal place
    muac_status: Optional[WastingStatus]
    # "manual" | "landmark_estimated" | "estimated_from_whz"
    muac_method: str
    age_in_range: bool              # True only for 6–59 months
    confidence: Optional[float] = None
    uncertainty_lower_cm: Optional[float] = None
    uncertainty_upper_cm: Optional[float] = None
    model_version: Optional[str] = None
    calibration_version: Optional[str] = None
    is_direct_measurement: bool = False
    requires_confirmation: bool = False
    referral_guidance: Optional[str] = None


@dataclass
class CombinedNutritionStatus:
    """
    Final SAM/MAM call after combining MUAC and WHZ via WHO OR-rule.
    """
    status: WastingStatus
    triggered_by: list[str]  # ["muac", "whz", ...]  why this status was chosen
    rationale: str           # human-readable explanation


class MUACService:
    """Estimate or classify MUAC for a child."""

    LANDMARK_MODEL_VERSION = "landmark-ratio-v1"
    LANDMARK_CALIBRATION_VERSION = "unvalidated-paired-tape-v0"
    LANDMARK_SAM_RECALL_VALIDATED = False
    MIN_AUTONOMOUS_SAM_RECALL = 0.80
    MIN_LANDMARK_CONFIDENCE = 0.80

    @staticmethod
    def estimate(
        age_months: float,
        sex: str,
        whz: Optional[float],
        manual_muac_cm: Optional[float] = None,
        upper_arm_length_cm: Optional[float] = None,
        shoulder_width_cm: Optional[float] = None,
        height_cm: Optional[float] = None,
        landmark_visibility: Optional[float] = None,
    ) -> MUACResult:
        """
        Return a MUACResult for the given child.

        Pathway priority:
          1. Manual tape measurement (manual_muac_cm).
          2. Landmark-based estimate, if upper_arm_length_cm and
             shoulder_width_cm and height_cm are all available.
          3. WHZ-based estimate, otherwise.

        Args:
            age_months:          Child's age in fractional months.
            sex:                 'M' or 'F'.
            whz:                 Weight-for-height z-score (may be None).
            manual_muac_cm:      Tape-measured MUAC in cm (takes priority).
            upper_arm_length_cm: Upper-arm bone length from pose (optional).
            shoulder_width_cm:   Shoulder width from pose (optional).
            height_cm:           Total height (optional, used for normalisation).
        """
        age_in_range = 6.0 <= age_months <= 59.9

        # ── 1. Manual measurement ────────────────────────────────────────
        if manual_muac_cm is not None and manual_muac_cm > 0:
            return MUACResult(
                muac_cm=round(manual_muac_cm, 1),
                muac_status=MUACService._classify(manual_muac_cm, age_in_range),
                muac_method="manual",
                age_in_range=age_in_range,
                confidence=1.0,
                uncertainty_lower_cm=round(manual_muac_cm, 1),
                uncertainty_upper_cm=round(manual_muac_cm, 1),
                model_version=None,
                calibration_version="direct-tape",
                is_direct_measurement=True,
            )

        # ── 2. Landmark-based estimate (independent of WHZ) ──────────────
        if (
            upper_arm_length_cm is not None
            and shoulder_width_cm is not None
            and height_cm is not None
            and height_cm > 0
        ):
            est = MUACService._estimate_from_landmarks(
                age_months=age_months,
                sex=sex,
                upper_arm_length_cm=upper_arm_length_cm,
                shoulder_width_cm=shoulder_width_cm,
                height_cm=height_cm,
            )
            if est is not None:
                visibility = landmark_visibility if landmark_visibility is not None else 0.5
                confidence = max(0.0, min(1.0, visibility))
                half_width = max(0.6, 2.0 * (1.0 - confidence))
                lower, upper = est - half_width, est + half_width
                crosses_threshold = any(lower < threshold <= upper for threshold in (11.5, 12.5))
                requires_confirmation = (
                    not MUACService.LANDMARK_SAM_RECALL_VALIDATED
                    or confidence < MUACService.MIN_LANDMARK_CONFIDENCE
                    or crosses_threshold
                )
                return MUACResult(
                    muac_cm=round(est, 1),
                    muac_status=(
                        MUACService._classify(est, age_in_range)
                        if not requires_confirmation
                        else None
                    ),
                    muac_method="landmark_estimated",
                    age_in_range=age_in_range,
                    confidence=round(confidence, 2),
                    uncertainty_lower_cm=round(lower, 1),
                    uncertainty_upper_cm=round(upper, 1),
                    model_version=MUACService.LANDMARK_MODEL_VERSION,
                    calibration_version=MUACService.LANDMARK_CALIBRATION_VERSION,
                    is_direct_measurement=False,
                    requires_confirmation=requires_confirmation,
                    referral_guidance=("Prompt tape MUAC confirmation and refer for clinical assessment; "
                                       "do not dismiss as Normal." if requires_confirmation else None),
                )

        # ── 3. WHZ-based estimate ────────────────────────────────────────
        if whz is None:
            return MUACResult(
                muac_cm=None,
                muac_status=None,
                muac_method="estimated_from_whz",
                age_in_range=age_in_range,
                confidence=None,
                uncertainty_lower_cm=None,
                uncertainty_upper_cm=None,
                model_version="whz-explanatory-v1",
                calibration_version="who-median-formula-v1",
                is_direct_measurement=False,
                requires_confirmation=True,
                referral_guidance="Obtain a direct tape MUAC measurement.",
            )

        median = MUACService._median_for_age(age_months, sex)
        whz_clamped = max(-3.0, min(3.0, whz))
        muac_cm = median * (1.0 + 0.087 * whz_clamped)
        muac_cm = round(muac_cm, 1)

        return MUACResult(
            muac_cm=muac_cm,
            # This estimate is transformed from WHZ and is explanatory only.
            # Returning no MUAC classification prevents downstream consumers
            # from treating the same evidence as an independent diagnostic arm.
            muac_status=None,
            muac_method="estimated_from_whz",
            age_in_range=age_in_range,
            confidence=0.4,
            uncertainty_lower_cm=round(muac_cm - 1.0, 1),
            uncertainty_upper_cm=round(muac_cm + 1.0, 1),
            model_version="whz-explanatory-v1",
            calibration_version="who-median-formula-v1",
            is_direct_measurement=False,
            requires_confirmation=True,
            referral_guidance="WHZ-derived MUAC is explanatory only; obtain a direct tape measurement.",
        )

    @staticmethod
    def _estimate_from_landmarks(
        age_months: float,
        sex: str,
        upper_arm_length_cm: float,
        shoulder_width_cm: float,
        height_cm: float,
    ) -> Optional[float]:
        """
        Estimate MUAC from pose landmarks, *partially* independent of WHZ.

        Empirical model (calibrated against WHO MUAC medians + Snyder 1975
        upper-arm proportions):

            MUAC ≈ median(age, sex)
                   × (upper_arm_length / expected_arm_length(age, height))^0.30
                   × (shoulder_width   / expected_shoulder(age, height)) ^0.50

        Rationale: the upper arm shortens proportionally less than its
        circumference under wasting, but is *not* a constant fraction of
        height in malnourished children either — the limb-to-trunk ratio
        shifts. Combining arm length and shoulder width gives a stoutness
        proxy that's only weakly correlated with the WHZ-derived weight,
        making this a useful independent signal for the OR-rule combiner.

        Returns None if inputs are out of plausible range.
        """
        if upper_arm_length_cm <= 0 or shoulder_width_cm <= 0 or height_cm <= 0:
            return None

        # Expected (well-nourished) arm and shoulder, age-interpolated
        if age_months < 12:
            arm_ratio, shoulder_ratio = 0.150, 0.193
        elif age_months < 24:
            arm_ratio, shoulder_ratio = 0.155, 0.207
        elif age_months < 48:
            arm_ratio, shoulder_ratio = 0.160, 0.212
        else:
            arm_ratio, shoulder_ratio = 0.165, 0.218

        expected_arm     = height_cm * arm_ratio
        expected_shoulder = height_cm * shoulder_ratio
        if expected_arm <= 0 or expected_shoulder <= 0:
            return None

        arm_factor      = (upper_arm_length_cm / expected_arm) ** 0.30
        shoulder_factor = (shoulder_width_cm   / expected_shoulder) ** 0.50

        median = MUACService._median_for_age(age_months, sex)
        muac_cm = median * arm_factor * shoulder_factor

        # Sanity range: real MUAC for 0-59 months is ~9-20 cm
        if muac_cm < 7.0 or muac_cm > 22.0:
            return None
        return float(muac_cm)

    @staticmethod
    def combine_with_whz_status(
        muac_status: Optional[WastingStatus],
        whz_status: Optional[WastingStatus],
        *,
        muac_method: Optional[str],
        is_direct_measurement: bool,
        landmark_autonomous_call_allowed: bool = False,
    ) -> CombinedNutritionStatus:
        """
        Combine MUAC and WHZ classifications via WHO OR-rule (2009/2013 CMAM).

        Rules (in order, severity descending):
          - If either says SAM           → SAM
          - Else if either says MAM      → MAM
          - Else applicable overweight categories follow WHZ severity
          - Else                          → NORMAL
          - If both inputs are None      → UNKNOWN

        Inputs must already use the canonical vocabulary. This strict boundary
        prevents descriptive presentation labels from entering clinical logic.
        """
        allowed = {status.value for status in WastingStatus}
        for source, value in (("muac_status", muac_status), ("whz_status", whz_status)):
            if value is not None and value not in allowed:
                raise ValueError(f"{source} must be a canonical wasting status, got {value!r}")

        # A WHZ-derived value is the same evidence transformed, not an
        # independent diagnostic arm. Experimental landmarks are similarly
        # non-autonomous until the documented safety floor has been met.
        muac_can_trigger = is_direct_measurement or (
            muac_method == "landmark_estimated" and landmark_autonomous_call_allowed
        )
        if not muac_can_trigger:
            muac_status = None

        triggered: list[str] = []

        # SAM
        if muac_status == WastingStatus.SAM:
            triggered.append("muac")
        if whz_status == WastingStatus.SAM:
            triggered.append("whz")
        if triggered:
            why = " or ".join(triggered)
            return CombinedNutritionStatus(
                status=WastingStatus.SAM,
                triggered_by=triggered,
                rationale=f"SAM flagged by {why} (WHO OR-rule)",
            )

        # MAM
        if muac_status == WastingStatus.MAM:
            triggered.append("muac")
        if whz_status == WastingStatus.MAM:
            triggered.append("whz")
        if triggered:
            why = " or ".join(triggered)
            return CombinedNutritionStatus(
                status=WastingStatus.MAM,
                triggered_by=triggered,
                rationale=f"MAM flagged by {why} (WHO OR-rule)",
            )

        # Overweight / Risk only governed by WHZ (MUAC has no upper threshold)
        if whz_status == WastingStatus.OBESE:
            return CombinedNutritionStatus(WastingStatus.OBESE, ["whz"], "Obesity from WHZ")
        if whz_status == WastingStatus.OVERWEIGHT:
            return CombinedNutritionStatus(
                status=WastingStatus.OVERWEIGHT,
                triggered_by=["whz"],
                rationale="Overweight from WHZ",
            )
        if whz_status == WastingStatus.RISK_OVERWEIGHT:
            return CombinedNutritionStatus(
                status=WastingStatus.RISK_OVERWEIGHT,
                triggered_by=["whz"],
                rationale="Risk of overweight from WHZ",
            )

        # Both None → Unknown
        if muac_status is None and whz_status is None:
            return CombinedNutritionStatus(
                status=WastingStatus.UNKNOWN,
                triggered_by=[],
                rationale="No MUAC or WHZ information available",
            )

        return CombinedNutritionStatus(
            status=WastingStatus.NORMAL,
            triggered_by=[],
            rationale="No MUAC or WHZ flag triggered",
        )

    @staticmethod
    def _median_for_age(age_months: float, sex: str) -> float:
        """
        Linear interpolation of WHO MUAC median for the given age and sex.
        Clamps to the nearest boundary for ages outside 3–60 months.
        """
        table = _MUAC_BOYS if sex.upper() == "M" else _MUAC_GIRLS

        if age_months <= table[0][0]:
            return table[0][1]
        if age_months >= table[-1][0]:
            return table[-1][1]

        for i in range(len(table) - 1):
            a0, m0 = table[i]
            a1, m1 = table[i + 1]
            if a0 <= age_months <= a1:
                t = (age_months - a0) / (a1 - a0)
                return m0 + t * (m1 - m0)

        return table[-1][1]  # fallback

    @staticmethod
    def _classify(
        muac_cm: float, age_in_range: bool
    ) -> Optional[WastingStatus]:
        """Classify MUAC using WHO absolute thresholds (6–59 months only)."""
        if not age_in_range:
            return None
        if muac_cm < 11.5:
            return WastingStatus.SAM
        if muac_cm < 12.5:
            return WastingStatus.MAM
        return WastingStatus.NORMAL
