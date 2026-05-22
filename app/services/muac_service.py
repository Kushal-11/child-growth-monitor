"""
MUAC (Mid-Upper Arm Circumference) estimation service.

MUAC is the WHO-recommended field screening tool for acute malnutrition.
Thresholds (WHO, 6–59 months, absolute — NOT age-adjusted):
  < 11.5 cm  → SAM (Severe Acute Malnutrition)
  11.5–12.5  → At Risk (MAM / Moderate Acute Malnutrition)
  ≥ 12.5 cm  → Normal

Three estimation pathways, in priority order:

  1. **Manual** — tape measurement provided, used directly.
  2. **Landmark-based** — derived from the upper-arm length and the body's
     overall stoutness (shoulder/hip widths) extracted from the pose. This
     is *partially* independent of the WHZ pathway: it depends mostly on
     lateral body proportions, not on the predicted weight. Makes the
     OR-rule combiner non-trivially additive.
  3. **WHZ-based** — fallback when landmarks aren't available. Uses the
     formula MUAC = median(age,sex) × (1 + 0.087 × clamp(WHZ, ±3)).

The OR-rule combiner (`combine_with_whz_status`) implements the WHO 2009/2013
community-management protocol: classify SAM/MAM if EITHER MUAC OR WHZ flags
the child. This *raises* sensitivity vs either single criterion alone.

Reference for MUAC medians:
    WHO Child Growth Standards (2006): Arm circumference-for-age.
    https://www.who.int/tools/child-growth-standards
"""

from dataclasses import dataclass
from typing import Optional


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
    muac_status: Optional[str]      # "SAM" | "At Risk (MAM)" | "Normal"
    # "manual" | "landmark_estimated" | "estimated_from_whz"
    muac_method: str
    age_in_range: bool              # True only for 6–59 months


@dataclass
class CombinedNutritionStatus:
    """
    Final SAM/MAM call after combining MUAC and WHZ via WHO OR-rule.
    """
    status: str              # SAM / MAM / Normal / Risk_Overweight / Overweight
    triggered_by: list[str]  # ["muac", "whz", ...]  why this status was chosen
    rationale: str           # human-readable explanation


class MUACService:
    """Estimate or classify MUAC for a child."""

    @staticmethod
    def estimate(
        age_months: float,
        sex: str,
        whz: Optional[float],
        manual_muac_cm: Optional[float] = None,
        upper_arm_length_cm: Optional[float] = None,
        shoulder_width_cm: Optional[float] = None,
        height_cm: Optional[float] = None,
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
                return MUACResult(
                    muac_cm=round(est, 1),
                    muac_status=MUACService._classify(est, age_in_range),
                    muac_method="landmark_estimated",
                    age_in_range=age_in_range,
                )

        # ── 3. WHZ-based estimate ────────────────────────────────────────
        if whz is None:
            return MUACResult(
                muac_cm=None,
                muac_status=None,
                muac_method="estimated_from_whz",
                age_in_range=age_in_range,
            )

        median = MUACService._median_for_age(age_months, sex)
        whz_clamped = max(-3.0, min(3.0, whz))
        muac_cm = median * (1.0 + 0.087 * whz_clamped)
        muac_cm = round(muac_cm, 1)

        return MUACResult(
            muac_cm=muac_cm,
            muac_status=MUACService._classify(muac_cm, age_in_range),
            muac_method="estimated_from_whz",
            age_in_range=age_in_range,
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
        muac_status: Optional[str],
        whz_status: Optional[str],
    ) -> CombinedNutritionStatus:
        """
        Combine MUAC and WHZ classifications via WHO OR-rule (2009/2013 CMAM).

        Rules (in order, severity descending):
          - If either says SAM           → SAM
          - Else if either says MAM      → MAM
          - Else if WHZ says Overweight  → Overweight
          - Else if WHZ says Risk_OW     → Risk_Overweight
          - Else                          → Normal
          - If both inputs are None      → status="Unknown"

        Note: MUAC service uses "At Risk (MAM)" as the moderate label; we
        normalise it to "MAM" in the combined status so downstream code
        sees a single vocabulary.
        """
        # Normalise the MUAC vocabulary
        if muac_status == "At Risk (MAM)":
            muac_status_n = "MAM"
        elif muac_status in ("SAM", "Normal"):
            muac_status_n = muac_status
        else:
            muac_status_n = None

        triggered: list[str] = []

        # SAM
        if muac_status_n == "SAM":
            triggered.append("muac")
        if whz_status == "SAM":
            triggered.append("whz")
        if triggered:
            why = " or ".join(triggered)
            return CombinedNutritionStatus(
                status="SAM",
                triggered_by=triggered,
                rationale=f"SAM flagged by {why} (WHO OR-rule)",
            )

        # MAM
        if muac_status_n == "MAM":
            triggered.append("muac")
        if whz_status == "MAM":
            triggered.append("whz")
        if triggered:
            why = " or ".join(triggered)
            return CombinedNutritionStatus(
                status="MAM",
                triggered_by=triggered,
                rationale=f"MAM flagged by {why} (WHO OR-rule)",
            )

        # Overweight / Risk only governed by WHZ (MUAC has no upper threshold)
        if whz_status == "Overweight":
            return CombinedNutritionStatus(
                status="Overweight",
                triggered_by=["whz"],
                rationale="Overweight from WHZ",
            )
        if whz_status == "Risk_Overweight":
            return CombinedNutritionStatus(
                status="Risk_Overweight",
                triggered_by=["whz"],
                rationale="Risk of overweight from WHZ",
            )

        # Both None → Unknown
        if muac_status_n is None and whz_status is None:
            return CombinedNutritionStatus(
                status="Unknown",
                triggered_by=[],
                rationale="No MUAC or WHZ information available",
            )

        return CombinedNutritionStatus(
            status="Normal",
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
    def _classify(muac_cm: float, age_in_range: bool) -> Optional[str]:
        """Classify MUAC using WHO absolute thresholds (6–59 months only)."""
        if not age_in_range:
            return None
        if muac_cm < 11.5:
            return "SAM"
        if muac_cm < 12.5:
            return "At Risk (MAM)"
        return "Normal"
