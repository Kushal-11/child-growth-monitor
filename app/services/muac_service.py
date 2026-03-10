"""
MUAC (Mid-Upper Arm Circumference) estimation service.

MUAC is the WHO-recommended field screening tool for acute malnutrition.
Thresholds (WHO, 6–59 months, absolute — NOT age-adjusted):
  < 11.5 cm  → SAM (Severe Acute Malnutrition)
  11.5–12.5  → At Risk (MAM / Moderate Acute Malnutrition)
  ≥ 12.5 cm  → Normal

When a tape measurement is provided (manual_muac_cm), it is used directly.

When no tape measurement is available, MUAC is ESTIMATED from age, sex, and the
child's WHZ (weight-for-height z-score) using:

    MUAC_estimate = median_MUAC(age, sex) × (1 + 0.087 × clamp(WHZ, −3, +3))

The coefficient 0.087 is calibrated so that WHZ = −3 maps a 24-month boy to
≈ 11.6 cm (just inside the SAM threshold of 11.5 cm).

This estimate is APPROXIMATE — it should be confirmed with a physical tape measurement.

Reference for MUAC medians:
    WHO Child Growth Standards (2006): Arm circumference-for-age.
    Available: https://www.who.int/tools/child-growth-standards
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
    muac_method: str                # "manual" | "estimated_from_whz"
    age_in_range: bool              # True only for 6–59 months


class MUACService:
    """Estimate or classify MUAC for a child."""

    @staticmethod
    def estimate(
        age_months: float,
        sex: str,
        whz: Optional[float],
        manual_muac_cm: Optional[float] = None,
    ) -> MUACResult:
        """
        Return a MUACResult for the given child.

        Args:
            age_months:     Child's age in fractional months.
            sex:            'M' or 'F'.
            whz:            Weight-for-height z-score (may be None).
            manual_muac_cm: Tape-measured MUAC in cm (takes priority).
        """
        age_in_range = 6.0 <= age_months <= 59.9

        # ── Manual measurement takes priority ────────────────────────────
        if manual_muac_cm is not None and manual_muac_cm > 0:
            return MUACResult(
                muac_cm=round(manual_muac_cm, 1),
                muac_status=MUACService._classify(manual_muac_cm, age_in_range),
                muac_method="manual",
                age_in_range=age_in_range,
            )

        # ── Cannot estimate without WHZ ───────────────────────────────────
        if whz is None:
            return MUACResult(
                muac_cm=None,
                muac_status=None,
                muac_method="estimated_from_whz",
                age_in_range=age_in_range,
            )

        # ── WHZ-based estimate ────────────────────────────────────────────
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
