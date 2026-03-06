"""
Synthetic training dataset generator for the wasting detection ML model.

Data provenance
---------------
VERIFIED from WHO data files in this project:
  - Height boundaries (z_minus_3..z_plus_3) by age/sex: who_haz_0_59m.csv
  - Weight-for-height LMS parameters (L, M, S): wfl/wfh Excel files
  - SAM/MAM thresholds: WHZ < -3 (SAM), -3 ≤ WHZ < -2 (MAM) per WHO standards

NOT from WHO — physics-based approximation (Snyder RG et al. 1975,
"Anthropometry of Infants, Children, and Youths to Age 18", NASA/SAE):
  - Baseline shoulder width / height ratios by age
  - Baseline hip width / height ratios (hip ≈ 0.88 × shoulder ratio)
  - Upper arm length / height ratios by age
  - Volumetric scaling of widths with weight: width ∝ (weight/median)^(1/3)

Run:  python ml/generate_synthetic_data.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = DATA_DIR / "training_data"

HAZ_CSV  = DATA_DIR / "who_haz_0_59m.csv"
WFL_BOYS  = DATA_DIR / "wfl_boys_0-to-2-years_zscores.xlsx"
WFL_GIRLS = DATA_DIR / "wfl_girls_0-to-2-years_zscores.xlsx"
WFH_BOYS  = DATA_DIR / "wfh_boys_2-to-5-years_zscores.xlsx"
WFH_GIRLS = DATA_DIR / "wfh_girls_2-to-5-years_zscores.xlsx"

N_SAMPLES = 60_000
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Non-WHO body proportion model (Snyder et al. 1975, US pediatric data)
# These are fraction-of-stature values for typically nourished children.
# Nodes: (age_months, ratio)
# ---------------------------------------------------------------------------
SHOULDER_RATIO_NODES = [(0, 0.191), (6, 0.193), (12, 0.198),
                         (24, 0.207), (36, 0.211), (48, 0.214), (60, 0.218)]
ARM_RATIO_NODES      = [(0, 0.145), (12, 0.150), (24, 0.155),
                         (36, 0.158), (48, 0.162), (60, 0.165)]
TORSO_RATIO_NODES    = [(0, 0.32), (12, 0.32), (24, 0.30), (48, 0.30), (60, 0.30)]


def _interp_ratio(nodes, age_months):
    """Linear interpolation through (age, ratio) nodes."""
    ages  = [n[0] for n in nodes]
    vals  = [n[1] for n in nodes]
    return float(np.interp(age_months, ages, vals))


# ---------------------------------------------------------------------------
# WHO data helpers (verified sources only)
# ---------------------------------------------------------------------------

def _load_haz() -> pd.DataFrame:
    df = pd.read_csv(HAZ_CSV)
    return df


def _load_lms() -> dict:
    """Load L/M/S tables keyed by ('M'|'F', 'wfl'|'wfh')."""
    result = {}
    for sex, wfl_path, wfh_path in [
        ("M", WFL_BOYS,  WFH_BOYS),
        ("F", WFL_GIRLS, WFH_GIRLS),
    ]:
        for key, path in [("wfl", wfl_path), ("wfh", wfh_path)]:
            df = pd.read_excel(path)
            idx_col = df.columns[0]
            df = df.rename(columns={idx_col: "index_value"})
            for col in ["L", "M", "S"]:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("\u2212", "-"), errors="coerce"
                )
            result[(sex, key)] = df.sort_values("index_value").reset_index(drop=True)
    return result


def _haz_to_height(haz_df: pd.DataFrame, sex: str, age_months: int, haz: float) -> float:
    """
    Convert a HAZ value to height in cm using linear interpolation of
    WHO HAZ boundaries. Source: who_haz_0_59m.csv (verified).
    """
    measure = "length" if age_months < 24 else "height"
    rows = haz_df[(haz_df["sex"] == sex) &
                  (haz_df["measure"] == measure) &
                  (haz_df["age_months"] == age_months)]
    if rows.empty:
        return float("nan")
    r = rows.iloc[0]
    z_points = [(-3, r.z_minus_3), (-2, r.z_minus_2), (-1, r.z_minus_1),
                (0, r.z_0), (1, r.z_plus_1), (2, r.z_plus_2), (3, r.z_plus_3)]
    zs = [p[0] for p in z_points]
    hs = [p[1] for p in z_points]
    return float(np.interp(haz, zs, hs))


def _get_lms(lms_tables: dict, sex: str, height_cm: float, age_months: float):
    """
    Retrieve (L, M, S) for a height via linear interpolation.
    Selects WFL (<24 months) or WFH (≥24 months). Source: Excel files (verified).
    """
    key = "wfl" if age_months < 24 else "wfh"
    df = lms_tables[(sex, key)]
    vals = df["index_value"].values
    idx = np.searchsorted(vals, height_cm)
    if idx == 0:
        row = df.iloc[0]
    elif idx >= len(df):
        row = df.iloc[-1]
    else:
        lo = df.iloc[idx - 1]
        hi = df.iloc[idx]
        denom = float(hi["index_value"] - lo["index_value"])
        if denom == 0:
            row = lo
        else:
            frac = (height_cm - float(lo["index_value"])) / denom
            row = lo.copy()
            for col in ["L", "M", "S"]:
                row[col] = float(lo[col]) + frac * (float(hi[col]) - float(lo[col]))
    return float(row["L"]), float(row["M"]), float(row["S"])


def _whz_to_weight(L: float, M: float, S: float, whz: float) -> float:
    """
    Inverse LMS: compute weight that corresponds to a given WHZ.
    Formula: weight = M * (1 + L * S * Z)^(1/L)   (L != 0)
             weight = M * exp(S * Z)                (L ≈ 0)
    Derived from the standard LMS Z-score formula; uses WHO Excel LMS data.
    """
    if abs(L) < 1e-6:
        return M * math.exp(S * whz)
    val = 1.0 + L * S * whz
    if val <= 0:
        # Extreme negative Z — clamp to near-zero weight
        return M * 0.5
    return M * (val ** (1.0 / L))


# ---------------------------------------------------------------------------
# Body proportion model (NON-WHO — Snyder et al. 1975 + volumetric scaling)
# ---------------------------------------------------------------------------

def _body_widths(height_cm: float, age_months: float,
                 weight_kg: float, median_weight: float,
                 rng: np.random.Generator) -> dict:
    """
    Compute simulated body width measurements as a camera would detect.

    The baseline ratios (Snyder 1975) represent normally nourished children.
    Width scaling with weight uses the volumetric cube-root assumption:
        width_actual = width_expected × (weight / median_weight)^(1/3)

    This is NOT from WHO standards. It is a physical approximation.
    Noise (σ=3%) simulates MediaPipe landmark detection variability.
    """
    width_scale = max((weight_kg / median_weight) ** (1 / 3), 0.7)

    shoulder_ratio   = _interp_ratio(SHOULDER_RATIO_NODES, age_months)
    shoulder_expected = height_cm * shoulder_ratio
    shoulder_actual   = shoulder_expected * width_scale
    shoulder_actual  *= rng.normal(1.0, 0.03)

    hip_expected = shoulder_expected * 0.88   # hip ≈ 88% of shoulder (Snyder 1975)
    hip_actual   = hip_expected * width_scale
    hip_actual  *= rng.normal(1.0, 0.03)

    arm_ratio    = _interp_ratio(ARM_RATIO_NODES, age_months)
    arm_actual   = height_cm * arm_ratio * rng.normal(1.0, 0.04)

    torso_ratio  = _interp_ratio(TORSO_RATIO_NODES, age_months)
    torso_actual = height_cm * torso_ratio * rng.normal(1.0, 0.03)

    return {
        "shoulder_width_cm":    max(shoulder_actual, 5.0),
        "hip_width_cm":         max(hip_actual,      4.0),
        "upper_arm_length_cm":  max(arm_actual,      3.0),
        "torso_length_cm":      max(torso_actual,    8.0),
        "width_scale":          width_scale,
    }


def _body_build_score(shoulder_width_cm: float, height_cm: float,
                      age_months: float) -> int:
    """
    Classify body build: -1=slender, 0=average, 1=stocky.
    Based on deviation from expected shoulder/height ratio.
    """
    expected = _interp_ratio(SHOULDER_RATIO_NODES, age_months)
    actual   = shoulder_width_cm / height_cm
    if actual < expected - 0.02:
        return -1
    if actual > expected + 0.02:
        return 1
    return 0


def _label(whz: float) -> str:
    if whz < -3:    return "SAM"
    if whz < -2:    return "MAM"
    if whz <= 1:    return "Normal"
    if whz <= 2:    return "Risk_Overweight"
    return "Overweight"


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    haz_df     = _load_haz()
    lms_tables = _load_lms()

    # Pre-compute valid age/sex pairs so we can sample uniformly
    sexes = ["M", "F"]
    ages  = list(range(0, 60))   # 0–59 months

    records = []
    for _ in range(n):
        sex       = rng.choice(sexes)
        age_mo    = int(rng.integers(0, 60))

        # Sample HAZ from a realistic global distribution
        # WHO global HAZ is approximately N(-0.6, 1.1) in low-income settings
        # Use N(0, 1) for a balanced training set (ensures enough normal samples)
        haz = float(rng.normal(0.0, 1.0))
        haz = float(np.clip(haz, -4.0, 4.0))

        height_cm = _haz_to_height(haz_df, sex, age_mo, haz)
        if math.isnan(height_cm) or height_cm <= 0:
            continue

        # Sample WHZ — slight left skew to create realistic class imbalance
        # Global prevalence: ~5% SAM, ~10% MAM, ~85% Normal (WHO 2022)
        # We use a mix: 70% from N(0,1), 15% from N(-2.5,0.5), 15% from N(-1.5,0.4)
        r = rng.random()
        if r < 0.70:
            whz = float(rng.normal(0.0, 1.0))
        elif r < 0.85:
            whz = float(rng.normal(-2.5, 0.5))   # MAM/SAM region
        else:
            whz = float(rng.normal(-1.5, 0.4))   # borderline
        whz = float(np.clip(whz, -4.0, 3.5))

        try:
            L, M, S = _get_lms(lms_tables, sex, height_cm, float(age_mo))
        except Exception:
            continue

        weight_kg     = _whz_to_weight(L, M, S, whz)
        median_weight = M

        if weight_kg <= 0 or median_weight <= 0:
            continue

        widths = _body_widths(height_cm, age_mo, weight_kg, median_weight, rng)

        shoulder_height_ratio = widths["shoulder_width_cm"] / height_cm
        hip_height_ratio      = widths["hip_width_cm"]      / height_cm
        bds = _body_build_score(widths["shoulder_width_cm"], height_cm, age_mo)

        records.append({
            # Features
            "age_months":           float(age_mo),
            "sex_binary":           1 if sex == "M" else 0,
            "height_cm":            round(height_cm, 2),
            "shoulder_width_cm":    round(widths["shoulder_width_cm"], 2),
            "hip_width_cm":         round(widths["hip_width_cm"], 2),
            "torso_length_cm":      round(widths["torso_length_cm"], 2),
            "upper_arm_length_cm":  round(widths["upper_arm_length_cm"], 2),
            "shoulder_height_ratio": round(shoulder_height_ratio, 4),
            "hip_height_ratio":      round(hip_height_ratio, 4),
            "body_build_score":      bds,
            # Targets
            "weight_kg":            round(weight_kg, 3),
            "whz":                  round(whz, 3),
            "haz":                  round(haz, 3),
            "label":                _label(whz),
            # Metadata
            "sex":                  sex,
            "median_weight_kg":     round(median_weight, 3),
        })

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} samples")
    print("Label distribution:\n" + df["label"].value_counts(normalize=True)
          .rename("fraction").to_string())
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    out = OUT_DIR / "synthetic_dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    sys.exit(main())
