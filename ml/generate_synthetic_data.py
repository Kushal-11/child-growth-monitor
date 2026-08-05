"""
Synthetic training dataset generator for the wasting detection ML model.

Data provenance
---------------
VERIFIED from WHO data files in this project:
  - Height boundaries (z_minus_3..z_plus_3) by age/sex: who_haz_0_59m.csv
  - Weight-for-height LMS parameters (L, M, S): wfl/wfh Excel files
  - SAM/MAM thresholds: WHZ < -3 (SAM), -3 ≤ WHZ < -2 (MAM) per WHO standards

NOT from WHO — physics-based approximation:
  - Baseline shoulder/hip/arm/torso ratios per age (Snyder RG et al. 1975,
    "Anthropometry of Infants, Children, and Youths to Age 18", NASA/SAE).
  - Compartment-based body width/depth scaling (fat / muscle / viscera).
    See Wells (2014), "Toward body composition reference data for infants,
    children, and adolescents" — wasting depletes fat first, then muscle,
    then viscera. Each compartment shrinks at a different rate; lateral
    width and AP depth see different fractions of each.

Version history
---------------
v2 (this revision):
  * Compartment-based scaling (fat/muscle/viscera) replacing single-exponent
    cube-root width and 0.5-exponent depth. Wasting now depletes compartments
    sequentially, matching clinical reality.
  * Ethnicity-as-latent-variable: Snyder ratios are perturbed per draw to
    cover cross-population variation (East Asian, South Asian, Sub-Saharan
    African, European/American baseline).
  * Boundary-aware WHZ sampling: extra density near WHZ = -3 and -2
    decision boundaries, to give the classifier more support there.
  * MediaPipe-realistic landmark noise: occlusion-based dropouts and
    foreshortening bias on top of additive Gaussian noise.

v1: simple cube-root width scaling, exponent-0.5 depth scaling, single
    Gaussian landmark noise.

Run:  python ml/generate_synthetic_data.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "training_data"
SPLIT_MANIFEST = OUT_DIR / "synthetic_split_manifest.json"

HAZ_CSV = DATA_DIR / "who_haz_0_59m.csv"
WFL_BOYS = DATA_DIR / "wfl_boys_0-to-2-years_zscores.xlsx"
WFL_GIRLS = DATA_DIR / "wfl_girls_0-to-2-years_zscores.xlsx"
WFH_BOYS = DATA_DIR / "wfh_boys_2-to-5-years_zscores.xlsx"
WFH_GIRLS = DATA_DIR / "wfh_girls_2-to-5-years_zscores.xlsx"

N_SAMPLES = 60_000
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Non-WHO body proportion model (Snyder et al. 1975, US pediatric data).
# Fraction-of-stature values for typically nourished children.
# Nodes: (age_months, ratio).
# ---------------------------------------------------------------------------
SHOULDER_RATIO_NODES = [
    (0, 0.191),
    (6, 0.193),
    (12, 0.198),
    (24, 0.207),
    (36, 0.211),
    (48, 0.214),
    (60, 0.218),
]
ARM_RATIO_NODES = [
    (0, 0.145),
    (12, 0.150),
    (24, 0.155),
    (36, 0.158),
    (48, 0.162),
    (60, 0.165),
]
TORSO_RATIO_NODES = [(0, 0.32), (12, 0.32), (24, 0.30), (48, 0.30), (60, 0.30)]


# ---------------------------------------------------------------------------
# Ethnicity-aware proportion perturbations.
# Multiplicative offsets applied per draw to Snyder baseline ratios.
# Values reflect *typical* magnitudes from comparative anthropometry studies
# (e.g. WHO MGRS, Frisancho 1990, Eveleth & Tanner 1990); they are
# illustrative, not authoritative — the goal is to break the model's
# dependence on any single population mean.
# ---------------------------------------------------------------------------
ETHNICITY_PROFILES = {
    "baseline": {"shoulder": 1.000, "torso": 1.000, "arm": 1.000},
    "east_asian": {"shoulder": 0.985, "torso": 1.020, "arm": 0.975},
    "south_asian": {"shoulder": 0.975, "torso": 1.000, "arm": 0.985},
    "sub_saharan": {"shoulder": 1.005, "torso": 0.985, "arm": 1.030},
    "european": {"shoulder": 1.000, "torso": 1.000, "arm": 1.000},
    "andean": {"shoulder": 1.010, "torso": 1.020, "arm": 0.965},
}
ETHNICITY_KEYS = list(ETHNICITY_PROFILES.keys())


# ---------------------------------------------------------------------------
# Compartment composition by body level.
# Wasting depletes compartments at very different rates: subcutaneous fat
# falls fastest, then muscle, then viscera. Each anatomical level (shoulder,
# hip, arm, chest-AP, abdomen-AP) is composed of these three compartments
# in different proportions, and lateral vs AP measurements see them
# differently. The numbers below are a defensible engineering approximation
# (Wells 2014, Forbes 1987) — not WHO standards.
# ---------------------------------------------------------------------------
# Fractions sum to ~1 per row; what matters is the relative weighting.
COMPARTMENTS = {
    # level         fat   muscle  viscera
    "shoulder": (0.30, 0.65, 0.05),  # mostly muscle (deltoid, trapezius)
    "hip": (0.30, 0.55, 0.15),  # muscle + visceral hint
    "arm": (0.40, 0.55, 0.05),  # MUAC-equivalent — fat + muscle
    "chest_ap": (0.45, 0.30, 0.25),  # AP depth has more visceral component
    "abdomen_ap": (0.55, 0.15, 0.30),  # AP abdomen dominated by fat + viscera
}

# Compartment shrinkage exponents under wasting (weight / median_weight)^exp.
#   fat:     small exponent → drops fast as weight falls
#   muscle:  medium exponent → drops more slowly
#   viscera: large exponent → relatively preserved
# Empirically, fat mass scales roughly as W^0.6, muscle as W^0.85, viscera as W^1.0
# in cross-sectional pediatric body composition data.
EXP_FAT = 0.60
EXP_MUSCLE = 0.85
EXP_VISCERA = 1.00


def _interp_ratio(nodes, age_months):
    """Linear interpolation through (age, ratio) nodes."""
    ages = [n[0] for n in nodes]
    vals = [n[1] for n in nodes]
    return float(np.interp(age_months, ages, vals))


def _compartment_scale(weight_kg: float, median_weight: float, level: str) -> float:
    """
    Combined volumetric scale factor for a body level under the given weight.

    Each compartment scales independently with weight; the level's overall
    "size" is the fat-, muscle-, and viscera-weighted geometric mean. We
    return a linear-dimension scale (so callers multiply this into the
    expected width or depth in the *same* dimension as the reference).
    """
    fat_frac, muscle_frac, viscera_frac = COMPARTMENTS[level]
    ratio = max(weight_kg / max(median_weight, 1e-3), 0.4)

    # Volume factor per compartment (then convert to linear dimension via cube root)
    vol_fat = ratio**EXP_FAT
    vol_muscle = ratio**EXP_MUSCLE
    vol_viscera = ratio**EXP_VISCERA

    weighted_vol = (
        fat_frac * vol_fat + muscle_frac * vol_muscle + viscera_frac * vol_viscera
    )

    return max(weighted_vol ** (1.0 / 3.0), 0.45)


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
        ("M", WFL_BOYS, WFH_BOYS),
        ("F", WFL_GIRLS, WFH_GIRLS),
    ]:
        for key, path in [("wfl", wfl_path), ("wfh", wfh_path)]:
            df = pd.read_excel(path)
            idx_col = df.columns[0]
            df = df.rename(columns={idx_col: "index_value"})
            for col in ["L", "M", "S"]:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("−", "-"), errors="coerce"
                )
            result[(sex, key)] = df.sort_values("index_value").reset_index(drop=True)
    return result


def _haz_to_height(
    haz_df: pd.DataFrame, sex: str, age_months: int, haz: float
) -> float:
    measure = "length" if age_months < 24 else "height"
    rows = haz_df[
        (haz_df["sex"] == sex)
        & (haz_df["measure"] == measure)
        & (haz_df["age_months"] == age_months)
    ]
    if rows.empty:
        return float("nan")
    r = rows.iloc[0]
    z_points = [
        (-3, r.z_minus_3),
        (-2, r.z_minus_2),
        (-1, r.z_minus_1),
        (0, r.z_0),
        (1, r.z_plus_1),
        (2, r.z_plus_2),
        (3, r.z_plus_3),
    ]
    zs = [p[0] for p in z_points]
    hs = [p[1] for p in z_points]
    return float(np.interp(haz, zs, hs))


def _get_lms(lms_tables: dict, sex: str, height_cm: float, age_months: float):
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
    if abs(L) < 1e-6:
        return M * math.exp(S * whz)
    val = 1.0 + L * S * whz
    if val <= 0:
        return M * 0.5
    return M * (val ** (1.0 / L))


# ---------------------------------------------------------------------------
# Noise model — MediaPipe-realistic
# ---------------------------------------------------------------------------


def _mediapipe_noise(
    rng: np.random.Generator,
    base_value: float,
    base_sigma: float,
    occlusion_prob: float = 0.05,
    foreshortening_bias: float = 0.0,
) -> float:
    """
    Apply MediaPipe-realistic noise to a measurement.

    - Occlusion: with small probability the measurement is degraded by an
      additional bias (the landmark is partially occluded).
    - Foreshortening: when the child is angled away from the camera,
      lateral measurements are biased *down*.
    - Gaussian jitter: standard landmark detection noise.
    """
    val = base_value * rng.normal(1.0, base_sigma)
    if rng.random() < occlusion_prob:
        val *= rng.uniform(0.85, 0.95)  # missing landmark = under-measured
    if foreshortening_bias > 0:
        val *= 1.0 - rng.uniform(0.0, foreshortening_bias)
    return val


# ---------------------------------------------------------------------------
# Body proportion model (NON-WHO)
# ---------------------------------------------------------------------------


def _body_widths(
    height_cm: float,
    age_months: float,
    weight_kg: float,
    median_weight: float,
    ethnicity: str,
    rng: np.random.Generator,
) -> dict:
    """
    Compute simulated body width and depth measurements as a camera would
    detect, using compartment-aware scaling and ethnicity perturbations.

    All measurements are noisy, in cm.
    """
    eth = ETHNICITY_PROFILES[ethnicity]

    # Compartment-aware scaling factors (linear dimension)
    s_shoulder = _compartment_scale(weight_kg, median_weight, "shoulder")
    s_hip = _compartment_scale(weight_kg, median_weight, "hip")
    s_arm = _compartment_scale(weight_kg, median_weight, "arm")
    s_chest_ap = _compartment_scale(weight_kg, median_weight, "chest_ap")
    s_abdomen_ap = _compartment_scale(weight_kg, median_weight, "abdomen_ap")

    # Random foreshortening — small fraction of samples have the child at an angle
    fore = rng.uniform(0.0, 0.05) if rng.random() < 0.10 else 0.0

    shoulder_ratio = _interp_ratio(SHOULDER_RATIO_NODES, age_months) * eth["shoulder"]
    shoulder_expected = height_cm * shoulder_ratio
    shoulder_actual = _mediapipe_noise(
        rng,
        shoulder_expected * s_shoulder,
        base_sigma=0.03,
        foreshortening_bias=fore,
    )

    hip_expected = shoulder_expected * 0.88  # hip ≈ 88% of shoulder (Snyder 1975)
    hip_actual = _mediapipe_noise(
        rng,
        hip_expected * s_hip,
        base_sigma=0.035,
        foreshortening_bias=fore,
    )

    arm_ratio = _interp_ratio(ARM_RATIO_NODES, age_months) * eth["arm"]
    arm_actual = _mediapipe_noise(
        rng,
        height_cm * arm_ratio * s_arm,
        base_sigma=0.04,
        occlusion_prob=0.07,
    )

    torso_ratio = _interp_ratio(TORSO_RATIO_NODES, age_months) * eth["torso"]
    torso_actual = _mediapipe_noise(
        rng,
        height_cm * torso_ratio,
        base_sigma=0.03,
    )

    # AP depth — chest depth ≈ 0.45 × shoulder lateral width (well nourished),
    # abdomen depth ≈ 0.50 × hip lateral width. Compartment scaling drops the
    # AP much faster than lateral because the AP composition is fat-heavy.
    chest_depth_actual = _mediapipe_noise(
        rng,
        shoulder_actual * 0.45 * s_chest_ap,
        base_sigma=0.04,
        occlusion_prob=0.10,
        foreshortening_bias=fore,
    )
    abd_depth_actual = _mediapipe_noise(
        rng,
        hip_actual * 0.50 * s_abdomen_ap,
        base_sigma=0.04,
        occlusion_prob=0.10,
        foreshortening_bias=fore,
    )

    return {
        "shoulder_width_cm": max(shoulder_actual, 5.0),
        "hip_width_cm": max(hip_actual, 4.0),
        "upper_arm_length_cm": max(arm_actual, 3.0),
        "torso_length_cm": max(torso_actual, 8.0),
        "chest_depth_cm": max(chest_depth_actual, 2.0),
        "abd_depth_cm": max(abd_depth_actual, 2.0),
    }


def _body_build_score(
    shoulder_width_cm: float, height_cm: float, age_months: float
) -> int:
    expected = _interp_ratio(SHOULDER_RATIO_NODES, age_months)
    actual = shoulder_width_cm / height_cm
    if actual < expected - 0.02:
        return -1
    if actual > expected + 0.02:
        return 1
    return 0


def _label(whz: float) -> str:
    if whz < -3:
        return "SAM"
    if whz < -2:
        return "MAM"
    if whz <= 1:
        return "Normal"
    if whz <= 2:
        return "Risk_Overweight"
    return "Overweight"


def _sample_whz(rng: np.random.Generator) -> float:
    """
    Boundary-aware WHZ sampler.

    The hard task is separating WHZ values on either side of -3 and -2.
    To give the classifier more support there, we sample with extra density
    in the bands [-3.5, -2.5] (SAM/MAM boundary) and [-2.5, -1.5] (MAM/Normal
    boundary) on top of the existing realistic distribution.
    """
    r = rng.random()
    if r < 0.50:
        # Bulk Normal-ish distribution
        whz = float(rng.normal(0.0, 1.0))
    elif r < 0.62:
        # Boundary band 1: SAM/MAM cliff
        whz = float(rng.uniform(-3.5, -2.5))
    elif r < 0.74:
        # Boundary band 2: MAM/Normal cliff
        whz = float(rng.uniform(-2.5, -1.5))
    elif r < 0.85:
        # Deep SAM tail (we still need extreme cases)
        whz = float(rng.normal(-3.5, 0.4))
    elif r < 0.95:
        # Borderline mild wasting
        whz = float(rng.normal(-1.5, 0.4))
    else:
        # Overweight tail
        whz = float(rng.normal(2.0, 0.6))
    return float(np.clip(whz, -4.0, 3.5))


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    haz_df = _load_haz()
    lms_tables = _load_lms()

    sexes = ["M", "F"]

    records = []
    for child_index in range(n):
        sex = rng.choice(sexes)
        age_mo = int(rng.integers(0, 60))
        ethnicity = str(rng.choice(ETHNICITY_KEYS))

        # Sample HAZ from a realistic global distribution
        haz = float(rng.normal(0.0, 1.0))
        haz = float(np.clip(haz, -4.0, 4.0))

        height_cm = _haz_to_height(haz_df, sex, age_mo, haz)
        if math.isnan(height_cm) or height_cm <= 0:
            continue

        whz = _sample_whz(rng)

        try:
            L, M, S = _get_lms(lms_tables, sex, height_cm, float(age_mo))
        except Exception:
            continue

        weight_kg = _whz_to_weight(L, M, S, whz)
        median_weight = M

        if weight_kg <= 0 or median_weight <= 0:
            continue

        widths = _body_widths(
            height_cm,
            age_mo,
            weight_kg,
            median_weight,
            ethnicity,
            rng,
        )

        shoulder_height_ratio = widths["shoulder_width_cm"] / height_cm
        hip_height_ratio = widths["hip_width_cm"] / height_cm
        chest_depth_ratio = widths["chest_depth_cm"] / height_cm
        abd_depth_ratio = widths["abd_depth_cm"] / height_cm
        bds = _body_build_score(widths["shoulder_width_cm"], height_cm, age_mo)

        records.append(
            {
                "child_id": f"synthetic-{seed}-{child_index:06d}",
                # Features
                "age_months": float(age_mo),
                "sex_binary": 1 if sex == "M" else 0,
                "height_cm": round(height_cm, 2),
                "shoulder_width_cm": round(widths["shoulder_width_cm"], 2),
                "hip_width_cm": round(widths["hip_width_cm"], 2),
                "torso_length_cm": round(widths["torso_length_cm"], 2),
                "upper_arm_length_cm": round(widths["upper_arm_length_cm"], 2),
                "shoulder_height_ratio": round(shoulder_height_ratio, 4),
                "hip_height_ratio": round(hip_height_ratio, 4),
                "body_build_score": bds,
                "chest_depth_cm": round(widths["chest_depth_cm"], 2),
                "abd_depth_cm": round(widths["abd_depth_cm"], 2),
                "chest_depth_ratio": round(chest_depth_ratio, 4),
                "abd_depth_ratio": round(abd_depth_ratio, 4),
                # Targets
                "weight_kg": round(weight_kg, 3),
                "whz": round(whz, 3),
                "haz": round(haz, 3),
                "label": _label(whz),
                # Metadata
                "sex": sex,
                "ethnicity": ethnicity,
                "median_weight_kg": round(median_weight, 3),
            }
        )

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} samples")
    print(
        "Label distribution:\n"
        + df["label"].value_counts(normalize=True).rename("fraction").to_string()
    )
    print("\nEthnicity distribution:")
    print(df["ethnicity"].value_counts(normalize=True).rename("fraction").to_string())
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    out = OUT_DIR / "synthetic_dataset.csv"
    df.to_csv(out, index=False)
    from ml.splits import write_split_manifest

    manifest = write_split_manifest(df, SPLIT_MANIFEST, seed=RANDOM_SEED)
    print(f"\nSaved → {out}")
    print(
        "Split manifest → "
        f"{SPLIT_MANIFEST} "
        + ", ".join(f"{name}={len(ids)}" for name, ids in manifest["splits"].items())
    )


if __name__ == "__main__":
    sys.exit(main())
