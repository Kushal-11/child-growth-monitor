"""Centralized configuration for the Child Growth Monitor application."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "growth_monitor.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Reference object dimensions (centimeters) - optional fallback for scale calibration
# Standard yellow packet dimensions (e.g., biscuit packet)
REFERENCE_OBJECT_LENGTH_CM = 12.7
REFERENCE_OBJECT_WIDTH_CM = 5.5
# Keep old name for backwards compatibility
PARLEG_LENGTH_CM = REFERENCE_OBJECT_LENGTH_CM
PARLEG_WIDTH_CM = REFERENCE_OBJECT_WIDTH_CM

# MediaPipe configuration
POSE_MODEL_PATH = DATA_DIR / "pose_landmarker_heavy.task"
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5

# WHO data file mappings
WHO_DATA_FILES = {
    "haz_0_59m": DATA_DIR / "who_haz_0_59m.csv",
    "wfh_0_59m": DATA_DIR / "who_wfh_0_59m.csv",
    "whz_reference": DATA_DIR / "who_whz_reference.csv",
    "wfh_boys_2_5": DATA_DIR / "wfh_boys_2-to-5-years_zscores.xlsx",
    "wfh_girls_2_5": DATA_DIR / "wfh_girls_2-to-5-years_zscores.xlsx",
    "wfl_boys_0_2": DATA_DIR / "wfl_boys_0-to-2-years_zscores.xlsx",
    "wfl_girls_0_2": DATA_DIR / "wfl_girls_0-to-2-years_zscores.xlsx",
}

# Classification thresholds (WHO standard)
ZSCORE_CLASSIFICATIONS = {
    "haz": {
        (-99, -3): "Severely Stunted",
        (-3, -2): "Stunted",
        (-2, 2): "Normal",
        (2, 99): "Tall",
    },
    "whz": {
        (-99, -3): "Severe Acute Malnutrition (SAM)",
        (-3, -2): "Moderate Acute Malnutrition (MAM)",
        (-2, 1): "Normal",
        (1, 2): "Possible Risk of Overweight",
        (2, 3): "Overweight",
        (3, 99): "Obese",
    },
}

# ============================================================
# ANTHROPOMETRIC RATIOS FOR HEIGHT ESTIMATION
# ============================================================
# Based on pediatric anthropometry literature.
# These ratios represent typical body segment proportions as fractions of total height.
# 
# Sources:
# - WHO Child Growth Standards
# - Roche AF, Davila GH. "Late adolescent growth in stature"
# - Fredriks AM et al. "Body index measurements in 1996-7 compared with 1980"

# Ratios by age group (segment length as fraction of total height)
ANTHROPOMETRIC_RATIOS = {
    # Infants 0-12 months: larger head relative to body
    "0-12_months": {
        "head_ratio": 0.28,   # Head is ~28% of height
        "torso_ratio": 0.32,  # Torso ~32%
        "leg_ratio": 0.40,    # Legs ~40%
    },
    # Toddlers 12-24 months: head proportion decreasing
    "12-24_months": {
        "head_ratio": 0.25,
        "torso_ratio": 0.32,
        "leg_ratio": 0.43,
    },
    # Young children 24-48 months
    "24-48_months": {
        "head_ratio": 0.22,
        "torso_ratio": 0.30,
        "leg_ratio": 0.48,
    },
    # Children 48-60 months: closer to adult proportions
    "48-60_months": {
        "head_ratio": 0.20,
        "torso_ratio": 0.30,
        "leg_ratio": 0.50,
    },
}

# Function to get ratios for a specific age
def get_anthropometric_ratios(age_months: float) -> dict:
    """Get body segment ratios for a given age in months."""
    if age_months < 12:
        return ANTHROPOMETRIC_RATIOS["0-12_months"]
    elif age_months < 24:
        return ANTHROPOMETRIC_RATIOS["12-24_months"]
    elif age_months < 48:
        return ANTHROPOMETRIC_RATIOS["24-48_months"]
    else:
        return ANTHROPOMETRIC_RATIOS["48-60_months"]

# Measurement validation thresholds
HEIGHT_VALIDATION_SD = 3.0  # Flag if height is >3 SD from WHO median
SEGMENT_AGREEMENT_THRESHOLD = 0.15  # Max 15% difference between segment-based estimates
MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimum pose confidence to use measurement

# Expected height ranges by age (cm) - for sanity checking
# Based on WHO growth charts (approximately -3 SD to +3 SD)
HEIGHT_RANGES_BY_AGE = {
    (0, 6): (45, 75),      # 0-6 months
    (6, 12): (60, 85),     # 6-12 months
    (12, 24): (70, 95),    # 12-24 months
    (24, 36): (80, 105),   # 24-36 months
    (36, 48): (85, 115),   # 36-48 months
    (48, 60): (95, 125),   # 48-60 months
}
