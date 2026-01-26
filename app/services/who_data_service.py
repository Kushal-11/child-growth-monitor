"""
Unified WHO reference data loader.

Loads CSV and Excel WHO growth standard files into normalized DataFrames.
Provides lookup methods for Z-score computation.

Data source priority:
  - Excel files (L-M-S parameters) are preferred for WFH/WFL (0-2y and 2-5y)
  - CSV files are used for HAZ (0-59 months) and as fallback for WHZ
  - The who_wfh_0_59m.csv has a defect (positive z-scores mirror negatives),
    so Excel L-M-S files are the authoritative source for weight-for-height.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import WHO_DATA_FILES


class WHODataService:
    """Loads and provides access to WHO child growth reference data."""

    def __init__(self):
        self._haz_df: Optional[pd.DataFrame] = None
        self._wfl_lms: Optional[pd.DataFrame] = None  # Weight-for-Length (0-2y)
        self._wfh_lms: Optional[pd.DataFrame] = None  # Weight-for-Height (2-5y)
        self._whz_reference: Optional[pd.DataFrame] = None
        self._loaded = False

    def load_all(self):
        """Load all WHO data files at application startup."""
        self._load_haz_csv()
        self._load_excel_lms_files()
        self._load_whz_reference()
        self._loaded = True

    def _load_haz_csv(self):
        """Load Height-for-Age Z-score boundaries from CSV."""
        self._haz_df = pd.read_csv(WHO_DATA_FILES["haz_0_59m"])

    def _load_excel_lms_files(self):
        """Load Weight-for-Length/Height from Excel files into unified DataFrames."""
        frames_wfl = []
        frames_wfh = []

        for key, sex, target in [
            ("wfl_boys_0_2", "M", frames_wfl),
            ("wfl_girls_0_2", "F", frames_wfl),
            ("wfh_boys_2_5", "M", frames_wfh),
            ("wfh_girls_2_5", "F", frames_wfh),
        ]:
            df = pd.read_excel(WHO_DATA_FILES[key])
            # Normalize: first column is 'Length' or 'Height'
            index_col = df.columns[0]
            df = df.rename(columns={index_col: "index_value"})
            df["sex"] = sex
            # Ensure numeric L, M, S columns (some Excel files use Unicode minus)
            for col in ["L", "M", "S"]:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("\u2212", "-"), errors="coerce"
                )
            target.append(df)

        self._wfl_lms = pd.concat(frames_wfl, ignore_index=True)
        self._wfh_lms = pd.concat(frames_wfh, ignore_index=True)

    def _load_whz_reference(self):
        """Load WHZ quick reference from CSV."""
        self._whz_reference = pd.read_csv(WHO_DATA_FILES["whz_reference"])

    def get_haz_boundaries(self, sex: str, age_months: int) -> Optional[dict]:
        """Get HAZ Z-score boundary values for given sex and age.

        Returns dict with keys: z_minus_3..z_plus_3 (height values in cm).
        """
        measure = "length" if age_months < 24 else "height"
        row = self._haz_df[
            (self._haz_df["sex"] == sex)
            & (self._haz_df["measure"] == measure)
            & (self._haz_df["age_months"] == age_months)
        ]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_wfh_lms(
        self, sex: str, height_cm: float, age_months: float
    ) -> Optional[Tuple[float, float, float]]:
        """Get L, M, S parameters for Weight-for-Height/Length.

        Selects the correct dataset based on age:
          - age < 24 months: use WFL (Weight-for-Length) data
          - age >= 24 months: use WFH (Weight-for-Height) data

        Interpolates between the two nearest height entries if exact match
        not found.

        Returns (L, M, S) tuple or None if out of range.
        """
        df = self._wfl_lms if age_months < 24 else self._wfh_lms
        subset = df[df["sex"] == sex].sort_values("index_value")

        if subset.empty:
            return None

        # Find exact or interpolate
        exact = subset[np.isclose(subset["index_value"], height_cm, atol=0.05)]
        if not exact.empty:
            row = exact.iloc[0]
            return (float(row["L"]), float(row["M"]), float(row["S"]))

        # Interpolate between two nearest
        below = subset[subset["index_value"] <= height_cm]
        above = subset[subset["index_value"] >= height_cm]
        if below.empty or above.empty:
            return None

        low = below.iloc[-1]
        high = above.iloc[0]
        denom = high["index_value"] - low["index_value"]
        if denom == 0:
            return None

        fraction = (height_cm - low["index_value"]) / denom
        L = float(low["L"] + fraction * (high["L"] - low["L"]))
        M = float(low["M"] + fraction * (high["M"] - low["M"]))
        S = float(low["S"] + fraction * (high["S"] - low["S"]))
        return (L, M, S)

    def get_median_weight_for_height(
        self, sex: str, height_cm: float, age_months: float = 36.0
    ) -> Optional[float]:
        """Get median weight for a given height using LMS parameters.

        Uses the authoritative Excel-based LMS data for accurate median weight.
        The age_months parameter determines whether to use WFL (< 24mo) or WFH (>= 24mo) data.
        
        Args:
            sex: 'M' or 'F'
            height_cm: Height in centimeters
            age_months: Age in months (default 36, used to select correct reference)
            
        Returns:
            Median weight in kg, or None if height is out of range.
        """
        # Use LMS data for accurate median (M parameter = median)
        lms = self.get_wfh_lms(sex, height_cm, age_months)
        if lms is not None:
            L, M, S = lms
            return round(M, 2)
        
        # Fallback to CSV reference if LMS not available
        ref = self._whz_reference
        row = ref[(ref["sex"] == sex) & (ref["height_cm"] == round(height_cm))]
        if row.empty:
            return None
        return float(row.iloc[0]["median_kg"])

    def get_median_height_for_age(
        self, sex: str, age_months: int
    ) -> Optional[float]:
        """Get median height (z=0) for a given sex and age.
        
        Returns the WHO median height in cm for the specified age.
        Used for statistical height estimation when no reference object is available.
        
        Args:
            sex: 'M' or 'F'
            age_months: Age in whole months (0-59)
            
        Returns:
            Median height in cm, or None if age is out of range.
        """
        if self._haz_df is None:
            return None
        
        # Determine measure type based on age
        measure = "length" if age_months < 24 else "height"
        
        row = self._haz_df[
            (self._haz_df["sex"] == sex)
            & (self._haz_df["measure"] == measure)
            & (self._haz_df["age_months"] == age_months)
        ]
        
        if row.empty:
            return None
        
        # z_0 column contains the median height
        return float(row.iloc[0]["z_0"])

    def get_height_sd_for_age(
        self, sex: str, age_months: int
    ) -> Optional[float]:
        """Get approximate standard deviation of height for a given age.
        
        Computes SD from the difference between median and -1 SD boundary.
        Used for validation of height estimates.
        
        Returns:
            Approximate SD in cm, or None if age is out of range.
        """
        if self._haz_df is None:
            return None
        
        measure = "length" if age_months < 24 else "height"
        
        row = self._haz_df[
            (self._haz_df["sex"] == sex)
            & (self._haz_df["measure"] == measure)
            & (self._haz_df["age_months"] == age_months)
        ]
        
        if row.empty:
            return None
        
        # SD ≈ (z_0 - z_minus_1)
        z_0 = float(row.iloc[0]["z_0"])
        z_minus_1 = float(row.iloc[0]["z_minus_1"])
        return z_0 - z_minus_1

    def get_height_range_for_age(
        self, sex: str, age_months: int, num_sd: float = 3.0
    ) -> Optional[Tuple[float, float]]:
        """Get valid height range for a given age (±num_sd standard deviations).
        
        Args:
            sex: 'M' or 'F'
            age_months: Age in whole months (0-59)
            num_sd: Number of standard deviations (default 3.0)
            
        Returns:
            Tuple of (min_height_cm, max_height_cm), or None if out of range.
        """
        if self._haz_df is None:
            return None
        
        measure = "length" if age_months < 24 else "height"
        
        row = self._haz_df[
            (self._haz_df["sex"] == sex)
            & (self._haz_df["measure"] == measure)
            & (self._haz_df["age_months"] == age_months)
        ]
        
        if row.empty:
            return None
        
        # Use pre-computed boundaries for ±3 SD
        z_minus_3 = float(row.iloc[0]["z_minus_3"])
        z_plus_3 = float(row.iloc[0]["z_plus_3"])
        
        if num_sd == 3.0:
            return (z_minus_3, z_plus_3)
        
        # For other SD values, interpolate
        z_0 = float(row.iloc[0]["z_0"])
        sd = z_0 - float(row.iloc[0]["z_minus_1"])
        return (z_0 - num_sd * sd, z_0 + num_sd * sd)
