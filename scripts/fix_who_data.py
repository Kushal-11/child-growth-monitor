"""
Fix corrupted WHO CSV files by regenerating them from the verified Excel LMS files.

Bugs found during data audit:
1. who_wfh_0_59m.csv: z_plus_1/2/3 are copies of z_minus_1/2/3 in reverse.
   Correct positive z-scores from Excel: at girls 65cm z_plus_1=7.9 (not 6.6).
2. who_whz_reference.csv: median weights don't match WHO LMS tables
   (e.g. boys 80cm shows 14.0 kg vs correct 10.6 kg from Excel).

The Excel WFH/WFL files are the authoritative source (verified against
official WHO Child Growth Standards publications).

Run:  python scripts/fix_who_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

EXCEL_FILES = {
    "wfl_boys":  DATA_DIR / "wfl_boys_0-to-2-years_zscores.xlsx",
    "wfl_girls": DATA_DIR / "wfl_girls_0-to-2-years_zscores.xlsx",
    "wfh_boys":  DATA_DIR / "wfh_boys_2-to-5-years_zscores.xlsx",
    "wfh_girls": DATA_DIR / "wfh_girls_2-to-5-years_zscores.xlsx",
}


def load_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Normalise column name (first col is 'Length' or 'Height')
    index_col = df.columns[0]
    df = df.rename(columns={index_col: "length_height_cm"})
    return df


def regenerate_wfh_csv():
    """
    Rebuild who_wfh_0_59m.csv with correct positive z-scores from Excel.

    Original file had: z_plus_1 = z_minus_1, z_plus_2 = z_minus_2, z_plus_3 = z_minus_3
    This fixes that by reading SD1, SD2, SD3 directly from the Excel files.
    """
    rows = []
    for sex, wfl_key, wfh_key in [
        ("F", "wfl_girls", "wfh_girls"),
        ("M", "wfl_boys",  "wfh_boys"),
    ]:
        wfl = load_excel(EXCEL_FILES[wfl_key])
        wfh = load_excel(EXCEL_FILES[wfh_key])

        for df, measure_label in [(wfl, "length"), (wfh, "height")]:
            for _, row in df.iterrows():
                rows.append({
                    "sex": sex,
                    "measure": measure_label,
                    "length_height_cm": float(row["length_height_cm"]),
                    "z_minus_3": float(row["SD3neg"]),
                    "z_minus_2": float(row["SD2neg"]),
                    "z_minus_1": float(row["SD1neg"]),
                    "z_0":       float(row["SD0"]),
                    "z_plus_1":  float(row["SD1"]),
                    "z_plus_2":  float(row["SD2"]),
                    "z_plus_3":  float(row["SD3"]),
                })

    out = pd.DataFrame(rows).sort_values(["sex", "measure", "length_height_cm"])
    out_path = DATA_DIR / "who_wfh_0_59m.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Wrote {len(out)} rows → {out_path}")

    # Spot-check: girls at 65cm should have z_plus_1 ≈ 7.9
    check = out[(out["sex"] == "F") & (out["measure"] == "height") &
                (np.isclose(out["length_height_cm"], 65.0))]
    if not check.empty:
        z_plus_1 = check.iloc[0]["z_plus_1"]
        ok = abs(z_plus_1 - 7.9) < 0.1
        status = "✓" if ok else "✗ UNEXPECTED"
        print(f"  Spot-check F height 65cm z_plus_1={z_plus_1:.2f} (expected ~7.9) {status}")


def regenerate_whz_reference():
    """
    Rebuild who_whz_reference.csv with correct median/-2SD/-3SD weights.

    Original file had wrong weight values (shifted ~15 cm, cause unknown).
    Regenerated from Excel LMS files using:
        -2SD weight = M * (1 + L * S * (-2))^(1/L)  when L != 0
        -3SD weight = M * (1 + L * S * (-3))^(1/L)  when L != 0
    """
    rows = []
    for sex, wfl_key, wfh_key in [
        ("F", "wfl_girls", "wfh_girls"),
        ("M", "wfl_boys",  "wfh_boys"),
    ]:
        wfl = load_excel(EXCEL_FILES[wfl_key])
        wfh = load_excel(EXCEL_FILES[wfh_key])

        for df in [wfl, wfh]:
            for _, row in df.iterrows():
                L = float(row["L"])
                M = float(row["M"])
                S = float(row["S"])
                h = float(row["length_height_cm"])

                def lms_weight(z):
                    if abs(L) < 1e-6:
                        return M * np.exp(S * z)
                    val = 1 + L * S * z
                    if val <= 0:
                        return None
                    return M * (val ** (1.0 / L))

                w_median = round(M, 2)
                w_minus2 = lms_weight(-2.0)
                w_minus3 = lms_weight(-3.0)
                if w_median and w_minus2 and w_minus3:
                    rows.append({
                        "sex": sex,
                        "height_cm": h,
                        "minus2sd_kg": round(w_minus2, 2),
                        "minus3sd_kg": round(w_minus3, 2),
                        "median_kg":   w_median,
                    })

    out = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["sex", "height_cm"])
        .sort_values(["sex", "height_cm"])
    )
    out_path = DATA_DIR / "who_whz_reference.csv"
    out.to_csv(out_path, index=False, float_format="%.2f")
    print(f"  Wrote {len(out)} rows → {out_path}")

    # Spot-check: boys at 80cm median should be ~10.6 kg
    check = out[(out["sex"] == "M") & (np.isclose(out["height_cm"], 80.0))]
    if not check.empty:
        median = check.iloc[0]["median_kg"]
        ok = abs(median - 10.6) < 0.3
        status = "✓" if ok else "✗ UNEXPECTED"
        print(f"  Spot-check M height 80cm median={median:.2f} kg (expected ~10.6) {status}")

        minus2 = check.iloc[0]["minus2sd_kg"]
        ok2 = abs(minus2 - 9.0) < 0.3
        status2 = "✓" if ok2 else "✗ UNEXPECTED"
        print(f"  Spot-check M height 80cm -2SD={minus2:.2f} kg (expected ~9.0) {status2}")


def main():
    print("Fixing WHO CSV files from verified Excel LMS sources...\n")

    print("1. Regenerating who_wfh_0_59m.csv (weight-for-height z-score boundaries):")
    regenerate_wfh_csv()

    print("\n2. Regenerating who_whz_reference.csv (SAM/MAM weight thresholds):")
    regenerate_whz_reference()

    print("\nDone. Both files now derived from the verified Excel LMS data.")


if __name__ == "__main__":
    sys.exit(main())
