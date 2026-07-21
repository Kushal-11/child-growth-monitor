"""
Validate field_data/ground_truth.csv before any assessment runs.

Typos in ground truth silently corrupt the whole study, so this gate is
mandatory: assessment must refuse to run until this passes.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/validate_ground_truth.py            # validate default path
    PYTHONPATH=. .venv/bin/python scripts/validate_ground_truth.py --template # write blank CSV
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from typing import Optional

DEFAULT_CSV = Path("field_data/ground_truth.csv")

REQUIRED_COLS = ["child_id", "sex", "date_of_birth", "measurement_date"]
ALL_COLS = REQUIRED_COLS + [
    "actual_height_cm", "actual_weight_kg", "muac_cm", "oedema", "notes",
]

# Plausibility ranges (spec: reject impossible values)
HEIGHT_RANGE_CM = (40.0, 130.0)
WEIGHT_RANGE_KG = (2.0, 30.0)
MUAC_RANGE_CM   = (8.0, 20.0)
AGE_RANGE_MONTHS = (0.0, 60.0)

TEMPLATE = """\
child_id,sex,date_of_birth,measurement_date,actual_height_cm,actual_weight_kg,muac_cm,oedema,notes
001,M,2023-04-12,2026-07-15,82.5,10.4,13.2,no,
002,F,2024-01-30,2026-07-15,74.0,8.1,12.1,,example row - delete me
"""


def _parse_date(val: str) -> Optional[date]:
    try:
        return date.fromisoformat(val.strip())
    except (ValueError, AttributeError):
        return None


def _parse_float(val: str) -> Optional[float]:
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return None


def _check_range(
    errors: list[str], tag: str, name: str,
    raw: str, lo: float, hi: float, warnings: list[str],
) -> None:
    if not (raw or "").strip():
        warnings.append(f"{tag}: {name} missing")
        return
    v = _parse_float(raw)
    if v is None or not (lo <= v <= hi):
        errors.append(f"{tag}: {name} '{raw}' out of range {lo}-{hi}")


def validate_rows(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Empty errors == safe to assess."""
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for i, row in enumerate(rows, start=2):  # row 1 is the header
        cid = (row.get("child_id") or "").strip()
        tag = f"row {i} (child {cid or '?'})"

        for col in REQUIRED_COLS:
            if not (row.get(col) or "").strip():
                errors.append(f"{tag}: {col} missing")

        if cid:
            if cid in seen_ids:
                errors.append(f"{tag}: duplicate child_id")
            seen_ids.add(cid)

        sex = (row.get("sex") or "").strip().upper()
        if sex and sex not in ("M", "F"):
            errors.append(f"{tag}: sex '{row.get('sex')}' must be M or F")

        oedema = (row.get("oedema") or "").strip().lower()
        if oedema and oedema not in ("yes", "no"):
            errors.append(f"{tag}: oedema '{row.get('oedema')}' must be yes/no/blank")

        dob = _parse_date(row.get("date_of_birth") or "")
        mdate = _parse_date(row.get("measurement_date") or "")
        if (row.get("date_of_birth") or "").strip() and dob is None:
            errors.append(f"{tag}: date_of_birth not YYYY-MM-DD")
        if (row.get("measurement_date") or "").strip() and mdate is None:
            errors.append(f"{tag}: measurement_date not YYYY-MM-DD")
        if dob and mdate:
            if mdate < dob:
                errors.append(f"{tag}: measurement_date before date_of_birth")
            age_months = (mdate - dob).days / 30.4375
            lo, hi = AGE_RANGE_MONTHS
            if not (lo <= age_months <= hi):
                errors.append(
                    f"{tag}: age {age_months:.1f} months out of range {lo}-{hi}"
                )
        if mdate and mdate > date.today():
            errors.append(f"{tag}: measurement_date in the future")

        _check_range(errors, tag, "height", row.get("actual_height_cm") or "",
                     *HEIGHT_RANGE_CM, warnings)
        _check_range(errors, tag, "weight", row.get("actual_weight_kg") or "",
                     *WEIGHT_RANGE_KG, warnings)
        _check_range(errors, tag, "muac", row.get("muac_cm") or "",
                     *MUAC_RANGE_CM, warnings)

    return errors, warnings


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--template", action="store_true",
                        help="Write a blank template CSV and exit.")
    args = parser.parse_args()

    if args.template:
        if args.csv_path.exists():
            print(f"Refusing to overwrite existing {args.csv_path}", file=sys.stderr)
            sys.exit(1)
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        args.csv_path.write_text(TEMPLATE)
        print(f"Template written to {args.csv_path}")
        return

    if not args.csv_path.exists():
        print(f"Not found: {args.csv_path} (use --template to create one)",
              file=sys.stderr)
        sys.exit(1)

    rows = load_csv(args.csv_path)
    errors, warnings = validate_rows(rows)
    for w in warnings:
        print(f"WARNING  {w}")
    for e in errors:
        print(f"ERROR    {e}")
    print(f"\n{len(rows)} rows: {len(errors)} errors, {len(warnings)} warnings")
    if errors:
        print("FAILED — fix the errors above before running assessment.")
        sys.exit(1)
    print("OK — ground truth is safe to use.")


if __name__ == "__main__":
    main()
