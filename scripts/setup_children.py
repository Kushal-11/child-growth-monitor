"""
Set up child folders and the pipeline's ground-truth file from one roster.

You maintain a single file, `field_data/roster.csv`, which holds the child's
NAME alongside their ID and measurements — so you can tell who is who while
entering data from the paper forms.

This script reads it and does two things:

  1. Creates `field_data/raw/<child_id>/` for every child in the roster.
  2. Writes `field_data/ground_truth.csv` with the identifying columns
     STRIPPED OUT, which is what the rest of the pipeline reads.

Names therefore never reach the assessment pipeline, the results CSV, or
the study report. The roster stays local (all of `field_data/` is
gitignored) and is the only place a child's name appears.

Re-run it any time you add children to the roster. It never deletes or
overwrites a photo folder that already exists.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/setup_children.py --template
    PYTHONPATH=. .venv/bin/python scripts/setup_children.py
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_ground_truth import ALL_COLS, validate_rows  # noqa: E402

DEFAULT_ROSTER = Path("field_data/roster.csv")
DEFAULT_RAW = Path("field_data/raw")
DEFAULT_GROUND_TRUTH = Path("field_data/ground_truth.csv")

# Columns that identify a real child. Kept in the roster, never written to
# ground_truth.csv, never seen by the assessment pipeline.
IDENTIFYING_COLS = ["child_name", "area"]

# The category the health worker wrote on the paper form. Recorded as a
# CROSS-CHECK, never as the gold standard: the pipeline computes status
# from the measurements via the WHO OR-rule, and a disagreement between
# the two means a transcription error worth investigating. Also excluded
# from ground_truth.csv so it cannot leak into the study's gold standard.
FIELD_CATEGORY_COL = "field_category"
VALID_CATEGORIES = {"SAM", "MAM", "NORMAL"}

ROSTER_COLS = (
    ["child_id"] + IDENTIFYING_COLS
    + [c for c in ALL_COLS if c != "child_id"]
    + [FIELD_CATEGORY_COL]
)

TEMPLATE = (
    ",".join(ROSTER_COLS) + "\n"
    + "001,Example Child,Example Village,"
      "F,12/04/2023,15/07/2026,82.5,10.4,13.2,no,"
      "DATES ARE DD/MM/YYYY - delete this example row,Normal\n"
)


DATE_COLS = ["date_of_birth", "measurement_date"]

# Indian convention, matching the CMAM paper forms: day first.
_DMY_SEPARATORS = "/-."


def normalize_date(raw: str) -> tuple[str, str]:
    """
    Accept a roster date and return (iso_value, error).

    Accepts DD/MM/YYYY as written on the paper forms (also DD-MM-YYYY and
    DD.MM.YYYY), and YYYY-MM-DD if already converted. Exactly one of the
    two is returned as ISO; anything else is an error, never a guess.

    Two-digit years are REJECTED. '30/5/25' cannot be distinguished from
    1925 by the parser, and a wrong century sails past the 0-60 month age
    gate as a merely-implausible date rather than an impossible one.

    Day-first is assumed because that is what the forms use. A date like
    05/06/2025 is therefore 5 June, never 6 May - which is why the caller
    prints the interpretation back for a human to eyeball.
    """
    from datetime import date

    text = (raw or "").strip()
    if not text:
        return "", ""

    # Already ISO (YYYY-MM-DD): 4-digit year first.
    if len(text) >= 8 and text[:4].isdigit() and "-" in text:
        try:
            return date.fromisoformat(text).isoformat(), ""
        except ValueError:
            return "", f"'{text}' is not a real date"

    sep = next((s for s in _DMY_SEPARATORS if s in text), None)
    if sep is None:
        return "", (
            f"'{text}' is not a date - write it as DD/MM/YYYY, e.g. 30/05/2025"
        )

    parts = [p.strip() for p in text.split(sep)]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return "", (
            f"'{text}' is not a date - write it as DD/MM/YYYY, e.g. 30/05/2025"
        )

    day, month, year = parts
    if len(year) != 4:
        return "", (
            f"'{text}' has a {len(year)}-digit year - write the full year, "
            f"e.g. 30/05/2025 not 30/05/25"
        )
    try:
        return date(int(year), int(month), int(day)).isoformat(), ""
    except ValueError:
        if int(month) > 12 and int(day) <= 12:
            return "", (
                f"'{text}' looks like MM/DD/YYYY - this file uses DD/MM/YYYY "
                f"(day first), so write it as {month}/{day}/{year}"
            )
        return "", f"'{text}' is not a real date (read as day/month/year)"


def normalize_roster_dates(
    rows: list[dict],
) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    """
    Rewrite date columns to ISO in place.

    Returns (errors, interpretations) where each interpretation is
    (child_id, column, original_text, iso_value) for rows whose date was
    written day-first, so the caller can show a human what it understood.
    """
    from datetime import date

    errors: list[str] = []
    interpretations: list[tuple[str, str, str, str]] = []
    for i, row in enumerate(rows, start=2):
        cid = (row.get("child_id") or "").strip() or "?"
        for col in DATE_COLS:
            original = (row.get(col) or "").strip()
            iso, err = normalize_date(original)
            if err:
                errors.append(f"row {i} (child {cid}): {col} {err}")
                continue
            row[col] = iso
            if iso and iso != original:
                pretty = date.fromisoformat(iso).strftime("%d %B %Y")
                interpretations.append((cid, col, original, pretty))
    return errors, interpretations


def _computed_category(row: dict) -> tuple[str, str]:
    """
    Compute WHO status from this row's measurements via the OR-rule.

    Returns (status, detail). status is '' when too little was measured to
    decide - which is not an error, just an unknown.
    """
    from datetime import date

    from app.services.nutrition_service import NutritionService
    from app.services.who_data_service import WHODataService

    from scripts.batch_assess import _combine_status, _muac_status

    global _WHO_SERVICES
    if _WHO_SERVICES is None:
        who = WHODataService()
        who.load_all()
        _WHO_SERVICES = (who, NutritionService(who))
    who, nutr = _WHO_SERVICES

    def _f(key: str):
        try:
            return float((row.get(key) or "").strip())
        except (TypeError, ValueError):
            return None

    sex = (row.get("sex") or "").strip().upper()
    height, weight, muac = _f("actual_height_cm"), _f("actual_weight_kg"), _f("muac_cm")
    oedema = (row.get("oedema") or "").strip().lower() == "yes"

    age_months = None
    try:
        dob = date.fromisoformat((row.get("date_of_birth") or "").strip())
        mdate = date.fromisoformat((row.get("measurement_date") or "").strip())
        age_months = (mdate - dob).days / 30.4375
    except ValueError:
        pass

    whz_status = None
    detail_bits = []
    if (height and weight and sex in ("M", "F")
            and age_months is not None and 0 <= age_months <= 60):
        whz = nutr.compute_whz(sex, age_months, height, weight)
        if whz is not None:
            whz_status = "SAM" if whz < -3 else "MAM" if whz < -2 else "Normal"
            detail_bits.append(f"WHZ {whz:+.2f}->{whz_status}")

    # age_months gates MUAC: WHO's absolute cutoffs are defined for
    # 6-59 months only, so _muac_status returns None outside that window.
    muac_status = _muac_status(muac, age_months)
    if muac_status:
        detail_bits.append(f"MUAC {muac}->{muac_status}")
    if oedema:
        detail_bits.append("oedema->SAM")

    combined = _combine_status(whz_status, muac_status, oedema)
    return (combined or ""), ", ".join(detail_bits)


_WHO_SERVICES = None


def check_field_categories(rows: list[dict]) -> list[str]:
    """
    Compare the health worker's written category against the category the
    WHO OR-rule computes from the same row's measurements.

    A mismatch is reported as a warning, never an error: it may be a
    transcription slip, but it may equally be clinical judgment the
    measurements alone do not capture. Either way a human should look.
    """
    warnings: list[str] = []
    for i, row in enumerate(rows, start=2):
        written = (row.get(FIELD_CATEGORY_COL) or "").strip()
        if not written:
            continue
        cid = (row.get("child_id") or "").strip() or "?"
        if written.upper() not in VALID_CATEGORIES:
            warnings.append(
                f"row {i} (child {cid}): field_category '{written}' is not "
                f"SAM, MAM, or Normal"
            )
            continue
        computed, detail = _computed_category(row)
        if not computed:
            continue
        if computed.upper() != written.upper():
            warnings.append(
                f"row {i} (child {cid}): form says {written.upper()}, "
                f"measurements compute {computed.upper()} ({detail}) "
                f"- check for a transcription error"
            )
    return warnings


def read_roster(path: Path) -> tuple[list[dict], list[str]]:
    """Return (rows, header) from the roster CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), (reader.fieldnames or [])


def to_ground_truth_rows(roster_rows: list[dict]) -> list[dict]:
    """Strip identifying columns, keeping only what the pipeline expects."""
    return [
        {col: (row.get(col) or "").strip() for col in ALL_COLS}
        for row in roster_rows
    ]


def make_folders(rows: list[dict], raw_dir: Path) -> tuple[list[str], list[str]]:
    """Create a photo folder per child. Returns (created, already_existed)."""
    created: list[str] = []
    existed: list[str] = []
    for row in rows:
        cid = (row.get("child_id") or "").strip()
        if not cid:
            continue
        folder = raw_dir / cid
        if folder.exists():
            existed.append(cid)
        else:
            folder.mkdir(parents=True)
            created.append(cid)
    return created, existed


def write_ground_truth(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--template", action="store_true",
                        help="Write a blank roster template and exit.")
    args = parser.parse_args()

    if args.template:
        if args.roster.exists():
            print(f"Refusing to overwrite existing {args.roster}", file=sys.stderr)
            sys.exit(1)
        args.roster.parent.mkdir(parents=True, exist_ok=True)
        args.roster.write_text(TEMPLATE)
        print(f"Roster template written to {args.roster}")
        print("Fill in one row per child, then re-run without --template.")
        return

    if not args.roster.exists():
        print(f"No roster found at {args.roster}", file=sys.stderr)
        print("Create one with:  scripts/setup_children.py --template",
              file=sys.stderr)
        sys.exit(1)

    roster_rows, header = read_roster(args.roster)
    if not roster_rows:
        print(f"{args.roster} has no rows yet — add one line per child.")
        return

    unknown = [c for c in header if c and c not in ROSTER_COLS]
    if unknown:
        print(f"Unexpected column(s) in roster: {', '.join(unknown)}",
              file=sys.stderr)
        print(f"Expected: {', '.join(ROSTER_COLS)}", file=sys.stderr)
        sys.exit(1)

    # Convert DD/MM/YYYY (as written on the paper forms) to ISO before
    # anything downstream sees it. Bad dates stop the run here.
    date_errors, interpretations = normalize_roster_dates(roster_rows)
    if date_errors:
        for e in date_errors:
            print(f"ERROR    {e}")
        print(f"\n{len(date_errors)} date problem(s) in {args.roster}. "
              f"Dates go in as DD/MM/YYYY, e.g. 30/05/2025.")
        print("Nothing was written.", file=sys.stderr)
        sys.exit(1)

    if interpretations:
        print("Dates read as day/month/year — check these are right:")
        for cid, col, original, pretty in interpretations:
            print(f"  child {cid}  {col}: {original}  ->  {pretty}")
        print()

    gt_rows = to_ground_truth_rows(roster_rows)

    # Validate BEFORE writing, so a bad roster never produces a
    # ground-truth file that looks usable.
    errors, warnings = validate_rows(gt_rows, fieldnames=ALL_COLS)
    warnings += check_field_categories(roster_rows)
    for w in warnings:
        print(f"WARNING  {w}")
    if errors:
        for e in errors:
            print(f"ERROR    {e}")
        print(f"\n{len(errors)} error(s) in {args.roster} — fix them and re-run.")
        print("Nothing was written.", file=sys.stderr)
        sys.exit(1)

    created, existed = make_folders(roster_rows, args.raw)
    write_ground_truth(gt_rows, args.ground_truth)

    print(f"{len(roster_rows)} children in roster")
    print(f"  photo folders created: {len(created)}")
    print(f"  photo folders already present: {len(existed)}")
    print(f"  wrote {args.ground_truth} (names stripped)")
    if created:
        preview = ", ".join(created[:10])
        more = f" ... (+{len(created) - 10} more)" if len(created) > 10 else ""
        print(f"\nPut each child's photos in {args.raw}/<id>/  e.g. {preview}{more}")


if __name__ == "__main__":
    main()
