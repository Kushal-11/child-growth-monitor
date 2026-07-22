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
IDENTIFYING_COLS = ["child_name", "caregiver_name", "village"]

ROSTER_COLS = (
    ["child_id"] + IDENTIFYING_COLS
    + [c for c in ALL_COLS if c != "child_id"]
)

TEMPLATE = (
    ",".join(ROSTER_COLS) + "\n"
    + "001,Example Child,Example Caregiver,Example Village,"
      "F,2023-04-12,2026-07-15,82.5,10.4,13.2,no,delete this example row\n"
)


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

    gt_rows = to_ground_truth_rows(roster_rows)

    # Validate BEFORE writing, so a bad roster never produces a
    # ground-truth file that looks usable.
    errors, warnings = validate_rows(gt_rows, fieldnames=ALL_COLS)
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
