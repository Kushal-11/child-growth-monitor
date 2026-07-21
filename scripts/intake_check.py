"""
Stage 1 — intake check. Read-only scan of field_data/raw/.

Reports, per child folder: photos/videos found, front/side named photos,
ground-truth row present, anomalies. Run it as often as you like while
gathering data; it is the progress dashboard for the manual organization
work (see docs/field_data_guide.md).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/intake_check.py
    PYTHONPATH=. .venv/bin/python scripts/intake_check.py --raw field_data/raw
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_ground_truth import load_csv  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

MANIFEST_COLS = [
    "child_id", "n_photos", "n_videos", "front_named", "side_named",
    "unnamed_photos", "gt_row", "issues",
]


def check_child(child_dir: Path, gt_ids: set[str]) -> dict:
    """Inspect one child folder. Never modifies anything."""
    cid = child_dir.name
    photos: list[Path] = []
    videos: list[Path] = []
    issues: list[str] = []

    for f in sorted(child_dir.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
            if f.stat().st_size == 0:
                issues.append(f"zero-byte file: {f.name}")
                continue
            (photos if ext in IMAGE_EXTENSIONS else videos).append(f)

    front_named = any(p.stem.lower().startswith("front") for p in photos)
    side_named = any(p.stem.lower().startswith("side") for p in photos)
    unnamed = sum(
        1 for p in photos
        if not p.stem.lower().startswith(("front", "side"))
    )

    if not photos and not videos:
        issues.append("no photos or videos")
    gt_row = cid in gt_ids
    if not gt_row:
        issues.append("no ground-truth row")

    return {
        "child_id": cid,
        "n_photos": len(photos),
        "n_videos": len(videos),
        "front_named": front_named,
        "side_named": side_named,
        "unnamed_photos": unnamed,
        "gt_row": gt_row,
        "issues": "; ".join(issues),
    }


def run_intake(raw_dir: Path, gt_csv: Path, manifest_out: Path) -> list[dict]:
    gt_ids: set[str] = set()
    if gt_csv.exists():
        gt_ids = {
            (r.get("child_id") or "").strip()
            for r in load_csv(gt_csv)
            if (r.get("child_id") or "").strip()
        }

    rows = [
        check_child(d, gt_ids)
        for d in sorted(raw_dir.iterdir())
        if d.is_dir()
    ]

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(rows)

    # Ground-truth rows that have no folder yet
    folder_ids = {r["child_id"] for r in rows}
    orphan_gt = sorted(gt_ids - folder_ids)

    complete = sum(
        1 for r in rows
        if r["gt_row"] and r["n_photos"] > 0 and not r["issues"]
    )
    missing_side = sum(
        1 for r in rows if r["n_photos"] > 0 and not r["side_named"]
    )
    missing_gt = sum(1 for r in rows if not r["gt_row"])

    print(f"{len(rows)} child folders in {raw_dir}")
    print(f"  complete (photos + ground truth, no issues): {complete}")
    print(f"  missing side-named photo: {missing_side}")
    print(f"  missing ground-truth row: {missing_gt}")
    for r in rows:
        if r["issues"]:
            print(f"  [{r['child_id']}] {r['issues']}")
    if orphan_gt:
        print(f"  ground-truth rows with no folder: {', '.join(orphan_gt)}")
    print(f"Manifest written to {manifest_out}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("field_data/raw"))
    parser.add_argument("--ground-truth", type=Path,
                        default=Path("field_data/ground_truth.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("field_data/reports/intake_manifest.csv"))
    args = parser.parse_args()

    if not args.raw.is_dir():
        print(f"Raw directory not found: {args.raw}", file=sys.stderr)
        sys.exit(1)
    run_intake(args.raw, args.ground_truth, args.manifest)


if __name__ == "__main__":
    main()
