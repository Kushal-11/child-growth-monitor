# Field Data Pipeline & Comparison Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the staged pipeline (intake check → photo QC/cleaning → ground-truth validation → assessment + study analysis) that turns per-child field folders into a defensible 200-child comparison report, per `docs/superpowers/specs/2026-07-21-field-data-pipeline-design.md`.

**Architecture:** File-based stages under a gitignored `field_data/` tree. Shared photo-scoring lives in a new `scripts/photo_qc.py`; each stage is a standalone CLI script in `scripts/` with pure, unit-testable core functions. `batch_assess.py` gains measurement-date-aware age, master-CSV ground truth, and WHO OR-rule statuses. A new `scripts/study_stats.py` + `scripts/analyze_results.py` produce the study report.

**Tech Stack:** Python 3.11 (`.venv`), OpenCV, MediaPipe PoseLandmarker (mocked in unit tests), numpy, scikit-learn (weighted κ), pytest.

## Global Constraints

- Run everything as `PYTHONPATH=. .venv/bin/python ...`; tests as `PYTHONPATH=. .venv/bin/python -m pytest tests/... -v`.
- Type hints on all function signatures (project convention).
- WHO MUAC thresholds are fixed: `<11.5` SAM, `11.5–12.5` MAM, `>=12.5` Normal. Never alter.
- No silent failures: every skipped/failed child must appear in a report with a reason.
- `field_data/` must never enter git (no photos, no measurements). Never commit `.db`, uploads, or model files.
- Unit tests must not load the MediaPipe model; pure functions take landmark lists / fake scores. (MediaPipe *is* installed in `.venv`, only the `.task` model load is expensive.)
- The pipeline never modifies `field_data/raw/`.
- Commit messages: imperative, concise, **no AI/Claude attribution of any kind** (project rule). Author must be `Kushal-11 <kushaltherokar1010@gmail.com>`.
- Quality thresholds are named constants with defaults from `extract_best_frame.py`; expected to be tuned once on the first real batch, then frozen.

---

### Task 1: `field_data/` gitignore + data-organization guide

**Files:**
- Modify: `.gitignore` (append at end)
- Create: `docs/field_data_guide.md`

**Interfaces:**
- Produces: the canonical folder layout (`field_data/raw/<id>/`, `field_data/cleaned/`, `field_data/reports/`, `field_data/ground_truth.csv`) that every later task assumes; the guide documents the `front*`/`side*` filename convention consumed by Tasks 3 and 6.

- [ ] **Step 1: Append the field_data rule to `.gitignore`**

Append after the existing `data/batch_results.csv` lines:

```gitignore

# Field study data (child photos + medical ground truth — never commit)
field_data/
```

- [ ] **Step 2: Verify git ignores the tree**

Run:
```bash
mkdir -p field_data/raw/000 && touch field_data/raw/000/front.jpg
git check-ignore -v field_data/raw/000/front.jpg
rm -r field_data/raw/000
```
Expected: output ends with `.gitignore:<line>:field_data/	field_data/raw/000/front.jpg` (exit code 0).

- [ ] **Step 3: Write `docs/field_data_guide.md`**

```markdown
# Field Data Organization Guide

How to arrange each child's photos, videos and paper measurements so the
pipeline can process them. Follow this while gathering; run the intake check
(Stage 1) at any time to see what is still missing.

## Layout

    field_data/                  <- created by you, ignored by git
      raw/                       <- you fill this, one folder per child
        001/
          front.jpg              <- frontal photo (best if named like this)
          side.jpg               <- side photo
          extra_01.jpg           <- any other photos: keep them, pipeline picks best
          walk.mp4               <- optional video
        002/
          ...
      ground_truth.csv           <- one row per child, typed from the paper forms
      cleaned/                   <- pipeline output. Never edit by hand.
      reports/                   <- pipeline output. Never edit by hand.

## Child IDs

- Folder name = child ID. Use zero-padded numbers: `001`, `002`, ... `250`.
- NEVER put the child's name in a folder name, file name, or the CSV.
- Keep the name-to-ID mapping on paper or a private sheet OUTSIDE this
  project folder. The repo and pipeline only ever see the numeric ID.
- Write the same ID on the child's paper form at measurement time.

## What each child folder should contain

Aim for, in order of importance:

1. **One frontal photo** — child standing straight, facing the camera,
   full body visible head to feet, arms slightly away from the body.
2. **One side photo** — child turned 90°, again full body.
3. Optional: extra shots, video clips. Keep everything; the cleaner
   scores all of them and picks the best automatically.

Name files `front...` / `side...` when you know which is which
(`front.jpg`, `front_2.jpg`, `side_a.jpg`). If a photo is unnamed the
pipeline guesses the orientation from the pose and flags the guess for
your confirmation in the QC report.

## Photo quality basics (saves recapture trips)

- Whole child in frame: head AND feet, nothing cropped.
- Camera at roughly the child's waist height, phone held vertically.
- Good light, child not in shadow; plain background if possible.
- Exactly one person in frame (no siblings/adults behind).
- Hold the phone still — motion blur is the top rejection reason.

## Ground truth CSV

`field_data/ground_truth.csv`, one row per child:

    child_id,sex,date_of_birth,measurement_date,actual_height_cm,actual_weight_kg,muac_cm,oedema,notes
    001,M,2023-04-12,2026-07-15,82.5,10.4,13.2,no,
    002,F,2024-01-30,2026-07-15,74.0,8.1,12.1,,left early

- `sex`: `M` or `F`.
- Dates: `YYYY-MM-DD`. `measurement_date` = the day height/weight/MUAC
  were taken (photos must be same-day — this drives the age used for
  z-scores).
- Height in cm, weight in kg, MUAC in cm. Decimal point, never a comma.
- `oedema`: `yes` / `no` / blank if not checked.
- Leave a value blank if it truly wasn't measured — never guess.
- After typing everything in, re-check a random 10–15% of rows against
  the paper forms, and run the validator (Stage 3) before any assessment.

## Rules

- Treat `raw/` as an archive: after dropping files in, don't rename,
  edit, or delete them. The pipeline never modifies `raw/` either.
- Don't commit `field_data/` — it is gitignored; `git status` must never
  show it.
- Videos welcome but photos preferred: a sharp photo beats a video frame.

## Workflow while gathering

1. Measure the child, fill the paper form, assign the next free ID.
2. Same day: photos (front, side, extras) into `field_data/raw/<id>/`.
3. Type the form into `ground_truth.csv` (or batch it, but soon —
   backlogs breed typos).
4. Any time: run the intake check to list gaps (missing side photo,
   missing CSV row, empty folders). Fix gaps while you still have field
   access to the child.
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore docs/field_data_guide.md
git commit -m "docs: add field-data organization guide; gitignore field_data tree"
```

---

### Task 2: Ground-truth validator (`scripts/validate_ground_truth.py`)

**Files:**
- Create: `scripts/validate_ground_truth.py`
- Test: `tests/test_validate_ground_truth.py`

**Interfaces:**
- Produces: `validate_rows(rows: list[dict]) -> tuple[list[str], list[str]]` (errors, warnings); `load_csv(path: Path) -> list[dict]`; CLI `--template` writes the header+examples to `field_data/ground_truth.csv`. Tasks 6 and 8 assume a validated CSV with exactly these columns: `child_id,sex,date_of_birth,measurement_date,actual_height_cm,actual_weight_kg,muac_cm,oedema,notes`.

- [ ] **Step 1: Write the failing tests**

`tests/test_validate_ground_truth.py`:

```python
"""Tests for scripts/validate_ground_truth.py."""
from scripts.validate_ground_truth import validate_rows


def _good_row(**over) -> dict:
    row = {
        "child_id": "001", "sex": "M",
        "date_of_birth": "2023-04-12", "measurement_date": "2026-07-15",
        "actual_height_cm": "82.5", "actual_weight_kg": "10.4",
        "muac_cm": "13.2", "oedema": "no", "notes": "",
    }
    row.update(over)
    return row


def test_valid_row_passes():
    errors, warnings = validate_rows([_good_row()])
    assert errors == []
    assert warnings == []


def test_height_out_of_range_rejected():
    errors, _ = validate_rows([_good_row(actual_height_cm="8.5")])
    assert len(errors) == 1 and "height" in errors[0]


def test_weight_out_of_range_rejected():
    errors, _ = validate_rows([_good_row(actual_weight_kg="110")])
    assert len(errors) == 1 and "weight" in errors[0]


def test_muac_out_of_range_rejected():
    errors, _ = validate_rows([_good_row(muac_cm="25.0")])
    assert len(errors) == 1 and "muac" in errors[0].lower()


def test_measurement_before_birth_rejected():
    errors, _ = validate_rows([_good_row(measurement_date="2022-01-01")])
    assert any("before date_of_birth" in e for e in errors)


def test_future_measurement_rejected():
    errors, _ = validate_rows([_good_row(measurement_date="2099-01-01")])
    assert any("future" in e for e in errors)


def test_age_over_60_months_rejected():
    errors, _ = validate_rows([_good_row(date_of_birth="2018-01-01")])
    assert any("age" in e for e in errors)


def test_bad_sex_rejected():
    errors, _ = validate_rows([_good_row(sex="X")])
    assert any("sex" in e for e in errors)


def test_bad_oedema_rejected():
    errors, _ = validate_rows([_good_row(oedema="maybe")])
    assert any("oedema" in e for e in errors)


def test_duplicate_child_id_rejected():
    errors, _ = validate_rows([_good_row(), _good_row()])
    assert any("duplicate" in e for e in errors)


def test_missing_required_field_rejected():
    errors, _ = validate_rows([_good_row(date_of_birth="")])
    assert any("date_of_birth" in e for e in errors)


def test_missing_optional_measurements_warn_not_error():
    errors, warnings = validate_rows(
        [_good_row(actual_height_cm="", actual_weight_kg="", muac_cm="", oedema="")]
    )
    assert errors == []
    assert len(warnings) == 3  # height, weight, muac missing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validate_ground_truth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.validate_ground_truth'`.

- [ ] **Step 3: Implement `scripts/validate_ground_truth.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_validate_ground_truth.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_ground_truth.py tests/test_validate_ground_truth.py
git commit -m "feat: add ground-truth CSV validator with plausibility gates"
```

---

### Task 3: Intake check (`scripts/intake_check.py`)

**Files:**
- Create: `scripts/intake_check.py`
- Test: `tests/test_intake_check.py`

**Interfaces:**
- Consumes: `load_csv` from `scripts.validate_ground_truth` (Task 2) to read ground-truth IDs.
- Produces: `check_child(child_dir: Path, gt_ids: set[str]) -> dict` returning keys `child_id, n_photos, n_videos, front_named, side_named, unnamed_photos, gt_row, issues`; `run_intake(raw_dir: Path, gt_csv: Path, manifest_out: Path) -> list[dict]`; manifest CSV at `field_data/reports/intake_manifest.csv` with exactly those columns (Task 8 reads it for coverage accounting).

- [ ] **Step 1: Write the failing tests**

`tests/test_intake_check.py`:

```python
"""Tests for scripts/intake_check.py."""
from pathlib import Path

from scripts.intake_check import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, check_child


def _make_child(tmp_path: Path, cid: str, files: list[str]) -> Path:
    d = tmp_path / cid
    d.mkdir()
    for name in files:
        (d / name).write_bytes(b"x" * 100)
    return d


def test_complete_child(tmp_path):
    d = _make_child(tmp_path, "001", ["front.jpg", "side.jpg", "walk.mp4"])
    row = check_child(d, gt_ids={"001"})
    assert row["child_id"] == "001"
    assert row["n_photos"] == 2 and row["n_videos"] == 1
    assert row["front_named"] and row["side_named"]
    assert row["gt_row"] is True
    assert row["issues"] == ""


def test_missing_side_and_gt(tmp_path):
    d = _make_child(tmp_path, "002", ["front.jpg"])
    row = check_child(d, gt_ids=set())
    assert not row["side_named"]
    assert row["gt_row"] is False
    assert "no ground-truth row" in row["issues"]


def test_unnamed_photos_counted(tmp_path):
    d = _make_child(tmp_path, "003", ["front.jpg", "IMG_1234.jpg", "IMG_1235.jpg"])
    row = check_child(d, gt_ids={"003"})
    assert row["unnamed_photos"] == 2


def test_empty_folder_flagged(tmp_path):
    d = tmp_path / "004"
    d.mkdir()
    row = check_child(d, gt_ids={"004"})
    assert "no photos or videos" in row["issues"]


def test_zero_byte_file_flagged(tmp_path):
    d = _make_child(tmp_path, "005", ["front.jpg"])
    (d / "side.jpg").write_bytes(b"")
    row = check_child(d, gt_ids={"005"})
    assert "zero-byte" in row["issues"]


def test_extension_sets_disjoint():
    assert not (IMAGE_EXTENSIONS & VIDEO_EXTENSIONS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_intake_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.intake_check'`.

- [ ] **Step 3: Implement `scripts/intake_check.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_intake_check.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/intake_check.py tests/test_intake_check.py
git commit -m "feat: add intake check manifest for field data gathering"
```

---

### Task 4: Photo QC scoring module (`scripts/photo_qc.py`)

**Files:**
- Create: `scripts/photo_qc.py`
- Test: `tests/test_photo_qc.py`

**Interfaces:**
- Produces (consumed by Task 5):
  - `@dataclass PhotoScore: pose_confidence, coverage, upright, frontal, sharpness: float; orientation: str; usable: bool; reason: str`
  - `landmark_metrics(lms: Sequence) -> Optional[dict]` — pure; landmark items need `.x`, `.y`, `.visibility`; returns `{"pose_confidence", "coverage", "upright", "frontal", "orientation"}` or `None` when required landmarks aren't visible.
  - `sharpness_of(image_bgr: np.ndarray) -> float`
  - `score_photo(image_bgr: np.ndarray, landmarker) -> PhotoScore`
  - `build_landmarker()` — real MediaPipe landmarker (never called in unit tests)
  - Threshold constants: `MIN_POSE_CONFIDENCE=0.5`, `MIN_COVERAGE=0.6`, `MIN_UPRIGHT=0.75`, `MIN_SHARPNESS=40.0`, `FRONT_RATIO=0.45`, `SIDE_RATIO=0.25`

- [ ] **Step 1: Write the failing tests**

`tests/test_photo_qc.py`:

```python
"""Tests for scripts/photo_qc.py — pure landmark math, no MediaPipe model."""
from dataclasses import dataclass

import numpy as np

from scripts.photo_qc import KP, landmark_metrics, sharpness_of


@dataclass
class FakeLandmark:
    x: float
    y: float
    visibility: float = 0.99


def _upright_front_pose() -> list:
    """33 landmarks for an ideal upright, camera-facing child."""
    lms = [FakeLandmark(0.5, 0.5) for _ in range(33)]
    lms[KP["nose"]] = FakeLandmark(0.50, 0.08)
    lms[KP["left_shoulder"]] = FakeLandmark(0.62, 0.25)
    lms[KP["right_shoulder"]] = FakeLandmark(0.38, 0.25)
    lms[KP["left_hip"]] = FakeLandmark(0.58, 0.50)
    lms[KP["right_hip"]] = FakeLandmark(0.42, 0.50)
    lms[KP["left_knee"]] = FakeLandmark(0.57, 0.68)
    lms[KP["right_knee"]] = FakeLandmark(0.43, 0.68)
    lms[KP["left_ankle"]] = FakeLandmark(0.56, 0.86)
    lms[KP["right_ankle"]] = FakeLandmark(0.44, 0.86)
    lms[KP["left_heel"]] = FakeLandmark(0.56, 0.88)
    lms[KP["right_heel"]] = FakeLandmark(0.44, 0.88)
    return lms


def test_ideal_front_pose_scores_high():
    m = landmark_metrics(_upright_front_pose())
    assert m is not None
    assert m["pose_confidence"] > 0.9
    assert m["coverage"] > 0.9          # 0.08 -> 0.88 span, normalised by 0.80
    assert m["upright"] == 1.0
    assert m["frontal"] > 0.8
    assert m["orientation"] == "front"


def test_side_pose_classified_side():
    lms = _upright_front_pose()
    # Shoulders and hips nearly overlap in x when the child is side-on
    lms[KP["left_shoulder"]] = FakeLandmark(0.51, 0.25)
    lms[KP["right_shoulder"]] = FakeLandmark(0.49, 0.25)
    lms[KP["left_hip"]] = FakeLandmark(0.51, 0.50)
    lms[KP["right_hip"]] = FakeLandmark(0.49, 0.50)
    m = landmark_metrics(lms)
    assert m is not None
    assert m["orientation"] == "side"


def test_low_visibility_returns_none():
    lms = _upright_front_pose()
    lms[KP["left_ankle"]] = FakeLandmark(0.56, 0.86, visibility=0.1)
    assert landmark_metrics(lms) is None


def test_upside_down_pose_scores_low_upright():
    lms = _upright_front_pose()
    for name in KP:
        lm = lms[KP[name]]
        lms[KP[name]] = FakeLandmark(lm.x, 1.0 - lm.y, lm.visibility)
    m = landmark_metrics(lms)
    assert m is not None
    assert m["upright"] == 0.0


def test_sharpness_flat_image_is_zero():
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert sharpness_of(flat) == 0.0


def test_sharpness_noisy_image_is_high():
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    assert sharpness_of(noisy) > 100.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_photo_qc.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.photo_qc'`.

- [ ] **Step 3: Implement `scripts/photo_qc.py`**

The scoring formulas intentionally mirror `scripts/extract_best_frame.py`
(`_score_frame`); that script stays untouched to avoid regressions.

```python
"""
Photo quality scoring for field-data cleaning (Stage 2).

Pure landmark math lives in landmark_metrics() so unit tests never need
the MediaPipe model; score_photo() wraps it for real images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

# MediaPipe pose landmark indices
KP = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_heel":      29,
    "right_heel":     30,
}

REQUIRED = ["nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_ankle", "right_ankle"]

# Quality thresholds — defaults derived from extract_best_frame.py scoring.
# Tune once on the first real batch, then freeze for the study.
MIN_VISIBILITY      = 0.4
MIN_POSE_CONFIDENCE = 0.5
MIN_COVERAGE        = 0.6    # normalised head-to-heel span (0.80 raw span = 1.0)
MIN_UPRIGHT         = 0.75
MIN_SHARPNESS       = 40.0   # Laplacian variance
FRONT_RATIO         = 0.45   # shoulder-width / torso-height above this = front
SIDE_RATIO          = 0.25   # below this = side; between = unknown

POSE_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "pose_landmarker_heavy.task"
)


@dataclass
class PhotoScore:
    pose_confidence: float
    coverage: float
    upright: float
    frontal: float
    sharpness: float
    orientation: str        # 'front' | 'side' | 'unknown'
    usable: bool
    reason: str             # empty when usable


def sharpness_of(image_bgr: np.ndarray) -> float:
    """Laplacian variance — motion-blur / focus measure (higher = sharper)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def landmark_metrics(lms: Sequence) -> Optional[dict]:
    """
    Compute pose sub-scores from a landmark list (items need .x/.y/.visibility).
    Returns None when a required landmark is not visible enough.
    """
    def vis(name: str) -> float:
        return lms[KP[name]].visibility or 0.0

    def y(name: str) -> float:
        return lms[KP[name]].y

    def x(name: str) -> float:
        return lms[KP[name]].x

    for req in REQUIRED:
        if vis(req) < MIN_VISIBILITY:
            return None

    key_joints = ["nose", "left_shoulder", "right_shoulder",
                  "left_hip", "right_hip", "left_knee", "right_knee",
                  "left_ankle", "right_ankle"]
    pose_confidence = float(np.mean([vis(j) for j in key_joints]))

    nose_y = y("nose")
    heel_y = max(y("left_heel"), y("right_heel"),
                 y("left_ankle"), y("right_ankle"))
    coverage = min(max(0.0, heel_y - nose_y) / 0.80, 1.0)

    order_pairs = [
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ]
    upright = sum(1 for a, b in order_pairs if y(a) < y(b)) / len(order_pairs)

    shoulder_y_diff = abs(y("left_shoulder") - y("right_shoulder"))
    shoulder_width = abs(x("left_shoulder") - x("right_shoulder")) + 1e-6
    frontal = max(0.0, 1.0 - (shoulder_y_diff / shoulder_width) * 3.0)

    torso_h = abs(
        (y("left_hip") + y("right_hip")) / 2
        - (y("left_shoulder") + y("right_shoulder")) / 2
    ) + 1e-6
    ratio = shoulder_width / torso_h
    if ratio >= FRONT_RATIO:
        orientation = "front"
    elif ratio <= SIDE_RATIO:
        orientation = "side"
    else:
        orientation = "unknown"

    return {
        "pose_confidence": pose_confidence,
        "coverage": coverage,
        "upright": upright,
        "frontal": frontal,
        "orientation": orientation,
    }


def _verdict(m: dict, sharpness: float) -> tuple[bool, str]:
    if m["pose_confidence"] < MIN_POSE_CONFIDENCE:
        return False, f"low pose confidence ({m['pose_confidence']:.2f})"
    if m["coverage"] < MIN_COVERAGE:
        return False, "body not fully in frame (head-to-feet)"
    if m["upright"] < MIN_UPRIGHT:
        return False, "child not standing upright"
    if sharpness < MIN_SHARPNESS:
        return False, f"image too blurry (sharpness {sharpness:.0f})"
    return True, ""


def score_photo(image_bgr: np.ndarray, landmarker) -> PhotoScore:
    """Run pose detection on one photo and score it."""
    import mediapipe as mp

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    sharp = sharpness_of(image_bgr)

    if not result.pose_landmarks:
        return PhotoScore(0, 0, 0, 0, sharp, "unknown", False, "no pose detected")

    m = landmark_metrics(result.pose_landmarks[0])
    if m is None:
        return PhotoScore(0, 0, 0, 0, sharp, "unknown", False,
                          "required landmarks not visible")

    usable, reason = _verdict(m, sharp)
    return PhotoScore(
        pose_confidence=m["pose_confidence"], coverage=m["coverage"],
        upright=m["upright"], frontal=m["frontal"], sharpness=sharp,
        orientation=m["orientation"], usable=usable, reason=reason,
    )


def build_landmarker():
    """Real MediaPipe landmarker (IMAGE mode). Not used in unit tests."""
    import mediapipe as mp

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_photo_qc.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/photo_qc.py tests/test_photo_qc.py
git commit -m "feat: add photo QC scoring module with orientation classifier"
```

---

### Task 5: Cleaning stage (`scripts/clean_media.py`)

**Files:**
- Create: `scripts/clean_media.py`
- Test: `tests/test_clean_media.py`

**Interfaces:**
- Consumes: `PhotoScore`, `score_photo`, `build_landmarker` from `scripts.photo_qc` (Task 4); `extract_best_frame` from `scripts.extract_best_frame`; `IMAGE_EXTENSIONS`, `VIDEO_EXTENSIONS` from `scripts.intake_check` (Task 3).
- Produces: `field_data/cleaned/<id>/front.jpg` (+ `side.jpg` when available) and `provenance.json` per child; `field_data/reports/qc_report.csv` with columns `child_id, verdict, front_source, front_via, side_source, side_via, needs_confirmation, reason`; verdict ∈ `ok | usable_no_side | failed`. Pure helpers `@dataclass Candidate(path, role_hint, score)` and `select_best(cands: list[Candidate]) -> dict` with keys `front: Candidate|None, side: Candidate|None, auto_classified: bool`. Task 8 reads `qc_report.csv` for coverage accounting.

- [ ] **Step 1: Write the failing tests**

`tests/test_clean_media.py`:

```python
"""Tests for scripts/clean_media.py — selection logic and orchestration.

score_photo is monkeypatched everywhere; no MediaPipe model is loaded.
"""
import json
from pathlib import Path

from scripts.photo_qc import PhotoScore
from scripts.clean_media import Candidate, select_best, clean_child


def _score(usable=True, orientation="front", conf=0.9, frontal=0.9,
           reason="") -> PhotoScore:
    return PhotoScore(
        pose_confidence=conf, coverage=0.9, upright=1.0, frontal=frontal,
        sharpness=100.0, orientation=orientation, usable=usable,
        reason=reason if not usable else "",
    )


def test_select_prefers_higher_confidence_front():
    a = Candidate(Path("front_1.jpg"), "front", _score(conf=0.7))
    b = Candidate(Path("front_2.jpg"), "front", _score(conf=0.95))
    sel = select_best([a, b])
    assert sel["front"] is b
    assert sel["side"] is None
    assert sel["auto_classified"] is False


def test_filename_hint_beats_pose_classification():
    # Named side.jpg but pose says front: the filename wins.
    c = Candidate(Path("side.jpg"), "side", _score(orientation="front"))
    sel = select_best([c])
    assert sel["side"] is c and sel["front"] is None


def test_unnamed_photo_auto_classified():
    c = Candidate(Path("IMG_1.jpg"), "", _score(orientation="front"))
    sel = select_best([c])
    assert sel["front"] is c
    assert sel["auto_classified"] is True


def test_unusable_photos_never_selected():
    c = Candidate(Path("front.jpg"), "front", _score(usable=False, reason="blurry"))
    sel = select_best([c])
    assert sel["front"] is None


def test_clean_child_writes_outputs(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "001"
    raw.mkdir(parents=True)
    # 1x1 white JPEG so cv2.imread succeeds
    import cv2
    import numpy as np
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(raw / "front.jpg"), img)
    cv2.imwrite(str(raw / "side.jpg"), img)

    # clean_child scores photos in sorted order: front.jpg then side.jpg
    pending = [_score(), _score(orientation="side")]
    monkeypatch.setattr(
        "scripts.clean_media.score_photo",
        lambda image_bgr, landmarker: pending.pop(0),
    )

    cleaned = tmp_path / "cleaned"
    row = clean_child(raw, cleaned, landmarker=None, force=False)
    assert row["verdict"] == "ok"
    assert (cleaned / "001" / "front.jpg").exists()
    assert (cleaned / "001" / "side.jpg").exists()
    prov = json.loads((cleaned / "001" / "provenance.json").read_text())
    assert prov["child_id"] == "001"
    assert prov["front"]["source"] == "front.jpg"


def test_clean_child_skips_when_already_cleaned(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "001"
    raw.mkdir(parents=True)
    cleaned = tmp_path / "cleaned" / "001"
    cleaned.mkdir(parents=True)
    prior = {"child_id": "001", "verdict": "ok",
             "front": {"source": "front.jpg", "via": "filename"},
             "side": None, "needs_confirmation": False, "reason": ""}
    (cleaned / "provenance.json").write_text(json.dumps(prior))

    called = []
    monkeypatch.setattr(
        "scripts.clean_media.score_photo",
        lambda *a, **k: called.append(1),
    )
    row = clean_child(raw, tmp_path / "cleaned", landmarker=None, force=False)
    assert called == []           # nothing rescored
    assert row["verdict"] == "ok"  # verdict recovered from provenance


def test_clean_child_fails_with_reason(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "002"
    raw.mkdir(parents=True)
    import cv2
    import numpy as np
    cv2.imwrite(str(raw / "front.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "scripts.clean_media.score_photo",
        lambda *a, **k: _score(usable=False, reason="image too blurry"),
    )
    row = clean_child(raw, tmp_path / "cleaned", landmarker=None, force=False)
    assert row["verdict"] == "failed"
    assert "blurry" in row["reason"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_clean_media.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.clean_media'`.

- [ ] **Step 3: Implement `scripts/clean_media.py`**

```python
"""
Stage 2 — cleaning. Select the best front and side photo per child.

Scores every photo in field_data/raw/<id>/ (pose confidence, coverage,
upright, orientation, sharpness), copies the winners to
field_data/cleaned/<id>/front.jpg / side.jpg, falls back to the best
video frame when no photo passes, and writes a QC report.

Never modifies raw/. Idempotent: children with an existing
provenance.json are skipped unless --force.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/clean_media.py
    PYTHONPATH=. .venv/bin/python scripts/clean_media.py --force
"""
import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from scripts.photo_qc import PhotoScore, build_landmarker, score_photo  # noqa: E402
from scripts.intake_check import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS  # noqa: E402

QC_COLS = [
    "child_id", "verdict", "front_source", "front_via",
    "side_source", "side_via", "needs_confirmation", "reason",
]


@dataclass
class Candidate:
    path: Path
    role_hint: str          # 'front' | 'side' | '' (from filename prefix)
    score: PhotoScore


def _rank(c: Candidate, role: str) -> float:
    s = c.score
    base = s.pose_confidence * s.coverage * s.upright
    if role == "front":
        base *= max(s.frontal, 0.05)
    return base * (0.7 + 0.3 * min(s.sharpness / 200.0, 1.0))


def select_best(cands: list[Candidate]) -> dict:
    """Pick best front and side among usable candidates.

    Filename hint wins over pose classification; unnamed photos use the
    pose-derived orientation and are flagged auto_classified.
    """
    auto = False
    pools: dict[str, list[Candidate]] = {"front": [], "side": []}
    for c in cands:
        if not c.score.usable:
            continue
        role = c.role_hint or c.score.orientation
        if role not in pools:
            continue                      # 'unknown' orientation: ineligible
        if not c.role_hint:
            auto = True
        pools[role].append(c)

    return {
        "front": max(pools["front"], key=lambda c: _rank(c, "front"), default=None),
        "side": max(pools["side"], key=lambda c: _rank(c, "side"), default=None),
        "auto_classified": auto,
    }


def _role_hint(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("front"):
        return "front"
    if stem.startswith("side"):
        return "side"
    return ""


def _score_dict(s: PhotoScore) -> dict:
    return {
        "pose_confidence": round(s.pose_confidence, 3),
        "coverage": round(s.coverage, 3),
        "upright": round(s.upright, 3),
        "frontal": round(s.frontal, 3),
        "sharpness": round(s.sharpness, 1),
        "orientation": s.orientation,
    }


def clean_child(
    child_dir: Path, cleaned_root: Path, landmarker, force: bool,
) -> dict:
    """Clean one child folder. Returns a QC report row (never raises)."""
    cid = child_dir.name
    out_dir = cleaned_root / cid
    prov_path = out_dir / "provenance.json"

    if prov_path.exists() and not force:
        prov = json.loads(prov_path.read_text())
        return _report_row_from_provenance(prov)

    photos = sorted(
        p for p in child_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.stat().st_size > 0
    )
    videos = sorted(
        p for p in child_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        and p.stat().st_size > 0
    )

    cands: list[Candidate] = []
    fail_reasons: list[str] = []
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            fail_reasons.append(f"{p.name}: unreadable")
            continue
        try:
            s = score_photo(img, landmarker)
        except Exception as e:              # one bad photo never aborts the child
            fail_reasons.append(f"{p.name}: {e}")
            continue
        if not s.usable:
            fail_reasons.append(f"{p.name}: {s.reason}")
        cands.append(Candidate(p, _role_hint(p), s))

    sel = select_best(cands)
    front, side = sel["front"], sel["side"]
    prov: dict = {
        "child_id": cid, "front": None, "side": None,
        "needs_confirmation": sel["auto_classified"], "reason": "",
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    if front is not None:
        shutil.copy2(front.path, out_dir / "front.jpg")
        prov["front"] = {
            "source": front.path.name,
            "via": "filename" if front.role_hint else "pose_classified",
            "scores": _score_dict(front.score),
        }
    elif videos:
        # Fallback: best frame from the first video that yields one
        from scripts.extract_best_frame import extract_best_frame
        for v in videos:
            try:
                extract_best_frame(v, out_dir / "front.jpg", verbose=False)
                prov["front"] = {"source": v.name, "via": "video_fallback"}
                break
            except Exception as e:
                fail_reasons.append(f"{v.name}: {e}")

    if side is not None:
        shutil.copy2(side.path, out_dir / "side.jpg")
        prov["side"] = {
            "source": side.path.name,
            "via": "filename" if side.role_hint else "pose_classified",
            "scores": _score_dict(side.score),
        }

    if prov["front"] is None:
        prov["verdict"] = "failed"
        prov["reason"] = "; ".join(fail_reasons) or "no usable photo or video"
    elif prov["side"] is None:
        prov["verdict"] = "usable_no_side"
    else:
        prov["verdict"] = "ok"

    prov_path.write_text(json.dumps(prov, indent=2))
    return _report_row_from_provenance(prov)


def _report_row_from_provenance(prov: dict) -> dict:
    front = prov.get("front") or {}
    side = prov.get("side") or {}
    return {
        "child_id": prov["child_id"],
        "verdict": prov["verdict"],
        "front_source": front.get("source", ""),
        "front_via": front.get("via", ""),
        "side_source": side.get("source", ""),
        "side_via": side.get("via", ""),
        "needs_confirmation": prov.get("needs_confirmation", False),
        "reason": prov.get("reason", ""),
    }


def run_clean(
    raw_dir: Path, cleaned_root: Path, report_path: Path, force: bool,
) -> list[dict]:
    landmarker = build_landmarker()
    rows: list[dict] = []
    for child_dir in sorted(d for d in raw_dir.iterdir() if d.is_dir()):
        print(f"  Cleaning {child_dir.name} ...")
        rows.append(clean_child(child_dir, cleaned_root, landmarker, force))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QC_COLS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["verdict"] == "ok")
    no_side = sum(1 for r in rows if r["verdict"] == "usable_no_side")
    failed = [r for r in rows if r["verdict"] == "failed"]
    confirm = [r for r in rows if r["needs_confirmation"] in (True, "True")]
    print(f"\n{len(rows)} children: {ok} ok, {no_side} usable without side, "
          f"{len(failed)} failed")
    for r in failed:
        print(f"  RECAPTURE [{r['child_id']}] {r['reason']}")
    for r in confirm:
        print(f"  CONFIRM   [{r['child_id']}] auto-classified orientation — "
              f"check front/side are not swapped")
    print(f"QC report written to {report_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("field_data/raw"))
    parser.add_argument("--cleaned", type=Path, default=Path("field_data/cleaned"))
    parser.add_argument("--report", type=Path,
                        default=Path("field_data/reports/qc_report.csv"))
    parser.add_argument("--force", action="store_true",
                        help="Re-clean children that already have provenance.json")
    args = parser.parse_args()

    if not args.raw.is_dir():
        print(f"Raw directory not found: {args.raw}", file=sys.stderr)
        sys.exit(1)
    run_clean(args.raw, args.cleaned, args.report, args.force)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_clean_media.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/clean_media.py tests/test_clean_media.py
git commit -m "feat: add cleaning stage selecting best front/side photo per child"
```

---

### Task 6: `batch_assess.py` — measurement date, master CSV, OR-rule statuses

**Files:**
- Modify: `scripts/batch_assess.py`
- Test: `tests/test_batch_assess_helpers.py`

**Interfaces:**
- Consumes: validated ground-truth CSV columns (Task 2).
- Produces (consumed by Task 8's analyzer): new result-CSV columns `measurement_date`, `muac_cm`, `actual_muac_status`, `actual_oedema`, `actual_combined_status`, `pred_status_final`; helpers `_compute_age_months(dob: date, at: Optional[date]) -> float`, `_muac_status(muac_cm: Optional[float]) -> Optional[str]`, `_collapse_wasting(status: Optional[str]) -> Optional[str]`, `_combine_status(whz_status, muac_status, oedema_present) -> Optional[str]`. Master CSV usable with per-child layout via `--ground-truth`, keyed by folder name == `child_id`.

- [ ] **Step 1: Write the failing tests**

`tests/test_batch_assess_helpers.py`:

```python
"""Tests for the pure helpers in scripts/batch_assess.py."""
from datetime import date

from scripts.batch_assess import (
    _collapse_wasting,
    _combine_status,
    _compute_age_months,
    _muac_status,
)


def test_age_uses_measurement_date_not_today():
    dob = date(2024, 1, 1)
    at = date(2026, 1, 1)
    assert abs(_compute_age_months(dob, at) - 24.0) < 0.2


def test_age_defaults_to_today_when_no_measurement_date():
    # Backward compatible: at=None still works (uses today's date)
    assert _compute_age_months(date(2024, 1, 1), None) > 24.0


def test_muac_thresholds_are_who_fixed():
    assert _muac_status(11.4) == "SAM"
    assert _muac_status(11.5) == "MAM"
    assert _muac_status(12.4) == "MAM"
    assert _muac_status(12.5) == "Normal"
    assert _muac_status(None) is None


def test_collapse_wasting():
    assert _collapse_wasting("SAM") == "SAM"
    assert _collapse_wasting("MAM") == "MAM"
    assert _collapse_wasting("Normal") == "Normal"
    assert _collapse_wasting("Risk_Overweight") == "Normal"
    assert _collapse_wasting("Overweight") == "Normal"
    assert _collapse_wasting(None) is None


def test_combine_status_or_rule():
    # Worst of the two arms wins
    assert _combine_status("Normal", "SAM", False) == "SAM"
    assert _combine_status("SAM", "Normal", False) == "SAM"
    assert _combine_status("MAM", "Normal", False) == "MAM"
    assert _combine_status("Normal", "Normal", False) == "Normal"
    # Oedema is an independent SAM trigger, regardless of the other arms
    assert _combine_status("Normal", "Normal", True) == "SAM"
    assert _combine_status(None, None, True) == "SAM"
    # Nothing known -> None, never a fabricated 'Normal'
    assert _combine_status(None, None, False) is None
    # One arm known
    assert _combine_status(None, "MAM", False) == "MAM"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_batch_assess_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name '_collapse_wasting'` (and friends).

- [ ] **Step 3: Fix `_compute_age_months` and add the status helpers**

In `scripts/batch_assess.py`, replace (currently at lines 195–197):

```python
def _compute_age_months(dob: date) -> float:
    delta = datetime.utcnow().date() - dob
    return delta.days / 30.4375
```

with:

```python
def _compute_age_months(dob: date, at: Optional[date] = None) -> float:
    """Age in months at `at` (the measurement date). Falls back to today,
    which is only correct when assessment runs the same day as measurement."""
    ref = at or datetime.utcnow().date()
    return (ref - dob).days / 30.4375


_WASTING_SEVERITY = {"SAM": 2, "MAM": 1, "Normal": 0}


def _muac_status(muac_cm: Optional[float]) -> Optional[str]:
    """WHO fixed thresholds: <11.5 SAM, 11.5-12.5 MAM, >=12.5 Normal."""
    if muac_cm is None:
        return None
    if muac_cm < 11.5:
        return "SAM"
    if muac_cm < 12.5:
        return "MAM"
    return "Normal"


def _collapse_wasting(status: Optional[str]) -> Optional[str]:
    """Collapse the 5-class scale to the wasting axis SAM/MAM/Normal."""
    if status is None:
        return None
    return status if status in ("SAM", "MAM") else "Normal"


def _combine_status(
    whz_status: Optional[str],
    muac_status: Optional[str],
    oedema_present: bool,
) -> Optional[str]:
    """WHO OR-rule: SAM if oedema OR MUAC<11.5 OR WHZ<-3 (worst arm wins)."""
    if oedema_present:
        return "SAM"
    arms = [s for s in (_collapse_wasting(whz_status), muac_status) if s]
    if not arms:
        return None
    return max(arms, key=lambda s: _WASTING_SEVERITY[s])
```

- [ ] **Step 4: Run the helper tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_batch_assess_helpers.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Thread measurement_date + OR-rule columns through `_process_child_image`**

Still in `scripts/batch_assess.py`. In `_process_child_image`, replace:

```python
    dob_str = (gt.get("date_of_birth") or "").strip()
    try:
        dob = date.fromisoformat(dob_str)
        age_months = _compute_age_months(dob)
    except ValueError:
        dob = None
        age_months = 24.0
```

with:

```python
    dob_str = (gt.get("date_of_birth") or "").strip()
    mdate_str = (gt.get("measurement_date") or "").strip()
    try:
        meas_date = date.fromisoformat(mdate_str) if mdate_str else None
    except ValueError:
        meas_date = None
    try:
        dob = date.fromisoformat(dob_str)
        age_months = _compute_age_months(dob, meas_date)
    except ValueError:
        dob = None
        age_months = 24.0
    oedema_present = (gt.get("oedema") or "").strip().lower() == "yes"
```

Then, directly after the existing `# --- Errors ---` block (after
`weight_error = ...`), insert:

```python
    # --- WHO OR-rule statuses (gold standard vs app) ---
    actual_muac_status = _muac_status(manual_muac)
    actual_combined_status = _combine_status(
        actual_whz_status, actual_muac_status, oedema_present,
    )
    pred_status_final = _collapse_wasting(
        wasting_status_ml if wasting_status_ml else pred_whz_status
    )
```

And extend the returned `row` dict — after the line `"sex": sex,` add:

```python
        "measurement_date":     mdate_str,
```

after `"actual_whz_status":    actual_whz_status,` add:

```python
        "muac_cm":              manual_muac,
        "actual_muac_status":   actual_muac_status,
        "actual_oedema":        "yes" if oedema_present else "",
        "actual_combined_status": actual_combined_status,
```

and after `"ml_wasting_status":    wasting_status_ml,` add:

```python
        "pred_status_final":    pred_status_final,
```

- [ ] **Step 6: Add the new columns to `_write_results` fieldnames**

Replace the `fieldnames` list in `_write_results` with:

```python
    fieldnames = [
        "image_file", "child_name", "age_months", "sex", "measurement_date",
        "actual_height_cm", "actual_weight_kg",
        "actual_haz_z", "actual_whz_z", "actual_haz_status", "actual_whz_status",
        "muac_cm", "actual_muac_status", "actual_oedema",
        "actual_combined_status",
        "pred_height_cm", "pred_weight_ml_kg",
        "pred_haz_z", "pred_whz_z", "pred_haz_status", "pred_whz_status",
        "ml_wasting_status", "pred_status_final",
        "sam_probability", "mam_probability",
        "height_error_cm", "weight_error_kg",
        "pose_confidence", "estimation_method", "annotated_image",
        "feat_shoulder_width_cm", "feat_hip_width_cm", "feat_torso_length_cm",
        "feat_upper_arm_length_cm", "feat_shoulder_height_ratio",
        "feat_hip_height_ratio", "feat_body_build_score",
        "finetune_label",
        "notes", "error",
    ]
```

- [ ] **Step 7: Support the master ground-truth CSV in per-child layout**

In `run_batch`, replace:

```python
    use_per_child = _looks_like_per_child_layout(images_dir)
    if use_per_child:
        print(f"Detected per-child layout in {images_dir}")
        per_child_entries = _enumerate_per_child(images_dir)
```

with:

```python
    use_per_child = _looks_like_per_child_layout(images_dir)
    if use_per_child:
        print(f"Detected per-child layout in {images_dir}")
        master_gt: dict[str, dict] = {}
        if ground_truth_csv and ground_truth_csv.exists():
            with open(ground_truth_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cid = (row.get("child_id") or "").strip()
                    if cid:
                        master_gt[cid] = row
            print(f"Master ground truth: {len(master_gt)} row(s).")
        per_child_entries = _enumerate_per_child(images_dir)
```

and pass it through — change `results = _run_per_child(` call to:

```python
        results = _run_per_child(
            per_child_entries, meas_svc, ml_svc, nutr_svc, who_data, verbose,
            master_gt=master_gt,
        )
```

In `_run_per_child`, change the signature and the `gt` construction:

```python
def _run_per_child(
    entries,
    meas_svc, ml_svc, nutr_svc, who_data,
    verbose: bool,
    master_gt: Optional[dict] = None,
):
    """
    Process a per-child layout: each entry is (child_id, front, side, values).
    Ground truth per child comes from the master CSV row (keyed by folder
    name == child_id) merged over any per-folder values.csv; the master
    CSV wins on conflicts.
    """
    import numpy as np  # noqa: F401

    master_gt = master_gt or {}
    results = []
    for child_id, front_path, side_path, values in entries:
        merged = dict(values)
        merged.update({
            k: v for k, v in (master_gt.get(child_id) or {}).items()
            if (v or "").strip()
        })
        gt = {
            "child_name":       merged.get("child_name", child_id),
            "sex":              merged.get("sex", ""),
            "date_of_birth":    merged.get("date_of_birth", ""),
            "measurement_date": merged.get("measurement_date", ""),
            "actual_height_cm": merged.get("actual_height_cm", ""),
            "actual_weight_kg": merged.get("actual_weight_kg", ""),
            "muac_cm":          merged.get("muac_cm", ""),
            "oedema":           merged.get("oedema", ""),
            "notes":            merged.get("notes", ""),
        }
```

(The rest of `_run_per_child` is unchanged.)

- [ ] **Step 8: Run the full Python test suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: all tests PASS (existing suite + new helpers; nothing regressed).

- [ ] **Step 9: Commit**

```bash
git add scripts/batch_assess.py tests/test_batch_assess_helpers.py
git commit -m "feat: measurement-date ages, master ground-truth CSV, WHO OR-rule statuses in batch assess"
```

---

### Task 7: Study statistics module (`scripts/study_stats.py`)

**Files:**
- Create: `scripts/study_stats.py`
- Test: `tests/test_study_stats.py`

**Interfaces:**
- Produces (consumed by Task 8):
  - `bland_altman(actual: list[float], predicted: list[float]) -> dict` — keys `n, bias, loa_low, loa_high, mae` (LoA = bias ± 1.96·SD, SD with ddof=1)
  - `wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]`
  - `confusion_binary(actual: list[str], pred: list[str], positive: set[str]) -> tuple[int, int, int, int]` — (tp, fp, tn, fn)
  - `binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict` — `sensitivity, specificity, ppv, npv` each as `(value, ci_low, ci_high)`, `None` when denominator is 0
  - `weighted_kappa(actual: list[str], pred: list[str], categories: list[str]) -> float` — linear weights via sklearn

- [ ] **Step 1: Write the failing tests**

`tests/test_study_stats.py`:

```python
"""Hand-computed examples — a formula bug must not misreport the study."""
import pytest

from scripts.study_stats import (
    bland_altman,
    binary_metrics,
    confusion_binary,
    weighted_kappa,
    wilson_ci,
)


def test_bland_altman_hand_computed():
    # diffs (pred - actual) = [+1, -1]: bias 0, sd(ddof=1) = sqrt(2)
    r = bland_altman(actual=[80.0, 90.0], predicted=[81.0, 89.0])
    assert r["n"] == 2
    assert r["bias"] == pytest.approx(0.0)
    assert r["loa_high"] == pytest.approx(1.96 * 2 ** 0.5, abs=1e-6)
    assert r["loa_low"] == pytest.approx(-1.96 * 2 ** 0.5, abs=1e-6)
    assert r["mae"] == pytest.approx(1.0)


def test_wilson_ci_hand_computed():
    # k=8, n=10, z=1.96 -> (0.490, 0.943) (standard worked example)
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3)
    assert hi == pytest.approx(0.943, abs=1e-3)


def test_confusion_binary():
    actual = ["SAM", "SAM", "MAM", "Normal", "Normal"]
    pred = ["SAM", "Normal", "SAM", "Normal", "SAM"]
    tp, fp, tn, fn = confusion_binary(actual, pred, positive={"SAM"})
    assert (tp, fp, tn, fn) == (1, 2, 1, 1)


def test_binary_metrics_hand_computed():
    m = binary_metrics(tp=8, fp=2, tn=88, fn=2)
    assert m["sensitivity"][0] == pytest.approx(0.8)
    assert m["specificity"][0] == pytest.approx(88 / 90)
    assert m["ppv"][0] == pytest.approx(0.8)
    assert m["npv"][0] == pytest.approx(88 / 90)


def test_binary_metrics_zero_denominator_is_none():
    m = binary_metrics(tp=0, fp=0, tn=10, fn=0)
    assert m["sensitivity"] is None
    assert m["ppv"] is None


CATS = ["SAM", "MAM", "Normal"]


def test_weighted_kappa_perfect_agreement():
    y = ["SAM", "MAM", "Normal", "SAM"]
    assert weighted_kappa(y, y, CATS) == pytest.approx(1.0)


def test_weighted_kappa_orders_error_severity():
    # SAM misread as Normal must cost more than SAM misread as MAM
    actual = ["SAM", "MAM", "Normal", "SAM", "MAM", "Normal"]
    near = ["MAM", "MAM", "Normal", "SAM", "MAM", "Normal"]   # SAM->MAM
    far = ["Normal", "MAM", "Normal", "SAM", "MAM", "Normal"]  # SAM->Normal
    assert weighted_kappa(actual, far, CATS) < weighted_kappa(actual, near, CATS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_study_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.study_stats'`.

- [ ] **Step 3: Implement `scripts/study_stats.py`**

```python
"""
Agreement statistics for the app-vs-manual comparison study.

Bland & Altman (Lancet 1986) for continuous agreement; Wilson score CIs
for proportions; linearly weighted Cohen's kappa for the ordered
SAM > MAM > Normal scale.
"""
import math
from typing import Optional

import numpy as np
from sklearn.metrics import cohen_kappa_score


def bland_altman(actual: list[float], predicted: list[float]) -> dict:
    """Mean bias and 95% limits of agreement (bias ± 1.96·SD, ddof=1)."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("need equal, non-empty actual/predicted lists")
    diffs = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return {
        "n": len(diffs),
        "bias": bias,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "mae": float(np.mean(np.abs(diffs))),
    }


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        raise ValueError("n must be > 0")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def confusion_binary(
    actual: list[str], pred: list[str], positive: set[str],
) -> tuple[int, int, int, int]:
    """(tp, fp, tn, fn) treating membership of `positive` as the positive class."""
    tp = fp = tn = fn = 0
    for a, p in zip(actual, pred):
        a_pos, p_pos = a in positive, p in positive
        if a_pos and p_pos:
            tp += 1
        elif not a_pos and p_pos:
            fp += 1
        elif not a_pos and not p_pos:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _rate(k: int, n: int) -> Optional[tuple[float, float, float]]:
    if n == 0:
        return None
    lo, hi = wilson_ci(k, n)
    return (k / n, lo, hi)


def binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Sensitivity/specificity/PPV/NPV, each (value, ci_low, ci_high) or None."""
    return {
        "sensitivity": _rate(tp, tp + fn),
        "specificity": _rate(tn, tn + fp),
        "ppv": _rate(tp, tp + fp),
        "npv": _rate(tn, tn + fn),
    }


def weighted_kappa(
    actual: list[str], pred: list[str], categories: list[str],
) -> float:
    """Linearly weighted Cohen's kappa over an ordered category scale."""
    return float(
        cohen_kappa_score(actual, pred, labels=categories, weights="linear")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_study_stats.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/study_stats.py tests/test_study_stats.py
git commit -m "feat: add Bland-Altman, Wilson CI, and weighted-kappa study statistics"
```

---

### Task 8: Report generator (`scripts/analyze_results.py`)

**Files:**
- Create: `scripts/analyze_results.py`
- Test: `tests/test_analyze_results.py`

**Interfaces:**
- Consumes: batch results CSV columns from Task 6 (`actual_height_cm`, `pred_height_cm`, `actual_weight_kg`, `pred_weight_ml_kg`, `actual_combined_status`, `pred_status_final`, `age_months`, `sex`, `error`); `field_data/reports/qc_report.csv` (Task 5) and `field_data/reports/intake_manifest.csv` (Task 3) for coverage; all stats from `scripts.study_stats` (Task 7).
- Produces: `analyze(results: list[dict]) -> dict` (pure); `coverage(intake_rows, qc_rows, results) -> dict`; `render_report(analysis: dict, cov: dict) -> str` (markdown); CLI writing `field_data/reports/study_report.md`.

- [ ] **Step 1: Write the failing tests**

`tests/test_analyze_results.py`:

```python
"""Tests for scripts/analyze_results.py — analysis and coverage accounting."""
from scripts.analyze_results import analyze, coverage, render_report


def _row(**over) -> dict:
    row = {
        "child_name": "001", "age_months": "30.0", "sex": "M",
        "actual_height_cm": "85.0", "pred_height_cm": "86.0",
        "actual_weight_kg": "10.0", "pred_weight_ml_kg": "9.5",
        "actual_combined_status": "Normal", "pred_status_final": "Normal",
        "error": "",
    }
    row.update(over)
    return row


def test_analyze_height_and_weight_pairs():
    a = analyze([_row(), _row(child_name="002", actual_height_cm="90.0",
                            pred_height_cm="89.0")])
    assert a["height"]["n"] == 2
    assert a["weight"]["n"] == 2


def test_analyze_skips_rows_missing_values():
    a = analyze([_row(), _row(child_name="002", actual_height_cm="",
                              pred_height_cm="")])
    assert a["height"]["n"] == 1


def test_analyze_sam_confusion():
    rows = [
        _row(actual_combined_status="SAM", pred_status_final="SAM"),
        _row(child_name="002", actual_combined_status="SAM",
             pred_status_final="Normal"),
        _row(child_name="003"),
    ]
    a = analyze(rows)
    assert a["sam"]["tp"] == 1 and a["sam"]["fn"] == 1 and a["sam"]["tn"] == 1


def test_analyze_subgroups_partition_rows():
    rows = [
        _row(sex="M", age_months="12.0"),
        _row(child_name="002", sex="F", age_months="30.0"),
    ]
    a = analyze(rows)
    assert a["subgroups"]["sex=M"]["status_n"] == 1
    assert a["subgroups"]["sex=F"]["status_n"] == 1
    assert a["subgroups"]["age 6-23m"]["status_n"] == 1
    assert a["subgroups"]["age 24-59m"]["status_n"] == 1


def test_coverage_buckets_sum_to_total():
    intake = [{"child_id": c} for c in ("001", "002", "003", "004")]
    qc = [
        {"child_id": "001", "verdict": "ok"},
        {"child_id": "002", "verdict": "ok"},
        {"child_id": "003", "verdict": "failed"},
    ]
    results = [_row(child_name="001"), _row(child_name="002", error="boom")]
    cov = coverage(intake, qc, results)
    assert cov["total"] == 4
    assert cov["assessed"] == 1        # 001
    assert cov["qc_failed"] == 1       # 003
    assert cov["missing_data"] == 2    # 002 errored, 004 never cleaned
    assert cov["assessed"] + cov["qc_failed"] + cov["missing_data"] == cov["total"]
    assert cov["discrepancy"] == ""


def test_render_report_contains_headline_sections():
    rows = [_row(actual_combined_status="SAM", pred_status_final="SAM")]
    text = render_report(
        analyze(rows),
        coverage([{"child_id": "001"}],
                 [{"child_id": "001", "verdict": "ok"}], rows),
    )
    assert "## Coverage" in text
    assert "## Height agreement" in text
    assert "## Weight agreement" in text
    assert "## Status agreement" in text
    assert "SAM sensitivity" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_analyze_results.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.analyze_results'`.

- [ ] **Step 3: Implement `scripts/analyze_results.py`**

```python
"""
Stage 4b — turn batch_assess results into the comparison-study report.

Reads the batch results CSV (+ QC report and intake manifest for coverage
accounting) and writes field_data/reports/study_report.md with
Bland-Altman agreement, SAM/MAM sensitivity-specificity with Wilson CIs,
weighted kappa, subgroup breakdowns, and strict coverage buckets.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/analyze_results.py
    PYTHONPATH=. .venv/bin/python scripts/analyze_results.py \
        --results field_data/reports/batch_results.csv
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.study_stats import (  # noqa: E402
    bland_altman, binary_metrics, confusion_binary, weighted_kappa,
)

STATUS_CATS = ["SAM", "MAM", "Normal"]

# Published yardsticks (see docs/ml_pipeline_improvement_and_feedback_loop.md)
SMART_HEIGHT_TOLERANCE_CM = 1.4
WHO_TEM_HEIGHT_CM = 0.7


def _f(row: dict, key: str) -> Optional[float]:
    try:
        return float((row.get(key) or "").strip())
    except (TypeError, ValueError):
        return None


def _pairs(rows: list[dict], a_key: str, p_key: str) -> tuple[list, list]:
    actual, pred = [], []
    for r in rows:
        a, p = _f(r, a_key), _f(r, p_key)
        if a is not None and p is not None:
            actual.append(a)
            pred.append(p)
    return actual, pred


def _status_pairs(rows: list[dict]) -> tuple[list[str], list[str]]:
    actual, pred = [], []
    for r in rows:
        a = (r.get("actual_combined_status") or "").strip()
        p = (r.get("pred_status_final") or "").strip()
        if a in STATUS_CATS and p in STATUS_CATS:
            actual.append(a)
            pred.append(p)
    return actual, pred


def _analyze_block(rows: list[dict]) -> dict:
    """Metrics for one set of rows (whole study or one subgroup)."""
    out: dict = {}
    ah, ph = _pairs(rows, "actual_height_cm", "pred_height_cm")
    aw, pw = _pairs(rows, "actual_weight_kg", "pred_weight_ml_kg")
    out["height"] = bland_altman(ah, ph) if ah else {"n": 0}
    out["weight"] = bland_altman(aw, pw) if aw else {"n": 0}

    sa, sp = _status_pairs(rows)
    out["status_n"] = len(sa)
    for name, positive in (("sam", {"SAM"}), ("sam_mam", {"SAM", "MAM"})):
        tp, fp, tn, fn = confusion_binary(sa, sp, positive)
        out[name] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     **binary_metrics(tp, fp, tn, fn)}
    out["kappa"] = weighted_kappa(sa, sp, STATUS_CATS) if len(set(sa)) > 1 else None
    return out


def analyze(results: list[dict]) -> dict:
    """Pure analysis over assessed rows (rows with error are excluded)."""
    rows = [r for r in results if not (r.get("error") or "").strip()]
    out = _analyze_block(rows)
    out["subgroups"] = {}
    partitions = {
        "sex=M": [r for r in rows if (r.get("sex") or "").upper() == "M"],
        "sex=F": [r for r in rows if (r.get("sex") or "").upper() == "F"],
        "age 6-23m": [r for r in rows
                      if (_f(r, "age_months") or 0) < 24],
        "age 24-59m": [r for r in rows
                       if (_f(r, "age_months") or 0) >= 24],
    }
    for name, part in partitions.items():
        out["subgroups"][name] = _analyze_block(part)
    return out


def coverage(
    intake_rows: list[dict], qc_rows: list[dict], results: list[dict],
) -> dict:
    """Every child in exactly one bucket; buckets must sum to total."""
    total_ids = {r["child_id"] for r in intake_rows}
    qc_failed_ids = {
        r["child_id"] for r in qc_rows if r.get("verdict") == "failed"
    }
    assessed_ids = {
        (r.get("child_name") or "").strip()
        for r in results if not (r.get("error") or "").strip()
    } & total_ids
    qc_failed_ids &= total_ids - assessed_ids
    missing_ids = total_ids - assessed_ids - qc_failed_ids

    cov = {
        "total": len(total_ids),
        "assessed": len(assessed_ids),
        "qc_failed": len(qc_failed_ids),
        "missing_data": len(missing_ids),
        "missing_ids": sorted(missing_ids),
        "discrepancy": "",
    }
    if cov["assessed"] + cov["qc_failed"] + cov["missing_data"] != cov["total"]:
        cov["discrepancy"] = (
            "BUCKET SUM MISMATCH — investigate before trusting this report"
        )
    return cov


def _fmt_rate(r: Optional[tuple]) -> str:
    if r is None:
        return "n/a (zero denominator)"
    v, lo, hi = r
    return f"{v:.3f} (95% CI {lo:.3f}-{hi:.3f})"


def _fmt_ba(b: dict, unit: str) -> list[str]:
    if b.get("n", 0) == 0:
        return ["No paired values."]
    return [
        f"- n = {b['n']}",
        f"- Mean bias (pred − actual): {b['bias']:+.2f} {unit}",
        f"- 95% limits of agreement: {b['loa_low']:+.2f} to {b['loa_high']:+.2f} {unit}",
        f"- MAE: {b['mae']:.2f} {unit}",
    ]


def render_report(analysis: dict, cov: dict) -> str:
    lines: list[str] = ["# App vs Manual Measurement — Comparison Study", ""]

    lines += ["## Coverage", ""]
    lines += [
        f"- Total children (intake manifest): {cov['total']}",
        f"- Assessed: {cov['assessed']}",
        f"- QC-failed (recapture list): {cov['qc_failed']}",
        f"- Missing data (no ground truth / not cleaned / errored): "
        f"{cov['missing_data']}",
    ]
    if cov["missing_ids"]:
        lines += [f"- Missing-data IDs: {', '.join(cov['missing_ids'])}"]
    if cov["discrepancy"]:
        lines += ["", f"**{cov['discrepancy']}**"]
    lines += [""]

    lines += ["## Height agreement (Bland–Altman)", ""]
    lines += _fmt_ba(analysis["height"], "cm")
    lines += [
        "",
        f"Yardsticks: SMART tolerance {SMART_HEIGHT_TOLERANCE_CM} cm; "
        f"WHO TEM {WHO_TEM_HEIGHT_CM} cm.",
        "",
    ]

    lines += ["## Weight agreement (Bland–Altman)", ""]
    lines += _fmt_ba(analysis["weight"], "kg")
    lines += [""]

    lines += ["## Status agreement (gold standard = WHO OR-rule on manual "
              "measurements)", ""]
    lines += [f"- Paired statuses: n = {analysis['status_n']}", ""]
    sam = analysis["sam"]
    lines += [
        f"### SAM (headline — missed SAM is the fatal error direction)",
        "",
        f"- Confusion: TP {sam['tp']}, FN {sam['fn']}, FP {sam['fp']}, "
        f"TN {sam['tn']}",
        f"- SAM sensitivity: {_fmt_rate(sam['sensitivity'])}",
        f"- SAM specificity: {_fmt_rate(sam['specificity'])}",
        f"- PPV: {_fmt_rate(sam['ppv'])}   NPV: {_fmt_rate(sam['npv'])}",
        "",
    ]
    sm = analysis["sam_mam"]
    lines += [
        "### SAM+MAM combined",
        "",
        f"- Sensitivity: {_fmt_rate(sm['sensitivity'])}",
        f"- Specificity: {_fmt_rate(sm['specificity'])}",
        "",
    ]
    kappa = analysis["kappa"]
    lines += [
        f"- Weighted κ (linear, SAM>MAM>Normal): "
        + (f"{kappa:.3f}" if kappa is not None else
           "n/a (needs ≥2 distinct actual categories)"),
        "",
    ]

    lines += ["## Subgroups", ""]
    for name, block in analysis["subgroups"].items():
        h, w = block["height"], block["weight"]
        s = block["sam"]
        lines += [
            f"### {name}",
            "",
            f"- Height n={h.get('n', 0)}"
            + (f", bias {h['bias']:+.2f} cm, MAE {h['mae']:.2f} cm"
               if h.get("n") else ""),
            f"- Weight n={w.get('n', 0)}"
            + (f", bias {w['bias']:+.2f} kg, MAE {w['mae']:.2f} kg"
               if w.get("n") else ""),
            f"- SAM sensitivity: {_fmt_rate(s['sensitivity'])}",
            "",
        ]

    lines += [
        "---",
        "",
        "Interpretation notes: Bland–Altman per Lancet 1986; Wilson CIs; "
        "weighted κ because SAM→Normal errors are worse than SAM→MAM. "
        "A small SAM count makes sensitivity CIs wide — enrich with "
        "known-malnourished sites if the CI is too wide to act on.",
    ]
    return "\n".join(lines)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("field_data/reports/batch_results.csv"))
    parser.add_argument("--qc-report", type=Path,
                        default=Path("field_data/reports/qc_report.csv"))
    parser.add_argument("--intake", type=Path,
                        default=Path("field_data/reports/intake_manifest.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("field_data/reports/study_report.md"))
    args = parser.parse_args()

    results = _load(args.results)
    if not results:
        print(f"No results at {args.results} — run batch_assess first.",
              file=sys.stderr)
        sys.exit(1)

    intake_rows = _load(args.intake)
    qc_rows = _load(args.qc_report)
    if not intake_rows:
        print("WARNING: no intake manifest — coverage uses results rows only.")
        intake_rows = [
            {"child_id": (r.get("child_name") or "").strip()} for r in results
        ]

    report = render_report(analyze(results),
                           coverage(intake_rows, qc_rows, results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_analyze_results.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_results.py tests/test_analyze_results.py
git commit -m "feat: add study report generator with coverage accounting"
```

---

### Task 9: End-to-end smoke test + runbook

**Files:**
- Create: `tests/test_field_pipeline_smoke.py`
- Modify: `docs/field_data_guide.md` (append runbook)

**Interfaces:**
- Consumes: everything from Tasks 2–8.

- [ ] **Step 1: Write the smoke test**

`tests/test_field_pipeline_smoke.py`:

```python
"""End-to-end smoke: real MediaPipe on one sample child, then full chain
with a stubbed cleaner so the chain runs even where the model is absent."""
import csv
import shutil
from pathlib import Path

import pytest

from scripts.intake_check import run_intake
from scripts.validate_ground_truth import validate_rows
from scripts.analyze_results import analyze, coverage, render_report

SAMPLE_DIR = Path("sample images")
POSE_MODEL = Path("data/pose_landmarker_heavy.task")

GT_HEADER = ("child_id,sex,date_of_birth,measurement_date,"
             "actual_height_cm,actual_weight_kg,muac_cm,oedema,notes\n")


def _build_raw(tmp_path: Path) -> Path:
    """Copy the first sample child into a field_data-shaped tree."""
    sample_children = [d for d in SAMPLE_DIR.iterdir() if d.is_dir()] \
        if SAMPLE_DIR.is_dir() else []
    if not sample_children:
        pytest.skip("no sample images available")
    raw = tmp_path / "field_data" / "raw"
    dest = raw / "001"
    dest.mkdir(parents=True)
    for f in sample_children[0].iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    return raw


def test_intake_and_ground_truth_chain(tmp_path):
    raw = _build_raw(tmp_path)
    gt = tmp_path / "field_data" / "ground_truth.csv"
    gt.write_text(GT_HEADER + "001,M,2023-04-12,2026-07-15,82.5,10.4,13.2,no,\n")

    rows = list(csv.DictReader(gt.open()))
    errors, _ = validate_rows(rows)
    assert errors == []

    manifest = tmp_path / "field_data" / "reports" / "intake_manifest.csv"
    intake_rows = run_intake(raw, gt, manifest)
    assert len(intake_rows) == 1
    assert manifest.exists()


def test_clean_media_with_real_model(tmp_path):
    """Slow: loads the real MediaPipe heavy model. Skips when absent."""
    if not POSE_MODEL.exists():
        pytest.skip("pose model not downloaded")
    raw = _build_raw(tmp_path)
    from scripts.clean_media import run_clean
    report = tmp_path / "field_data" / "reports" / "qc_report.csv"
    rows = run_clean(raw, tmp_path / "field_data" / "cleaned", report, False)
    assert len(rows) == 1
    assert rows[0]["verdict"] in ("ok", "usable_no_side", "failed")
    assert report.exists()


def test_analysis_chain_from_synthetic_results():
    results = [{
        "child_name": "001", "age_months": "39.0", "sex": "M",
        "actual_height_cm": "82.5", "pred_height_cm": "84.0",
        "actual_weight_kg": "10.4", "pred_weight_ml_kg": "10.9",
        "actual_combined_status": "Normal", "pred_status_final": "Normal",
        "error": "",
    }]
    text = render_report(
        analyze(results),
        coverage([{"child_id": "001"}],
                 [{"child_id": "001", "verdict": "ok"}], results),
    )
    assert "## Status agreement" in text
```

- [ ] **Step 2: Run the smoke tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_field_pipeline_smoke.py -v`
Expected: PASS (the slow test may SKIP where the model file is absent; that is fine).

- [ ] **Step 3: Append the runbook to `docs/field_data_guide.md`**

Append:

```markdown

## Runbook — commands in order

All commands from the project root.

    # 0. One-time: create the ground-truth template
    PYTHONPATH=. .venv/bin/python scripts/validate_ground_truth.py --template

    # 1. While gathering: what's still missing?
    PYTHONPATH=. .venv/bin/python scripts/intake_check.py

    # 2. Validate the typed-in measurements (must pass before assessing)
    PYTHONPATH=. .venv/bin/python scripts/validate_ground_truth.py

    # 3. Clean: pick best front/side per child, get the recapture list
    PYTHONPATH=. .venv/bin/python scripts/clean_media.py

    # 4. Assess every cleaned child against ground truth
    PYTHONPATH=. .venv/bin/python scripts/batch_assess.py \
        --images field_data/cleaned \
        --ground-truth field_data/ground_truth.csv \
        --output field_data/reports/batch_results.csv

    # 5. Generate the study report
    PYTHONPATH=. .venv/bin/python scripts/analyze_results.py

    # Read: field_data/reports/study_report.md

Re-run any stage at any time; stages never modify `raw/` and cleaning
skips already-cleaned children (add `--force` to redo them).
```

- [ ] **Step 4: Run the full test suite one last time**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: all tests PASS (slow test may SKIP).

- [ ] **Step 5: Commit**

```bash
git add tests/test_field_pipeline_smoke.py docs/field_data_guide.md
git commit -m "test: add field-pipeline smoke tests; document runbook"
```

---

## Plan Self-Review (completed)

- **Spec coverage:** layout/IDs/privacy → Task 1; intake check → Task 3; cleaning + video fallback + QC report + idempotency → Tasks 4–5; ground-truth CSV, validator, measurement-date bug, master-CSV lookup → Tasks 2 and 6; OR-rule gold standard → Task 6; Bland–Altman/sensitivity/κ/subgroups/coverage → Tasks 7–8; error handling (per-child try/except, reasons in reports) → Tasks 5, 6, 8; testing incl. MediaPipe-free units + real-model smoke → every task + Task 9. Model improvement intentionally absent (out of scope per spec).
- **Placeholder scan:** no TBDs; all steps carry code or exact commands.
- **Type consistency:** `PhotoScore`/`Candidate`/`select_best`/`clean_child` names match between Tasks 4, 5, 9; `_muac_status`/`_combine_status`/`_collapse_wasting` match between Tasks 6 and 8 consumers; stats signatures match between Tasks 7 and 8.
