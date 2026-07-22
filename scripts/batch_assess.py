"""
Batch assessment: run all images through the growth monitor and compare with ground truth.

Two layouts are supported:

A. **Flat layout** — all photos in one directory; ground truth in a single CSV
   keyed by filename:

       /path/to/photos/
         child1.jpg
         child2.jpg
         …
       data/ground_truth.csv

B. **Per-child layout** — one subfolder per child, each containing
   front.jpeg, side.{jpeg,png}, back.jpeg (optional), and values.csv:

       /path/to/photos/
         child001/
           front.jpeg
           side.png
           back.jpeg          (ignored)
           values.csv         (ground truth for this child)
         child002/
           …

   The values.csv inside each folder must have a single non-header row
   (or column-form key,value pairs) with at least:

       sex            : M or F
       date_of_birth  : YYYY-MM-DD
       actual_height_cm  (optional, in cm)
       actual_weight_kg  (optional, in kg)
       muac_cm           (optional, tape-measured MUAC in cm)
       notes             (optional)

   Empty values are tolerated. Layout B uses the front photo for pose and
   the side photo for AP-depth features (when available).

Usage
-----
# Generate a blank template to fill in:
    python scripts/batch_assess.py --template

# Run batch assessment (flat layout):
    python scripts/batch_assess.py \\
        --images  /path/to/photos/ \\
        --ground-truth  data/ground_truth.csv \\
        --output  data/batch_results.csv

# Run batch assessment (per-child layout, auto-detected):
    python scripts/batch_assess.py --images "/path/to/sample images/"

# Images without ground truth (just run the system):
    python scripts/batch_assess.py --images /path/to/photos/

Ground-truth CSV columns (flat layout)
--------------------------------------
image_file         : filename (just the name) — must match a file in --images.
                      This is the ONLY identifier this layout uses. Do not
                      add a child_name/ID column — no child names belong in
                      folder names, filenames, or CSVs anywhere in this
                      pipeline (see docs/field_data_guide.md). The results
                      CSV's own "child_name" column is always populated from
                      image_file (or the per-child folder id), never from
                      an operator-supplied value.
date_of_birth      : YYYY-MM-DD
measurement_date   : YYYY-MM-DD, the date the photo/measurements were taken
                      (leave blank to fall back to today's date — but every
                      WHO z-score depends on age, so fill this in whenever
                      the assessment doesn't run the same day as measurement)
sex                : M or F
actual_height_cm   : measured height in cm  (leave blank if unknown)
actual_weight_kg   : measured weight in kg  (leave blank if unknown)
muac_cm            : measured MUAC in cm    (leave blank if unknown)
oedema             : "yes" if bilateral pitting oedema is present, else
                      blank or "no" — an independent WHO SAM trigger
notes              : free-text (optional)

All other columns are ignored and preserved in the output.
"""

import argparse
import csv
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_ground_truth import AGE_RANGE_MONTHS, validate_rows  # noqa: E402

TEMPLATE_CSV = """\
image_file,date_of_birth,measurement_date,sex,actual_height_cm,actual_weight_kg,muac_cm,oedema,notes
child1.jpg,2022-03-15,2024-03-20,M,85.0,11.2,14.5,no,
child2.jpg,2023-07-01,2024-03-20,F,78.5,,13.2,no,weight not available
child3.jpg,2021-11-20,2024-03-20,M,,,,,height and weight unknown
"""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Per-child layout filenames (case-insensitive prefix match)
PER_CHILD_FRONT_NAMES = ("front",)
PER_CHILD_SIDE_NAMES  = ("side",)
PER_CHILD_VALUES_NAME = "values.csv"


def _read_per_child_values(path: Path) -> dict:
    """
    Read a per-child values.csv and return a normalised dict.

    Accepts two formats:

      Row format (header + one data row):
        sex,date_of_birth,actual_height_cm,actual_weight_kg,muac_cm,notes
        M,2023-04-12,82.5,11.4,13.2,

      Key,value format (one row per attribute, no header required):
        sex,M
        date_of_birth,2023-04-12
        actual_height_cm,82.5
    """
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return {}
    if not text:
        return {}

    lines = [l for l in text.splitlines() if l.strip()]

    # Try header-row format first
    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames or []
    data_rows = list(reader)
    if data_rows and any(k.lower() in {
        "sex", "date_of_birth", "actual_height_cm", "actual_weight_kg",
        "muac_cm", "notes",
    } for k in fieldnames):
        first = data_rows[0]
        return {
            (k or "").strip().lower(): (v or "").strip()
            for k, v in first.items() if k
        }

    # Fall back to key,value rows
    out: dict[str, str] = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            out[parts[0].lower()] = parts[1]
    return out


def _looks_like_per_child_layout(images_dir: Path) -> bool:
    """A directory uses the per-child layout if at least one immediate
    subdirectory contains a front-* image and (optionally) a values.csv."""
    if not images_dir.is_dir():
        return False
    for child in images_dir.iterdir():
        if not child.is_dir():
            continue
        for f in child.iterdir():
            if (f.suffix.lower() in IMAGE_EXTENSIONS
                and f.stem.lower().startswith(PER_CHILD_FRONT_NAMES)):
                return True
    return False


def _enumerate_per_child(images_dir: Path) -> list[tuple[str, Path, Optional[Path], dict]]:
    """
    For per-child layout, yield (child_id, front_path, side_path, values_dict).
    Skips folders without a front image.
    """
    out = []
    for child in sorted(images_dir.iterdir()):
        if not child.is_dir():
            continue
        front = None
        side  = None
        for f in sorted(child.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = f.stem.lower()
            if stem.startswith(PER_CHILD_FRONT_NAMES) and front is None:
                front = f
            elif stem.startswith(PER_CHILD_SIDE_NAMES) and side is None:
                side = f
        if front is None:
            print(f"  Skipping {child.name}: no front image found")
            continue
        values = _read_per_child_values(child / PER_CHILD_VALUES_NAME)
        out.append((child.name, front, side, values))
    return out


def generate_template(out_path: Path = Path("data/ground_truth_template.csv")):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(TEMPLATE_CSV)
    print(f"Template written to {out_path}")
    print("Fill in the measurements and save as data/ground_truth.csv")


def _compute_age_months(dob: date, at: Optional[date] = None) -> float:
    """Age in months at `at` (the measurement date). Falls back to today,
    which is only correct when assessment runs the same day as measurement."""
    ref = at or datetime.now(timezone.utc).date()
    return (ref - dob).days / 30.4375


def _parse_dob(dob_str: str) -> tuple[Optional[date], Optional[str]]:
    """Parse a date_of_birth string.

    Returns (dob, error_message). On success `error_message` is None. On
    failure `dob` is None and `error_message` names the problem so the
    caller can surface it in the result row instead of fabricating an age
    and emitting a verdict derived from it.

    Empty/blank (nothing supplied, e.g. the --images-only mode with no
    ground truth) is distinguished from a garbled value (something was
    supplied but doesn't parse) so a working "just run the system" mode
    isn't mislabelled as broken.
    """
    if not dob_str:
        return (
            None,
            "no date_of_birth supplied - age and all z-scores unavailable",
        )
    try:
        return date.fromisoformat(dob_str), None
    except ValueError:
        return (
            None,
            f"unparseable date_of_birth: '{dob_str}' - age and all z-scores unavailable",
        )


_WASTING_SEVERITY = {"SAM": 2, "MAM": 1, "Normal": 0}

# WHO MUAC cutoffs are defined for 6-59 months only. Mirrors
# app/services/muac_service.py's `age_in_range` bounds exactly.
MUAC_AGE_RANGE_MONTHS = (6.0, 59.9)


def _muac_status(
    muac_cm: Optional[float], age_months: Optional[float]
) -> Optional[str]:
    """WHO fixed thresholds: <11.5 SAM, 11.5-12.5 MAM, >=12.5 Normal.

    WHO defines these absolute cutoffs for 6-59 months ONLY; below 6 months
    an 11.4 cm arm is normal infant anatomy, not wasting. Applying them
    anyway would mint gold-standard SAM labels the app is then scored
    against, inflating the apparent false-negative count with no visible
    signal. Returns None outside the window (unknown, not "Normal") so
    `_combine_status` drops the arm rather than treating it as a negative.

    The window matches app/services/muac_service.py's `age_in_range`
    (6.0-59.9) so the study's gold standard and the app's own classifier
    cannot drift apart. `age_months` is None when age is unknown, which is
    likewise not a licence to classify.
    """
    if muac_cm is None or age_months is None:
        return None
    lo, hi = MUAC_AGE_RANGE_MONTHS
    if not (lo <= age_months <= hi):
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


def _whz_status_from_z(z: Optional[float]) -> Optional[str]:
    if z is None:
        return None
    if z < -3:
        return "SAM"
    if z < -2:
        return "MAM"
    if z <= 1:
        return "Normal"
    if z <= 2:
        return "Risk_Overweight"
    return "Overweight"


def _haz_status_from_z(z: Optional[float]) -> Optional[str]:
    if z is None:
        return None
    if z < -3:
        return "Severely Stunted"
    if z < -2:
        return "Stunted"
    if z <= 2:
        return "Normal"
    return "Tall"


def run_batch(
    images_dir: Path,
    ground_truth_csv: Optional[Path],
    output_csv: Path,
    verbose: bool = True,
):
    # Late imports so template generation works without TF loaded
    import numpy as np
    from app.services.measurement_service import MeasurementService
    from app.services.ml_service import MLService
    from app.services.who_data_service import WHODataService
    from app.services.nutrition_service import NutritionService

    who_data = WHODataService()
    meas_svc = MeasurementService()
    nutr_svc = NutritionService(who_data)
    ml_svc   = MLService()

    use_per_child = _looks_like_per_child_layout(images_dir)
    if use_per_child:
        print(f"Detected per-child layout in {images_dir}")
        master_gt: dict[str, dict] = {}
        if ground_truth_csv and ground_truth_csv.exists():
            # Hard gate: a master ground-truth CSV with impossible values
            # must never silently produce a study — abort before any
            # child is assessed. See _load_master_ground_truth.
            master_gt = _load_master_ground_truth(ground_truth_csv)
            print(f"Master ground truth: {len(master_gt)} row(s).")
        per_child_entries = _enumerate_per_child(images_dir)
        if not per_child_entries:
            print(f"No usable child folders in {images_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(per_child_entries)} child folder(s).\n")
        results = _run_per_child(
            per_child_entries, meas_svc, ml_svc, nutr_svc, who_data, verbose,
            master_gt=master_gt,
        )
    else:
        # Build ground-truth lookup keyed by image filename
        gt_lookup: dict[str, dict] = {}
        if ground_truth_csv and ground_truth_csv.exists():
            with open(ground_truth_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("image_file", "").strip()
                    if fname:
                        gt_lookup[fname] = row

        image_files = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_files:
            print(f"No images found in {images_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(image_files)} image(s). Ground truth: {len(gt_lookup)} row(s).\n")
        results = _run_flat(
            image_files, gt_lookup, meas_svc, ml_svc, nutr_svc, who_data, verbose,
        )

    _write_results(output_csv, results)


def _run_flat(
    image_files: list,
    gt_lookup: dict,
    meas_svc, ml_svc, nutr_svc, who_data,
    verbose: bool,
):
    results = []
    for img_path in image_files:
        fname = img_path.name
        gt = gt_lookup.get(fname, {})
        try:
            row = _process_child_image(
                fname=fname,
                img_path=img_path,
                gt=gt,
                meas_svc=meas_svc,
                ml_svc=ml_svc,
                nutr_svc=nutr_svc,
                who_data=who_data,
                side_image_bytes=None,
                child_id=None,
                verbose=verbose,
            )
        except Exception as e:
            print(f"    [ERROR] {fname}: {e}")
            row = _error_row(
                fname=fname,
                child_name=fname,  # I-3: identity, never an operator-supplied name
                age_months=24.0,
                sex=(gt.get("sex") or "M").strip().upper(),
                actual_height=_parse_float(gt.get("actual_height_cm")),
                actual_weight=_parse_float(gt.get("actual_weight_kg")),
                error_msg=str(e),
            )
        results.append(row)
    return results


def _process_child_image(
    fname: str,
    img_path: Path,
    gt: dict,
    meas_svc, ml_svc, nutr_svc, who_data,
    side_image_bytes: Optional[bytes],
    child_id: Optional[str],
    verbose: bool,
) -> dict:
    """Run the full assessment pipeline for one image. Returns a result row."""
    # I-3: the child is keyed strictly on the folder id (per-child layout)
    # or the image filename (flat layout) — never on an operator-supplied
    # name. This is what ends up in the results CSV's "child_name" column,
    # which analyze_results.coverage() uses as its join key against the
    # intake manifest's child_id; a personal name there both violates the
    # no-PII-in-CSVs rule and silently breaks that join (see final-review
    # I-3). `_display_label` is a display-only convenience for the console
    # warning below - it is never persisted to any file.
    identity = child_id or fname
    _display_label = gt.get("child_name") or identity
    sex = (gt.get("sex") or "M").strip().upper() or "M"

    dob_str = (gt.get("date_of_birth") or "").strip()
    mdate_str = (gt.get("measurement_date") or "").strip()
    meas_date = None
    mdate_error = None
    # Provenance of the date every z-score is ultimately computed against,
    # persisted to the results CSV so a reader can tell a real measurement
    # date from a fallback instead of having to trust that one was supplied.
    measurement_date_source = "supplied"
    if mdate_str:
        try:
            meas_date = date.fromisoformat(mdate_str)
        except ValueError:
            # Same treatment as an unparseable date_of_birth: age drives
            # every WHO z-score AND selects the WFL (<24m) vs WFH (>=24m)
            # LMS table, so silently substituting today's date yields a
            # clean-looking verdict computed from an age that is wrong by
            # however long elapsed between the field visit and this run.
            # A console WARN is not a signal on a 250-child batch.
            mdate_error = (
                f"unparseable measurement_date: '{mdate_str}' - age and all "
                "z-scores unavailable"
            )
            measurement_date_source = "unparseable"
    else:
        # Blank remains permitted (the flat layout and --images-only mode
        # predate this column), but it is recorded and counted rather than
        # passing as if a date had been supplied.
        measurement_date_source = "today_fallback"

    dob, dob_error = _parse_dob(dob_str)
    age_range_error = None
    if dob is not None and mdate_error is None:
        age_months = _compute_age_months(dob, meas_date)
        lo, hi = AGE_RANGE_MONTHS
        if not (lo <= age_months <= hi):
            # WHO growth standards are only defined for 0-60 months. A
            # parseable-but-impossible age (swapped dob/measurement_date
            # columns, a future dob, a one-digit year typo, ...) must be
            # rejected the same way an unparseable dob is: never let a
            # fabricated age drive pose/ML/WHO lookups into a clean-looking
            # verdict. Matches scripts/validate_ground_truth.py's range.
            # Display precision: 1 decimal is plenty for a value that's
            # unambiguously out of range (e.g. -12.0, 600.0), but a value
            # that rounds to exactly the boundary itself at 1 decimal
            # (e.g. 60.0246 -> "60.0") would read as self-contradictory
            # next to "out of range 0-60" - show extra precision only
            # then, so the rejection is self-explanatory.
            rounded_1dp = round(age_months, 1)
            age_disp = (
                f"{age_months:.2f}" if rounded_1dp in (lo, hi) else f"{rounded_1dp:.1f}"
            )
            age_range_error = (
                f"age_months {age_disp} out of range {lo:.0f}-{hi:.0f} "
                f"(date_of_birth='{dob_str}', "
                f"measurement_date='{mdate_str or 'unset - fell back to today'}')"
            )
    else:
        # Placeholder only, to keep the pose/ML pipeline running below;
        # dob_error/mdate_error forces error + pred_status_final=None on the
        # row so nothing derived from this fabricated age is presented as
        # usable.
        age_months = 24.0
    # True age, or None when it could not be established. Distinct from the
    # 24.0 placeholder above, which exists only to keep pose/ML running —
    # anything that classifies against age must use this, never the
    # placeholder.
    known_age_months = (
        age_months if (dob is not None and mdate_error is None and not age_range_error)
        else None
    )
    oedema_present = (gt.get("oedema") or "").strip().lower() == "yes"

    actual_height = _parse_float(gt.get("actual_height_cm"))
    actual_weight = _parse_float(gt.get("actual_weight_kg"))
    manual_muac   = _parse_float(gt.get("muac_cm"))

    if verbose:
        tag = f"{child_id}/" if child_id else ""
        print(f"  Processing {tag}{fname} ...")

    # --- Pose estimation ---
    meas = meas_svc.process_image_with_estimation(
        image_path=str(img_path),
        age_months=age_months,
        sex=sex,
        who_data=who_data,
    )
    pred_height = meas.predicted_height_cm
    effective_height = pred_height if pred_height else actual_height

    # --- Side-view AP depth (per-child layout only) ---
    side_segments = None
    if side_image_bytes is not None and effective_height is not None:
        side_segments = meas_svc.process_side_image(side_image_bytes, effective_height)

    # --- ML prediction ---
    ml_pred = None
    if effective_height and meas.body_segments:
        try:
            ml_pred = ml_svc.predict(
                meas.body_segments, age_months, sex, effective_height, side_segments,
            )
        except Exception as e:
            print(f"    [WARN] ML prediction failed: {e}")

    pred_weight_ml    = ml_pred.estimated_weight_kg if ml_pred else None
    wasting_status_ml = ml_pred.wasting_status if ml_pred else None
    sam_prob = round(ml_pred.sam_probability, 4) if ml_pred else None
    mam_prob = round(ml_pred.mam_probability, 4) if ml_pred else None

    # --- Z-scores from ACTUAL measurements ---
    actual_haz_z      = None
    actual_whz_z      = None
    actual_haz_status = None
    actual_whz_status = None
    # Guarded on `known_age_months`, NOT on `dob`. A parseable dob combined
    # with an unparseable measurement_date leaves `dob` truthy while
    # `age_months` holds the 24.0 placeholder — which is both fabricated and
    # sitting exactly on the WFL/WFH table boundary. Keying on `dob` here
    # would compute the gold standard at that fabricated age and write it to
    # the CSV as if it were real. WHO tables are indexed by COMPLETED months,
    # so truncate rather than round: 23.7 months is row 23.
    if actual_height and known_age_months is not None:
        actual_haz_z = nutr_svc.compute_haz(sex, int(known_age_months), actual_height)
        if actual_haz_z is not None:
            actual_haz_status = _haz_status_from_z(actual_haz_z)
            actual_haz_z = round(actual_haz_z, 3)
    if actual_height and actual_weight and known_age_months is not None:
        actual_whz_z = nutr_svc.compute_whz(
            sex, known_age_months, actual_height, actual_weight,
        )
        if actual_whz_z is not None:
            actual_whz_status = _whz_status_from_z(actual_whz_z)
            actual_whz_z = round(actual_whz_z, 3)

    # --- Z-scores from PREDICTED measurements ---
    pred_haz_z      = None
    pred_whz_z      = None
    pred_haz_status = None
    pred_whz_status = None
    if pred_height and known_age_months is not None:
        pred_haz_z = nutr_svc.compute_haz(sex, int(known_age_months), pred_height)
        if pred_haz_z is not None:
            pred_haz_status = _haz_status_from_z(pred_haz_z)
            pred_haz_z = round(pred_haz_z, 3)
    if pred_height and pred_weight_ml and known_age_months is not None:
        pred_whz_z = nutr_svc.compute_whz(
            sex, known_age_months, pred_height, pred_weight_ml,
        )
        if pred_whz_z is not None:
            pred_whz_status = _whz_status_from_z(pred_whz_z)
            pred_whz_z = round(pred_whz_z, 3)

    # --- Errors ---
    height_error = round(pred_height - actual_height, 2) if pred_height and actual_height else None
    weight_error = round(pred_weight_ml - actual_weight, 2) if pred_weight_ml and actual_weight else None

    # --- WHO OR-rule statuses (gold standard vs app) ---
    actual_muac_status = _muac_status(manual_muac, known_age_months)
    actual_combined_status = _combine_status(
        actual_whz_status, actual_muac_status, oedema_present,
    )
    # The app's own verdict combines its two available arms by the same
    # worst-arm-wins rule the gold standard uses above. The previous
    # `ml if ml else whz` let a "Normal" ML verdict silently discard a
    # predicted WHZ < -3 — a manufactured false negative in a system whose
    # stated hard rule is that false negatives endanger lives. The app has
    # no oedema input, so that arm is absent (not negative) on this side.
    _pred_arms = [
        s for s in (
            _collapse_wasting(wasting_status_ml),
            _collapse_wasting(pred_whz_status),
        ) if s
    ]
    pred_status_final = (
        max(_pred_arms, key=lambda s: _WASTING_SEVERITY[s]) if _pred_arms else None
    )
    if dob_error or age_range_error or mdate_error:
        # dob could not be parsed, or the resulting age is outside the
        # 0-60 month range WHO growth standards are defined for: never
        # present a verdict derived from a fabricated or impossible age.
        pred_status_final = None

    row = {
        "image_file":           fname,
        "child_name":           identity,
        "age_months":           round(age_months, 1),
        "sex":                  sex,
        "measurement_date":     mdate_str,
        "measurement_date_source": measurement_date_source,
        # Ground truth
        "actual_height_cm":     actual_height,
        "actual_weight_kg":     actual_weight,
        "actual_haz_z":         actual_haz_z,
        "actual_whz_z":         actual_whz_z,
        "actual_haz_status":    actual_haz_status,
        "actual_whz_status":    actual_whz_status,
        "muac_cm":              manual_muac,
        "actual_muac_status":   actual_muac_status,
        "actual_oedema":        "yes" if oedema_present else "",
        "actual_combined_status": actual_combined_status,
        # Predictions
        "pred_height_cm":       round(pred_height, 1) if pred_height else None,
        "pred_weight_ml_kg":    round(pred_weight_ml, 2) if pred_weight_ml else None,
        "pred_haz_z":           pred_haz_z,
        "pred_whz_z":           pred_whz_z,
        "pred_haz_status":      pred_haz_status,
        "pred_whz_status":      pred_whz_status,
        "ml_wasting_status":    wasting_status_ml,
        "pred_status_final":    pred_status_final,
        "sam_probability":      sam_prob,
        "mam_probability":      mam_prob,
        # Errors
        "height_error_cm":      height_error,
        "weight_error_kg":      weight_error,
        # Metadata
        "pose_confidence":      round(meas.confidence_score, 3) if meas.confidence_score else None,
        "estimation_method":    meas.estimation_method,
        "annotated_image":      meas.annotated_image_filename,
        "notes":                gt.get("notes", ""),
        "error":                dob_error or age_range_error or mdate_error or "",
    }

    if ml_svc.is_available and meas.body_segments and effective_height:
        features = ml_svc.extract_features(
            meas.body_segments, age_months, sex, effective_height, side_segments,
        )
        if features:
            row.update({
                "feat_shoulder_width_cm":     round(features.shoulder_width_cm, 2),
                "feat_hip_width_cm":          round(features.hip_width_cm, 2),
                "feat_torso_length_cm":       round(features.torso_length_cm, 2),
                "feat_upper_arm_length_cm":   round(features.upper_arm_length_cm, 2),
                "feat_shoulder_height_ratio": round(features.shoulder_height_ratio, 4),
                "feat_hip_height_ratio":      round(features.hip_height_ratio, 4),
                "feat_body_build_score":      features.body_build_score,
                "finetune_label":             actual_whz_status if actual_whz_z is not None else "",
            })

    if verbose:
        ht = f"{pred_height:.1f}cm" if pred_height else "N/A"
        wt = f"{pred_weight_ml:.1f}kg" if pred_weight_ml else "N/A"
        err_h = f" (err={height_error:+.1f})" if height_error is not None else ""
        err_w = f" (err={weight_error:+.1f})" if weight_error is not None else ""
        wst = wasting_status_ml or "N/A"
        muac_tag = f"  MUAC(manual): {manual_muac}" if manual_muac else ""
        side_tag = "  +side" if side_image_bytes else ""
        print(f"    Height: {ht}{err_h}  Weight(ML): {wt}{err_w}  "
              f"Wasting: {wst}{muac_tag}{side_tag}")

    return row


def _clean_master_row(row: Optional[dict]) -> tuple[dict[str, str], Optional[str]]:
    """Sanitize a master ground-truth CSV row before merging it into a
    per-child dict.

    A row with more fields than its header makes csv.DictReader stash the
    overflow under a `None` key whose value is a `list` (a ragged row) —
    every column after the missing/extra one is shifted, so its values are
    not trustworthy. Guard against any non-string key/value here so that
    one malformed master row can never raise and take down the whole
    batch.

    Returns `(cleaned, ragged_error)`. `ragged_error` is `None` for a
    well-formed row; otherwise it names the problem so the caller can
    surface it in that child's `error` field instead of silently
    assessing shifted data — belt and braces alongside the
    `validate_ground_truth` gate in `_load_master_ground_truth`, in case a
    caller bypasses that gate.
    """
    if not row:
        return {}, None
    out: dict[str, str] = {}
    ragged = False
    for k, v in row.items():
        if not isinstance(k, str):
            ragged = True
            continue  # e.g. the None key csv.DictReader uses for overflow
        if isinstance(v, list):
            v = ",".join(str(x) for x in v if x is not None)
        elif v is None:
            v = ""
        elif not isinstance(v, str):
            v = str(v)
        if v.strip():
            out[k] = v
    ragged_error = (
        "master ground-truth CSV row is ragged (more fields than the "
        "header) - columns after the mismatch are shifted, values not "
        "trusted"
    ) if ragged else None
    return out, ragged_error


def _load_master_ground_truth(ground_truth_csv: Path) -> dict[str, dict]:
    """Load the master ground-truth CSV for the per-child layout and gate
    on `validate_ground_truth.validate_rows` before any child is assessed.

    A ground-truth file with impossible values (e.g. a ragged row whose
    shift produces an out-of-range height/weight/MUAC, a bad date, an age
    outside 0-60 months, ...) must never silently produce a study — refuse
    to run at all rather than assess every child against untrustworthy
    data. Warnings (missing optional measurements) are printed but do not
    block.
    """
    with open(ground_truth_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    errors, warnings = validate_rows(rows, fieldnames=fieldnames)
    for w in warnings:
        print(f"  WARNING  {w}")
    if errors:
        for e in errors:
            print(f"  ERROR    {e}", file=sys.stderr)
        print(
            f"\n{ground_truth_csv}: {len(errors)} error(s) found in the "
            "master ground-truth CSV - refusing to assess any child.\n"
            "Fix the file and re-check it with:\n"
            f"  PYTHONPATH=. .venv/bin/python scripts/validate_ground_truth.py {ground_truth_csv}",
            file=sys.stderr,
        )
        sys.exit(1)

    master_gt: dict[str, dict] = {}
    for row in rows:
        cid = (row.get("child_id") or "").strip()
        if cid:
            master_gt[cid] = row
    return master_gt


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
        # The merge (and everything derived from it) lives inside the
        # per-child try: a malformed master CSV row must degrade only this
        # child's row, never abort the batch for every other child.
        try:
            merged = dict(values)
            cleaned_master, ragged_error = _clean_master_row(master_gt.get(child_id))
            merged.update(cleaned_master)
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
            side_bytes = side_path.read_bytes() if side_path else None
            row = _process_child_image(
                fname=front_path.name,
                img_path=front_path,
                gt=gt,
                meas_svc=meas_svc,
                ml_svc=ml_svc,
                nutr_svc=nutr_svc,
                who_data=who_data,
                side_image_bytes=side_bytes,
                child_id=child_id,
                verbose=verbose,
            )
            if ragged_error:
                # Belt and braces: never let a ragged master row present a
                # verdict derived from shifted columns as clean, even
                # though _load_master_ground_truth should already have
                # refused to run before reaching this point.
                row["error"] = (
                    f"{row['error']}; {ragged_error}" if row.get("error") else ragged_error
                )
                row["pred_status_final"] = None
        except Exception as e:
            print(f"    [ERROR] {child_id}: {e}")
            row = _error_row(
                fname=front_path.name,
                child_name=child_id,  # I-3: identity, never an operator-supplied name
                age_months=24.0,
                sex=(values.get("sex") or "M"),
                actual_height=_parse_float(values.get("actual_height_cm")),
                actual_weight=_parse_float(values.get("actual_weight_kg")),
                error_msg=str(e),
            )
        results.append(row)

    return results


def _write_results(output_csv: Path, results: list[dict]):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_file", "child_name", "age_months", "sex", "measurement_date",
        "measurement_date_source",
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

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_csv}")
    _print_summary(results)


def _print_summary(results: list[dict]):
    total = len(results)
    errors = sum(1 for r in results if r.get("error"))
    with_height = [r for r in results if r.get("actual_height_cm") and r.get("pred_height_cm")]
    with_weight = [r for r in results if r.get("actual_weight_kg") and r.get("pred_weight_ml_kg")]
    with_labels = [r for r in results if r.get("actual_whz_status") and r.get("ml_wasting_status")]

    print(f"\n{'='*60}")
    print(f"SUMMARY  |  {total} images  |  {errors} errors")
    print(f"{'='*60}")

    # Surface the today-fallback count once per batch: individually these are
    # silent, but in aggregate they say "N children's ages — and therefore
    # every z-score derived from them — were computed against the run date
    # rather than the field-visit date."
    fallback = sum(
        1 for r in results if r.get("measurement_date_source") == "today_fallback"
    )
    if fallback:
        print(
            f"  [WARN] {fallback} child(ren) had no measurement_date; age "
            f"computed from today's date. Every z-score for those children "
            f"is wrong by the time elapsed since their field visit."
        )

    if with_height:
        errs = [abs(float(r["height_error_cm"])) for r in with_height]
        import numpy as np
        print(f"\nHeight estimation ({len(with_height)} with ground truth):")
        print(f"  MAE  = {np.mean(errs):.1f} cm")
        print(f"  Max error = {np.max(errs):.1f} cm")

    if with_weight:
        errs = [abs(float(r["weight_error_kg"])) for r in with_weight]
        import numpy as np
        print(f"\nML weight estimation ({len(with_weight)} with ground truth):")
        print(f"  MAE  = {np.mean(errs):.2f} kg")
        print(f"  Max error = {np.max(errs):.2f} kg")

    if with_labels:
        correct = sum(
            1 for r in with_labels
            if r["actual_whz_status"] == r["ml_wasting_status"]
        )
        print(f"\nWasting classification ({len(with_labels)} with ground truth):")
        print(f"  Accuracy = {correct}/{len(with_labels)} ({correct/len(with_labels)*100:.0f}%)")

        # SAM/MAM recall
        sam_actual = [r for r in with_labels if r["actual_whz_status"] == "SAM"]
        if sam_actual:
            sam_caught = sum(1 for r in sam_actual if r["ml_wasting_status"] == "SAM")
            print(f"  SAM recall = {sam_caught}/{len(sam_actual)}")
        mam_actual = [r for r in with_labels if r["actual_whz_status"] == "MAM"]
        if mam_actual:
            mam_caught = sum(1 for r in mam_actual if r["ml_wasting_status"] in ("SAM", "MAM"))
            print(f"  MAM recall (SAM+MAM predicted) = {mam_caught}/{len(mam_actual)}")

    # Fine-tuning dataset stats
    finetune_rows = [r for r in results if r.get("finetune_label")]
    if finetune_rows:
        from collections import Counter
        label_counts = Counter(r["finetune_label"] for r in finetune_rows)
        print(f"\nFine-tuning dataset: {len(finetune_rows)} labeled rows")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

    print(f"\nFine-tuning: load the output CSV in notebooks/finetune_with_real_data.ipynb")


def _parse_float(val) -> Optional[float]:
    try:
        v = float(str(val).strip())
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _error_row(fname, child_name, age_months, sex, actual_height, actual_weight, error_msg):
    return {
        "image_file":   fname,
        "child_name":   child_name,
        "age_months":   round(age_months, 1),
        "sex":          sex,
        "actual_height_cm": actual_height,
        "actual_weight_kg": actual_weight,
        "error":        error_msg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch-assess photos and compare against ground-truth measurements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--template", action="store_true",
        help="Generate a blank ground-truth CSV template and exit."
    )
    parser.add_argument(
        "--images", type=Path, default=None,
        help="Directory containing JPEG/PNG images to process."
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=None,
        metavar="CSV",
        help="CSV with manual measurements (see --template for format)."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/batch_results.csv"),
        help="Output CSV path (default: data/batch_results.csv)."
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.template:
        generate_template()
        return

    if not args.images:
        parser.error("--images is required (or use --template to generate a blank CSV).")

    if not args.images.is_dir():
        parser.error(f"--images path does not exist or is not a directory: {args.images}")

    run_batch(
        images_dir=args.images,
        ground_truth_csv=args.ground_truth,
        output_csv=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
