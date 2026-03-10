"""
Batch assessment: run all images through the growth monitor and compare with ground truth.

Workflow
--------
1. Prepare a CSV with your manual measurements (run --template to generate a blank one).
2. Point the script at your image folder and the CSV.
3. Get back a detailed comparison report + a labeled dataset ready for model fine-tuning.

Usage
-----
# Generate a blank template to fill in:
    python scripts/batch_assess.py --template

# Run batch assessment:
    python scripts/batch_assess.py \\
        --images  /path/to/photos/ \\
        --ground-truth  data/ground_truth.csv \\
        --output  data/batch_results.csv

# Images without ground truth (just run the system):
    python scripts/batch_assess.py --images /path/to/photos/

Ground-truth CSV columns
------------------------
image_file         : filename (just the name, not full path) — must match a file in --images
child_name         : child's name or ID
date_of_birth      : YYYY-MM-DD
sex                : M or F
actual_height_cm   : measured height in cm  (leave blank if unknown)
actual_weight_kg   : measured weight in kg  (leave blank if unknown)
notes              : free-text (optional)

All other columns are ignored and preserved in the output.
"""

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEMPLATE_CSV = """\
image_file,child_name,date_of_birth,sex,actual_height_cm,actual_weight_kg,notes
child1.jpg,Child 1,2022-03-15,M,85.0,11.2,
child2.jpg,Child 2,2023-07-01,F,78.5,,weight not available
child3.jpg,Child 3,2021-11-20,M,,,height and weight unknown
"""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def generate_template(out_path: Path = Path("data/ground_truth_template.csv")):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(TEMPLATE_CSV)
    print(f"Template written to {out_path}")
    print("Fill in the measurements and save as data/ground_truth.csv")


def _compute_age_months(dob: date) -> float:
    delta = datetime.utcnow().date() - dob
    return delta.days / 30.4375


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

    # Build ground-truth lookup keyed by image filename
    gt_lookup: dict[str, dict] = {}
    if ground_truth_csv and ground_truth_csv.exists():
        with open(ground_truth_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("image_file", "").strip()
                if fname:
                    gt_lookup[fname] = row

    # Collect images
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_files)} image(s). Ground truth: {len(gt_lookup)} row(s).\n")

    results = []

    for img_path in image_files:
        fname = img_path.name
        gt = gt_lookup.get(fname, {})

        # Parse ground truth
        child_name = gt.get("child_name", fname)
        sex = gt.get("sex", "M").strip().upper() or "M"

        dob_str = (gt.get("date_of_birth") or "").strip()
        try:
            dob = date.fromisoformat(dob_str)
            age_months = _compute_age_months(dob)
        except ValueError:
            dob = None
            age_months = 24.0  # fallback for systems with no DOB

        actual_height = _parse_float(gt.get("actual_height_cm"))
        actual_weight = _parse_float(gt.get("actual_weight_kg"))

        if verbose:
            print(f"  Processing {fname} ...")

        # --- Run pose estimation ---
        try:
            meas = meas_svc.process_image_with_estimation(
                image_path=str(img_path),
                age_months=age_months,
                sex=sex,
                who_data=who_data,
            )
        except Exception as e:
            print(f"    [ERROR] Pose estimation failed: {e}")
            results.append(_error_row(fname, child_name, age_months, sex, actual_height, actual_weight, str(e)))
            continue

        pred_height = meas.predicted_height_cm
        effective_height = pred_height if pred_height else actual_height

        # --- ML prediction ---
        ml_pred = None
        if effective_height and meas.body_segments:
            try:
                ml_pred = ml_svc.predict(meas.body_segments, age_months, sex, effective_height)
            except Exception as e:
                print(f"    [WARN] ML prediction failed: {e}")

        pred_weight_ml = ml_pred.estimated_weight_kg if ml_pred else None
        wasting_status_ml = ml_pred.wasting_status if ml_pred else None
        sam_prob = round(ml_pred.sam_probability, 4) if ml_pred else None
        mam_prob = round(ml_pred.mam_probability, 4) if ml_pred else None

        # --- Z-scores from ACTUAL measurements (ground truth evaluation) ---
        actual_haz_z    = None
        actual_whz_z    = None
        actual_haz_status = None
        actual_whz_status = None
        if actual_height and dob:
            actual_haz_z = nutr_svc.compute_haz(sex, int(round(age_months)), actual_height)
            if actual_haz_z:
                actual_haz_status = _haz_status_from_z(actual_haz_z)
                actual_haz_z = round(actual_haz_z, 3)
        if actual_height and actual_weight and dob:
            actual_whz_z = nutr_svc.compute_whz(sex, age_months, actual_height, actual_weight)
            if actual_whz_z:
                actual_whz_status = _whz_status_from_z(actual_whz_z)
                actual_whz_z = round(actual_whz_z, 3)

        # --- Z-scores from PREDICTED measurements ---
        pred_haz_z    = None
        pred_whz_z    = None
        pred_haz_status = None
        pred_whz_status = None
        if pred_height and dob:
            pred_haz_z = nutr_svc.compute_haz(sex, int(round(age_months)), pred_height)
            if pred_haz_z:
                pred_haz_status = _haz_status_from_z(pred_haz_z)
                pred_haz_z = round(pred_haz_z, 3)
        if pred_height and pred_weight_ml and dob:
            pred_whz_z = nutr_svc.compute_whz(sex, age_months, pred_height, pred_weight_ml)
            if pred_whz_z:
                pred_whz_status = _whz_status_from_z(pred_whz_z)
                pred_whz_z = round(pred_whz_z, 3)

        # --- Height/weight errors ---
        height_error = round(pred_height - actual_height, 2) if pred_height and actual_height else None
        weight_error = round(pred_weight_ml - actual_weight, 2) if pred_weight_ml and actual_weight else None

        row = {
            "image_file":           fname,
            "child_name":           child_name,
            "age_months":           round(age_months, 1),
            "sex":                  sex,
            # Ground truth
            "actual_height_cm":     actual_height,
            "actual_weight_kg":     actual_weight,
            "actual_haz_z":         actual_haz_z,
            "actual_whz_z":         actual_whz_z,
            "actual_haz_status":    actual_haz_status,
            "actual_whz_status":    actual_whz_status,
            # Predictions
            "pred_height_cm":       round(pred_height, 1) if pred_height else None,
            "pred_weight_ml_kg":    round(pred_weight_ml, 2) if pred_weight_ml else None,
            "pred_haz_z":           pred_haz_z,
            "pred_whz_z":           pred_whz_z,
            "pred_haz_status":      pred_haz_status,
            "pred_whz_status":      pred_whz_status,
            "ml_wasting_status":    wasting_status_ml,
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
            "error":                "",
        }

        # ML features for fine-tuning dataset
        if ml_svc.is_available and meas.body_segments and effective_height:
            features = ml_svc.extract_features(meas.body_segments, age_months, sex, effective_height)
            if features:
                row.update({
                    "feat_shoulder_width_cm":   round(features.shoulder_width_cm, 2),
                    "feat_hip_width_cm":        round(features.hip_width_cm, 2),
                    "feat_torso_length_cm":     round(features.torso_length_cm, 2),
                    "feat_upper_arm_length_cm": round(features.upper_arm_length_cm, 2),
                    "feat_shoulder_height_ratio": round(features.shoulder_height_ratio, 4),
                    "feat_hip_height_ratio":    round(features.hip_height_ratio, 4),
                    "feat_body_build_score":    features.body_build_score,
                    # Ground-truth label for fine-tuning (only if actual WHZ computed)
                    "finetune_label":           actual_whz_status if actual_whz_z is not None else "",
                })

        results.append(row)

        if verbose:
            ht = f"{pred_height:.1f}cm" if pred_height else "N/A"
            wt = f"{pred_weight_ml:.1f}kg" if pred_weight_ml else "N/A"
            err_h = f" (err={height_error:+.1f})" if height_error is not None else ""
            err_w = f" (err={weight_error:+.1f})" if weight_error is not None else ""
            wst = wasting_status_ml or "N/A"
            print(f"    Height: {ht}{err_h}  Weight(ML): {wt}{err_w}  Wasting: {wst}")

    # --- Write output CSV ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_file", "child_name", "age_months", "sex",
        "actual_height_cm", "actual_weight_kg",
        "actual_haz_z", "actual_whz_z", "actual_haz_status", "actual_whz_status",
        "pred_height_cm", "pred_weight_ml_kg",
        "pred_haz_z", "pred_whz_z", "pred_haz_status", "pred_whz_status",
        "ml_wasting_status", "sam_probability", "mam_probability",
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
