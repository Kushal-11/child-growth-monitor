"""
Diagnose and calibrate the Parle-G reference-object scale.

Height from a single photo needs an absolute scale. The pipeline's primary
estimator does not have one - it returns the WHO median height for the
child's age, so predicted height-for-age is ~0 for every child and stunting
cannot be detected. A reference object of known size in frame is the way
out, and this script tells you whether that actually works on your photos.

It answers two questions:

  1. DETECTION - on what fraction of photos is the packet found at all, and
     does the found rectangle look like a packet (aspect ratio, size)?

  2. CALIBRATION - for children who also have a tape-measured height, what
     packet length would make the photo agree with the tape?

         implied_cm = measured_height_cm * packet_px / child_height_px

     Run over many children this inverts the problem: you never have to know
     which edge the detector latched onto. A TIGHT cluster means the
     detector is consistent and its centre is your constant. A WIDE spread
     means it is finding different things in different photos, and
     colour-based detection is not viable - no single constant will fix it.

Nothing is written to config; this only reports. Set the constant yourself
once you have seen the spread.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/calibrate_reference.py
    PYTHONPATH=. .venv/bin/python scripts/calibrate_reference.py --annotate
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_ground_truth import load_csv  # noqa: E402

DEFAULT_IMAGES = Path("field_data/cleaned")
DEFAULT_GROUND_TRUTH = Path("field_data/ground_truth.csv")
DEFAULT_OUT = Path("field_data/reports/reference_calibration.csv")

OUT_COLS = [
    "child_id", "detected", "packet_px", "packet_aspect", "pose_confidence",
    "child_height_px", "measured_height_cm", "implied_packet_cm", "note",
]


def _f(row: dict, key: str) -> Optional[float]:
    try:
        return float((row.get(key) or "").strip())
    except (TypeError, ValueError):
        return None


def measure_one(image_path: Path, svc) -> dict:
    """Detect the packet and the child's pixel height in one photo."""
    import cv2

    out = {
        "detected": False, "packet_px": None, "packet_aspect": None,
        "pose_confidence": None,
        "child_height_px": None, "note": "",
    }
    img = cv2.imread(str(image_path))
    if img is None:
        out["note"] = "unreadable image"
        return out

    scale_factor, detected, box = svc._detect_parlegi(img)
    if detected and box is not None:
        rect = cv2.minAreaRect(box.astype("float32"))
        w, h = rect[1]
        longer, shorter = max(w, h), min(w, h)
        out["detected"] = True
        out["packet_px"] = round(longer, 1)
        out["packet_aspect"] = round(longer / shorter, 2) if shorter else None
    else:
        out["note"] = "packet not detected"

    # Child height in pixels: head_y to heel_y, exactly as the pipeline
    # derives it (measurement_service.py:751), so the calibration matches
    # what the estimator would actually use.
    try:
        pose = svc._detect_pose(str(image_path), img.shape[:2])
    except Exception as e:
        out["note"] = (out["note"] + f"; pose failed: {e}").strip("; ")
        return out

    head_y, heel_y = pose.get("head_y"), pose.get("heel_y")
    out["pose_confidence"] = round(pose.get("confidence") or 0.0, 3)
    if head_y is None or heel_y is None:
        out["note"] = (out["note"] + "; no pose detected").strip("; ")
        return out
    if not pose.get("posture_valid", True):
        issues = "; ".join(pose.get("posture_issues") or [])
        out["note"] = (out["note"] + f"; posture: {issues}").strip("; ")

    out["child_height_px"] = round(abs(heel_y - head_y), 1)
    return out


def summarize(rows: list[dict]) -> None:
    total = len(rows)
    detected = [r for r in rows if r["detected"]]
    implied = [
        r["implied_packet_cm"] for r in rows
        if r.get("implied_packet_cm") is not None
    ]

    print(f"\n{'=' * 62}")
    print(f"DETECTION: packet found in {len(detected)}/{total} photos "
          f"({len(detected) / total * 100:.0f}%)" if total else "no photos")

    if detected:
        aspects = [r["packet_aspect"] for r in detected if r["packet_aspect"]]
        if aspects:
            print(f"  detected aspect ratio: median {statistics.median(aspects):.2f} "
                  f"(range {min(aspects):.2f}-{max(aspects):.2f})")
            print("  a real packet is about 14.75/5.2 = 2.84 long-edge, or "
                  "11.4/5.0 = 2.28 for the biscuit block")

    if not implied:
        print("\nCALIBRATION: no children had BOTH a detected packet and a "
              "measured height, so the packet length cannot be calibrated.")
        print(f"{'=' * 62}")
        return

    med = statistics.median(implied)
    print(f"\nCALIBRATION: {len(implied)} children with packet + measured height")
    print(f"  implied packet length: median {med:.2f} cm")
    print(f"                         range  {min(implied):.2f} - {max(implied):.2f} cm")
    if len(implied) > 1:
        sd = statistics.stdev(implied)
        cv = sd / med * 100 if med else float("inf")
        print(f"                         sd     {sd:.2f} cm  "
              f"(coefficient of variation {cv:.1f}%)")
        print()
        # The verdict. A height error propagates straight into HAZ, so the
        # spread of the implied constant IS the method's error bar.
        if cv < 5:
            print("  VERDICT: tight cluster - the detector is consistent.")
            print(f"  Set REFERENCE_OBJECT_LENGTH_CM = {med:.1f} in config.py "
                  f"and re-run the pilot.")
        elif cv < 15:
            print("  VERDICT: moderate scatter. Usable but noisy - expect "
                  f"roughly {cv:.0f}% height error, which is "
                  f"{cv / 100 * 75:.1f} cm on a 75 cm child.")
            print("  Inspect the annotated images before trusting this.")
        else:
            print("  VERDICT: wide scatter - the detector is NOT finding the "
                  "same object consistently.")
            print("  No single constant will fix this. Colour-based detection "
                  "is not viable on these photos;")
            print("  consider a printed marker (ArUco) or a depth sensor.")
    print(f"{'=' * 62}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--annotate", action="store_true",
                        help="Write annotated copies showing the detected box.")
    args = parser.parse_args()

    if not args.images.is_dir():
        print(f"Images directory not found: {args.images}", file=sys.stderr)
        sys.exit(1)

    from app.services.measurement_service import MeasurementService
    svc = MeasurementService()

    gt: dict[str, dict] = {}
    if args.ground_truth.exists():
        for row in load_csv(args.ground_truth):
            cid = (row.get("child_id") or "").strip()
            if cid:
                gt[cid] = row

    rows: list[dict] = []
    for child_dir in sorted(d for d in args.images.iterdir() if d.is_dir()):
        front = child_dir / "front.jpg"
        if not front.exists():
            continue
        cid = child_dir.name
        print(f"  {cid} ...", end=" ", flush=True)
        r = measure_one(front, svc)
        r["child_id"] = cid

        measured = _f(gt.get(cid, {}), "actual_height_cm")
        r["measured_height_cm"] = measured
        r["implied_packet_cm"] = None
        if measured and r["packet_px"] and r["child_height_px"]:
            r["implied_packet_cm"] = round(
                measured * r["packet_px"] / r["child_height_px"], 2
            )
        rows.append(r)
        print(
            f"packet {'yes' if r['detected'] else 'NO '}"
            + (f"  implied {r['implied_packet_cm']} cm"
               if r["implied_packet_cm"] else "")
        )

        if args.annotate and r["detected"]:
            import cv2
            img = cv2.imread(str(front))
            _, _, box = svc._detect_parlegi(img)
            if box is not None:
                cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
                out_dir = args.out.parent / "annotated"
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"{cid}_packet.jpg"), img)

    if not rows:
        print(f"No front.jpg found under {args.images}", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-child detail written to {args.out}")
    summarize(rows)


if __name__ == "__main__":
    main()
