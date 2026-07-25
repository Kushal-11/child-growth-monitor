"""
Extract representative scan views from rotating child videos.

This stage is intentionally conservative: it samples frames, scores each frame
with the existing pose QC gate, groups usable frames by pose-derived
orientation, and writes the best frame per requested role under
field_data/derived/<child_id>/video_views/.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/extract_scan_views.py
    PYTHONPATH=. .venv/bin/python scripts/extract_scan_views.py --force
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.intake_check import VIDEO_EXTENSIONS  # noqa: E402

DEFAULT_RAW = Path("field_data/raw")
DEFAULT_OUT = Path("field_data/derived")
DEFAULT_REPORT = Path("field_data/reports/video_views.csv")
DEFAULT_SAMPLE_EVERY = 15
DEFAULT_MAX_FRAMES = 120
FRONTAL_SCORE_FLOOR = 0.05
SHARPNESS_SCORE_CAP = 200.0
POSE_RANK_WEIGHT = 0.7
SHARPNESS_RANK_WEIGHT = 0.3
REPORT_COLS = [
    "child_id", "video_file", "frames_sampled", "usable_frames",
    "front_best", "side_best", "unknown_best", "reason",
]


@dataclass
class FrameCandidate:
    frame_idx: int
    image_bgr: object
    score: object


def _score_dict(s: object) -> dict:
    return {
        "pose_confidence": round(s.pose_confidence, 3),
        "coverage": round(s.coverage, 3),
        "upright": round(s.upright, 3),
        "frontal": round(s.frontal, 3),
        "sharpness": round(s.sharpness, 1),
        "orientation": s.orientation,
    }


def rank_frame(cand: FrameCandidate, role: str) -> float:
    """Rank a usable video frame for a target view."""
    s = cand.score
    base = s.pose_confidence * s.coverage * s.upright
    if role == "front":
        base *= max(s.frontal, FRONTAL_SCORE_FLOOR)
    # Prefer sharp frames, capped so a blurry high-confidence pose can still
    # lose to a crisp frame with similar pose quality.
    sharpness = min(s.sharpness / SHARPNESS_SCORE_CAP, 1.0)
    return base * (
        POSE_RANK_WEIGHT + SHARPNESS_RANK_WEIGHT * sharpness
    )


def choose_best_by_orientation(
    candidates: Iterable[FrameCandidate],
) -> dict[str, FrameCandidate | None]:
    """Return the best usable front/side/unknown frame candidates."""
    pools: dict[str, list[FrameCandidate]] = {"front": [], "side": [], "unknown": []}
    for cand in candidates:
        if not cand.score.usable:
            continue
        pools.setdefault(cand.score.orientation, []).append(cand)
    return {
        role: max(items, key=lambda c: rank_frame(c, role), default=None)
        for role, items in pools.items()
    }


def _sample_video(
    video_path: Path, landmarker: object, sample_every: int, max_frames: int
) -> tuple[list[FrameCandidate], int, list[str]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0, ["could not open video"]

    candidates: list[FrameCandidate] = []
    failures: list[str] = []
    sampled = 0
    frame_idx = -1
    try:
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue
            sampled += 1
            try:
                from scripts.photo_qc import score_photo

                score = score_photo(frame, landmarker)
            except Exception as e:
                failures.append(f"frame {frame_idx}: {e}")
                continue
            if not score.usable:
                failures.append(
                    f"frame {frame_idx}: {score.reason or 'failed pose QC'}"
                )
                continue
            candidates.append(FrameCandidate(frame_idx, frame.copy(), score))
    finally:
        cap.release()
    return candidates, sampled, failures


def _video_inputs(videos: Iterable[Path]) -> list[dict]:
    """Build a stable cache key for the current video inputs."""
    return [
        {
            "name": video.name,
            "size": video.stat().st_size,
            "mtime_ns": video.stat().st_mtime_ns,
        }
        for video in videos
    ]


def _manifest_outputs_exist(
    manifest: dict, child_out_root: Path,
) -> bool:
    """Return whether every selected path recorded by a manifest still exists."""
    for row in manifest.get("rows") or []:
        for role in ("front", "side", "unknown"):
            relative_path = row.get(f"{role}_best")
            if relative_path and not (child_out_root / relative_path).is_file():
                return False
    return True


def extract_child_video_views(
    child_dir: Path,
    out_root: Path,
    landmarker: object,
    sample_every: int,
    max_frames: int,
    force: bool,
) -> list[dict]:
    """Extract best orientation frames for every video in one child folder."""
    rows: list[dict] = []
    child_id = child_dir.name
    videos = sorted(
        p for p in child_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and p.stat().st_size > 0
    )
    out_dir = out_root / child_id / "video_views"
    manifest_path = out_dir / "manifest.json"
    inputs = _video_inputs(videos)
    if manifest_path.exists() and not force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_matches = (
                manifest.get("inputs") == inputs
                and manifest.get("sample_every") == sample_every
                and manifest.get("max_frames") == max_frames
                and isinstance(manifest.get("rows"), list)
                and _manifest_outputs_exist(manifest, out_root / child_id)
            )
            if cache_matches:
                return manifest["rows"]
        except (OSError, json.JSONDecodeError):
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "child_id": child_id,
        "inputs": inputs,
        "sample_every": sample_every,
        "max_frames": max_frames,
        "videos": [],
        "rows": rows,
    }
    for video in videos:
        candidates, sampled, failures = _sample_video(
            video, landmarker, sample_every, max_frames
        )
        best = choose_best_by_orientation(candidates)
        video_record = {
            "source": video.name,
            "frames_sampled": sampled,
            "usable_frames": len(candidates),
            "selected": {},
            "failures": failures,
        }
        row = {
            "child_id": child_id,
            "video_file": video.name,
            "frames_sampled": sampled,
            "usable_frames": video_record["usable_frames"],
            "front_best": "",
            "side_best": "",
            "unknown_best": "",
            "reason": "",
        }
        for role in ("front", "side", "unknown"):
            cand = best.get(role)
            if cand is None:
                continue
            suffix = video.suffix.lower().lstrip(".")
            out_name = f"{video.stem}_{suffix}_{role}_best.jpg"
            out_path = out_dir / out_name
            import cv2

            try:
                written = cv2.imwrite(str(out_path), cand.image_bgr)
            except Exception as error:
                failures.append(f"{role} output write failed: {error}")
                continue
            if not written:
                failures.append(
                    f"{role} output write failed: {out_path.name}"
                )
                continue
            rel = str(out_path.relative_to(out_root / child_id))
            row[f"{role}_best"] = rel
            video_record["selected"][role] = {
                "path": rel,
                "frame_idx": cand.frame_idx,
                "scores": _score_dict(cand.score),
            }
        if not any(row[f"{role}_best"] for role in ("front", "side", "unknown")):
            row["reason"] = "; ".join(failures) or "no usable frame"
        elif failures:
            row["reason"] = f"{len(failures)} frame/output failure(s); see manifest"
        rows.append(row)
        manifest["videos"].append(video_record)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows


def run_extract(
    raw_dir: Path,
    out_root: Path,
    report_path: Path,
    sample_every: int,
    max_frames: int,
    force: bool,
) -> list[dict]:
    from scripts.photo_qc import build_landmarker

    landmarker = build_landmarker()
    rows: list[dict] = []
    for child_dir in sorted(d for d in raw_dir.iterdir() if d.is_dir()):
        print(f"  Extracting video views for {child_dir.name} ...")
        rows.extend(
            extract_child_video_views(
                child_dir, out_root, landmarker, sample_every, max_frames, force
            )
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Video-view report written to {report_path}")
    return rows


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--sample-every", type=_positive_int, default=DEFAULT_SAMPLE_EVERY,
    )
    parser.add_argument(
        "--max-frames", type=_positive_int, default=DEFAULT_MAX_FRAMES,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.raw.is_dir():
        print(f"Raw directory not found: {args.raw}", file=sys.stderr)
        sys.exit(1)
    run_extract(
        args.raw, args.out, args.report, args.sample_every, args.max_frames,
        args.force,
    )


if __name__ == "__main__":
    main()
