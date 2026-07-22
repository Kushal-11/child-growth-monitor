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
    pools: dict[str, list[Candidate]] = {"front": [], "side": []}
    for c in cands:
        if not c.score.usable:
            continue
        role = c.role_hint or c.score.orientation
        if role not in pools:
            continue                      # 'unknown' orientation: ineligible
        pools[role].append(c)

    front = max(pools["front"], key=lambda c: _rank(c, "front"), default=None)
    side = max(pools["side"], key=lambda c: _rank(c, "side"), default=None)
    # auto_classified means an auto-classified photo was actually selected —
    # not merely that one was in the running.
    auto = (front is not None and not front.role_hint) or \
           (side is not None and not side.role_hint)

    return {
        "front": front,
        "side": side,
        "auto_classified": auto,
    }


# Roles the assessment pipeline consumes. Anything else is archived in
# raw/ but never fed to pose measurement.
SELECTABLE_ROLES = ("front", "side")

# Roles that are captured in the field but are NOT usable for whole-body
# pose measurement. They must be recognised by NAME because the pose
# classifier cannot identify them:
#   back - a rear view has the same shoulder-width/torso-height ratio as a
#          front view, so landmark_metrics() reports it as 'front'. Left
#          unnamed, a sharp back photo can outscore the real front photo
#          and be measured as if the child were facing the camera.
#   arm  - a MUAC close-up has no full-body pose at all, so it is rejected
#          as unusable; naming it keeps it out of the failure list, where
#          it would read as a photo problem rather than a different shot.
ARCHIVED_ROLE_PREFIXES = {
    "back": ("back", "rear"),
    "arm": ("arm", "muac"),
}


def _role_hint(path: Path) -> str:
    """Map a filename to a role. '' means 'let the pose classifier decide'."""
    stem = path.stem.lower()
    for role in SELECTABLE_ROLES:
        if stem.startswith(role):
            return role
    for role, prefixes in ARCHIVED_ROLE_PREFIXES.items():
        if stem.startswith(prefixes):
            return role
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
    child_dir: Path, cleaned_root: Path, landmarker: object, force: bool,
) -> dict:
    """Clean one child folder. Returns a QC report row (never raises)."""
    cid = child_dir.name
    out_dir = cleaned_root / cid
    prov_path = out_dir / "provenance.json"

    if prov_path.exists() and not force:
        try:
            prov = json.loads(prov_path.read_text())
            if "child_id" not in prov or "verdict" not in prov:
                raise KeyError("provenance missing required keys")
            return _report_row_from_provenance(prov)
        except (json.JSONDecodeError, OSError, KeyError):
            # Truncated (crash mid-write) or written by an older version:
            # treat as "not yet cleaned" and fall through to re-clean rather
            # than aborting the whole batch.
            pass

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
    side_reject_reasons: list[str] = []
    archived: dict[str, list[str]] = {}
    for p in photos:
        hint = _role_hint(p)
        if hint in ARCHIVED_ROLE_PREFIXES:
            # Not a whole-body pose shot. Record that it exists and move on
            # without scoring: a MUAC close-up has no full-body pose, so
            # scoring it only produces a "no pose detected" entry that reads
            # like a photo problem in the recapture list.
            archived.setdefault(hint, []).append(p.name)
            continue
        img = cv2.imread(str(p))
        if img is None:
            reason = f"{p.name}: unreadable"
            fail_reasons.append(reason)
            if hint == "side":
                side_reject_reasons.append(reason)
            continue
        try:
            s = score_photo(img, landmarker)
        except Exception as e:              # one bad photo never aborts the child
            reason = f"{p.name}: {e}"
            fail_reasons.append(reason)
            if hint == "side":
                side_reject_reasons.append(reason)
            continue
        if not s.usable:
            reason = f"{p.name}: {s.reason}"
            fail_reasons.append(reason)
            if hint == "side" or (not hint and s.orientation == "side"):
                side_reject_reasons.append(reason)
        cands.append(Candidate(p, hint, s))

    sel = select_best(cands)
    front, side = sel["front"], sel["side"]
    prov: dict = {
        "child_id": cid, "front": None, "side": None,
        "needs_confirmation": sel["auto_classified"], "reason": "",
        # Shots kept in raw/ that this pipeline does not measure. Recorded
        # so a later stage (e.g. an image-MUAC model) can find them without
        # rescanning, and so their absence is visible now.
        "archived": archived,
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
        # Fallback: best frame from the first video that yields a *usable*
        # frame. Imported here (not at module level) because extract_best_frame
        # pulls in mediapipe/tensorflow, which are heavy and should stay
        # lazy for consumers that never hit the video-fallback path.
        from scripts.extract_best_frame import extract_best_frame
        frame_path = out_dir / "front.jpg"
        for v in videos:
            try:
                extract_best_frame(v, frame_path, verbose=False)
            except Exception as e:
                fail_reasons.append(f"{v.name}: {e}")
                continue
            # extract_best_frame only checks landmark visibility (>= 0.4);
            # it never applies MIN_COVERAGE/MIN_UPRIGHT/MIN_SHARPNESS/
            # MIN_POSE_CONFIDENCE. Without this re-score, a frame with the
            # child's feet cut off (or heavily blurred) would be promoted
            # into the study with no QC signal at all — re-run it through
            # the exact same gate a photo goes through.
            frame_img = cv2.imread(str(frame_path))
            if frame_img is None:
                fail_reasons.append(f"{v.name}: extracted frame unreadable")
                frame_path.unlink(missing_ok=True)
                continue
            try:
                frame_score = score_photo(frame_img, landmarker)
            except Exception as e:
                fail_reasons.append(f"{v.name}: {e}")
                frame_path.unlink(missing_ok=True)
                continue
            if not frame_score.usable:
                fail_reasons.append(f"{v.name}: {frame_score.reason}")
                frame_path.unlink(missing_ok=True)
                continue
            prov["front"] = {
                "source": v.name,
                "via": "video_fallback",
                "scores": _score_dict(frame_score),
            }
            break

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
        # Distinguish "a side photo existed but failed QC" from "no side
        # photo was ever taken" — the latter leaves reason empty.
        if side_reject_reasons:
            prov["reason"] = "; ".join(side_reject_reasons)
    else:
        prov["verdict"] = "ok"

    prov_path.write_text(json.dumps(prov, indent=2))
    return _report_row_from_provenance(prov)


def _report_row_from_provenance(prov: dict) -> dict:
    """Build a QC row from a provenance dict, defensively.

    Uses .get() with sensible defaults throughout so a provenance dict
    missing an optional key still yields a usable row instead of raising.
    """
    front = prov.get("front") or {}
    side = prov.get("side") or {}
    return {
        "child_id": prov.get("child_id", ""),
        "verdict": prov.get("verdict", "failed"),
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
    landmarker: object = build_landmarker()
    rows: list[dict] = []
    for child_dir in sorted(d for d in raw_dir.iterdir() if d.is_dir()):
        print(f"  Cleaning {child_dir.name} ...")
        try:
            rows.append(clean_child(child_dir, cleaned_root, landmarker, force))
        except Exception as e:
            # One child's unexpected failure must never abort the batch —
            # record it as failed and keep going so the QC report still
            # covers every child.
            print(f"  ERROR     [{child_dir.name}] {e}")
            rows.append({
                "child_id": child_dir.name,
                "verdict": "failed",
                "front_source": "",
                "front_via": "",
                "side_source": "",
                "side_via": "",
                "needs_confirmation": False,
                "reason": f"unexpected error: {e}",
            })

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
