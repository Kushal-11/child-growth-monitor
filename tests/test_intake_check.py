"""Tests for scripts/intake_check.py."""
from pathlib import Path

from scripts.intake_check import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    check_child,
    run_intake,
)


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


# --- Finding 1: video-only child must be flagged, not reported as fine ---

def test_video_only_flagged(tmp_path):
    d = _make_child(tmp_path, "006", ["walk.mp4"])
    row = check_child(d, gt_ids={"006"})
    assert row["n_photos"] == 0
    assert row["n_videos"] == 1
    assert "no photos (video only - will rely on frame extraction)" in row["issues"]


# --- Finding 2: zero-padding mismatch must be reported against the folder,
# not silently normalized, and must not also show up as an orphan gt row ---

def test_padding_variant_reported_on_folder(tmp_path):
    d = _make_child(tmp_path, "007", ["front.jpg"])
    row = check_child(d, gt_ids={"7"})
    assert row["gt_row"] is False
    assert (
        "no exact ground-truth row; id '7' looks like a padding variant"
        in row["issues"]
    )


def test_padding_variant_excluded_from_orphan_list(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_child(raw, "007", ["front.jpg"])
    gt_csv = tmp_path / "ground_truth.csv"
    gt_csv.write_text("child_id\n7\n")
    manifest = tmp_path / "reports" / "manifest.csv"

    run_intake(raw, gt_csv, manifest)

    out = capsys.readouterr().out
    assert "padding variant" in out
    assert "ground-truth rows with no folder" not in out


# --- Finding 3: media nested in a subdirectory must not be reported as
# "no photos or videos" (misleading; the top-level scan is intentional) ---

def test_media_in_subdir_flagged_distinctly(tmp_path):
    d = tmp_path / "008"
    d.mkdir()
    sub = d / "DCIM"
    sub.mkdir()
    (sub / "front.jpg").write_bytes(b"x" * 100)

    row = check_child(d, gt_ids={"008"})
    assert row["n_photos"] == 0
    assert row["n_videos"] == 0
    assert "no photos or videos at top level" in row["issues"]
    assert "DCIM" in row["issues"]
