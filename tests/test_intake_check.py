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
