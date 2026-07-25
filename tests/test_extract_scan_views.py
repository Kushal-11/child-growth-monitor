"""Tests for scripts/extract_scan_views.py selection and extraction behavior."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pytest import MonkeyPatch

from scripts.extract_scan_views import (
    FrameCandidate,
    _positive_int,
    choose_best_by_orientation,
    extract_child_video_views,
)


def _score(
    orientation: str,
    conf: float,
    sharpness: float = 100.0,
    usable: bool = True,
) -> object:
    class Score:
        pass

    score = Score()
    score.pose_confidence = conf
    score.coverage = 0.9
    score.upright = 1.0
    score.frontal = 0.9 if orientation == "front" else 0.2
    score.sharpness = sharpness
    score.orientation = orientation
    score.usable = usable
    score.reason = "" if usable else "bad frame"
    return score


def test_choose_best_by_orientation_ignores_unusable_frames() -> None:
    bad_front = FrameCandidate(1, object(), _score("front", 0.99, usable=False))
    good_front = FrameCandidate(2, object(), _score("front", 0.7))
    side = FrameCandidate(3, object(), _score("side", 0.8))

    best = choose_best_by_orientation([bad_front, good_front, side])

    assert best["front"] is good_front
    assert best["side"] is side
    assert best["unknown"] is None


def test_choose_best_by_orientation_uses_rank_not_frame_order() -> None:
    weaker = FrameCandidate(1, object(), _score("front", 0.5, sharpness=40))
    stronger = FrameCandidate(2, object(), _score("front", 0.8, sharpness=150))

    best = choose_best_by_orientation([weaker, stronger])

    assert best["front"] is stronger


def _candidate() -> FrameCandidate:
    return FrameCandidate(
        15,
        np.zeros((8, 8, 3), dtype=np.uint8),
        _score("front", 0.9),
    )


def _child_with_videos(tmp_path: Path, *names: str) -> Path:
    child_dir = tmp_path / "raw" / "001"
    child_dir.mkdir(parents=True)
    for name in names:
        (child_dir / name).write_bytes(b"video")
    return child_dir


def test_extract_distinguishes_same_stem_video_extensions(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    child_dir = _child_with_videos(tmp_path, "rotate.mp4", "rotate.mov")
    monkeypatch.setattr(
        "scripts.extract_scan_views._sample_video",
        lambda *_: ([_candidate()], 1, []),
    )

    rows = extract_child_video_views(
        child_dir, tmp_path / "derived", None, 15, 120, force=True,
    )

    outputs = [row["front_best"] for row in rows]
    assert outputs == [
        "video_views/rotate_mov_front_best.jpg",
        "video_views/rotate_mp4_front_best.jpg",
    ]
    assert len(set(outputs)) == 2
    assert all((tmp_path / "derived" / "001" / path).is_file() for path in outputs)


def test_failed_image_write_is_reported_and_not_selected(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    child_dir = _child_with_videos(tmp_path, "rotate.mp4")
    monkeypatch.setattr(
        "scripts.extract_scan_views._sample_video",
        lambda *_: ([_candidate()], 1, []),
    )
    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)

    rows = extract_child_video_views(
        child_dir, tmp_path / "derived", None, 15, 120, force=True,
    )
    manifest = json.loads(
        (tmp_path / "derived/001/video_views/manifest.json").read_text()
    )

    assert rows[0]["front_best"] == ""
    assert "output write failed" in rows[0]["reason"]
    assert manifest["videos"][0]["selected"] == {}
    assert "output write failed" in manifest["videos"][0]["failures"][0]


def test_partial_success_preserves_and_surfaces_all_failures(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    child_dir = _child_with_videos(tmp_path, "rotate.mp4")
    failures = [f"frame {index}: scoring failed" for index in range(12)]
    monkeypatch.setattr(
        "scripts.extract_scan_views._sample_video",
        lambda *_: ([_candidate()], 13, failures.copy()),
    )

    rows = extract_child_video_views(
        child_dir, tmp_path / "derived", None, 15, 120, force=True,
    )
    manifest = json.loads(
        (tmp_path / "derived/001/video_views/manifest.json").read_text()
    )

    assert rows[0]["front_best"]
    assert rows[0]["reason"] == "12 frame/output failure(s); see manifest"
    assert manifest["videos"][0]["failures"] == failures


def test_manifest_cache_invalidates_when_video_is_added(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    child_dir = _child_with_videos(tmp_path, "first.mp4")
    calls: list[str] = []

    def fake_sample(video: Path, *_: object) -> tuple[list[FrameCandidate], int, list[str]]:
        calls.append(video.name)
        return [_candidate()], 1, []

    monkeypatch.setattr(
        "scripts.extract_scan_views._sample_video",
        fake_sample,
    )
    out_root = tmp_path / "derived"
    extract_child_video_views(
        child_dir, out_root, None, 15, 120, force=False,
    )
    (child_dir / "second.mov").write_bytes(b"video")
    rows = extract_child_video_views(
        child_dir, out_root, None, 15, 120, force=False,
    )

    assert len(rows) == 2
    assert calls == ["first.mp4", "first.mp4", "second.mov"]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_non_positive_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int(value)
