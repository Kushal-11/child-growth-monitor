"""Tests for scripts/clean_media.py — selection logic and orchestration.

score_photo is monkeypatched everywhere; no MediaPipe model is loaded.
"""
import json
from pathlib import Path

from scripts.photo_qc import PhotoScore
from scripts.clean_media import Candidate, select_best, clean_child, run_clean


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


# ---------------------------------------------------------------------------
# Finding 1: a malformed/incomplete provenance.json must not abort the child
# or the batch — it should be treated as "not yet cleaned" and re-cleaned.
# ---------------------------------------------------------------------------

def test_clean_child_recleans_on_corrupt_provenance_json(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "010"
    raw.mkdir(parents=True)
    import cv2
    import numpy as np
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(raw / "front.jpg"), img)

    cleaned_child_dir = tmp_path / "cleaned" / "010"
    cleaned_child_dir.mkdir(parents=True)
    # Truncated JSON, as if the process crashed mid-write.
    (cleaned_child_dir / "provenance.json").write_text('{"child_id": "010", "verd')

    monkeypatch.setattr(
        "scripts.clean_media.score_photo",
        lambda *a, **k: _score(),
    )
    row = clean_child(raw, tmp_path / "cleaned", landmarker=None, force=False)
    assert row["verdict"] == "usable_no_side"
    prov = json.loads((cleaned_child_dir / "provenance.json").read_text())
    assert prov["child_id"] == "010"


def test_clean_child_recleans_on_incomplete_provenance(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "011"
    raw.mkdir(parents=True)
    import cv2
    import numpy as np
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(raw / "front.jpg"), img)

    cleaned_child_dir = tmp_path / "cleaned" / "011"
    cleaned_child_dir.mkdir(parents=True)
    # Valid JSON, but missing "verdict" — as an older version of the script
    # might have written before the field existed.
    (cleaned_child_dir / "provenance.json").write_text(json.dumps({"child_id": "011"}))

    called = []

    def fake_score(*a, **k):
        called.append(1)
        return _score()

    monkeypatch.setattr("scripts.clean_media.score_photo", fake_score)
    row = clean_child(raw, tmp_path / "cleaned", landmarker=None, force=False)
    assert called, "should have re-scored rather than trusting incomplete provenance"
    assert row["verdict"] == "usable_no_side"


def test_run_clean_isolates_failing_child(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "001").mkdir(parents=True)
    (raw / "002").mkdir(parents=True)
    cleaned = tmp_path / "cleaned"
    report = tmp_path / "reports" / "qc_report.csv"

    monkeypatch.setattr("scripts.clean_media.build_landmarker", lambda: None)

    def fake_clean_child(child_dir, cleaned_root, landmarker, force):
        if child_dir.name == "001":
            raise RuntimeError("boom: corrupt state")
        return {
            "child_id": "002", "verdict": "ok",
            "front_source": "front.jpg", "front_via": "filename",
            "side_source": "side.jpg", "side_via": "filename",
            "needs_confirmation": False, "reason": "",
        }

    monkeypatch.setattr("scripts.clean_media.clean_child", fake_clean_child)

    rows = run_clean(raw, cleaned, report, force=False)

    assert report.exists()          # QC report is still written for the batch
    row_001 = next(r for r in rows if r["child_id"] == "001")
    assert row_001["verdict"] == "failed"
    assert "boom" in row_001["reason"]
    row_002 = next(r for r in rows if r["child_id"] == "002")
    assert row_002["verdict"] == "ok"


# ---------------------------------------------------------------------------
# Finding 2: auto_classified must reflect the SELECTED candidate, not merely
# whether an unnamed candidate was ever added to a pool.
# ---------------------------------------------------------------------------

def test_auto_classified_only_when_selected_is_unnamed():
    named = Candidate(Path("front.jpg"), "front", _score(conf=0.9))
    unnamed = Candidate(Path("IMG_1.jpg"), "", _score(orientation="front", conf=0.5))
    sel = select_best([named, unnamed])
    assert sel["front"] is named
    assert sel["auto_classified"] is False


# ---------------------------------------------------------------------------
# Finding 3: usable_no_side must carry the rejection reason when a side photo
# existed but failed QC (distinct from "no side photo was ever taken").
# ---------------------------------------------------------------------------

def test_usable_no_side_reports_rejection_reason(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    import cv2
    import numpy as np
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(raw / "front.jpg"), img)
    cv2.imwrite(str(raw / "side.jpg"), img)

    pending = [
        _score(),                                                     # front.jpg usable
        _score(usable=False, orientation="side", reason="image too blurry"),  # side.jpg rejected
    ]
    monkeypatch.setattr(
        "scripts.clean_media.score_photo",
        lambda image_bgr, landmarker: pending.pop(0),
    )
    row = clean_child(raw, tmp_path / "cleaned", landmarker=None, force=False)
    assert row["verdict"] == "usable_no_side"
    assert "blurry" in row["reason"]


# ---------------------------------------------------------------------------
# Finding 4: video-fallback path (no usable photo, but a video exists).
# ---------------------------------------------------------------------------

def test_clean_child_video_fallback_success(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "004"
    raw.mkdir(parents=True)
    import cv2
    import numpy as np
    cv2.imwrite(str(raw / "front.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    (raw / "clip.mp4").write_bytes(b"not a real video")

    monkeypatch.setattr(
        "scripts.clean_media.score_photo",
        lambda *a, **k: _score(usable=False, reason="image too blurry"),
    )

    calls = []

    def fake_extract(video_path, output_path, verbose=False):
        calls.append((video_path, output_path))
        Path(output_path).write_bytes(b"fake-jpg-bytes")
        return Path(output_path)

    # extract_best_frame is imported lazily (function-local, not module-level)
    # so that importing scripts.clean_media never pulls in mediapipe/tensorflow.
    # A function-level import re-resolves the name from its source module on
    # every call, so the source must be patched rather than the destination.
    monkeypatch.setattr("scripts.extract_best_frame.extract_best_frame", fake_extract)

    row = clean_child(raw, tmp_path / "cleaned", landmarker=None, force=False)
    assert calls, "fake_extract was never called — fallback path not exercised"
    assert row["verdict"] == "usable_no_side"
    prov = json.loads((tmp_path / "cleaned" / "004" / "provenance.json").read_text())
    assert prov["front"]["via"] == "video_fallback"
    assert (tmp_path / "cleaned" / "004" / "front.jpg").exists()


def test_clean_child_video_fallback_raises_reports_failed(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "005"
    raw.mkdir(parents=True)
    (raw / "clip.mp4").write_bytes(b"not a real video")

    calls = []

    def fake_extract(video_path, output_path, verbose=False):
        calls.append((video_path, output_path))
        raise RuntimeError("no pose found in any sampled frame")

    # See test_clean_child_video_fallback_success for why the source module
    # (not scripts.clean_media) is patched here.
    monkeypatch.setattr("scripts.extract_best_frame.extract_best_frame", fake_extract)

    row = clean_child(raw, tmp_path / "cleaned", landmarker=None, force=False)
    assert calls, "fake_extract was never called — fallback path not exercised"
    assert row["verdict"] == "failed"
    assert "no pose found" in row["reason"]
