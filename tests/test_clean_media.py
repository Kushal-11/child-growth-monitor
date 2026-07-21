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
