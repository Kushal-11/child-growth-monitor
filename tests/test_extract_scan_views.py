"""Tests for scripts/extract_scan_views.py pure selection helpers."""

from scripts.extract_scan_views import FrameCandidate, choose_best_by_orientation


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


def test_choose_best_by_orientation_ignores_unusable_frames():
    bad_front = FrameCandidate(1, object(), _score("front", 0.99, usable=False))
    good_front = FrameCandidate(2, object(), _score("front", 0.7))
    side = FrameCandidate(3, object(), _score("side", 0.8))

    best = choose_best_by_orientation([bad_front, good_front, side])

    assert best["front"] is good_front
    assert best["side"] is side
    assert best["unknown"] is None


def test_choose_best_by_orientation_uses_rank_not_frame_order():
    weaker = FrameCandidate(1, object(), _score("front", 0.5, sharpness=40))
    stronger = FrameCandidate(2, object(), _score("front", 0.8, sharpness=150))

    best = choose_best_by_orientation([weaker, stronger])

    assert best["front"] is stronger
