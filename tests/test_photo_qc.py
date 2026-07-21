"""Tests for scripts/photo_qc.py — pure landmark math, no MediaPipe model."""
from dataclasses import dataclass

import numpy as np

from scripts.photo_qc import KP, landmark_metrics, sharpness_of


@dataclass
class FakeLandmark:
    x: float
    y: float
    visibility: float = 0.99


def _upright_front_pose() -> list:
    """33 landmarks for an ideal upright, camera-facing child."""
    lms = [FakeLandmark(0.5, 0.5) for _ in range(33)]
    lms[KP["nose"]] = FakeLandmark(0.50, 0.08)
    lms[KP["left_shoulder"]] = FakeLandmark(0.62, 0.25)
    lms[KP["right_shoulder"]] = FakeLandmark(0.38, 0.25)
    lms[KP["left_hip"]] = FakeLandmark(0.58, 0.50)
    lms[KP["right_hip"]] = FakeLandmark(0.42, 0.50)
    lms[KP["left_knee"]] = FakeLandmark(0.57, 0.68)
    lms[KP["right_knee"]] = FakeLandmark(0.43, 0.68)
    lms[KP["left_ankle"]] = FakeLandmark(0.56, 0.86)
    lms[KP["right_ankle"]] = FakeLandmark(0.44, 0.86)
    lms[KP["left_heel"]] = FakeLandmark(0.56, 0.88)
    lms[KP["right_heel"]] = FakeLandmark(0.44, 0.88)
    return lms


def test_ideal_front_pose_scores_high():
    m = landmark_metrics(_upright_front_pose())
    assert m is not None
    assert m["pose_confidence"] > 0.9
    assert m["coverage"] > 0.9          # 0.08 -> 0.88 span, normalised by 0.80
    assert m["upright"] == 1.0
    assert m["frontal"] > 0.8
    assert m["orientation"] == "front"


def test_side_pose_classified_side():
    lms = _upright_front_pose()
    # Shoulders and hips nearly overlap in x when the child is side-on
    lms[KP["left_shoulder"]] = FakeLandmark(0.51, 0.25)
    lms[KP["right_shoulder"]] = FakeLandmark(0.49, 0.25)
    lms[KP["left_hip"]] = FakeLandmark(0.51, 0.50)
    lms[KP["right_hip"]] = FakeLandmark(0.49, 0.50)
    m = landmark_metrics(lms)
    assert m is not None
    assert m["orientation"] == "side"


def test_low_visibility_returns_none():
    lms = _upright_front_pose()
    lms[KP["left_ankle"]] = FakeLandmark(0.56, 0.86, visibility=0.1)
    assert landmark_metrics(lms) is None


def test_upside_down_pose_scores_low_upright():
    lms = _upright_front_pose()
    for name in KP:
        lm = lms[KP[name]]
        lms[KP[name]] = FakeLandmark(lm.x, 1.0 - lm.y, lm.visibility)
    m = landmark_metrics(lms)
    assert m is not None
    assert m["upright"] == 0.0


def test_sharpness_flat_image_is_zero():
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert sharpness_of(flat) == 0.0


def test_sharpness_noisy_image_is_high():
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    assert sharpness_of(noisy) > 100.0
