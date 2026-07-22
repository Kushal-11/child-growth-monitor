"""
Photo quality scoring for field-data cleaning (Stage 2).

Pure landmark math lives in landmark_metrics() so unit tests never need
the MediaPipe model; score_photo() wraps it for real images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

# MediaPipe pose landmark indices
KP = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_heel":      29,
    "right_heel":     30,
}

REQUIRED = ["nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_ankle", "right_ankle"]

# Quality thresholds — defaults derived from extract_best_frame.py scoring.
# Tune once on the first real batch, then freeze for the study.
MIN_VISIBILITY      = 0.4
MIN_POSE_CONFIDENCE = 0.5
MIN_COVERAGE        = 0.6    # normalised head-to-heel span (0.80 raw span = 1.0)
MIN_UPRIGHT         = 0.75
MIN_SHARPNESS       = 40.0   # Laplacian variance
FRONT_RATIO         = 0.45   # shoulder-width / torso-height above this = front
SIDE_RATIO          = 0.25   # below this = side; between = unknown
COVERAGE_NORMALIZER = 0.80   # raw head-to-heel span (fraction of frame
                              # height) that normalises to coverage == 1.0.
                              # Shared with extract_best_frame.py — see the
                              # import note there for why they must match.

POSE_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "pose_landmarker_heavy.task"
)


@dataclass
class PhotoScore:
    pose_confidence: float
    coverage: float
    upright: float
    frontal: float
    sharpness: float
    orientation: str        # 'front' | 'side' | 'unknown'
    usable: bool
    reason: str             # empty when usable


def sharpness_of(image_bgr: np.ndarray) -> float:
    """Laplacian variance — motion-blur / focus measure (higher = sharper)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def landmark_metrics(lms: Sequence) -> Optional[dict]:
    """
    Compute pose sub-scores from a landmark list (items need .x/.y/.visibility).
    Returns None when a required landmark is not visible enough.
    """
    def vis(name: str) -> float:
        return lms[KP[name]].visibility or 0.0

    def y(name: str) -> float:
        return lms[KP[name]].y

    def x(name: str) -> float:
        return lms[KP[name]].x

    for req in REQUIRED:
        if vis(req) < MIN_VISIBILITY:
            return None

    key_joints = ["nose", "left_shoulder", "right_shoulder",
                  "left_hip", "right_hip", "left_knee", "right_knee",
                  "left_ankle", "right_ankle"]
    pose_confidence = float(np.mean([vis(j) for j in key_joints]))

    nose_y = y("nose")
    heel_y = max(y("left_heel"), y("right_heel"),
                 y("left_ankle"), y("right_ankle"))
    coverage = min(max(0.0, heel_y - nose_y) / COVERAGE_NORMALIZER, 1.0)

    order_pairs = [
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ]
    upright = sum(1 for a, b in order_pairs if y(a) < y(b)) / len(order_pairs)

    shoulder_y_diff = abs(y("left_shoulder") - y("right_shoulder"))
    shoulder_width = abs(x("left_shoulder") - x("right_shoulder")) + 1e-6
    frontal = max(0.0, 1.0 - (shoulder_y_diff / shoulder_width) * 3.0)

    torso_h = abs(
        (y("left_hip") + y("right_hip")) / 2
        - (y("left_shoulder") + y("right_shoulder")) / 2
    ) + 1e-6
    ratio = shoulder_width / torso_h
    if ratio >= FRONT_RATIO:
        orientation = "front"
    elif ratio <= SIDE_RATIO:
        orientation = "side"
    else:
        orientation = "unknown"

    return {
        "pose_confidence": pose_confidence,
        "coverage": coverage,
        "upright": upright,
        "frontal": frontal,
        "orientation": orientation,
    }


def _verdict(m: dict, sharpness: float) -> tuple[bool, str]:
    if m["pose_confidence"] < MIN_POSE_CONFIDENCE:
        return False, f"low pose confidence ({m['pose_confidence']:.2f})"
    if m["coverage"] < MIN_COVERAGE:
        return False, "body not fully in frame (head-to-feet)"
    if m["upright"] < MIN_UPRIGHT:
        return False, "child not standing upright"
    if sharpness < MIN_SHARPNESS:
        return False, f"image too blurry (sharpness {sharpness:.0f})"
    return True, ""


def score_photo(image_bgr: np.ndarray, landmarker) -> PhotoScore:
    """Run pose detection on one photo and score it."""
    import mediapipe as mp

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    sharp = sharpness_of(image_bgr)

    if not result.pose_landmarks:
        return PhotoScore(0, 0, 0, 0, sharp, "unknown", False, "no pose detected")

    m = landmark_metrics(result.pose_landmarks[0])
    if m is None:
        return PhotoScore(0, 0, 0, 0, sharp, "unknown", False,
                          "required landmarks not visible")

    usable, reason = _verdict(m, sharp)
    return PhotoScore(
        pose_confidence=m["pose_confidence"], coverage=m["coverage"],
        upright=m["upright"], frontal=m["frontal"], sharpness=sharp,
        orientation=m["orientation"], usable=usable, reason=reason,
    )


def build_landmarker() -> object:
    """Real MediaPipe landmarker (IMAGE mode). Not used in unit tests.

    Return type is `object` (not the real MediaPipe type) so this
    signature doesn't force importing mediapipe at module import time.
    """
    import mediapipe as mp

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)
