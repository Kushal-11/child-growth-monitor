"""
Extract the best standing-upright frame from a child walking video.

Scores each sampled frame on four criteria:
  1. Pose confidence   — average landmark visibility for key joints
  2. Body coverage     — fraction of frame height used by head-to-heel span
  3. Upright score     — landmarks in correct top-to-bottom order
  4. Frontal score     — shoulders level (not turned sideways)
  5. Sharpness         — Laplacian variance (motion blur rejection)

Usage:
    python scripts/extract_best_frame.py VIDEO_FILE [OUTPUT_FILE] [--every N]

Examples:
    python scripts/extract_best_frame.py footage/child1.mp4
    python scripts/extract_best_frame.py footage/child1.mp4 best_frames/child1.jpg
    python scripts/extract_best_frame.py footage/child1.mp4 --every 5   # sample every 5th frame

    # Batch-process a whole folder:
    for f in footage/*.mp4; do
        python scripts/extract_best_frame.py "$f" "best_frames/$(basename ${f%.mp4}).jpg"
    done
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Key landmark indices used for scoring
_KP = {
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

# Landmarks that must be visible for a frame to be considered at all
_REQUIRED = ["nose", "left_shoulder", "right_shoulder",
             "left_hip", "right_hip", "left_ankle", "right_ankle"]

POSE_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "pose_landmarker_heavy.task"


def _build_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.3,   # lower threshold to catch partial poses
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)


def _score_frame(frame_rgb: np.ndarray, landmarker) -> tuple[float, list]:
    """
    Run pose detection on one frame and return (score, landmarks).
    Returns (0.0, []) if no pose detected or required landmarks are missing.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return 0.0, []

    lms = result.pose_landmarks[0]  # first (only) person

    def vis(name):
        return lms[_KP[name]].visibility or 0.0

    def y(name):
        return lms[_KP[name]].y  # normalised 0-1, top=0

    def x(name):
        return lms[_KP[name]].x

    # 1. Check required landmarks are visible enough
    for req in _REQUIRED:
        if vis(req) < 0.4:
            return 0.0, lms

    # 2. Pose confidence: mean visibility of key joints
    key_joints = ["nose", "left_shoulder", "right_shoulder",
                  "left_hip", "right_hip", "left_knee", "right_knee",
                  "left_ankle", "right_ankle"]
    pose_confidence = float(np.mean([vis(j) for j in key_joints]))

    # 3. Body coverage: head-to-heel span as fraction of frame height
    nose_y    = y("nose")
    heel_y    = max(y("left_heel"), y("right_heel"),
                    y("left_ankle"), y("right_ankle"))
    coverage  = max(0.0, heel_y - nose_y)        # 0–1; want ~0.7–0.95
    coverage_score = min(coverage / 0.80, 1.0)   # normalise: 0.80 coverage = 1.0

    # 4. Upright score: key joints in correct vertical order
    order_pairs = [
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ]
    order_ok = sum(1 for a, b in order_pairs if y(a) < y(b))
    upright_score = order_ok / len(order_pairs)

    # 5. Frontal score: shoulders at similar y (not rotated)
    shoulder_y_diff = abs(y("left_shoulder") - y("right_shoulder"))
    shoulder_width  = abs(x("left_shoulder") - x("right_shoulder")) + 1e-6
    frontal_score   = max(0.0, 1.0 - (shoulder_y_diff / shoulder_width) * 3.0)

    combined = (
        pose_confidence ** 1.5    # weight confidence highest
        * coverage_score
        * upright_score
        * frontal_score
    )
    return float(combined), lms


def _sharpness(frame_bgr: np.ndarray) -> float:
    """Laplacian variance as a motion-blur / focus measure (higher = sharper)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_best_frame(
    video_path: str | Path,
    output_path: str | Path | None = None,
    every_n: int = 3,
    verbose: bool = True,
) -> Path:
    """
    Extract the best upright frame from a video.

    Args:
        video_path:  Path to the input video file.
        output_path: Where to save the extracted JPEG.
                     Defaults to <video_stem>_best.jpg next to the video.
        every_n:     Sample every Nth frame (default 3).
        verbose:     Print progress.

    Returns:
        Path to the saved JPEG.
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_name(video_path.stem + "_best.jpg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    if verbose:
        print(f"Video: {video_path.name}  |  {total_frames} frames @ {fps:.1f} fps")

    landmarker = _build_landmarker()

    best_score  = -1.0
    best_frame  = None
    best_sharp  = 0.0
    frame_idx   = 0
    scored      = 0
    candidate_pool: list[tuple[float, float, np.ndarray]] = []  # (pose_score, sharpness, frame)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_score, _ = _score_frame(frame_rgb, landmarker)
            if pose_score > 0:
                sharp = _sharpness(frame_bgr)
                candidate_pool.append((pose_score, sharp, frame_bgr.copy()))
                scored += 1

        frame_idx += 1

    cap.release()

    if not candidate_pool:
        raise RuntimeError(
            "No valid upright pose found in the video. "
            "Ensure the full body is visible from head to feet in at least a few frames."
        )

    # Normalise sharpness across candidates and combine with pose score
    max_sharp = max(s for _, s, _ in candidate_pool) or 1.0
    best_combined = -1.0
    for pose_score, sharp, frame in candidate_pool:
        sharp_norm = sharp / max_sharp
        combined = pose_score * (0.7 + 0.3 * sharp_norm)  # 70% pose, 30% sharpness
        if combined > best_combined:
            best_combined = combined
            best_frame = frame
            best_score = pose_score
            best_sharp = sharp

    cv2.imwrite(str(output_path), best_frame)

    if verbose:
        print(f"Sampled {frame_idx} frames, scored {scored} with visible pose.")
        print(f"Best frame  |  pose_score={best_score:.3f}  sharpness={best_sharp:.0f}")
        print(f"Saved → {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract the best upright frame from a child walking video."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "output", nargs="?", default=None,
        help="Output JPEG path (default: <video_stem>_best.jpg next to video)"
    )
    parser.add_argument(
        "--every", type=int, default=3, metavar="N",
        help="Sample every Nth frame (default: 3)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    try:
        out = extract_best_frame(
            args.video,
            output_path=args.output,
            every_n=args.every,
            verbose=not args.quiet,
        )
        print(f"\nDone: {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
