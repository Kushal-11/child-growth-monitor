"""Convert sklearn StandardScaler pickle into JSON for the Flutter app.

Reads:  data/models/feature_scaler.pkl
Writes: flutter_app/assets/models/feature_scaler.json

Run from repo root:
    PYTHONPATH=. .venv/bin/python flutter_app/scripts/export_scaler.py

NOTE: pickle.load is used to read a file produced by this repo's own
ml/train.py. Do not run this script against pickle files from
untrusted sources.
"""
import json
import pickle  # nosec B403 - reading repo-local training artifact only
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKL = REPO_ROOT / "data" / "models" / "feature_scaler.pkl"
OUT = REPO_ROOT / "flutter_app" / "assets" / "models" / "feature_scaler.json"

FEATURE_NAMES = [
    "age_months",
    "sex_binary",
    "height_cm",
    "shoulder_width_cm",
    "hip_width_cm",
    "torso_length_cm",
    "upper_arm_length_cm",
    "shoulder_height_ratio",
    "hip_height_ratio",
    "body_build_score",
    "chest_depth_cm",
    "abd_depth_cm",
    "chest_depth_ratio",
    "abd_depth_ratio",
]


def main() -> None:
    with PKL.open("rb") as fh:
        scaler = pickle.load(fh)  # nosec B301 - trusted local artifact
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    if len(mean) != 14 or len(scale) != 14:
        raise SystemExit(
            f"Expected 14 features in scaler, got mean={len(mean)} scale={len(scale)}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(
            {"mean": mean, "scale": scale, "feature_names": FEATURE_NAMES},
            fh,
            indent=2,
        )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
