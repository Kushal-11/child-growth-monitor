import json
from pathlib import Path

import numpy as np
import pytest

from ml.evaluate_tflite import ASSET_DIR, _run_model, verify_shipped_assets

CASES = json.loads(
    (
        Path(__file__).resolve().parents[1] / "shared" / "ml_parity_cases.json"
    ).read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_python_feature_expansion_matches_cross_language_fixture(case):
    feature = case["features"]
    chest = feature.get("chest_depth_cm", feature["shoulder_width_cm"] * 0.45)
    abdomen = feature.get("abd_depth_cm", feature["hip_width_cm"] * 0.50)
    actual = [
        feature["age_months"],
        feature["sex_binary"],
        feature["height_cm"],
        feature["shoulder_width_cm"],
        feature["hip_width_cm"],
        feature["torso_length_cm"],
        feature["upper_arm_length_cm"],
        feature["shoulder_height_ratio"],
        feature["hip_height_ratio"],
        feature["body_build_score"],
        chest,
        abdomen,
        feature.get("chest_depth_ratio", chest / feature["height_cm"]),
        feature.get("abd_depth_ratio", abdomen / feature["height_cm"]),
    ]
    assert actual == pytest.approx(case["expected_raw_features"], rel=1e-7, abs=1e-7)


def test_fixture_matches_exact_shipped_tflite_outputs():
    verify_shipped_assets()
    scaler = json.loads((ASSET_DIR / "feature_scaler.json").read_text())
    raw = np.asarray([case["expected_raw_features"] for case in CASES], dtype="float32")
    scaled = (raw - np.asarray(scaler["mean"], dtype="float32")) / np.asarray(
        scaler["scale"], dtype="float32"
    )
    weights = _run_model(ASSET_DIR / "weight_estimator.tflite", scaled).reshape(-1)
    probabilities = _run_model(ASSET_DIR / "wasting_classifier.tflite", scaled)
    assert weights == pytest.approx(
        [case["expected_weight_kg"] for case in CASES], rel=1e-6, abs=1e-6
    )
    np.testing.assert_allclose(
        probabilities,
        np.asarray(
            [case["expected_class_probabilities"] for case in CASES],
            dtype="float32",
        ),
        rtol=1e-6,
        atol=1e-6,
    )
