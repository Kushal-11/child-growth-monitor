import json
from pathlib import Path

import numpy as np

from ml.inference import WastingFeatures


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wasting_features_parity.json"


def test_python_feature_vector_matches_shared_mobile_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    values = fixture["input"]
    features = WastingFeatures(
        age_months=values["age_months"],
        sex_binary=values["sex_binary"],
        height_cm=values["height_cm"],
        shoulder_width_cm=values["shoulder_width_cm"],
        hip_width_cm=values["hip_width_cm"],
        torso_length_cm=values["torso_length_cm"],
        upper_arm_length_cm=values["upper_arm_length_cm"],
        shoulder_height_ratio=values["shoulder_height_ratio"],
        hip_height_ratio=values["hip_height_ratio"],
        body_build_score=values["body_build_score"],
        chest_depth_cm=values["chest_depth_cm"],
        abd_depth_cm=values["abd_depth_cm"],
    )

    np.testing.assert_allclose(
        features.to_array(),
        np.asarray(fixture["expected_vector"], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
