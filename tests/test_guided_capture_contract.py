"""Cross-language guided-capture contract tests."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.guided_capture import (
    CameraResultSubmission,
    CaptureAssetSubmission,
    MeasuredDetailsSubmission,
    VisitTransitionRequest,
)
from app.services.guided_capture_contract import (
    ALLOWED_CAPTURE_TRANSITIONS,
    REQUIRED_CAPTURE_ROLES,
    CaptureAssetRole,
    CaptureState,
    MeasurementMode,
    OedemaStatus,
    can_transition_capture_state,
    require_capture_transition,
)


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "contracts"
    / "guided_capture_v1.json"
)


@pytest.fixture(scope="module")
def fixture():
    return json.loads(CONTRACT_PATH.read_text())


def test_python_wire_values_match_language_neutral_fixture(fixture):
    assert [value.value for value in CaptureState] == fixture["visit_states"]
    assert [value.value for value in CaptureAssetRole] == fixture["asset_roles"]
    assert [value.value for value in MeasurementMode] == fixture["measurement_modes"]
    assert [value.value for value in OedemaStatus] == fixture["oedema_values"]
    assert [value.value for value in REQUIRED_CAPTURE_ROLES] == fixture[
        "required_asset_roles"
    ]
    assert {
        state.value: [target.value for target in targets]
        for state, targets in ALLOWED_CAPTURE_TRANSITIONS.items()
    } == fixture["allowed_transitions"]


@pytest.mark.parametrize(
    ("model", "field", "valid_value"),
    [
        (VisitTransitionRequest, "capture_state", "processing"),
        (CaptureAssetSubmission, "role", "front"),
        (MeasuredDetailsSubmission, "measurement_mode", "standing_height"),
        (MeasuredDetailsSubmission, "oedema", "not_checked"),
    ],
)
def test_unknown_canonical_values_fail_closed(model, field, valid_value):
    common = {
        VisitTransitionRequest: {"capture_state": valid_value},
        CaptureAssetSubmission: {
            "asset_uuid": uuid4(),
            "visit_uuid": uuid4(),
            "role": valid_value,
            "captured_at": datetime.now(timezone.utc),
        },
        MeasuredDetailsSubmission: {
            "measurement_mode": "standing_height",
            "oedema": "not_checked",
            "measured_at": datetime.now(timezone.utc),
        },
    }[model]
    common[field] = "unknown"

    with pytest.raises(ValidationError):
        model.model_validate(common)


def _camera_payload():
    return {
        "result_uuid": uuid4(),
        "visit_uuid": uuid4(),
        "version": 1,
        "estimated_height_cm": 88.2,
        "estimated_weight_kg": 12.1,
        "estimated_haz": -1.2,
        "estimated_whz": -0.8,
        "height_source": "who_statistical",
        "weight_source": "ml_estimated",
        "component_probabilities": {"SAM": 0.1, "MAM": 0.2, "Normal": 0.7},
        "body_proportion_features": {"shoulder_hip_ratio": 0.82},
        "capture_quality_summary": {"overall": 0.91},
        "method": "camera_screening_v1",
        "model_version": "test-model",
        "manifest_checksum": "a" * 64,
        "training_data_label": "research_only",
        "non_clinical": True,
        "created_at": datetime.now(timezone.utc),
    }


def test_camera_submission_requires_non_clinical_true_and_snake_case_json():
    payload = _camera_payload()
    result = CameraResultSubmission.model_validate(payload)
    serialized = result.model_dump(mode="json")

    assert serialized["non_clinical"] is True
    assert serialized["estimated_height_cm"] == 88.2
    assert "estimatedHeightCm" not in serialized

    payload["non_clinical"] = False
    with pytest.raises(ValidationError):
        CameraResultSubmission.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("estimated_height_cm", math.nan),
        ("estimated_weight_kg", math.inf),
        ("estimated_haz", -math.inf),
        ("estimated_whz", math.nan),
    ],
)
def test_camera_submission_rejects_non_finite_measurements(field, value):
    payload = _camera_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        CameraResultSubmission.model_validate(payload)


def test_camera_submission_rejects_non_finite_or_out_of_range_probabilities():
    for invalid in (math.nan, math.inf, -0.1, 1.1):
        payload = _camera_payload()
        payload["component_probabilities"] = {"SAM": invalid}
        with pytest.raises(ValidationError):
            CameraResultSubmission.model_validate(payload)


def test_only_documented_capture_state_transitions_are_allowed(fixture):
    for current in CaptureState:
        expected = set(fixture["allowed_transitions"][current.value])
        actual = {
            candidate.value
            for candidate in CaptureState
            if can_transition_capture_state(current, candidate)
        }
        assert actual == expected

    require_capture_transition(CaptureState.PROCESSING, CaptureState.ESTIMATED_REPORT)
    with pytest.raises(ValueError, match="Invalid capture-state transition"):
        require_capture_transition(
            CaptureState.MEASURED_REPORT,
            CaptureState.ESTIMATED_REPORT,
        )
