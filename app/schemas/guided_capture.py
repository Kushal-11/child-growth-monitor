"""Validated API contracts for guided capture and measured details."""

import math
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.guided_capture_contract import (
    CaptureAssetRole,
    CaptureState,
    MeasurementMode,
    OedemaStatus,
)


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VisitTransitionRequest(_StrictContract):
    capture_state: CaptureState


class CaptureAssetSubmission(_StrictContract):
    asset_uuid: UUID
    visit_uuid: UUID
    role: CaptureAssetRole
    local_path: Optional[str] = None
    server_object_id: Optional[str] = None
    captured_at: datetime
    selected_rank: Optional[int] = Field(default=None, ge=0)
    pose_score: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    coverage_score: Optional[float] = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    orientation_score: Optional[float] = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    sharpness_score: Optional[float] = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    lighting_score: Optional[float] = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    overall_score: Optional[float] = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    quality_verdict: Optional[str] = Field(default=None, max_length=64)
    rejection_reason: Optional[str] = Field(default=None, max_length=500)
    image_width: Optional[int] = Field(default=None, gt=0)
    image_height: Optional[int] = Field(default=None, gt=0)
    display_orientation: Optional[int] = None
    device_camera_metadata: dict[str, Any] = Field(default_factory=dict)


class CameraResultSubmission(_StrictContract):
    result_uuid: UUID
    visit_uuid: UUID
    version: int = Field(ge=1)
    supersedes_result_uuid: Optional[UUID] = None
    estimated_height_cm: Optional[float] = Field(
        default=None, ge=35, le=145, allow_inf_nan=False
    )
    estimated_weight_kg: Optional[float] = Field(
        default=None, ge=0.5, le=80, allow_inf_nan=False
    )
    estimated_muac_cm: Optional[float] = Field(
        default=None, ge=7, le=24, allow_inf_nan=False
    )
    height_range_lower_cm: Optional[float] = Field(
        default=None, ge=20, le=160, allow_inf_nan=False
    )
    height_range_upper_cm: Optional[float] = Field(
        default=None, ge=20, le=160, allow_inf_nan=False
    )
    weight_range_lower_kg: Optional[float] = Field(
        default=None, ge=0.1, le=100, allow_inf_nan=False
    )
    weight_range_upper_kg: Optional[float] = Field(
        default=None, ge=0.1, le=100, allow_inf_nan=False
    )
    muac_range_lower_cm: Optional[float] = Field(
        default=None, ge=5, le=30, allow_inf_nan=False
    )
    muac_range_upper_cm: Optional[float] = Field(
        default=None, ge=5, le=30, allow_inf_nan=False
    )
    estimated_haz: Optional[float] = Field(default=None, allow_inf_nan=False)
    estimated_whz: Optional[float] = Field(default=None, allow_inf_nan=False)
    estimated_stunting_status: Optional[str] = Field(default=None, max_length=100)
    estimated_wasting_status: Optional[str] = Field(default=None, max_length=100)
    experimental_overall_category: Optional[str] = Field(
        default=None, max_length=100
    )
    height_source: Optional[str] = Field(default=None, max_length=100)
    weight_source: Optional[str] = Field(default=None, max_length=100)
    muac_source: Optional[str] = Field(default=None, max_length=100)
    component_probabilities: dict[str, float] = Field(default_factory=dict)
    body_proportion_features: dict[str, Any] = Field(default_factory=dict)
    capture_quality_summary: dict[str, Any] = Field(default_factory=dict)
    method: Literal[
        "camera_screening_v1",
        "camera_screening_contactless_v2",
    ] = "camera_screening_v1"
    model_version: str = Field(min_length=1, max_length=100)
    manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_data_label: str = Field(min_length=1, max_length=100)
    non_clinical: Literal[True] = True
    created_at: datetime

    @field_validator("component_probabilities")
    @classmethod
    def validate_component_probabilities(
        cls,
        probabilities: dict[str, float],
    ) -> dict[str, float]:
        for name, value in probabilities.items():
            if not name or isinstance(value, bool):
                raise ValueError("probability names and values must be valid")
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or not 0 <= numeric_value <= 1:
                raise ValueError(
                    "component probabilities must be finite values from 0 to 1"
                )
        return probabilities

    @model_validator(mode="after")
    def validate_estimate_ranges(self):
        for name, estimate, lower, upper in (
            (
                "height_range_cm",
                self.estimated_height_cm,
                self.height_range_lower_cm,
                self.height_range_upper_cm,
            ),
            (
                "weight_range_kg",
                self.estimated_weight_kg,
                self.weight_range_lower_kg,
                self.weight_range_upper_kg,
            ),
            (
                "muac_range_cm",
                self.estimated_muac_cm,
                self.muac_range_lower_cm,
                self.muac_range_upper_cm,
            ),
        ):
            if lower is None and upper is None:
                continue
            if (
                estimate is None
                or lower is None
                or upper is None
                or not lower <= estimate <= upper
            ):
                raise ValueError(f"{name} must contain its estimate")
        return self


class MeasuredDetailsSubmission(_StrictContract):
    measurement_mode: MeasurementMode
    oedema: OedemaStatus
    height_cm: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    weight_kg: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    muac_cm: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    measured_at: datetime
    notes: Optional[str] = Field(default=None, max_length=2000)
    reason: Optional[str] = Field(default=None, max_length=500)
