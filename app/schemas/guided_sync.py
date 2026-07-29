"""Strict JSON contracts and per-entity acknowledgements for guided sync."""

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.guided_capture import CameraResultSubmission
from app.services.guided_capture_contract import (
    CaptureAssetRole,
    CaptureState,
    MeasurementMode,
    OedemaStatus,
)


class _StrictSyncModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuidedVisitSyncRequest(_StrictSyncModel):
    local_uuid: UUID
    child_id: int = Field(gt=0)
    visit_date: datetime
    age_months: float = Field(ge=0, lt=60, allow_inf_nan=False)
    capture_state: CaptureState
    device_metadata: dict[str, Any] = Field(default_factory=dict)
    consent_version: str = Field(min_length=1, max_length=50)
    consent_timestamp: datetime
    consent_operator_identifier: str = Field(min_length=1, max_length=100)
    capture_started_at: Optional[datetime] = None
    capture_completed_at: Optional[datetime] = None


class GuidedAssetQuality(_StrictSyncModel):
    pose: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    coverage: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    orientation: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    sharpness: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    lighting: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    overall: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    threshold_version: Optional[str] = Field(default=None, max_length=100)


class GuidedAssetSyncRequest(_StrictSyncModel):
    asset_uuid: UUID
    visit_uuid: UUID
    role: CaptureAssetRole
    captured_at: datetime
    selected_rank: Optional[int] = Field(default=None, ge=0)
    quality: GuidedAssetQuality
    image_width: Optional[int] = Field(default=None, gt=0)
    image_height: Optional[int] = Field(default=None, gt=0)
    exif_orientation: Optional[int] = None
    display_orientation: Optional[int] = None
    device_camera_metadata: dict[str, Any] = Field(default_factory=dict)
    content_type: Literal["image/jpeg", "image/png", "image/webp"]
    content_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_base64: str = Field(min_length=1)


class GuidedCameraResultSyncRequest(CameraResultSubmission):
    """Camera submission reused verbatim with the visit carried in JSON."""


class MeasuredSnapshotSync(_StrictSyncModel):
    height_cm: Optional[float] = Field(default=None, allow_inf_nan=False)
    weight_kg: Optional[float] = Field(default=None, allow_inf_nan=False)
    muac_cm: Optional[float] = Field(default=None, allow_inf_nan=False)
    measurement_mode: Optional[MeasurementMode] = None
    oedema: Optional[OedemaStatus] = None
    measured_at: Optional[datetime] = None
    notes: Optional[str] = Field(default=None, max_length=2000)
    haz_zscore: Optional[float] = Field(default=None, allow_inf_nan=False)
    whz_zscore: Optional[float] = Field(default=None, allow_inf_nan=False)
    haz_status: Optional[str] = None
    whz_status: Optional[str] = None
    who_acute_status: Optional[str] = None
    poshan_status: Optional[str] = None


class GuidedMeasuredRevisionSyncRequest(_StrictSyncModel):
    revision_uuid: UUID
    visit_uuid: UUID
    revision_number: int = Field(ge=1)
    before: MeasuredSnapshotSync
    after: MeasuredSnapshotSync
    editor_user_id: Optional[int] = None
    created_at: datetime
    reason: Optional[str] = Field(default=None, max_length=500)


class GuidedSyncAcknowledgement(BaseModel):
    entity_type: Literal[
        "visit",
        "capture_asset",
        "camera_result",
        "measured_revision",
        "media_deletion",
    ]
    entity_uuid: UUID
    status: Literal["accepted", "already_accepted"]
    server_id: Optional[int] = None
    server_object_id: Optional[str] = None
    checksum: Optional[str] = None
    acknowledged_at: datetime
