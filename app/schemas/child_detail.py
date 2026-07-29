"""Typed child-detail and guided-visit timeline responses."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.services.guided_capture_contract import CaptureState


class ChildVisitMeasurementResponse(BaseModel):
    predicted_height_cm: Optional[float] = None
    predicted_weight_kg: Optional[float] = None
    manual_height_cm: Optional[float] = None
    manual_weight_kg: Optional[float] = None
    reference_object_detected: bool = False
    scale_factor: Optional[float] = None
    haz_zscore: Optional[float] = None
    whz_zscore: Optional[float] = None
    haz_status: Optional[str] = None
    whz_status: Optional[str] = None
    confidence_score: Optional[float] = None
    effective_height_cm: Optional[float] = None
    effective_weight_kg: Optional[float] = None
    height_method: Optional[str] = None
    weight_method: Optional[str] = None
    estimation_method: Optional[str] = None
    bmi: Optional[float] = None
    bmi_status: Optional[str] = None
    height_confidence: Optional[float] = None
    weight_confidence: Optional[float] = None
    classification_confidence: Optional[float] = None
    body_build: Optional[str] = None
    side_view_used: Optional[bool] = None
    chest_depth_cm: Optional[float] = None
    abd_depth_cm: Optional[float] = None
    ml_estimated_weight_kg: Optional[float] = None
    ml_wasting_status: Optional[str] = None
    ml_wasting_method: Optional[str] = None
    sam_probability: Optional[float] = None
    mam_probability: Optional[float] = None
    normal_probability: Optional[float] = None
    risk_probability: Optional[float] = None
    overweight_probability: Optional[float] = None
    muac_cm: Optional[float] = None
    muac_status: Optional[str] = None
    muac_method: Optional[str] = None
    muac_age_in_range: Optional[bool] = None
    muac_confidence: Optional[float] = None
    muac_uncertainty_lower_cm: Optional[float] = None
    muac_uncertainty_upper_cm: Optional[float] = None
    muac_model_version: Optional[str] = None
    muac_calibration_version: Optional[str] = None
    muac_is_direct_measurement: Optional[bool] = None
    muac_requires_confirmation: Optional[bool] = None
    muac_referral_guidance: Optional[str] = None
    combined_status: Optional[str] = None
    combined_triggered_by: list[str] = Field(default_factory=list)
    combined_rationale: Optional[str] = None
    combined_method: Optional[str] = None
    combined_confidence_score: Optional[float] = None
    combined_protocol_version: Optional[str] = None
    poshan_status: Optional[str] = None
    poshan_triggered_by: list[str] = Field(default_factory=list)
    classification_method: Optional[str] = None
    classification_rationale: Optional[str] = None
    poshan_complete: Optional[bool] = None


class CameraResultSummaryResponse(BaseModel):
    result_uuid: str
    version: int
    estimated_height_cm: Optional[float] = None
    estimated_weight_kg: Optional[float] = None
    estimated_stunting_status: Optional[str] = None
    estimated_wasting_status: Optional[str] = None
    experimental_overall_category: Optional[str] = None
    method: str
    model_version: str
    non_clinical: Literal[True] = True


AssetAcknowledgementState = Literal["missing", "pending", "acknowledged"]


class RequiredAssetAcknowledgementResponse(BaseModel):
    front: AssetAcknowledgementState
    side: AssetAcknowledgementState


class ChildVisitResponse(BaseModel):
    visit_id: int
    local_uuid: Optional[str] = None
    visit_date: Optional[datetime] = None
    age_months: Optional[float] = None
    entry_method: Optional[str] = None
    capture_state: Optional[CaptureState] = None
    camera_result_summary: Optional[CameraResultSummaryResponse] = None
    has_measured_report: bool = False
    required_asset_acknowledgement: RequiredAssetAcknowledgementResponse
    required_assets_acknowledged: bool = False
    media_deleted_at: Optional[datetime] = None
    measurement: Optional[ChildVisitMeasurementResponse] = None


class ChildDetailResponse(BaseModel):
    id: int
    name: str
    date_of_birth: str
    sex: str
    guardian_name: Optional[str] = None
    location: Optional[str] = None
    visits: list[ChildVisitResponse] = Field(default_factory=list)
