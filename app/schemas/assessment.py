"""Pydantic schemas for API request/response validation."""
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from app.services.age_service import AgeService
from config import WastingStatus


class AssessmentRequest(BaseModel):
    """Metadata submitted alongside the uploaded image."""

    child_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date
    sex: str = Field(..., pattern="^[MF]$")
    weight_kg: Optional[float] = Field(
        None,
        gt=0,
        le=50,
        description="Manually entered weight in kg. If omitted, weight is estimated from WHO median.",
    )
    height_cm: Optional[float] = Field(
        None,
        gt=0,
        le=200,
        description="Manually entered height in cm. Overrides an image-based estimate.",
    )
    guardian_name: Optional[str] = None
    location: Optional[str] = None

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, value: date, info: ValidationInfo) -> date:
        """Reject dates that cannot be assessed against the WHO tables."""
        context = info.context or {}
        age_service = context.get("age_service") or AgeService()
        age_service.validate_clinical_age(value, context.get("as_of"))
        return value


class MeasurementDetail(BaseModel):
    effective_height_cm: Optional[float] = None
    height_method: str = "unavailable"  # "manual" | "image_estimated" | "unavailable"
    predicted_height_cm: Optional[float] = None
    predicted_weight_kg: Optional[float] = None
    manual_height_cm: Optional[float] = None
    manual_weight_kg: Optional[float] = None
    reference_object_detected: bool = False
    scale_factor: Optional[float] = None
    confidence_score: Optional[float] = None
    annotated_image: Optional[str] = None  # filename of pose-annotated image
    estimation_method: str = "none"  # "who_statistical", "reference_object", "manual", "none"
    body_build: Optional[str] = None  # "slender", "average", "stocky", or None
    # Side-view depth measurements (None when no side photo provided)
    side_view_used: bool = False
    chest_depth_cm: Optional[float] = None  # AP chest diameter from side view
    abd_depth_cm: Optional[float] = None    # AP abdomen diameter from side view


class NutritionDetail(BaseModel):
    haz_zscore: Optional[float] = None
    whz_zscore: Optional[float] = None
    haz_status: Optional[str] = None
    whz_status: Optional[WastingStatus] = None
    age_months: float


class MLPrediction(BaseModel):
    """Output from the ML wasting detection models."""
    estimated_weight_kg: Optional[float] = None
    sam_probability: float = 0.0
    mam_probability: float = 0.0
    normal_probability: float = 0.0
    risk_probability: float = 0.0
    overweight_probability: float = 0.0
    wasting_status: Optional[str] = None
    wasting_method: str = "ml_classifier"


class MUACDetail(BaseModel):
    """MUAC measurement or estimate."""
    muac_cm: Optional[float] = None
    muac_status: Optional[WastingStatus] = None
    # "manual" | "landmark_estimated" | "estimated_from_whz"
    muac_method: str = "estimated_from_whz"
    age_in_range: bool = True  # False if age outside 6-59 months
    confidence: Optional[float] = Field(None, ge=0, le=1)
    uncertainty_lower_cm: Optional[float] = None
    uncertainty_upper_cm: Optional[float] = None
    model_version: Optional[str] = None
    calibration_version: Optional[str] = None
    is_direct_measurement: bool = False
    requires_confirmation: bool = False
    referral_guidance: Optional[str] = None


class CombinedNutritionDetail(BaseModel):
    """Final clinical wasting verdict after combining all applicable arms."""

    status: WastingStatus
    triggered_by: list[str]
    rationale: str
    method: str = "who_or_rule"
    confidence_score: Optional[float] = None


class AssessmentResponse(BaseModel):
    child_name: str
    sex: str
    age_months: float
    measurement: MeasurementDetail
    nutrition: NutritionDetail
    ml_prediction: Optional[MLPrediction] = None
    muac: Optional[MUACDetail] = None
    combined_nutrition: CombinedNutritionDetail
    summary: str
    warnings: List[str] = Field(default_factory=list)
