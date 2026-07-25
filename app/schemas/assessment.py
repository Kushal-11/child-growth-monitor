"""Pydantic schemas for API request/response validation."""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class AssessmentRequest(BaseModel):
    """Metadata submitted alongside the uploaded image."""

    child_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date
    sex: str = Field(..., pattern="^[MF]$")
    weight_kg: Optional[float] = Field(
        None,
        ge=0.5,
        le=40,
        description=(
            "Manually entered weight in kg. Statistical/ML estimates, when "
            "available, remain non-authoritative screening references."
        ),
    )
    height_cm: Optional[float] = Field(
        None,
        ge=30,
        le=130,
        description="Manually entered height in cm. Takes precedence over image estimates.",
    )
    muac_cm: Optional[float] = Field(
        None,
        ge=5,
        le=25,
        description="Manual/tape MUAC in cm.",
    )
    assessment_date: Optional[date] = Field(
        None,
        description="Date measurements were collected; defaults to today.",
    )
    guardian_name: Optional[str] = None
    location: Optional[str] = None


class MeasurementDetail(BaseModel):
    predicted_height_cm: Optional[float] = None
    predicted_weight_kg: Optional[float] = None
    manual_height_cm: Optional[float] = None
    manual_weight_kg: Optional[float] = None
    effective_height_cm: Optional[float] = None
    effective_weight_kg: Optional[float] = None
    height_source: str = "unavailable"
    weight_source: str = "unavailable"
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
    whz_status: Optional[str] = None
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
    model_version: Optional[str] = None
    training_data: Optional[str] = None
    non_clinical: bool = True


class MUACDetail(BaseModel):
    """MUAC measurement or estimate."""
    muac_cm: Optional[float] = None
    muac_status: str = "Indeterminate"
    muac_method: str = "unavailable"
    age_in_range: bool = True  # False if age outside 6-59 months


class PoshanDetail(BaseModel):
    """Canonical Poshan Setu v1 classification and calculation provenance."""

    bmi: Optional[float] = None
    bmi_status: str
    muac_status: str
    final_status: str
    triggered_by: list[str] = Field(default_factory=list)
    classification_method: str = "poshan_setu_v1"
    rationale: str
    complete: bool


class AssessmentResponse(BaseModel):
    child_name: str
    sex: str
    age_months: float
    measurement: MeasurementDetail
    nutrition: NutritionDetail
    ml_prediction: Optional[MLPrediction] = None
    muac: Optional[MUACDetail] = None
    poshan: PoshanDetail
    summary: str
