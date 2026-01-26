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
        gt=0,
        le=50,
        description="Manually entered weight in kg. If omitted, weight is estimated from WHO median.",
    )
    height_cm: Optional[float] = Field(
        None,
        gt=0,
        le=200,
        description="Manually entered height in cm. Used as fallback when image-based estimation fails.",
    )
    guardian_name: Optional[str] = None
    location: Optional[str] = None


class MeasurementDetail(BaseModel):
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


class NutritionDetail(BaseModel):
    haz_zscore: Optional[float] = None
    whz_zscore: Optional[float] = None
    haz_status: Optional[str] = None
    whz_status: Optional[str] = None
    age_months: float


class AssessmentResponse(BaseModel):
    child_name: str
    sex: str
    age_months: float
    measurement: MeasurementDetail
    nutrition: NutritionDetail
    summary: str
