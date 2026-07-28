"""MeasurementResult model storing assessment outputs."""
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.models.database import Base


class MeasurementResult(Base):
    __tablename__ = "measurement_results"

    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("visits.id"), nullable=False, unique=True)

    # Measured/estimated values
    predicted_height_cm = Column(Float, nullable=True)
    predicted_weight_kg = Column(Float, nullable=True)  # estimated from WHO median
    manual_height_cm = Column(Float, nullable=True)  # if manually entered
    manual_weight_kg = Column(Float, nullable=True)  # if manually entered
    effective_height_cm = Column(Float, nullable=True)
    effective_weight_kg = Column(Float, nullable=True)
    height_method = Column(String(50), nullable=True)
    weight_method = Column(String(50), nullable=True)
    estimation_method = Column(String(50), nullable=True)

    # Derived anthropometry.  BMI status uses the same WHO weight-for-height
    # classification as the clinical wasting result; the protocol version
    # below makes that interpretation explicit for historical records.
    bmi = Column(Float, nullable=True)
    bmi_status = Column(String(50), nullable=True)

    # Calibration info
    reference_object_detected = Column(String(10), default="false")
    scale_factor = Column(Float, nullable=True)  # cm per pixel

    # Z-scores
    haz_zscore = Column(Float, nullable=True)
    whz_zscore = Column(Float, nullable=True)

    # Classifications
    haz_status = Column(String(50), nullable=True)
    whz_status = Column(String(50), nullable=True)
    bmi = Column(Float, nullable=True)
    bmi_status = Column(String(20), nullable=True)
    combined_status = Column(String(20), nullable=True)
    combined_triggered_by = Column(String(100), nullable=True)
    height_method = Column(String(50), nullable=True)
    weight_method = Column(String(50), nullable=True)

    # Metadata
    confidence_score = Column(Float, nullable=True)
    height_confidence = Column(Float, nullable=True)
    weight_confidence = Column(Float, nullable=True)
    classification_confidence = Column(Float, nullable=True)

    # Body build + side view (from on-device measurement)
    body_build = Column(String(50), nullable=True)  # slender / average / stocky
    side_view_used = Column(Boolean, default=False, nullable=True)
    chest_depth_cm = Column(Float, nullable=True)
    abd_depth_cm = Column(Float, nullable=True)

    # ML wasting classifier output (5-class softmax)
    ml_estimated_weight_kg = Column(Float, nullable=True)
    ml_wasting_status = Column(String(50), nullable=True)
    ml_wasting_method = Column(String(50), nullable=True)
    sam_probability = Column(Float, nullable=True)
    mam_probability = Column(Float, nullable=True)
    normal_probability = Column(Float, nullable=True)
    risk_probability = Column(Float, nullable=True)
    overweight_probability = Column(Float, nullable=True)

    # MUAC
    muac_cm = Column(Float, nullable=True)
    muac_status = Column(String(50), nullable=True)
    muac_method = Column(String(50), nullable=True)  # manual / estimated_from_whz
    muac_age_in_range = Column(Boolean, nullable=True)
    muac_confidence = Column(Float, nullable=True)
    muac_uncertainty_lower_cm = Column(Float, nullable=True)
    muac_uncertainty_upper_cm = Column(Float, nullable=True)
    muac_model_version = Column(String(100), nullable=True)
    muac_calibration_version = Column(String(100), nullable=True)
    muac_is_direct_measurement = Column(Boolean, nullable=True)
    muac_requires_confirmation = Column(Boolean, nullable=True)
    muac_referral_guidance = Column(Text, nullable=True)

    # Final combined clinical classification (MUAC + WHZ WHO OR-rule)
    combined_status = Column(String(30), nullable=True)
    combined_triggered_by = Column(String(100), nullable=True)  # JSON list
    combined_rationale = Column(String(255), nullable=True)
    combined_method = Column(String(50), nullable=True)
    combined_confidence_score = Column(Float, nullable=True)
    combined_protocol_version = Column(String(50), nullable=True)

    # Final provenance-gated Poshan Setu programme classification.
    poshan_status = Column(String(30), nullable=True)
    poshan_triggered_by = Column(String(100), nullable=True)  # JSON list
    classification_method = Column(String(50), nullable=True)
    classification_rationale = Column(Text, nullable=True)
    poshan_complete = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    visit = relationship("Visit", back_populates="measurement")
