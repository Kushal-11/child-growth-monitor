"""MeasurementResult model storing assessment outputs."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
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

    # Calibration info
    reference_object_detected = Column(String(10), default="false")
    scale_factor = Column(Float, nullable=True)  # cm per pixel

    # Z-scores
    haz_zscore = Column(Float, nullable=True)
    whz_zscore = Column(Float, nullable=True)

    # Classifications
    haz_status = Column(String(50), nullable=True)
    whz_status = Column(String(50), nullable=True)

    # Metadata
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    visit = relationship("Visit", back_populates="measurement")
