"""Immutable, explicitly non-clinical camera inference snapshot."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import relationship

from app.models.database import Base


class CameraResult(Base):
    __tablename__ = "camera_results"

    id = Column(Integer, primary_key=True, index=True)
    result_uuid = Column(String(36), nullable=False, unique=True)
    visit_id = Column(
        Integer,
        ForeignKey("visits.id", ondelete="CASCADE"),
        nullable=False,
    )
    version = Column(Integer, nullable=False)
    supersedes_result_uuid = Column(String(36), nullable=True)

    estimated_height_cm = Column(Float, nullable=True)
    estimated_weight_kg = Column(Float, nullable=True)
    height_source = Column(String(100), nullable=True)
    weight_source = Column(String(100), nullable=True)
    estimated_haz = Column(Float, nullable=True)
    estimated_whz = Column(Float, nullable=True)
    estimated_stunting_status = Column(String(100), nullable=True)
    estimated_wasting_status = Column(String(100), nullable=True)
    experimental_overall_category = Column(String(100), nullable=True)
    component_probabilities_json = Column(JSON, nullable=True)
    body_proportion_features_json = Column(JSON, nullable=True)
    capture_quality_summary_json = Column(JSON, nullable=True)

    method = Column(String(50), nullable=False)
    model_version = Column(String(100), nullable=False)
    manifest_checksum = Column(String(64), nullable=False)
    training_data_label = Column(String(100), nullable=False)
    non_clinical = Column(
        Boolean,
        default=True,
        server_default=text("1"),
        nullable=False,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "non_clinical = 1",
            name="ck_camera_results_non_clinical",
        ),
        UniqueConstraint(
            "visit_id",
            "version",
            name="uq_camera_results_visit_version",
        ),
        Index("ix_camera_results_visit_version", "visit_id", "version"),
    )

    visit = relationship("Visit", back_populates="camera_results")
