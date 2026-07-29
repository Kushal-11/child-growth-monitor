"""Retained still image and its guided-capture provenance."""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.models.database import Base


class CaptureAsset(Base):
    __tablename__ = "capture_assets"

    id = Column(Integer, primary_key=True, index=True)
    asset_uuid = Column(String(36), nullable=False, unique=True)
    visit_id = Column(
        Integer,
        ForeignKey("visits.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String(20), nullable=False)
    local_path = Column(String(500), nullable=True)
    server_object_id = Column(String(200), nullable=True)
    captured_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    selected_rank = Column(Integer, nullable=True)

    pose_score = Column(Float, nullable=True)
    coverage_score = Column(Float, nullable=True)
    orientation_score = Column(Float, nullable=True)
    sharpness_score = Column(Float, nullable=True)
    lighting_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    quality_verdict = Column(String(64), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    quality_threshold_version = Column(String(100), nullable=True)

    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    exif_orientation = Column(Integer, nullable=True)
    display_orientation = Column(Integer, nullable=True)
    device_camera_metadata_json = Column(JSON, nullable=True)

    sync_state = Column(String(30), default="pending", nullable=False)
    server_acknowledged_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_capture_assets_visit_role", "visit_id", "role"),
    )

    visit = relationship("Visit", back_populates="capture_assets")
