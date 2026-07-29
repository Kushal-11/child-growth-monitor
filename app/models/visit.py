"""Visit model representing a single assessment visit."""
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
    text,
)
from sqlalchemy.orm import relationship

from app.models.database import Base
from app.services.guided_capture_contract import CaptureState


class Visit(Base):
    __tablename__ = "visits"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("children.id"), nullable=False)
    visit_date = Column(DateTime, default=datetime.utcnow)
    age_months = Column(Float, nullable=False)
    image_path = Column(String(500), nullable=True)
    side_image_path = Column(String(500), nullable=True)
    back_image_path = Column(String(500), nullable=True)
    notes = Column(Text, nullable=True)
    local_uuid = Column(String(36), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    entry_method = Column(String(20), default="assessment", nullable=False)  # "assessment" | "manual"
    capture_state = Column(
        String(30),
        default=CaptureState.ESTIMATED_REPORT.value,
        server_default=CaptureState.ESTIMATED_REPORT.value,
        nullable=False,
    )
    capture_started_at = Column(DateTime, nullable=True)
    capture_completed_at = Column(DateTime, nullable=True)
    device_metadata_json = Column(JSON, nullable=True)
    consent_version = Column(String(50), nullable=True)
    consent_timestamp = Column(DateTime, nullable=True)
    consent_operator_identifier = Column(String(100), nullable=True)
    media_deleted_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index(
            "ix_visits_local_uuid",
            "local_uuid",
            unique=True,
            sqlite_where=text("local_uuid IS NOT NULL"),
        ),
        Index(
            "ix_visits_owner_local_uuid",
            "user_id",
            "local_uuid",
            unique=True,
            sqlite_where=text("local_uuid IS NOT NULL"),
        ),
    )

    child = relationship("Child", back_populates="visits")
    measurement = relationship(
        "MeasurementResult",
        back_populates="visit",
        uselist=False,
        cascade="all, delete-orphan",
    )
    capture_assets = relationship(
        "CaptureAsset",
        back_populates="visit",
        cascade="all, delete-orphan",
        order_by="CaptureAsset.captured_at",
    )
    camera_results = relationship(
        "CameraResult",
        back_populates="visit",
        cascade="all, delete-orphan",
        order_by="CameraResult.version",
    )
    measured_revisions = relationship(
        "MeasuredDetailRevision",
        back_populates="visit",
        cascade="all, delete-orphan",
        order_by="MeasuredDetailRevision.revision_number",
    )
