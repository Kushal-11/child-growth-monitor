"""Visit model representing a single assessment visit."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import relationship

from app.models.database import Base


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

    __table_args__ = (
        Index(
            "ix_visits_local_uuid",
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
