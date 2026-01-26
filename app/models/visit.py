"""Visit model representing a single assessment visit."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.models.database import Base


class Visit(Base):
    __tablename__ = "visits"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("children.id"), nullable=False)
    visit_date = Column(DateTime, default=datetime.utcnow)
    age_months = Column(Float, nullable=False)
    image_path = Column(String(500), nullable=True)
    notes = Column(Text, nullable=True)

    child = relationship("Child", back_populates="visits")
    measurement = relationship(
        "MeasurementResult",
        back_populates="visit",
        uselist=False,
        cascade="all, delete-orphan",
    )
