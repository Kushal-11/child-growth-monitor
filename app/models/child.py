"""Child model representing a registered child."""
from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Integer, String
from sqlalchemy.orm import relationship

from app.models.database import Base


class Child(Base):
    __tablename__ = "children"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    sex = Column(String(1), nullable=False)  # 'M' or 'F'
    guardian_name = Column(String(100), nullable=True)
    location = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    visits = relationship("Visit", back_populates="child", cascade="all, delete-orphan")
