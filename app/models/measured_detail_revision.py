"""Immutable audit revision for a visit's measured details."""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.models.database import Base


class MeasuredDetailRevision(Base):
    __tablename__ = "measured_detail_revisions"

    id = Column(Integer, primary_key=True, index=True)
    revision_uuid = Column(String(36), nullable=False, unique=True)
    visit_id = Column(
        Integer,
        ForeignKey("visits.id", ondelete="CASCADE"),
        nullable=False,
    )
    revision_number = Column(Integer, nullable=False)
    before_json = Column(JSON, nullable=False)
    after_json = Column(JSON, nullable=False)
    editor_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reason = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "visit_id",
            "revision_number",
            name="uq_measured_revisions_visit_revision",
        ),
        Index(
            "ix_measured_revisions_visit_revision",
            "visit_id",
            "revision_number",
        ),
    )

    visit = relationship("Visit", back_populates="measured_revisions")
