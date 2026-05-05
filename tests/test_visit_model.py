"""Visit model schema tests — local_uuid column + partial unique index."""
import pytest
from sqlalchemy.exc import IntegrityError

from app.models.database import Base, SessionLocal, engine, init_db
from app.models.visit import Visit


def setup_module(_module):
    # Import all models so FK-referenced tables are registered in metadata.
    from app.models.child import Child  # noqa: F401
    from app.models.measurement import MeasurementResult  # noqa: F401
    Base.metadata.drop_all(bind=engine)
    init_db()


def test_visit_has_local_uuid_column():
    columns = {c.name for c in Visit.__table__.columns}
    assert "local_uuid" in columns


def test_local_uuid_column_is_nullable():
    col = Visit.__table__.columns["local_uuid"]
    assert col.nullable is True


def test_partial_unique_index_exists():
    indexes = {idx.name: idx for idx in Visit.__table__.indexes}
    assert "ix_visits_local_uuid" in indexes
    idx = indexes["ix_visits_local_uuid"]
    assert idx.unique is True


def test_multiple_nulls_allowed():
    """Partial unique index excludes NULL — many rows may have local_uuid IS NULL."""
    # Need a child to satisfy the FK.
    from app.models.child import Child
    from datetime import date
    db = SessionLocal()
    try:
        child = Child(name="test", date_of_birth=date(2024, 1, 1), sex="M")
        db.add(child)
        db.flush()
        v1 = Visit(child_id=child.id, age_months=12.0, local_uuid=None)
        v2 = Visit(child_id=child.id, age_months=13.0, local_uuid=None)
        db.add_all([v1, v2])
        db.commit()
        assert v1.id != v2.id
    finally:
        db.rollback()
        db.close()


def test_duplicate_non_null_local_uuid_rejected():
    """Two rows with the same non-null local_uuid must raise IntegrityError."""
    from app.models.child import Child
    from datetime import date
    db = SessionLocal()
    try:
        child = Child(name="test2", date_of_birth=date(2024, 1, 2), sex="F")
        db.add(child)
        db.flush()
        uuid = "11111111-1111-1111-1111-111111111111"
        v1 = Visit(child_id=child.id, age_months=12.0, local_uuid=uuid)
        db.add(v1)
        db.commit()
        v2 = Visit(child_id=child.id, age_months=13.0, local_uuid=uuid)
        db.add(v2)
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()
