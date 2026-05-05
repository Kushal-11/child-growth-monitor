"""Visit model schema tests."""
from datetime import datetime

from app.models.database import Base, engine
from app.models.child import Child
from app.models.visit import Visit


def setup_module(_module):
    Base.metadata.create_all(bind=engine)


def test_visit_has_local_uuid_column():
    columns = {c.name for c in Visit.__table__.columns}
    assert "local_uuid" in columns


def test_local_uuid_is_unique_nullable():
    col = Visit.__table__.columns["local_uuid"]
    assert col.unique is True
    assert col.nullable is True
