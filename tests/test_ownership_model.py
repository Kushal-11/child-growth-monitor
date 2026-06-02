"""Ownership / new-column model tests."""
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base
from app.models.user import User
from app.models.child import Child
from app.models.visit import Visit
from app.models.measurement import MeasurementResult  # noqa: F401


def _session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_child_owner_and_new_columns():
    db = _session()
    u = User(username="w", full_name="W", hashed_password="x")
    db.add(u)
    db.flush()
    c = Child(name="Kid", date_of_birth=date(2024, 1, 1), sex="M",
              user_id=u.id, photo_path="/p.jpg", is_archived=False)
    db.add(c)
    db.commit()
    db.refresh(c)
    assert c.owner.username == "w"
    assert c.photo_path == "/p.jpg"
    assert c.is_archived is False


def test_visit_owner_and_entry_method_default():
    db = _session()
    c = Child(name="Kid", date_of_birth=date(2024, 1, 1), sex="M")
    db.add(c)
    db.flush()
    v = Visit(child_id=c.id, age_months=12.0, user_id=None)
    db.add(v)
    db.commit()
    db.refresh(v)
    assert v.entry_method == "assessment"
