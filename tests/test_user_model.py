"""User model tests."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base
from app.models.user import User


def _session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_create_user_defaults():
    db = _session()
    u = User(username="worker1", full_name="Asha W", hashed_password="x")
    db.add(u)
    db.commit()
    db.refresh(u)
    assert u.id is not None
    assert u.role == "worker"
    assert u.is_active is True
    assert u.created_at is not None


def test_username_unique():
    db = _session()
    db.add(User(username="dup", full_name="A", hashed_password="x"))
    db.commit()
    db.add(User(username="dup", full_name="B", hashed_password="y"))
    import pytest
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        db.commit()
