"""CLI user-creation tests."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base
from app.models.user import User
from app.services import auth_service
from scripts.create_user import create_user


def _session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_create_user_persists_and_hashes():
    db = _session()
    create_user(db, username="admin1", full_name="Admin", password="pw", role="admin")
    u = db.query(User).filter(User.username == "admin1").first()
    assert u is not None
    assert u.role == "admin"
    assert u.hashed_password != "pw"
    assert auth_service.verify_password("pw", u.hashed_password)


def test_create_user_duplicate_raises():
    db = _session()
    create_user(db, username="dup", full_name="A", password="pw", role="worker")
    import pytest
    with pytest.raises(ValueError):
        create_user(db, username="dup", full_name="B", password="pw2", role="worker")
