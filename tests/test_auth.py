"""Auth endpoint integration tests."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.database import Base, get_db
from app.models.user import User
from app.services import auth_service
from app.api.auth import router as auth_router


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    db = TestingSession()
    db.add(User(username="asha", full_name="Asha", hashed_password=auth_service.hash_password("pw123")))
    db.commit()
    db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_login_success(client):
    r = client.post("/api/v1/auth/login", json={"username": "asha", "password": "pw123"})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["user"]["username"] == "asha"


def test_login_wrong_password(client):
    r = client.post("/api/v1/auth/login", json={"username": "asha", "password": "bad"})
    assert r.status_code == 401


def test_login_unknown_user(client):
    r = client.post("/api/v1/auth/login", json={"username": "ghost", "password": "x"})
    assert r.status_code == 401


def test_me_with_token(client):
    token = client.post("/api/v1/auth/login", json={"username": "asha", "password": "pw123"}).json()["access_token"]
    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["username"] == "asha"


def test_me_without_token(client):
    r = client.get("/api/v1/auth/me")
    assert r.status_code == 401
