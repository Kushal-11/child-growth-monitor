"""Children API auth + ownership tests."""
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User
from app.models.child import Child
from app.services import auth_service
from app.api.auth import router as auth_router
from app.api.routes import router as api_router


@pytest.fixture
def ctx():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    db = TestingSession()
    u1 = User(username="u1", full_name="One", hashed_password=auth_service.hash_password("pw"))
    u2 = User(username="u2", full_name="Two", hashed_password=auth_service.hash_password("pw"))
    db.add_all([u1, u2]); db.flush()
    db.add(Child(name="A", date_of_birth=date(2024, 1, 1), sex="M", user_id=u1.id))
    db.add(Child(name="B", date_of_birth=date(2024, 1, 1), sex="F", user_id=u2.id))
    db.commit(); db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(api_router)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    def token(username):
        return client.post("/api/v1/auth/login", json={"username": username, "password": "pw"}).json()["access_token"]

    return client, token


def test_children_requires_auth(ctx):
    client, _ = ctx
    assert client.get("/api/v1/children").status_code == 401


def test_children_only_own(ctx):
    client, token = ctx
    r = client.get("/api/v1/children", headers={"Authorization": f"Bearer {token('u1')}"})
    assert r.status_code == 200
    names = [c["name"] for c in r.json()]
    assert names == ["A"]


def test_other_users_child_404(ctx):
    client, token = ctx
    listing = client.get("/api/v1/children", headers={"Authorization": f"Bearer {token('u2')}"}).json()
    other_id = listing[0]["id"]
    r = client.get(f"/api/v1/children/{other_id}", headers={"Authorization": f"Bearer {token('u1')}"})
    assert r.status_code == 404
