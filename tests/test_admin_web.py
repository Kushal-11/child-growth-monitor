"""Admin web UI tests."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User
from app.services import auth_service
from app.web.admin import router as admin_router


@pytest.fixture
def client():
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
    db.add(User(username="boss", full_name="Boss", role="admin",
                hashed_password=auth_service.hash_password("pw")))
    db.add(User(username="worker", full_name="Worker", role="worker",
                hashed_password=auth_service.hash_password("pw")))
    db.commit(); db.close()

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test")
    app.include_router(admin_router)
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_users_page_redirects_when_logged_out(client):
    r = client.get("/admin/users", follow_redirects=False)
    assert r.status_code in (302, 303)
    assert "/admin/login" in r.headers["location"]


def test_admin_login_and_list(client):
    r = client.post("/admin/login", data={"username": "boss", "password": "pw"}, follow_redirects=False)
    assert r.status_code in (302, 303)
    page = client.get("/admin/users")
    assert page.status_code == 200
    assert "worker" in page.text


def test_worker_cannot_admin_login(client):
    r = client.post("/admin/login", data={"username": "worker", "password": "pw"}, follow_redirects=False)
    assert r.status_code == 200
    assert "Invalid" in r.text or "admin" in r.text.lower()


def test_create_user_via_admin(client):
    client.post("/admin/login", data={"username": "boss", "password": "pw"})
    r = client.post("/admin/users/create", data={
        "username": "newworker", "full_name": "New", "password": "pw2", "role": "worker",
    }, follow_redirects=True)
    assert r.status_code == 200
    assert "newworker" in r.text


def test_admin_self_row_has_no_toggle(client):
    """The logged-in admin's own row must not expose a deactivate button
    (prevents self-lockout). With seeded users boss(admin)+worker, only the
    worker row should render a toggle form."""
    client.post("/admin/login", data={"username": "boss", "password": "pw"})
    page = client.get("/admin/users")
    assert page.status_code == 200
    assert "boss" in page.text
    assert "worker" in page.text
    # Exactly one toggle form (the worker's); boss's own row is gated out.
    assert page.text.count("/toggle") == 1


def test_protected_posts_require_admin_session(client):
    """Without an admin session, user-management POSTs must redirect to login
    and must NOT mutate data."""
    client.get("/admin/logout")
    r = client.post(
        "/admin/users/create",
        data={"username": "sneaky", "full_name": "S", "password": "x", "role": "admin"},
        follow_redirects=False,
    )
    assert r.status_code in (302, 303)
    assert "/admin/login" in r.headers["location"]
    # Confirm nothing was created: log in as admin and check the list.
    client.post("/admin/login", data={"username": "boss", "password": "pw"})
    page = client.get("/admin/users")
    assert "sneaky" not in page.text
