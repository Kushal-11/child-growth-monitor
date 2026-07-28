"""Tests for the FastAPI API endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_headers():
    """Create a real user in the app DB and mint a bearer token for it.

    The children endpoints are auth-protected and owner-scoped; tests that hit
    them must authenticate as a real, active user resolvable by get_current_user.
    """
    from app.models.database import SessionLocal
    from app.models.user import User
    from app.services import auth_service

    db = SessionLocal()
    try:
        username = "test_api_worker"
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            user = User(
                username=username,
                full_name="Test API Worker",
                hashed_password=auth_service.hash_password("pw"),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        token = auth_service.create_access_token(user_id=user.id, username=user.username)
    finally:
        db.close()
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestChildrenEndpoints:
    def test_list_children_requires_auth(self, client):
        response = client.get("/api/v1/children")
        assert response.status_code == 401

    def test_list_children_returns_list(self, client, auth_headers):
        response = client.get("/api/v1/children", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_child_not_found(self, client, auth_headers):
        response = client.get("/api/v1/children/99999", headers=auth_headers)
        assert response.status_code == 404


class TestAssessEndpoint:
    def test_missing_fields(self, client):
        """Missing required form fields should return 422."""
        response = client.post("/api/v1/assess", data={"child_name": "Test"})
        assert response.status_code == 422

    def test_invalid_sex(self, client):
        """Invalid sex value should return 400."""
        # Create a minimal 1x1 white pixel PNG
        import io

        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        response = client.post(
            "/api/v1/assess",
            data={
                "child_name": "Test",
                "date_of_birth": "2023-01-01",
                "sex": "X",
            },
            files={"image": ("test.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        ("dob", "message"),
        [
            ((date.today() + timedelta(days=1)).isoformat(), "cannot be in the future"),
            ((date.today() - timedelta(days=6 * 365)).isoformat(), "0 through 59 months"),
        ],
    )
    def test_invalid_clinical_age_returns_422(self, client, dob, message):
        response = client.post(
            "/api/v1/assess",
            data={"child_name": "Test", "date_of_birth": dob, "sex": "F"},
            files={"image": ("test.png", b"not-read-before-validation", "image/png")},
        )
        assert response.status_code == 422
        assert message in response.json()["detail"]


class TestWebUI:
    def test_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Child Growth Assessment" in response.text

    def test_children_page(self, client):
        response = client.get("/children")
        assert response.status_code == 200
        assert "Registered Children" in response.text
