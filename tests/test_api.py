"""Tests for the FastAPI API endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        ("field", "value"),
        (
            ("height_cm", "10"),
            ("weight_kg", "nan"),
            ("muac_cm", "30"),
        ),
    )
    def test_rejects_implausible_or_non_finite_measurements(
        self, client, field, value
    ):
        import io

        response = client.post(
            "/api/v1/assess",
            data={
                "child_name": "Validation Child",
                "date_of_birth": "2024-01-01",
                "sex": "M",
                field: value,
            },
            files={"image": ("test.jpg", io.BytesIO(b"image"), "image/jpeg")},
        )
        assert response.status_code == 400

    def test_rejects_future_dob(self, client):
        import io

        response = client.post(
            "/api/v1/assess",
            data={
                "child_name": "Future Child",
                "date_of_birth": "2099-01-01",
                "sex": "F",
            },
            files={"image": ("test.jpg", io.BytesIO(b"image"), "image/jpeg")},
        )
        assert response.status_code == 400
        assert "date_of_birth" in response.json()["detail"]

    def test_pose_runtime_unavailable_returns_503(
        self, client, tmp_path, monkeypatch
    ):
        import io

        from app.api import routes as routes_module
        from app.services.measurement_service import PoseRuntimeUnavailableError

        class UnavailableService:
            @staticmethod
            def _compute_age_months(_dob, _today):
                return 24.0

            @staticmethod
            def _validate_inputs(**_kwargs):
                return None

            @staticmethod
            def assess(**_kwargs):
                raise PoseRuntimeUnavailableError("model asset is missing")

        original = app.dependency_overrides.get(
            routes_module.get_assessment_service
        )
        app.dependency_overrides[routes_module.get_assessment_service] = (
            lambda: UnavailableService()
        )
        monkeypatch.setattr(routes_module, "UPLOAD_DIR", tmp_path)
        try:
            response = client.post(
                "/api/v1/assess",
                data={
                    "child_name": "Runtime Child",
                    "date_of_birth": "2024-01-01",
                    "sex": "M",
                },
                files={
                    "image": (
                        "test.jpg",
                        io.BytesIO(b"image"),
                        "image/jpeg",
                    )
                },
            )
        finally:
            if original is None:
                app.dependency_overrides.pop(
                    routes_module.get_assessment_service, None
                )
            else:
                app.dependency_overrides[
                    routes_module.get_assessment_service
                ] = original

        assert response.status_code == 503
        assert "Pose measurement runtime is unavailable" in response.json()["detail"]
        assert not list(tmp_path.iterdir())


class TestWebUI:
    def test_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Child Growth Assessment" in response.text

    def test_children_page(self, client):
        response = client.get("/children")
        assert response.status_code == 200
        assert "Registered Children" in response.text
