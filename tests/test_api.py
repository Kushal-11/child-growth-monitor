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


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestChildrenEndpoints:
    def test_list_children_empty(self, client):
        response = client.get("/api/v1/children")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_child_not_found(self, client):
        response = client.get("/api/v1/children/99999")
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


class TestWebUI:
    def test_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Child Growth Assessment" in response.text

    def test_children_page(self, client):
        response = client.get("/children")
        assert response.status_code == 200
        assert "Registered Children" in response.text
