"""Auth service unit tests."""
import pytest

from app.services import auth_service


def test_hash_and_verify_password():
    h = auth_service.hash_password("s3cret")
    assert h != "s3cret"
    assert auth_service.verify_password("s3cret", h) is True
    assert auth_service.verify_password("wrong", h) is False


def test_create_and_decode_token():
    token = auth_service.create_access_token(user_id=42, username="worker1")
    payload = auth_service.decode_token(token)
    assert payload["sub"] == "42"
    assert payload["username"] == "worker1"


def test_decode_invalid_token_raises():
    with pytest.raises(auth_service.AuthError):
        auth_service.decode_token("not.a.valid.token")
