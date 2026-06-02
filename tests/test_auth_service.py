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


def test_get_current_user_non_numeric_sub_raises_401():
    from fastapi import HTTPException
    from jose import jwt
    from config import JWT_SECRET, JWT_ALGORITHM
    from app.services.auth_service import get_current_user

    # Build a validly-signed token whose sub is non-numeric.
    bad_token = jwt.encode({"sub": "notanint", "username": "x"}, JWT_SECRET, algorithm=JWT_ALGORITHM)

    class _DummyQuery:
        def filter(self, *a, **k): return self
        def first(self): return None
    class _DummyDB:
        def query(self, *a, **k): return _DummyQuery()

    with pytest.raises(HTTPException) as exc:
        get_current_user(authorization=f"Bearer {bad_token}", db=_DummyDB())
    assert exc.value.status_code == 401
