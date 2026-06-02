# Login + Child Management System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add JWT auth (admin-provisioned accounts) and a dedicated child-management module (profile+photo CRUD, manual monthly measurement entry, growth timeline, archive) to the Child Growth Monitor backend and Flutter app, preserving the offline-first sync architecture.

**Architecture:** Backend gets a `User` model + `auth_service` (bcrypt + JWT), `user_id`/photo/archive columns on `children` and `visits` (added via idempotent startup `ALTER TABLE`), auth routes, an auth-protected sync endpoint, an admin web UI, and a CLI seed script. Flutter gets an `auth` feature (login screen, auth provider, secure token storage), `child_management` screens, a bearer-token-aware API/sync layer, a router auth gate, and mirrored Drift columns via a schema migration.

**Tech Stack:** FastAPI, SQLAlchemy (SQLite, no Alembic), passlib[bcrypt], python-jose, Jinja2; Flutter, Riverpod, Drift, go_router, flutter_secure_storage, http, image_picker.

**Spec:** `docs/superpowers/specs/2026-06-02-login-child-management-design.md`

---

## File Structure

### Backend — new files
- `app/models/user.py` — `User` SQLAlchemy model
- `app/services/auth_service.py` — password hashing, JWT create/verify, `get_current_user` dependency
- `app/schemas/auth.py` — Pydantic schemas (`LoginRequest`, `TokenResponse`, `UserOut`)
- `app/api/auth.py` — `/api/v1/auth/login`, `/api/v1/auth/me`
- `app/web/admin.py` — admin web UI router (`/admin/login`, `/admin/users`, ...)
- `app/web/templates/admin_login.html`, `app/web/templates/admin_users.html`
- `scripts/create_user.py` — CLI to seed/create users
- `tests/test_auth.py`, `tests/test_admin_web.py`, `tests/test_migration.py`, `tests/test_sync_auth.py`

### Backend — modified files
- `config.py` — add `JWT_SECRET`, `JWT_ALGORITHM`, `JWT_EXPIRE_DAYS`
- `app/models/database.py` — import `User` in `init_db`; add `run_migrations()` (idempotent ALTER TABLE)
- `app/models/child.py` — add `user_id`, `photo_path`, `is_archived`, `owner` relationship
- `app/models/visit.py` — add `user_id`, `entry_method`
- `app/api/sync.py` — require auth, stamp `user_id`, accept `photo`/`entry_method`/`is_archived`
- `app/api/routes.py` — protect `/children`, `/children/{id}` with auth + owner filter
- `main.py` — call `run_migrations()`; include `auth_router` + `admin_router`; add `SessionMiddleware`
- `requirements.txt` — add auth deps

### Flutter — new files
- `lib/services/auth_service.dart` — login/logout/me HTTP + token via flutter_secure_storage
- `lib/providers/auth_provider.dart` — `AuthState`, `AuthNotifier`, providers
- `lib/screens/auth/login_screen.dart`
- `lib/screens/child_management/child_form_screen.dart`
- `lib/screens/child_management/manual_measurement_screen.dart`
- `lib/database/daos/manual_visit_dao.dart` — create a manual visit (no image required)
- test files under `flutter_app/test/`

### Flutter — modified files
- `pubspec.yaml` — add `flutter_secure_storage`
- `lib/database/tables/children_table.dart` — add `ownerUserId`, `photoPath`, `isArchived`
- `lib/database/tables/visits_table.dart` — add `ownerUserId`, `entryMethod`; make `imagePath` nullable
- `lib/database/database.dart` — bump `schemaVersion` to 3 + migration
- `lib/database/daos/child_dao.dart` — owner-scoped create/update/archive
- `lib/services/api_service.dart` — accept auth token, send `Authorization` header
- `lib/services/sync_service.dart` — send bearer token + new fields; surface 401
- `lib/providers/api_provider.dart` / `sync_provider.dart` — wire token in
- `lib/router.dart` — auth redirect gate + new routes
- `lib/main.dart` — load token at startup, pass to ProviderScope
- `lib/screens/children/children_list_screen.dart` — "+ New child", archive, owner filter
- `lib/screens/children/child_detail_screen.dart` — profile header + "Add measurement"/"Edit"

---

## PHASE 1 — BACKEND AUTH FOUNDATION

### Task 1: Add auth dependencies & config

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py:1-10`

- [ ] **Step 1: Add deps to requirements.txt**

Add after the `# Validation` block (after `pydantic>=2.5.0`):
```
# Authentication
passlib[bcrypt]>=1.7.4
python-jose[cryptography]>=3.3.0
```

- [ ] **Step 2: Install**

Run: `.venv/bin/pip install "passlib[bcrypt]>=1.7.4" "python-jose[cryptography]>=3.3.0"`
Expected: "Successfully installed ..." (passlib, python-jose, bcrypt, ecdsa, rsa)

- [ ] **Step 3: Add JWT config to config.py**

Add after the `DATABASE_URL` line (config.py:9):
```python
import os

# Authentication / JWT
JWT_SECRET = os.environ.get("CGM_JWT_SECRET", "dev-insecure-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30
```

- [ ] **Step 4: Verify config imports**

Run: `PYTHONPATH=. .venv/bin/python -c "import config; print(config.JWT_ALGORITHM, config.JWT_EXPIRE_DAYS)"`
Expected: `HS256 30`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config.py
git commit -m "feat(backend): add auth dependencies and JWT config"
```

---

### Task 2: User model

**Files:**
- Create: `app/models/user.py`
- Modify: `app/models/database.py:23-31` (init_db imports)

- [ ] **Step 1: Write the failing test**

Create `tests/test_user_model.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_user_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models.user'`

- [ ] **Step 3: Create the User model**

Create `app/models/user.py`:
```python
"""User model representing a health worker or admin account."""
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from app.models.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="worker", nullable=False)  # "worker" | "admin"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    children = relationship("Child", back_populates="owner")
```

- [ ] **Step 4: Register User in init_db**

In `app/models/database.py`, modify `init_db` (currently lines 23-31) to import User first:
```python
def init_db():
    """Create all tables."""
    from app.models.user import User  # noqa: F401
    from app.models.child import Child  # noqa: F401
    from app.models.visit import Visit  # noqa: F401
    from app.models.measurement import MeasurementResult  # noqa: F401

    Base.metadata.create_all(bind=engine)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_user_model.py -v`
Expected: 2 passed

> Note: the `children` relationship references `Child.owner`, added in Task 4. Until then `User` alone compiles; the relationship is only resolved when both mappers are configured (Task 4 tests cover that).

- [ ] **Step 6: Commit**

```bash
git add app/models/user.py app/models/database.py tests/test_user_model.py
git commit -m "feat(backend): add User model"
```

---

### Task 3: Auth service (hashing + JWT + current-user dependency)

**Files:**
- Create: `app/services/auth_service.py`
- Test: `tests/test_auth_service.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_service.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_auth_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.auth_service'`

- [ ] **Step 3: Create the auth service**

Create `app/services/auth_service.py`:
```python
"""Authentication: password hashing, JWT tokens, current-user dependency."""
from datetime import datetime, timedelta

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.user import User
from config import JWT_ALGORITHM, JWT_EXPIRE_DAYS, JWT_SECRET

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthError(Exception):
    """Raised when a token is invalid or expired."""


def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)


def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise AuthError(str(exc))


def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency: resolve the authenticated, active user from the bearer token."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except AuthError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == int(user_id)).first() if user_id else None
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
    return user
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_auth_service.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/auth_service.py tests/test_auth_service.py
git commit -m "feat(backend): add auth service (bcrypt + JWT + current-user dependency)"
```

---

### Task 4: Add owner/photo/archive columns to Child & Visit models

**Files:**
- Modify: `app/models/child.py:10-22`
- Modify: `app/models/visit.py:11-38`
- Test: `tests/test_ownership_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ownership_model.py`:
```python
"""Ownership / new-column model tests."""
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base
from app.models.user import User
from app.models.child import Child
from app.models.visit import Visit
from app.models.measurement import MeasurementResult  # noqa: F401


def _session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_child_owner_and_new_columns():
    db = _session()
    u = User(username="w", full_name="W", hashed_password="x")
    db.add(u)
    db.flush()
    c = Child(name="Kid", date_of_birth=date(2024, 1, 1), sex="M",
              user_id=u.id, photo_path="/p.jpg", is_archived=False)
    db.add(c)
    db.commit()
    db.refresh(c)
    assert c.owner.username == "w"
    assert c.photo_path == "/p.jpg"
    assert c.is_archived is False


def test_visit_owner_and_entry_method_default():
    db = _session()
    c = Child(name="Kid", date_of_birth=date(2024, 1, 1), sex="M")
    db.add(c)
    db.flush()
    v = Visit(child_id=c.id, age_months=12.0, user_id=None)
    db.add(v)
    db.commit()
    db.refresh(v)
    assert v.entry_method == "assessment"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ownership_model.py -v`
Expected: FAIL — `TypeError: 'user_id' is an invalid keyword argument for Child` (or AttributeError on `owner`)

- [ ] **Step 3: Update Child model**

Replace the body of `app/models/child.py` with:
```python
"""Child model representing a registered child."""
from datetime import datetime

from sqlalchemy import Boolean, Column, Date, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.models.database import Base


class Child(Base):
    __tablename__ = "children"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    sex = Column(String(1), nullable=False)  # 'M' or 'F'
    guardian_name = Column(String(100), nullable=True)
    location = Column(String(200), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    photo_path = Column(String(500), nullable=True)
    is_archived = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="children")
    visits = relationship("Visit", back_populates="child", cascade="all, delete-orphan")
```

- [ ] **Step 4: Update Visit model**

In `app/models/visit.py`, add two columns. After the `local_uuid` line (visit.py:19) add:
```python
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    entry_method = Column(String(20), default="assessment", nullable=False)  # "assessment" | "manual"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_ownership_model.py tests/test_user_model.py -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add app/models/child.py app/models/visit.py tests/test_ownership_model.py
git commit -m "feat(backend): add owner/photo/archive columns to Child and Visit"
```

---

### Task 5: Idempotent startup migration (ALTER TABLE)

The DB has no Alembic. Existing `growth_monitor.db` files lack the new columns; `create_all` won't add columns to existing tables. Add an idempotent migration that runs at startup.

**Files:**
- Modify: `app/models/database.py` (add `run_migrations`)
- Modify: `main.py:33-34` (call it)
- Test: `tests/test_migration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_migration.py`:
```python
"""Idempotent ALTER TABLE migration tests."""
from sqlalchemy import create_engine, inspect, text

from app.models import database as dbmod


def _legacy_engine():
    """Create an engine with the OLD children/visits schema (no new columns)."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE children (id INTEGER PRIMARY KEY, name VARCHAR, "
            "date_of_birth DATE, sex VARCHAR, guardian_name VARCHAR, location VARCHAR, "
            "created_at DATETIME, updated_at DATETIME)"
        ))
        conn.execute(text(
            "CREATE TABLE visits (id INTEGER PRIMARY KEY, child_id INTEGER, "
            "visit_date DATETIME, age_months FLOAT, image_path VARCHAR, "
            "side_image_path VARCHAR, back_image_path VARCHAR, notes TEXT, local_uuid VARCHAR)"
        ))
    return engine


def test_migration_adds_missing_columns():
    engine = _legacy_engine()
    dbmod.run_migrations(engine)
    insp = inspect(engine)
    child_cols = {c["name"] for c in insp.get_columns("children")}
    visit_cols = {c["name"] for c in insp.get_columns("visits")}
    assert {"user_id", "photo_path", "is_archived"} <= child_cols
    assert {"user_id", "entry_method"} <= visit_cols


def test_migration_is_idempotent():
    engine = _legacy_engine()
    dbmod.run_migrations(engine)
    dbmod.run_migrations(engine)  # second run must not raise
    insp = inspect(engine)
    assert "user_id" in {c["name"] for c in insp.get_columns("children")}


def test_migration_noop_when_table_absent():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    dbmod.run_migrations(engine)  # no tables yet — must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_migration.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'run_migrations'`

- [ ] **Step 3: Implement run_migrations**

Add to `app/models/database.py` (after `init_db`):
```python
from sqlalchemy import inspect, text  # add to existing imports at top


# Columns added after the original schema. (table, column, DDL type with default)
_MIGRATIONS = [
    ("children", "user_id", "INTEGER"),
    ("children", "photo_path", "VARCHAR(500)"),
    ("children", "is_archived", "BOOLEAN NOT NULL DEFAULT 0"),
    ("visits", "user_id", "INTEGER"),
    ("visits", "entry_method", "VARCHAR(20) NOT NULL DEFAULT 'assessment'"),
]


def run_migrations(target_engine=None):
    """Idempotently add columns missing from existing tables (SQLite, no Alembic)."""
    eng = target_engine or engine
    insp = inspect(eng)
    existing_tables = set(insp.get_table_names())
    with eng.begin() as conn:
        for table, column, ddl in _MIGRATIONS:
            if table not in existing_tables:
                continue
            cols = {c["name"] for c in insp.get_columns(table)}
            if column in cols:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
```

Put the `from sqlalchemy import inspect, text` import at the top with the other imports (the existing import line is `from sqlalchemy import create_engine` — change it to `from sqlalchemy import create_engine, inspect, text`).

- [ ] **Step 4: Call run_migrations at startup**

In `main.py`, in `create_app()`, change the init block (main.py:33-34) from:
```python
    # Initialize database tables
    init_db()
```
to:
```python
    # Initialize database tables, then apply additive migrations to existing DBs
    init_db()
    from app.models.database import run_migrations
    run_migrations()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_migration.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add app/models/database.py main.py tests/test_migration.py
git commit -m "feat(backend): add idempotent ALTER TABLE migration for new columns"
```

---

### Task 6: Auth schemas + auth routes (/login, /me)

**Files:**
- Create: `app/schemas/auth.py`
- Create: `app/api/auth.py`
- Modify: `main.py` (include router)
- Test: `tests/test_auth.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth.py`:
```python
"""Auth endpoint integration tests."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User
from app.services import auth_service
from app.api.auth import router as auth_router


@pytest.fixture
def client():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api.auth'`

- [ ] **Step 3: Create auth schemas**

Create `app/schemas/auth.py`:
```python
"""Auth request/response schemas."""
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1)


class UserOut(BaseModel):
    id: int
    username: str
    full_name: str
    role: str

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
```

- [ ] **Step 4: Create auth routes**

Create `app/api/auth.py`:
```python
"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse, UserOut
from app.services import auth_service

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if user is None or not user.is_active or not auth_service.verify_password(
        payload.password, user.hashed_password
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    token = auth_service.create_access_token(user_id=user.id, username=user.username)
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(current: User = Depends(auth_service.get_current_user)):
    return UserOut.model_validate(current)
```

- [ ] **Step 5: Include router in main.py**

In `main.py`, add the import near the other api imports (after `from app.api.sync import router as sync_router`):
```python
from app.api.auth import router as auth_router
```
And register it where the other routers are included (after `app.include_router(sync_router)`):
```python
    app.include_router(auth_router)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_auth.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add app/schemas/auth.py app/api/auth.py main.py tests/test_auth.py
git commit -m "feat(backend): add /auth/login and /auth/me endpoints"
```

---

### Task 7: CLI user-seed script

**Files:**
- Create: `scripts/create_user.py`
- Test: `tests/test_create_user_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_create_user_cli.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_create_user_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.create_user'`

- [ ] **Step 3: Create the script**

Create `scripts/__init__.py` (empty file).
Create `scripts/create_user.py`:
```python
"""CLI: create a health-worker or admin account.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/create_user.py \
      --username admin --full-name "Site Admin" --role admin
(prompts for password)
"""
import argparse
import getpass

from sqlalchemy.orm import Session

from app.models.database import SessionLocal, init_db, run_migrations
from app.models.user import User
from app.services import auth_service


def create_user(db: Session, *, username: str, full_name: str, password: str, role: str) -> User:
    if db.query(User).filter(User.username == username).first() is not None:
        raise ValueError(f"User '{username}' already exists")
    user = User(
        username=username,
        full_name=full_name,
        hashed_password=auth_service.hash_password(password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CGM user account")
    parser.add_argument("--username", required=True)
    parser.add_argument("--full-name", required=True)
    parser.add_argument("--role", default="worker", choices=["worker", "admin"])
    args = parser.parse_args()

    init_db()
    run_migrations()
    password = getpass.getpass("Password: ")
    if not password:
        raise SystemExit("Password cannot be empty")

    db = SessionLocal()
    try:
        user = create_user(
            db, username=args.username, full_name=args.full_name,
            password=password, role=args.role,
        )
        print(f"Created {user.role} '{user.username}' (id={user.id})")
    except ValueError as exc:
        raise SystemExit(str(exc))
    finally:
        db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_create_user_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/create_user.py tests/test_create_user_cli.py
git commit -m "feat(backend): add CLI user-seed script"
```

---

## PHASE 2 — BACKEND: PROTECT DATA ROUTES & SYNC

### Task 8: Protect & owner-filter the children API routes

**Files:**
- Modify: `app/api/routes.py:101-155`
- Test: `tests/test_children_auth.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_children_auth.py`:
```python
"""Children API auth + ownership tests."""
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User
from app.models.child import Child
from app.services import auth_service
from app.api.auth import router as auth_router
from app.api.routes import router as api_router


@pytest.fixture
def ctx():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
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
    # u2's child is id 2; u1 must not see it
    listing = client.get("/api/v1/children", headers={"Authorization": f"Bearer {token('u2')}"}).json()
    other_id = listing[0]["id"]
    r = client.get(f"/api/v1/children/{other_id}", headers={"Authorization": f"Bearer {token('u1')}"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_children_auth.py -v`
Expected: FAIL — `test_children_requires_auth` gets 200, not 401 (route is currently public)

- [ ] **Step 3: Add auth + owner filter to routes**

In `app/api/routes.py`, add imports near the top (after `from app.models.child import Child`):
```python
from app.models.user import User
from app.services.auth_service import get_current_user
```

Replace `list_children` (routes.py:101-120) with:
```python
@router.get("/children")
def list_children(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """List the authenticated worker's non-archived children."""
    children = (
        db.query(Child)
        .filter(Child.user_id == current.id, Child.is_archived == False)  # noqa: E712
        .order_by(Child.name)
        .all()
    )
    return [
        {
            "id": c.id,
            "name": c.name,
            "date_of_birth": c.date_of_birth.isoformat(),
            "sex": c.sex,
            "photo_path": c.photo_path,
            "visit_count": len(c.visits),
        }
        for c in children
    ]
```

Replace the `get_child` signature + ownership check (routes.py:123-126). Change the signature to add the dependency and the filter:
```python
@router.get("/children/{child_id}")
def get_child(
    child_id: int,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """Get child detail with full visit history (owner-scoped)."""
    child = (
        db.query(Child)
        .filter(Child.id == child_id, Child.user_id == current.id)
        .first()
    )
    if not child:
        raise HTTPException(404, "Child not found")
```
(Leave the rest of `get_child`'s body — the visits loop and return — unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_children_auth.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/routes.py tests/test_children_auth.py
git commit -m "feat(backend): require auth and owner-filter children endpoints"
```

---

### Task 9: Auth-protect sync + stamp user_id + accept photo/entry_method/is_archived

**Files:**
- Modify: `app/api/sync.py`
- Test: `tests/test_sync_auth.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sync_auth.py`:
```python
"""Sync auth + ownership + new-field tests."""
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User
from app.models.child import Child
from app.models.visit import Visit
from app.services import auth_service
from app.api.auth import router as auth_router
from app.api.sync import router as sync_router


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    import app.api.sync as syncmod
    monkeypatch.setattr(syncmod, "UPLOAD_DIR", tmp_path)
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    db = TestingSession()
    db.add(User(username="w", full_name="W", hashed_password=auth_service.hash_password("pw")))
    db.commit(); db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(sync_router)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    token = client.post("/api/v1/auth/login", json={"username": "w", "password": "pw"}).json()["access_token"]
    return client, token, TestingSession


def _payload():
    return {
        "local_uuid": "11111111-1111-1111-1111-111111111111",
        "child_name": "Kid", "date_of_birth": "2024-01-01", "sex": "M",
        "age_months": "12.0", "visit_date": "2026-06-01T00:00:00",
        "manual_height_cm": "75.0", "manual_weight_kg": "9.0",
        "entry_method": "manual",
    }


def _files():
    return {"image": ("img.jpg", io.BytesIO(b"fakejpeg"), "image/jpeg")}


def test_sync_requires_auth(ctx):
    client, _, _ = ctx
    r = client.post("/api/v1/sync", data=_payload(), files=_files())
    assert r.status_code == 401


def test_sync_stamps_user_and_entry_method(ctx):
    client, token, Session = ctx
    r = client.post("/api/v1/sync", data=_payload(), files=_files(),
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    db = Session()
    visit = db.query(Visit).first()
    child = db.query(Child).first()
    assert visit.user_id is not None
    assert visit.entry_method == "manual"
    assert child.user_id == visit.user_id
    db.close()


def test_sync_idempotent(ctx):
    client, token, _ = ctx
    h = {"Authorization": f"Bearer {token}"}
    r1 = client.post("/api/v1/sync", data=_payload(), files=_files(), headers=h)
    r2 = client.post("/api/v1/sync", data=_payload(), files=_files(), headers=h)
    assert r1.json()["status"] == "synced"
    assert r2.json()["status"] == "already_synced"
    assert r1.json()["server_visit_id"] == r2.json()["server_visit_id"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_sync_auth.py -v`
Expected: FAIL — `test_sync_requires_auth` returns 200 (sync is public)

- [ ] **Step 3: Update sync.py**

In `app/api/sync.py`, add imports (after `from app.models.visit import Visit`):
```python
from app.models.user import User
from app.services.auth_service import get_current_user
```

Add three new `Form` params to the `sync_assessment` signature — insert them just before the `guardian_name: Optional[str] = Form(None),` line:
```python
    entry_method: str = Form("assessment"),
    is_archived: str = Form("false"),
```
And add a new optional file param after `image_back` (after the `image_back: Optional[UploadFile] = File(None),` line):
```python
    photo: Optional[UploadFile] = File(None),
```
And add the auth dependency — change the final param `db: Session = Depends(get_db),` to:
```python
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
```

Now stamp ownership. In the child find-or-create block, change the lookup to be owner-scoped and set `user_id` + photo on create. Replace the existing child block (sync.py:93-112) with:
```python
    child = (
        db.query(Child)
        .filter(
            Child.name == child_name,
            Child.date_of_birth == dob,
            Child.sex == sex,
            Child.user_id == current.id,
        )
        .first()
    )
    photo_path = _save_upload(photo) if photo is not None else None
    if child is None:
        child = Child(
            name=child_name,
            date_of_birth=dob,
            sex=sex,
            guardian_name=guardian_name,
            location=location,
            user_id=current.id,
            photo_path=photo_path,
        )
        db.add(child)
        db.flush()
    elif photo_path is not None:
        child.photo_path = photo_path
```

In the `Visit(...)` constructor (sync.py:114-122) add `user_id` and `entry_method`:
```python
    visit = Visit(
        child_id=child.id,
        visit_date=visit_dt,
        age_months=age_months,
        image_path=image_path,
        side_image_path=side_path,
        back_image_path=back_path,
        local_uuid=local_uuid,
        user_id=current.id,
        entry_method=entry_method,
    )
```

> Note: `image` remains required by the current contract. Manual entries from Flutter will send a placeholder 1x1 image (handled in the Flutter manual-sync task) so this endpoint stays unchanged in that respect. `is_archived` is accepted now for forward-compat; the archive-propagation wiring is Task 18.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_sync_auth.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full backend suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: all passing (pre-existing + new)

- [ ] **Step 6: Commit**

```bash
git add app/api/sync.py tests/test_sync_auth.py
git commit -m "feat(backend): auth-protect sync, stamp user_id, accept photo/entry_method"
```

---

## PHASE 3 — BACKEND ADMIN WEB UI

### Task 10: Admin web UI (session login + user management)

**Files:**
- Create: `app/web/admin.py`
- Create: `app/web/templates/admin_login.html`, `app/web/templates/admin_users.html`
- Modify: `main.py` (SessionMiddleware + include admin_router)
- Modify: `requirements.txt` (itsdangerous for SessionMiddleware)
- Test: `tests/test_admin_web.py`

- [ ] **Step 1: Add itsdangerous dependency**

Add to `requirements.txt` under the Authentication block:
```
itsdangerous>=2.1.0
```
Run: `.venv/bin/pip install "itsdangerous>=2.1.0"`
Expected: Successfully installed itsdangerous

- [ ] **Step 2: Write the failing test**

Create `tests/test_admin_web.py`:
```python
"""Admin web UI tests."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User
from app.services import auth_service
from app.web.admin import router as admin_router


@pytest.fixture
def client():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
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
    # non-admin rejected
    assert r.status_code == 200
    assert "Invalid" in r.text or "admin" in r.text.lower()


def test_create_user_via_admin(client):
    client.post("/admin/login", data={"username": "boss", "password": "pw"})
    r = client.post("/admin/users/create", data={
        "username": "newworker", "full_name": "New", "password": "pw2", "role": "worker",
    }, follow_redirects=True)
    assert r.status_code == 200
    assert "newworker" in r.text
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_admin_web.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.web.admin'`

- [ ] **Step 4: Create the admin router**

Create `app/web/admin.py`:
```python
"""Admin web UI: session-cookie login + user management."""
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.user import User
from app.services import auth_service

router = APIRouter(prefix="/admin", tags=["Admin"])
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _current_admin(request: Request, db: Session) -> User | None:
    user_id = request.session.get("admin_user_id")
    if not user_id:
        return None
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active or user.role != "admin":
        return None
    return user


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse(request, "admin_login.html", {"error": None})


@router.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if (user is None or not user.is_active or user.role != "admin"
            or not auth_service.verify_password(password, user.hashed_password)):
        return templates.TemplateResponse(
            request, "admin_login.html",
            {"error": "Invalid admin credentials"}, status_code=200,
        )
    request.session["admin_user_id"] = user.id
    return RedirectResponse("/admin/users", status_code=303)


@router.get("/logout")
def logout(request: Request):
    request.session.pop("admin_user_id", None)
    return RedirectResponse("/admin/login", status_code=303)


@router.get("/users", response_class=HTMLResponse)
def list_users(request: Request, db: Session = Depends(get_db)):
    admin = _current_admin(request, db)
    if admin is None:
        return RedirectResponse("/admin/login", status_code=303)
    users = db.query(User).order_by(User.username).all()
    return templates.TemplateResponse(
        request, "admin_users.html", {"users": users, "admin": admin, "error": None},
    )


@router.post("/users/create")
def create_user_web(
    request: Request,
    username: str = Form(...),
    full_name: str = Form(...),
    password: str = Form(...),
    role: str = Form("worker"),
    db: Session = Depends(get_db),
):
    admin = _current_admin(request, db)
    if admin is None:
        return RedirectResponse("/admin/login", status_code=303)
    error = None
    if db.query(User).filter(User.username == username).first() is not None:
        error = f"Username '{username}' already exists"
    elif role not in ("worker", "admin"):
        error = "Invalid role"
    else:
        db.add(User(
            username=username, full_name=full_name, role=role,
            hashed_password=auth_service.hash_password(password),
        ))
        db.commit()
    users = db.query(User).order_by(User.username).all()
    return templates.TemplateResponse(
        request, "admin_users.html", {"users": users, "admin": admin, "error": error},
    )


@router.post("/users/{user_id}/toggle")
def toggle_user(request: Request, user_id: int, db: Session = Depends(get_db)):
    admin = _current_admin(request, db)
    if admin is None:
        return RedirectResponse("/admin/login", status_code=303)
    user = db.query(User).filter(User.id == user_id).first()
    if user is not None and user.id != admin.id:
        user.is_active = not user.is_active
        db.commit()
    return RedirectResponse("/admin/users", status_code=303)
```

- [ ] **Step 5: Create admin_login.html**

Create `app/web/templates/admin_login.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Admin Login — Child Growth Monitor</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
  <div class="container" style="max-width: 420px; margin-top: 8vh;">
    <h1 class="h4 mb-4 text-center">Admin Login</h1>
    {% if error %}<div class="alert alert-danger">{{ error }}</div>{% endif %}
    <form method="post" action="/admin/login" class="card card-body shadow-sm">
      <div class="mb-3">
        <label class="form-label" for="username">Username</label>
        <input class="form-control" id="username" name="username" required autofocus>
      </div>
      <div class="mb-3">
        <label class="form-label" for="password">Password</label>
        <input class="form-control" id="password" name="password" type="password" required>
      </div>
      <button class="btn btn-primary w-100" type="submit">Log in</button>
    </form>
  </div>
</body>
</html>
```

- [ ] **Step 6: Create admin_users.html**

Create `app/web/templates/admin_users.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Manage Users — Child Growth Monitor</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
  <nav class="navbar navbar-dark bg-primary">
    <div class="container">
      <span class="navbar-brand">CGM Admin — Users</span>
      <a class="btn btn-outline-light btn-sm" href="/admin/logout">Log out</a>
    </div>
  </nav>
  <main class="container mt-4">
    {% if error %}<div class="alert alert-danger">{{ error }}</div>{% endif %}
    <div class="row">
      <div class="col-md-7">
        <h2 class="h5 mb-3">Accounts</h2>
        <table class="table table-striped">
          <thead><tr><th>Username</th><th>Name</th><th>Role</th><th>Active</th><th></th></tr></thead>
          <tbody>
            {% for u in users %}
            <tr>
              <td>{{ u.username }}</td>
              <td>{{ u.full_name }}</td>
              <td>{{ u.role }}</td>
              <td>{{ "Yes" if u.is_active else "No" }}</td>
              <td>
                {% if u.id != admin.id %}
                <form method="post" action="/admin/users/{{ u.id }}/toggle" class="d-inline">
                  <button class="btn btn-sm btn-outline-secondary" type="submit">
                    {{ "Deactivate" if u.is_active else "Activate" }}
                  </button>
                </form>
                {% endif %}
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <div class="col-md-5">
        <h2 class="h5 mb-3">Create account</h2>
        <form method="post" action="/admin/users/create" class="card card-body">
          <div class="mb-2"><label class="form-label" for="cu">Username</label>
            <input class="form-control" id="cu" name="username" required></div>
          <div class="mb-2"><label class="form-label" for="cf">Full name</label>
            <input class="form-control" id="cf" name="full_name" required></div>
          <div class="mb-2"><label class="form-label" for="cp">Password</label>
            <input class="form-control" id="cp" name="password" type="password" required></div>
          <div class="mb-3"><label class="form-label" for="cr">Role</label>
            <select class="form-select" id="cr" name="role">
              <option value="worker">worker</option>
              <option value="admin">admin</option>
            </select></div>
          <button class="btn btn-primary" type="submit">Create</button>
        </form>
      </div>
    </div>
  </main>
</body>
</html>
```

- [ ] **Step 7: Wire SessionMiddleware + router in main.py**

In `main.py`, add imports at the top:
```python
from starlette.middleware.sessions import SessionMiddleware
from app.web.admin import router as admin_router
from config import JWT_SECRET
```
After `app = FastAPI(...)` is created (inside `create_app`, before `init_db()`), add:
```python
    app.add_middleware(SessionMiddleware, secret_key=JWT_SECRET)
```
Where routers are included (after `app.include_router(web_router)`), add:
```python
    app.include_router(admin_router)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_admin_web.py -v`
Expected: 4 passed

- [ ] **Step 9: Run full backend suite + smoke-start the app**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: all passing
Run: `PYTHONPATH=. .venv/bin/python -c "import main; print('app ok')"`
Expected: `app ok` (no import/wiring errors)

- [ ] **Step 10: Commit**

```bash
git add app/web/admin.py app/web/templates/admin_login.html app/web/templates/admin_users.html main.py requirements.txt tests/test_admin_web.py
git commit -m "feat(backend): add admin web UI for user management"
```

---

## PHASE 4 — FLUTTER AUTH

### Task 11: Add flutter_secure_storage + auth service

**Files:**
- Modify: `flutter_app/pubspec.yaml`
- Create: `flutter_app/lib/services/auth_service.dart`
- Test: `flutter_app/test/auth_service_test.dart`

- [ ] **Step 1: Add dependency**

In `flutter_app/pubspec.yaml`, under `dependencies:` (after `shared_preferences: ^2.3.2`) add:
```yaml
  flutter_secure_storage: ^9.2.2
```
Run: `cd flutter_app && flutter pub get`
Expected: "Got dependencies!"

- [ ] **Step 2: Write the failing test**

Create `flutter_app/test/auth_service_test.dart`:
```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';

void main() {
  group('AuthUser', () {
    test('parses from login json', () {
      final json = {
        'access_token': 'tok123',
        'token_type': 'bearer',
        'user': {'id': 1, 'username': 'asha', 'full_name': 'Asha', 'role': 'worker'},
      };
      final result = AuthLoginResult.fromJson(json);
      expect(result.token, 'tok123');
      expect(result.user.username, 'asha');
      expect(result.user.role, 'worker');
    });
  });
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/auth_service_test.dart`
Expected: FAIL — `Error: Couldn't resolve the package 'child_growth_monitor_app' ... auth_service.dart` (file missing)

- [ ] **Step 4: Create the auth service**

Create `flutter_app/lib/services/auth_service.dart`:
```dart
import 'dart:async';
import 'dart:convert';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;

class AuthUser {
  AuthUser({required this.id, required this.username, required this.fullName, required this.role});

  final int id;
  final String username;
  final String fullName;
  final String role;

  factory AuthUser.fromJson(Map<String, dynamic> json) => AuthUser(
        id: json['id'] as int,
        username: json['username'] as String,
        fullName: json['full_name'] as String,
        role: json['role'] as String,
      );

  Map<String, dynamic> toJson() =>
      {'id': id, 'username': username, 'full_name': fullName, 'role': role};
}

class AuthLoginResult {
  AuthLoginResult({required this.token, required this.user});
  final String token;
  final AuthUser user;

  factory AuthLoginResult.fromJson(Map<String, dynamic> json) => AuthLoginResult(
        token: json['access_token'] as String,
        user: AuthUser.fromJson(json['user'] as Map<String, dynamic>),
      );
}

class AuthException implements Exception {
  AuthException(this.message, {this.statusCode});
  final String message;
  final int? statusCode;
  @override
  String toString() => message;
}

/// Handles login HTTP + secure persistence of the token & user.
class AuthService {
  AuthService({
    required this.baseUrl,
    FlutterSecureStorage? storage,
    http.Client? httpClient,
  })  : _storage = storage ?? const FlutterSecureStorage(),
        _client = httpClient ?? http.Client();

  final String baseUrl;
  final FlutterSecureStorage _storage;
  final http.Client _client;

  static const _kToken = 'auth_token';
  static const _kUser = 'auth_user';
  static const Duration _timeout = Duration(seconds: 30);

  Future<AuthLoginResult> login(String username, String password) async {
    final uri = Uri.parse('$baseUrl/api/v1/auth/login');
    late final http.Response resp;
    try {
      resp = await _client
          .post(uri,
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({'username': username, 'password': password}))
          .timeout(_timeout);
    } on TimeoutException {
      throw AuthException('Login timed out. Check your connection.');
    } on http.ClientException catch (e) {
      throw AuthException('Network error during login: $e');
    }
    if (resp.statusCode == 200) {
      final result = AuthLoginResult.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
      await _persist(result);
      return result;
    }
    if (resp.statusCode == 401) {
      throw AuthException('Invalid username or password', statusCode: 401);
    }
    throw AuthException('Login failed (${resp.statusCode})', statusCode: resp.statusCode);
  }

  Future<void> _persist(AuthLoginResult result) async {
    await _storage.write(key: _kToken, value: result.token);
    await _storage.write(key: _kUser, value: jsonEncode(result.user.toJson()));
  }

  Future<String?> readToken() => _storage.read(key: _kToken);

  Future<AuthUser?> readUser() async {
    final raw = await _storage.read(key: _kUser);
    if (raw == null) return null;
    return AuthUser.fromJson(jsonDecode(raw) as Map<String, dynamic>);
  }

  Future<void> logout() async {
    await _storage.delete(key: _kToken);
    await _storage.delete(key: _kUser);
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/auth_service_test.dart`
Expected: All tests passed!

- [ ] **Step 6: Commit**

```bash
git add flutter_app/pubspec.yaml flutter_app/pubspec.lock flutter_app/lib/services/auth_service.dart flutter_app/test/auth_service_test.dart
git commit -m "feat(flutter): add flutter_secure_storage and AuthService"
```

---

### Task 12: Auth provider (state + notifier)

**Files:**
- Create: `flutter_app/lib/providers/auth_provider.dart`
- Test: `flutter_app/test/auth_provider_test.dart`

- [ ] **Step 1: Write the failing test**

Create `flutter_app/test/auth_provider_test.dart`:
```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';

class _FakeAuthService implements AuthService {
  _FakeAuthService();
  String? _token;
  AuthUser? _user;

  @override
  String get baseUrl => 'http://test';

  @override
  Future<AuthLoginResult> login(String username, String password) async {
    if (password != 'good') throw AuthException('bad', statusCode: 401);
    _token = 'tok';
    _user = AuthUser(id: 1, username: username, fullName: 'X', role: 'worker');
    return AuthLoginResult(token: _token!, user: _user!);
  }

  @override
  Future<String?> readToken() async => _token;
  @override
  Future<AuthUser?> readUser() async => _user;
  @override
  Future<void> logout() async {
    _token = null;
    _user = null;
  }
}

void main() {
  test('initial restore with no token => unauthenticated', () async {
    final container = ProviderContainer(overrides: [
      authServiceProvider.overrideWithValue(_FakeAuthService()),
    ]);
    addTearDown(container.dispose);
    await container.read(authProvider.notifier).restore();
    expect(container.read(authProvider).status, AuthStatus.unauthenticated);
  });

  test('login success => authenticated with user', () async {
    final container = ProviderContainer(overrides: [
      authServiceProvider.overrideWithValue(_FakeAuthService()),
    ]);
    addTearDown(container.dispose);
    await container.read(authProvider.notifier).login('asha', 'good');
    final state = container.read(authProvider);
    expect(state.status, AuthStatus.authenticated);
    expect(state.user?.username, 'asha');
    expect(state.token, 'tok');
  });

  test('login failure keeps unauthenticated and surfaces error', () async {
    final container = ProviderContainer(overrides: [
      authServiceProvider.overrideWithValue(_FakeAuthService()),
    ]);
    addTearDown(container.dispose);
    await expectLater(
      container.read(authProvider.notifier).login('asha', 'bad'),
      throwsA(isA<AuthException>()),
    );
    expect(container.read(authProvider).status, AuthStatus.unauthenticated);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/auth_provider_test.dart`
Expected: FAIL — auth_provider.dart not found

- [ ] **Step 3: Create the auth provider**

Create `flutter_app/lib/providers/auth_provider.dart`:
```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/auth_service.dart';
import 'api_provider.dart';

enum AuthStatus { unknown, unauthenticated, authenticated }

class AuthState {
  const AuthState({this.status = AuthStatus.unknown, this.user, this.token});

  final AuthStatus status;
  final AuthUser? user;
  final String? token;

  AuthState copyWith({AuthStatus? status, AuthUser? user, String? token}) =>
      AuthState(
        status: status ?? this.status,
        user: user ?? this.user,
        token: token ?? this.token,
      );

  static const unauthenticated = AuthState(status: AuthStatus.unauthenticated);
}

/// Overridable so widgets/tests can inject a configured AuthService.
final authServiceProvider = Provider<AuthService>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  return AuthService(baseUrl: effectiveBaseUrl(baseUrl));
});

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier(this._service) : super(const AuthState());

  final AuthService _service;

  /// Loads any cached token/user. Offline-tolerant: presence of a token = authenticated.
  Future<void> restore() async {
    final token = await _service.readToken();
    final user = await _service.readUser();
    if (token != null && user != null) {
      state = AuthState(status: AuthStatus.authenticated, user: user, token: token);
    } else {
      state = AuthState.unauthenticated;
    }
  }

  Future<void> login(String username, String password) async {
    final result = await _service.login(username, password);
    state = AuthState(
      status: AuthStatus.authenticated,
      user: result.user,
      token: result.token,
    );
  }

  Future<void> logout() async {
    await _service.logout();
    state = AuthState.unauthenticated;
  }

  /// Called when the backend rejects the token (401 during sync).
  Future<void> onTokenRejected() async {
    await _service.logout();
    state = AuthState.unauthenticated;
  }
}

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  return AuthNotifier(ref.watch(authServiceProvider));
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/auth_provider_test.dart`
Expected: All tests passed!

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/providers/auth_provider.dart flutter_app/test/auth_provider_test.dart
git commit -m "feat(flutter): add auth provider (state + notifier)"
```

---

### Task 13: Login screen + router auth gate

**Files:**
- Create: `flutter_app/lib/screens/auth/login_screen.dart`
- Modify: `flutter_app/lib/router.dart`
- Modify: `flutter_app/lib/main.dart`
- Test: `flutter_app/test/login_screen_test.dart`

- [ ] **Step 1: Write the failing widget test**

Create `flutter_app/test/login_screen_test.dart`:
```dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:child_growth_monitor_app/screens/auth/login_screen.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';

class _FakeAuthService implements AuthService {
  @override
  String get baseUrl => 'http://test';
  @override
  Future<AuthLoginResult> login(String u, String p) async {
    if (p != 'good') throw AuthException('Invalid username or password', statusCode: 401);
    return AuthLoginResult(token: 't', user: AuthUser(id: 1, username: u, fullName: 'X', role: 'worker'));
  }
  @override
  Future<String?> readToken() async => null;
  @override
  Future<AuthUser?> readUser() async => null;
  @override
  Future<void> logout() async {}
}

void main() {
  testWidgets('shows error on bad login', (tester) async {
    await tester.pumpWidget(ProviderScope(
      overrides: [authServiceProvider.overrideWithValue(_FakeAuthService())],
      child: const MaterialApp(home: LoginScreen()),
    ));
    await tester.enterText(find.byKey(const Key('login_username')), 'asha');
    await tester.enterText(find.byKey(const Key('login_password')), 'bad');
    await tester.tap(find.byKey(const Key('login_submit')));
    await tester.pumpAndSettle();
    expect(find.textContaining('Invalid'), findsOneWidget);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/login_screen_test.dart`
Expected: FAIL — login_screen.dart not found

- [ ] **Step 3: Create the login screen**

Create `flutter_app/lib/screens/auth/login_screen.dart`:
```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/auth_provider.dart';
import '../../services/auth_service.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});

  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _username = TextEditingController();
  final _password = TextEditingController();
  bool _submitting = false;
  String? _error;

  @override
  void dispose() {
    _username.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() {
      _submitting = true;
      _error = null;
    });
    try {
      await ref.read(authProvider.notifier).login(
            _username.text.trim(),
            _password.text,
          );
      // Router redirect handles navigation on auth state change.
    } on AuthException catch (e) {
      setState(() => _error = e.message);
    } catch (e) {
      setState(() => _error = 'Login failed: $e');
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Sign in')),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text('Child Growth Monitor',
                    style: Theme.of(context).textTheme.headlineSmall,
                    textAlign: TextAlign.center),
                const SizedBox(height: 24),
                if (_error != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Text(_error!,
                        style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  ),
                TextField(
                  key: const Key('login_username'),
                  controller: _username,
                  decoration: const InputDecoration(labelText: 'Username', border: OutlineInputBorder()),
                  textInputAction: TextInputAction.next,
                ),
                const SizedBox(height: 16),
                TextField(
                  key: const Key('login_password'),
                  controller: _password,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Password', border: OutlineInputBorder()),
                  onSubmitted: (_) => _submit(),
                ),
                const SizedBox(height: 24),
                FilledButton(
                  key: const Key('login_submit'),
                  onPressed: _submitting ? null : _submit,
                  child: _submitting
                      ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Text('Log in'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/login_screen_test.dart`
Expected: All tests passed!

- [ ] **Step 5: Add auth gate + login route to router**

Replace `flutter_app/lib/router.dart` with (adds a `ref`-aware router factory + redirect):
```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import 'providers/auth_provider.dart';
import 'screens/assessment/assessment_screen.dart';
import 'screens/assessment/result_screen.dart';
import 'screens/auth/login_screen.dart';
import 'screens/children/child_detail_screen.dart';
import 'screens/children/children_list_screen.dart';
import 'screens/child_management/child_form_screen.dart';
import 'screens/child_management/manual_measurement_screen.dart';
import 'screens/settings/settings_screen.dart';

GoRouter buildRouter(Ref ref) {
  return GoRouter(
    initialLocation: '/',
    refreshListenable: _AuthListenable(ref),
    redirect: (context, state) {
      final status = ref.read(authProvider).status;
      final loggingIn = state.matchedLocation == '/login';
      if (status == AuthStatus.unknown) return null; // wait for restore()
      if (status == AuthStatus.unauthenticated) {
        return loggingIn ? null : '/login';
      }
      if (loggingIn) return '/';
      return null;
    },
    routes: [
      GoRoute(path: '/login', builder: (c, s) => const LoginScreen()),
      GoRoute(path: '/', builder: (c, s) => const AssessmentScreen()),
      GoRoute(path: '/result', builder: (c, s) => const ResultScreen()),
      GoRoute(path: '/children', builder: (c, s) => const ChildrenListScreen()),
      GoRoute(
        path: '/children/new',
        builder: (c, s) => const ChildFormScreen(),
      ),
      GoRoute(
        path: '/children/:id',
        builder: (c, s) => ChildDetailScreen(childId: int.parse(s.pathParameters['id']!)),
      ),
      GoRoute(
        path: '/children/:id/edit',
        builder: (c, s) => ChildFormScreen(childId: int.parse(s.pathParameters['id']!)),
      ),
      GoRoute(
        path: '/children/:id/measure',
        builder: (c, s) => ManualMeasurementScreen(childId: int.parse(s.pathParameters['id']!)),
      ),
      GoRoute(path: '/settings', builder: (c, s) => const SettingsScreen()),
    ],
  );
}

/// Bridges Riverpod auth state changes to GoRouter's refresh mechanism.
class _AuthListenable extends ChangeNotifier {
  _AuthListenable(Ref ref) {
    ref.listen(authProvider, (_, __) => notifyListeners());
  }
}

final routerProvider = Provider<GoRouter>((ref) => buildRouter(ref));
```
Add the missing import at the top of router.dart: `import 'package:flutter/foundation.dart';` (for `ChangeNotifier`).

> Note: `ChildFormScreen`, `ManualMeasurementScreen` are created in Tasks 16-17. To keep the app compiling between tasks, create minimal stub files now (Step 6).

- [ ] **Step 6: Create stub screens so the router compiles**

Create `flutter_app/lib/screens/child_management/child_form_screen.dart`:
```dart
import 'package:flutter/material.dart';

class ChildFormScreen extends StatelessWidget {
  const ChildFormScreen({super.key, this.childId});
  final int? childId;

  @override
  Widget build(BuildContext context) =>
      const Scaffold(body: Center(child: Text('Child form (stub)')));
}
```
Create `flutter_app/lib/screens/child_management/manual_measurement_screen.dart`:
```dart
import 'package:flutter/material.dart';

class ManualMeasurementScreen extends StatelessWidget {
  const ManualMeasurementScreen({super.key, required this.childId});
  final int childId;

  @override
  Widget build(BuildContext context) =>
      const Scaffold(body: Center(child: Text('Manual measurement (stub)')));
}
```

- [ ] **Step 7: Wire router + restore() in main.dart**

Replace `flutter_app/lib/main.dart` with:
```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/api_provider.dart';
import 'providers/auth_provider.dart';
import 'providers/sync_provider.dart';
import 'router.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final savedUrl = await loadSavedBaseUrl();
  runApp(
    ProviderScope(
      overrides: [
        baseUrlProvider.overrideWith((ref) => savedUrl),
      ],
      child: const ChildGrowthApp(),
    ),
  );
}

class ChildGrowthApp extends ConsumerStatefulWidget {
  const ChildGrowthApp({super.key});

  @override
  ConsumerState<ChildGrowthApp> createState() => _ChildGrowthAppState();
}

class _ChildGrowthAppState extends ConsumerState<ChildGrowthApp> {
  @override
  void initState() {
    super.initState();
    // Restore cached auth (offline-tolerant), then start sync listener.
    ref.read(authProvider.notifier).restore();
    ref.read(syncTriggerProvider);
  }

  @override
  Widget build(BuildContext context) {
    final router = ref.watch(routerProvider);
    return MaterialApp.router(
      title: 'SNEH Growth Monitor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      routerConfig: router,
    );
  }
}
```

- [ ] **Step 8: Verify analyze + tests pass**

Run: `cd flutter_app && flutter analyze`
Expected: No issues (or only pre-existing warnings)
Run: `cd flutter_app && flutter test test/login_screen_test.dart test/auth_provider_test.dart`
Expected: All tests passed!

- [ ] **Step 9: Commit**

```bash
git add flutter_app/lib/screens/auth/login_screen.dart flutter_app/lib/router.dart flutter_app/lib/main.dart flutter_app/lib/screens/child_management/ flutter_app/test/login_screen_test.dart
git commit -m "feat(flutter): add login screen + router auth gate"
```

---

### Task 14: Attach bearer token to API + sync; surface 401

**Files:**
- Modify: `flutter_app/lib/services/api_service.dart`
- Modify: `flutter_app/lib/providers/api_provider.dart`
- Modify: `flutter_app/lib/services/sync_service.dart`
- Modify: `flutter_app/lib/providers/sync_provider.dart`
- Test: `flutter_app/test/api_service_auth_test.dart`

- [ ] **Step 1: Write the failing test**

Create `flutter_app/test/api_service_auth_test.dart`:
```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/api_service.dart';

void main() {
  test('authToken is included in header builder', () {
    final svc = ApiService(baseUrl: 'http://test', authToken: 'abc');
    expect(svc.authToken, 'abc');
    expect(svc.authHeaders['Authorization'], 'Bearer abc');
  });

  test('no token => empty auth headers', () {
    final svc = ApiService(baseUrl: 'http://test');
    expect(svc.authHeaders.containsKey('Authorization'), false);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/api_service_auth_test.dart`
Expected: FAIL — `authToken`/`authHeaders` not defined

- [ ] **Step 3: Add token support to ApiService**

In `flutter_app/lib/services/api_service.dart`, change the constructor + add a header helper. Replace lines 19-23:
```dart
class ApiService {
  ApiService({required this.baseUrl});

  final String baseUrl;
  static const Duration apiTimeout = Duration(seconds: 60);
```
with:
```dart
class ApiService {
  ApiService({required this.baseUrl, this.authToken});

  final String baseUrl;
  final String? authToken;
  static const Duration apiTimeout = Duration(seconds: 60);

  Map<String, String> get authHeaders =>
      authToken == null ? const {} : {'Authorization': 'Bearer $authToken'};
```

Now add `authHeaders` to the GET calls. Change the `getChildren` request (api_service.dart:57-58):
```dart
      final response =
          await http.get(_uri('/api/v1/children'), headers: authHeaders).timeout(apiTimeout);
```
Change the `getChildDetail` request (api_service.dart:78-79):
```dart
      final response =
          await http.get(_uri('/api/v1/children/$childId'), headers: authHeaders).timeout(apiTimeout);
```
And in `submitAssessment`, after the request is created (api_service.dart:116, before the files are added), add:
```dart
    request.headers.addAll(authHeaders);
```

- [ ] **Step 4: Wire token into apiProvider**

In `flutter_app/lib/providers/api_provider.dart`, change the `apiProvider` (lines 69-72) to read the auth token:
```dart
final apiProvider = Provider<ApiService>((ref) {
  final url = ref.watch(baseUrlProvider);
  final token = ref.watch(authProvider).token;
  return ApiService(baseUrl: effectiveBaseUrl(url), authToken: token);
});
```
Add the import at the top of api_provider.dart:
```dart
import 'auth_provider.dart';
```

> Note: this creates an import edge api_provider → auth_provider, and auth_provider imports api_provider for `baseUrlProvider`/`effectiveBaseUrl`. Dart handles mutual imports between libraries fine (no cycle error) since they reference top-level providers, not each other's private state at load time.

- [ ] **Step 5: Add token + 401 callback to SyncService**

In `flutter_app/lib/services/sync_service.dart`, add to the constructor. Change lines 14-26:
```dart
  SyncService({
    required AppDatabase db,
    required VisitDao visitDao,
    required ChildDao childDao,
    required SyncQueueDao syncDao,
    required String baseUrl,
    String? authToken,
    void Function()? onUnauthorized,
    http.Client? httpClient,
  })  : _db = db,
        _visitDao = visitDao,
        _childDao = childDao,
        _syncDao = syncDao,
        _baseUrl = baseUrl,
        _authToken = authToken,
        _onUnauthorized = onUnauthorized,
        _client = httpClient ?? http.Client();
```
Add fields after `final http.Client _client;` (line 33):
```dart
  final String? _authToken;
  final void Function()? _onUnauthorized;
```
In `_syncOne`, after `final req = http.MultipartRequest('POST', uri);` (line 77) add:
```dart
      if (_authToken != null) {
        req.headers['Authorization'] = 'Bearer $_authToken';
      }
```
Then handle 401 in the response branch. Replace the `else` branch (lines 149-152) with:
```dart
      } else if (response.statusCode == 401) {
        _onUnauthorized?.call();
        await _syncDao.markFailed(entry.id, 'Unauthorized (401) — re-login required');
      } else {
        await _syncDao.markFailed(
            entry.id, 'HTTP ${response.statusCode}: ${response.body}');
      }
```

- [ ] **Step 6: Wire token + callback into syncServiceProvider**

In `flutter_app/lib/providers/sync_provider.dart`, update `syncServiceProvider` (lines 10-19):
```dart
final syncServiceProvider = Provider<SyncService>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  final token = ref.watch(authProvider).token;
  return SyncService(
    db: ref.watch(databaseProvider),
    visitDao: ref.watch(visitDaoProvider),
    childDao: ref.watch(childDaoProvider),
    syncDao: ref.watch(syncQueueDaoProvider),
    baseUrl: effectiveBaseUrl(baseUrl),
    authToken: token,
    onUnauthorized: () => ref.read(authProvider.notifier).onTokenRejected(),
  );
});
```
Add the import at the top of sync_provider.dart:
```dart
import 'auth_provider.dart';
```

- [ ] **Step 7: Run tests + analyze**

Run: `cd flutter_app && flutter test test/api_service_auth_test.dart`
Expected: All tests passed!
Run: `cd flutter_app && flutter analyze`
Expected: No new issues

- [ ] **Step 8: Commit**

```bash
git add flutter_app/lib/services/api_service.dart flutter_app/lib/providers/api_provider.dart flutter_app/lib/services/sync_service.dart flutter_app/lib/providers/sync_provider.dart flutter_app/test/api_service_auth_test.dart
git commit -m "feat(flutter): attach bearer token to API/sync and surface 401"
```

---

## PHASE 5 — FLUTTER DRIFT SCHEMA + CHILD MANAGEMENT

### Task 15: Add Drift columns + schema migration

**Files:**
- Modify: `flutter_app/lib/database/tables/children_table.dart`
- Modify: `flutter_app/lib/database/tables/visits_table.dart`
- Modify: `flutter_app/lib/database/database.dart`
- Regenerate: `flutter_app/lib/database/database.g.dart` (build_runner)
- Test: `flutter_app/test/db_migration_test.dart`

- [ ] **Step 1: Add columns to children table**

In `flutter_app/lib/database/tables/children_table.dart`, add before `createdAt`:
```dart
  IntColumn get ownerUserId => integer().nullable()();
  TextColumn get photoPath => text().nullable()();
  BoolColumn get isArchived => boolean().withDefault(const Constant(false))();
```

- [ ] **Step 2: Add columns to visits table + make imagePath nullable**

In `flutter_app/lib/database/tables/visits_table.dart`, change `imagePath` from `text()()` to nullable (manual visits have no image), and add two columns. Replace the `imagePath` line and add after `notes`:
```dart
  TextColumn get imagePath => text().nullable()();
  IntColumn get ownerUserId => integer().nullable()();
  TextColumn get entryMethod =>
      text().withDefault(const Constant('assessment'))();
```

- [ ] **Step 3: Bump schemaVersion + migration**

In `flutter_app/lib/database/database.dart`, change `schemaVersion` to 3 and extend the migration. Replace the `schemaVersion` getter and `migration` getter:
```dart
  @override
  int get schemaVersion => 3;

  @override
  MigrationStrategy get migration => MigrationStrategy(
        onUpgrade: (migrator, from, to) async {
          if (from < 2) {
            await migrator.deleteTable('sync_queue');
            await migrator.deleteTable('measurements');
            await migrator.deleteTable('visits');
            await migrator.createTable(visits);
            await migrator.createTable(measurements);
            await migrator.createTable(syncQueue);
          }
          if (from < 3) {
            await migrator.addColumn(children, children.ownerUserId);
            await migrator.addColumn(children, children.photoPath);
            await migrator.addColumn(children, children.isArchived);
            await migrator.addColumn(visits, visits.ownerUserId);
            await migrator.addColumn(visits, visits.entryMethod);
          }
        },
      );
```

> Note: `imagePath` changing from non-null to nullable is a relaxation; existing rows already have values, and SQLite stores no NOT NULL constraint difference that breaks reads. For `from < 2` the visits table is recreated fresh anyway.

- [ ] **Step 4: Regenerate Drift code**

Run: `cd flutter_app && dart run build_runner build --delete-conflicting-outputs`
Expected: "Succeeded after ..." with database.g.dart updated

- [ ] **Step 5: Write the migration test**

Create `flutter_app/test/db_migration_test.dart`:
```dart
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/database/database.dart';

void main() {
  test('schema v3 has new columns and inserts work', () async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final id = await db.into(db.children).insert(
          ChildrenCompanion.insert(
            name: 'Kid',
            dateOfBirth: '2024-01-01',
            sex: 'M',
          ),
        );
    final child = await (db.select(db.children)..where((c) => c.id.equals(id))).getSingle();
    expect(child.isArchived, false);
    expect(child.ownerUserId, isNull);
    expect(child.photoPath, isNull);
    await db.close();
  });
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/db_migration_test.dart`
Expected: All tests passed!

- [ ] **Step 7: Commit**

```bash
git add flutter_app/lib/database/ flutter_app/test/db_migration_test.dart
git commit -m "feat(flutter): add owner/photo/archive/entryMethod Drift columns (schema v3)"
```

---

### Task 16: Owner-aware child DAO methods + child form screen

**Files:**
- Modify: `flutter_app/lib/database/daos/child_dao.dart`
- Replace stub: `flutter_app/lib/screens/child_management/child_form_screen.dart`
- Test: `flutter_app/test/child_dao_test.dart`

- [ ] **Step 1: Write the failing DAO test**

Create `flutter_app/test/child_dao_test.dart`:
```dart
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';

void main() {
  late AppDatabase db;
  late ChildDao dao;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    dao = ChildDao(db);
  });
  tearDown(() => db.close());

  test('createChild stores owner + photo', () async {
    final id = await dao.createChild(
      name: 'Kid', dateOfBirth: '2024-01-01', sex: 'M',
      guardianName: 'Mom', location: 'Village', ownerUserId: 7, photoPath: '/p.jpg',
    );
    final child = await dao.getById(id);
    expect(child!.ownerUserId, 7);
    expect(child.photoPath, '/p.jpg');
    expect(child.isArchived, false);
  });

  test('updateChild changes fields', () async {
    final id = await dao.createChild(name: 'Kid', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await dao.updateChild(id: id, name: 'Renamed', guardianName: 'Dad', location: 'Town', photoPath: '/q.jpg');
    final child = await dao.getById(id);
    expect(child!.name, 'Renamed');
    expect(child.guardianName, 'Dad');
    expect(child.photoPath, '/q.jpg');
  });

  test('archive sets isArchived and watchAll(owner) excludes archived', () async {
    final id = await dao.createChild(name: 'Kid', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await dao.setArchived(id, true);
    final child = await dao.getById(id);
    expect(child!.isArchived, true);
    final visible = await dao.watchForOwner(1).first;
    expect(visible.where((c) => c.id == id), isEmpty);
  });

  test('watchForOwner only returns that owner', () async {
    await dao.createChild(name: 'A', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await dao.createChild(name: 'B', dateOfBirth: '2024-01-01', sex: 'F', ownerUserId: 2);
    final forOne = await dao.watchForOwner(1).first;
    expect(forOne.map((c) => c.name), ['A']);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/child_dao_test.dart`
Expected: FAIL — `createChild`/`updateChild`/`setArchived`/`watchForOwner` not defined

- [ ] **Step 3: Add DAO methods**

In `flutter_app/lib/database/daos/child_dao.dart`, add these methods inside the `ChildDao` class (after `findOrCreate`):
```dart
  Future<int> createChild({
    required String name,
    required String dateOfBirth,
    required String sex,
    String? guardianName,
    String? location,
    int? ownerUserId,
    String? photoPath,
  }) {
    return _db.into(_db.children).insert(
          ChildrenCompanion.insert(
            name: name,
            dateOfBirth: dateOfBirth,
            sex: sex,
            guardianName: Value(guardianName),
            location: Value(location),
            ownerUserId: Value(ownerUserId),
            photoPath: Value(photoPath),
          ),
        );
  }

  Future<void> updateChild({
    required int id,
    String? name,
    String? guardianName,
    String? location,
    String? photoPath,
  }) {
    return (_db.update(_db.children)..where((c) => c.id.equals(id))).write(
      ChildrenCompanion(
        name: name == null ? const Value.absent() : Value(name),
        guardianName: Value(guardianName),
        location: Value(location),
        photoPath: photoPath == null ? const Value.absent() : Value(photoPath),
        updatedAt: Value(DateTime.now()),
      ),
    );
  }

  Future<void> setArchived(int id, bool archived) {
    return (_db.update(_db.children)..where((c) => c.id.equals(id))).write(
      ChildrenCompanion(isArchived: Value(archived), updatedAt: Value(DateTime.now())),
    );
  }

  Stream<List<ChildrenData>> watchForOwner(int ownerUserId, {String? search}) {
    final query = _db.select(_db.children)
      ..where((c) => c.ownerUserId.equals(ownerUserId) & c.isArchived.equals(false))
      ..orderBy([(c) => OrderingTerm.desc(c.updatedAt)]);
    if (search != null && search.isNotEmpty) {
      query.where((c) => c.name.like('%$search%'));
    }
    return query.watch();
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/child_dao_test.dart`
Expected: All tests passed!

- [ ] **Step 5: Implement the child form screen**

Replace `flutter_app/lib/screens/child_management/child_form_screen.dart`:
```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';

import '../../providers/auth_provider.dart';
import '../../providers/database_provider.dart';
import '../../services/image_storage_service.dart';

class ChildFormScreen extends ConsumerStatefulWidget {
  const ChildFormScreen({super.key, this.childId});
  final int? childId;

  @override
  ConsumerState<ChildFormScreen> createState() => _ChildFormScreenState();
}

class _ChildFormScreenState extends ConsumerState<ChildFormScreen> {
  final _formKey = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _guardian = TextEditingController();
  final _location = TextEditingController();
  DateTime? _dob;
  String _sex = 'M';
  String? _photoPath;
  bool _loading = false;
  bool _saving = false;

  bool get _isEdit => widget.childId != null;

  @override
  void initState() {
    super.initState();
    if (_isEdit) _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    final child = await ref.read(childDaoProvider).getById(widget.childId!);
    if (child != null) {
      _name.text = child.name;
      _guardian.text = child.guardianName ?? '';
      _location.text = child.location ?? '';
      _dob = DateTime.tryParse(child.dateOfBirth);
      _sex = child.sex;
      _photoPath = child.photoPath;
    }
    if (mounted) setState(() => _loading = false);
  }

  @override
  void dispose() {
    _name.dispose();
    _guardian.dispose();
    _location.dispose();
    super.dispose();
  }

  Future<void> _pickPhoto() async {
    final picked = await ImagePicker().pickImage(source: ImageSource.camera, maxWidth: 1024);
    if (picked == null) return;
    final stored = await ImageStorageService().persist(picked.path);
    setState(() => _photoPath = stored);
  }

  Future<void> _pickDob() async {
    final now = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: _dob ?? DateTime(now.year - 1, now.month, now.day),
      firstDate: DateTime(now.year - 6),
      lastDate: now,
    );
    if (picked != null) setState(() => _dob = picked);
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate() || _dob == null) {
      if (_dob == null) {
        ScaffoldMessenger.of(context)
            .showSnackBar(const SnackBar(content: Text('Please select date of birth')));
      }
      return;
    }
    setState(() => _saving = true);
    final dao = ref.read(childDaoProvider);
    final dobStr = DateFormat('yyyy-MM-dd').format(_dob!);
    final ownerId = ref.read(authProvider).user?.id;
    try {
      if (_isEdit) {
        await dao.updateChild(
          id: widget.childId!,
          name: _name.text.trim(),
          guardianName: _guardian.text.trim().isEmpty ? null : _guardian.text.trim(),
          location: _location.text.trim().isEmpty ? null : _location.text.trim(),
          photoPath: _photoPath,
        );
        if (mounted) context.pop();
      } else {
        final id = await dao.createChild(
          name: _name.text.trim(),
          dateOfBirth: dobStr,
          sex: _sex,
          guardianName: _guardian.text.trim().isEmpty ? null : _guardian.text.trim(),
          location: _location.text.trim().isEmpty ? null : _location.text.trim(),
          ownerUserId: ownerId,
          photoPath: _photoPath,
        );
        if (mounted) context.go('/children/$id');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Save failed: $e')));
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(title: Text(_isEdit ? 'Edit child' : 'New child')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Center(
              child: Column(children: [
                CircleAvatar(
                  radius: 48,
                  backgroundImage: _photoPath != null ? FileImageFromPath(_photoPath!) : null,
                  child: _photoPath == null ? const Icon(Icons.person, size: 48) : null,
                ),
                TextButton.icon(
                  onPressed: _pickPhoto,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Photo'),
                ),
              ]),
            ),
            TextFormField(
              controller: _name,
              decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
              validator: (v) => (v == null || v.trim().isEmpty) ? 'Name is required' : null,
            ),
            const SizedBox(height: 12),
            InputDecorator(
              decoration: const InputDecoration(labelText: 'Date of birth', border: OutlineInputBorder()),
              child: InkWell(
                onTap: _pickDob,
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  child: Text(_dob == null ? 'Select date' : DateFormat('yyyy-MM-dd').format(_dob!)),
                ),
              ),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: _sex,
              decoration: const InputDecoration(labelText: 'Sex', border: OutlineInputBorder()),
              items: const [
                DropdownMenuItem(value: 'M', child: Text('Male')),
                DropdownMenuItem(value: 'F', child: Text('Female')),
              ],
              onChanged: _isEdit ? null : (v) => setState(() => _sex = v ?? 'M'),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _guardian,
              decoration: const InputDecoration(labelText: 'Guardian (optional)', border: OutlineInputBorder()),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _location,
              decoration: const InputDecoration(labelText: 'Location (optional)', border: OutlineInputBorder()),
            ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: _saving ? null : _save,
              child: _saving
                  ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2))
                  : Text(_isEdit ? 'Save changes' : 'Create child'),
            ),
          ],
        ),
      ),
    );
  }
}

/// Helper to build a FileImage from a path without importing dart:io in the widget tree directly.
ImageProvider FileImageFromPath(String path) => _fileImage(path);
```

Add this helper at the bottom of the file (kept separate so the widget body stays import-light):
```dart
// ignore_for_file: non_constant_identifier_names
```
And at the top of the file add the dart:io + FileImage import line and the helper. Replace the import block header by adding:
```dart
import 'dart:io';
```
and define the helper near the bottom:
```dart
ImageProvider _fileImage(String path) => FileImage(File(path));
```

> Note: simplify — instead of the `FileImageFromPath` indirection, use `FileImage(File(_photoPath!))` directly in `backgroundImage`. Replace the `backgroundImage:` line with:
> `backgroundImage: _photoPath != null ? FileImage(File(_photoPath!)) : null,`
> and delete the `FileImageFromPath`/`_fileImage` helpers. Keep only `import 'dart:io';`.

- [ ] **Step 6: Verify analyze + tests**

Run: `cd flutter_app && flutter analyze`
Expected: No issues
Run: `cd flutter_app && flutter test test/child_dao_test.dart`
Expected: All tests passed!

- [ ] **Step 7: Commit**

```bash
git add flutter_app/lib/database/daos/child_dao.dart flutter_app/lib/screens/child_management/child_form_screen.dart flutter_app/test/child_dao_test.dart
git commit -m "feat(flutter): owner-aware child DAO + child profile form"
```

---

### Task 17: Manual measurement entry (DAO + screen + z-score reuse)

**Files:**
- Create: `flutter_app/lib/database/daos/manual_visit_dao.dart`
- Replace stub: `flutter_app/lib/screens/child_management/manual_measurement_screen.dart`
- Test: `flutter_app/test/manual_visit_dao_test.dart`

- [ ] **Step 1: Inspect the nutrition + muac service signatures**

Run: `cd flutter_app && grep -n "class NutritionService\|class MuacService\|class MuacResult\|String classify\|compute\|zscore\|Zscore" lib/services/nutrition_service.dart lib/services/muac_service.dart`
Expected: prints the public method signatures. Read both files fully before Step 3 so the manual screen calls them correctly. (These reuse the existing on-device pipeline; do not duplicate WHO logic.)

> The screen MUST compute HAZ/WHZ via the existing NutritionService and MUAC status via MuacService. If a service method name differs from what this plan assumes, adapt the call — never bypass z-score computation (safety rule).

- [ ] **Step 2: Write the failing DAO test**

Create `flutter_app/test/manual_visit_dao_test.dart`:
```dart
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/manual_visit_dao.dart';

void main() {
  late AppDatabase db;
  late ChildDao childDao;
  late ManualVisitDao dao;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    childDao = ChildDao(db);
    dao = ManualVisitDao(db);
  });
  tearDown(() => db.close());

  test('createManualVisit stores visit (entry_method=manual) + measurement + sync queue', () async {
    final childId = await childDao.createChild(
        name: 'Kid', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 5);
    final visitId = await dao.createManualVisit(
      childId: childId,
      ownerUserId: 5,
      ageMonths: 18.0,
      visitDate: DateTime(2026, 6, 1),
      heightCm: 80.0,
      weightKg: 10.5,
      muacCm: 13.0,
      hazZscore: -1.2,
      whzZscore: -0.5,
      hazStatus: 'Normal',
      whzStatus: 'Normal',
      muacStatus: 'Normal',
      notes: 'monthly visit',
    );

    final visit = await (db.select(db.visits)..where((v) => v.id.equals(visitId))).getSingle();
    expect(visit.entryMethod, 'manual');
    expect(visit.ownerUserId, 5);
    expect(visit.imagePath, isNull);
    expect(visit.notes, 'monthly visit');

    final m = await (db.select(db.measurements)..where((x) => x.visitId.equals(visitId))).getSingle();
    expect(m.manualHeightCm, 80.0);
    expect(m.manualWeightKg, 10.5);
    expect(m.muacCm, 13.0);
    expect(m.muacMethod, 'manual');

    final queued = await db.select(db.syncQueue).get();
    expect(queued.length, 1);
    expect(queued.first.visitId, visitId);
  });
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/manual_visit_dao_test.dart`
Expected: FAIL — manual_visit_dao.dart not found

- [ ] **Step 4: Create the manual visit DAO**

Create `flutter_app/lib/database/daos/manual_visit_dao.dart`:
```dart
import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';
import '../database.dart';

/// Creates a manually-entered visit (no image, no ML) with its measurement and
/// a sync-queue entry, in a single transaction. Mirrors VisitDao but for the
/// manual-entry path (entry_method = 'manual').
class ManualVisitDao {
  ManualVisitDao(this._db);
  final AppDatabase _db;
  static const _uuid = Uuid();

  Future<int> createManualVisit({
    required int childId,
    required int? ownerUserId,
    required double ageMonths,
    required DateTime visitDate,
    required double heightCm,
    required double weightKg,
    double? muacCm,
    double? hazZscore,
    double? whzZscore,
    String? hazStatus,
    String? whzStatus,
    String? muacStatus,
    String? notes,
  }) {
    return _db.transaction(() async {
      final visitId = await _db.into(_db.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: _uuid.v4(),
              ageMonths: ageMonths,
              visitDate: Value(visitDate),
              imagePath: const Value(null),
              notes: Value(notes),
              ownerUserId: Value(ownerUserId),
              entryMethod: const Value('manual'),
            ),
          );
      await _db.into(_db.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: Value(visitId),
              manualHeightCm: Value(heightCm),
              manualWeightKg: Value(weightKg),
              hazZscore: Value(hazZscore),
              whzZscore: Value(whzZscore),
              hazStatus: Value(hazStatus),
              whzStatus: Value(whzStatus),
              muacCm: Value(muacCm),
              muacStatus: Value(muacStatus),
              muacMethod: const Value('manual'),
            ),
          );
      await _db.into(_db.syncQueue).insert(SyncQueueCompanion.insert(visitId: visitId));
      return visitId;
    });
  }
}
```

> Note: `VisitsCompanion.insert` requires `imagePath` only if it is still non-nullable. After Task 15 made it nullable, `imagePath` becomes an optional `Value`; pass `const Value(null)`. If codegen marks it required-nullable, `imagePath: const Value(null)` still satisfies it. `MeasurementsCompanion.insert` requires `visitId` — provided. All other measurement columns are nullable.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd flutter_app && flutter test test/manual_visit_dao_test.dart`
Expected: All tests passed!

- [ ] **Step 6: Add a manualVisitDaoProvider**

In `flutter_app/lib/providers/database_provider.dart`, add:
```dart
import '../database/daos/manual_visit_dao.dart';
```
and after `syncQueueDaoProvider`:
```dart
final manualVisitDaoProvider =
    Provider<ManualVisitDao>((ref) => ManualVisitDao(ref.watch(databaseProvider)));
```

- [ ] **Step 7: Implement the manual measurement screen**

Replace `flutter_app/lib/screens/child_management/manual_measurement_screen.dart`. Read `lib/services/nutrition_service.dart`, `lib/services/muac_service.dart`, and `lib/services/who_data_service.dart` first (Step 1) to match their real method names. Use this structure, substituting the real service calls where marked:
```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';

import '../../providers/auth_provider.dart';
import '../../providers/database_provider.dart';

class ManualMeasurementScreen extends ConsumerStatefulWidget {
  const ManualMeasurementScreen({super.key, required this.childId});
  final int childId;

  @override
  ConsumerState<ManualMeasurementScreen> createState() => _ManualMeasurementScreenState();
}

class _ManualMeasurementScreenState extends ConsumerState<ManualMeasurementScreen> {
  final _formKey = GlobalKey<FormState>();
  final _height = TextEditingController();
  final _weight = TextEditingController();
  final _muac = TextEditingController();
  final _notes = TextEditingController();
  DateTime _visitDate = DateTime.now();
  bool _saving = false;
  String? _error;

  @override
  void dispose() {
    _height.dispose();
    _weight.dispose();
    _muac.dispose();
    _notes.dispose();
    super.dispose();
  }

  Future<void> _pickDate() async {
    final picked = await showDatePicker(
      context: context,
      initialDate: _visitDate,
      firstDate: DateTime(2015),
      lastDate: DateTime.now(),
    );
    if (picked != null) setState(() => _visitDate = picked);
  }

  double _ageMonths(String dob) {
    final birth = DateTime.parse(dob);
    return _visitDate.difference(birth).inDays / 30.4375;
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      final child = await ref.read(childDaoProvider).getById(widget.childId);
      if (child == null) throw Exception('Child not found');
      final ageMonths = _ageMonths(child.dateOfBirth);
      final heightCm = double.parse(_height.text);
      final weightKg = double.parse(_weight.text);
      final muacCm = _muac.text.trim().isEmpty ? null : double.parse(_muac.text);

      // SAFETY: compute WHO z-scores via the existing on-device pipeline.
      // Replace these calls with the real NutritionService / MuacService API
      // discovered in Step 1. Do NOT skip z-score computation.
      final nutrition = await ref.read(nutritionServiceProvider).computeZScores(
            sex: child.sex,
            ageMonths: ageMonths,
            heightCm: heightCm,
            weightKg: weightKg,
          );
      final muacStatus = muacCm == null
          ? null
          : ref.read(muacServiceProvider).classify(muacCm: muacCm, ageMonths: ageMonths);

      final ownerId = ref.read(authProvider).user?.id;
      final visitId = await ref.read(manualVisitDaoProvider).createManualVisit(
            childId: widget.childId,
            ownerUserId: ownerId,
            ageMonths: ageMonths,
            visitDate: _visitDate,
            heightCm: heightCm,
            weightKg: weightKg,
            muacCm: muacCm,
            hazZscore: nutrition.hazZscore,
            whzZscore: nutrition.whzZscore,
            hazStatus: nutrition.hazStatus,
            whzStatus: nutrition.whzStatus,
            muacStatus: muacStatus,
            notes: _notes.text.trim().isEmpty ? null : _notes.text.trim(),
          );
      // Kick a sync attempt opportunistically.
      ref.read(syncServiceProvider).runOnce();
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Measurement saved (visit $visitId)')));
        context.pop();
      }
    } catch (e) {
      // No silent failures (safety rule): surface and do not save partial data.
      setState(() => _error = 'Could not save: $e');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Monthly measurement')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            if (_error != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
              ),
            InputDecorator(
              decoration: const InputDecoration(labelText: 'Visit date', border: OutlineInputBorder()),
              child: InkWell(
                onTap: _pickDate,
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  child: Text(DateFormat('yyyy-MM-dd').format(_visitDate)),
                ),
              ),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _height,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Height (cm)', border: OutlineInputBorder()),
              validator: _positiveNumber,
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _weight,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Weight (kg)', border: OutlineInputBorder()),
              validator: _positiveNumber,
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _muac,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'MUAC (cm, optional)', border: OutlineInputBorder()),
              validator: (v) => (v == null || v.trim().isEmpty) ? null : _positiveNumber(v),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _notes,
              maxLines: 3,
              decoration: const InputDecoration(labelText: 'Notes (optional)', border: OutlineInputBorder()),
            ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: _saving ? null : _save,
              child: _saving
                  ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Text('Save measurement'),
            ),
          ],
        ),
      ),
    );
  }

  String? _positiveNumber(String? v) {
    if (v == null || v.trim().isEmpty) return 'Required';
    final n = double.tryParse(v);
    if (n == null || n <= 0) return 'Enter a positive number';
    return null;
  }
}
```

> IMPORTANT: `nutritionServiceProvider`, `muacServiceProvider`, and the shapes `nutrition.hazZscore/whzZscore/hazStatus/whzStatus` and `muacServiceProvider.classify(...)` are placeholders for the REAL on-device services. In Step 1 you read those files; wire the actual provider names and method signatures here. If no Riverpod provider exists for them yet, instantiate the service directly (e.g. `NutritionService(...)`) exactly as `assessment_service.dart` does — check how the existing assessment flow constructs and calls them, and copy that. Ensure `who_data_service` is loaded if those services need it.

- [ ] **Step 8: Verify analyze + tests**

Run: `cd flutter_app && flutter analyze`
Expected: No issues (resolve any service-name mismatches surfaced here)
Run: `cd flutter_app && flutter test test/manual_visit_dao_test.dart`
Expected: All tests passed!

- [ ] **Step 9: Commit**

```bash
git add flutter_app/lib/database/daos/manual_visit_dao.dart flutter_app/lib/providers/database_provider.dart flutter_app/lib/screens/child_management/manual_measurement_screen.dart flutter_app/test/manual_visit_dao_test.dart
git commit -m "feat(flutter): manual monthly measurement entry with WHO z-score reuse"
```

---

## PHASE 6 — FLUTTER UI INTEGRATION

### Task 18: Children list (owner filter, +new, archive) + detail entry points + sync archive field

**Files:**
- Modify: `flutter_app/lib/screens/children/children_list_screen.dart`
- Modify: `flutter_app/lib/screens/children/child_detail_screen.dart`
- Modify: `flutter_app/lib/services/sync_service.dart` (send entry_method + is_archived + photo)
- Test: `flutter_app/test/children_list_screen_test.dart`

- [ ] **Step 1: Read the two screens fully**

Run: `cd flutter_app && wc -l lib/screens/children/children_list_screen.dart lib/screens/children/child_detail_screen.dart`
Then read both completely. They currently load children/detail from the **API** (`apiProvider`). Decide per existing pattern whether to switch the list to the local `watchForOwner` stream (offline-first) — preferred — or keep API. This plan switches the LIST to local Drift (owner-scoped, works offline); detail keeps its current source but gains action buttons.

- [ ] **Step 2: Write the failing widget test**

Create `flutter_app/test/children_list_screen_test.dart`:
```dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/providers/auth_provider.dart';
import 'package:child_growth_monitor_app/services/auth_service.dart';
import 'package:child_growth_monitor_app/screens/children/children_list_screen.dart';

class _FakeAuth implements AuthService {
  @override
  String get baseUrl => 'http://t';
  @override
  Future<AuthLoginResult> login(String u, String p) async => throw UnimplementedError();
  @override
  Future<String?> readToken() async => 't';
  @override
  Future<AuthUser?> readUser() async => AuthUser(id: 1, username: 'a', fullName: 'A', role: 'worker');
  @override
  Future<void> logout() async {}
}

void main() {
  testWidgets('shows only owner children from local db', (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final dao = ChildDao(db);
    await dao.createChild(name: 'Mine', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await dao.createChild(name: 'Theirs', dateOfBirth: '2024-01-01', sex: 'F', ownerUserId: 2);

    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      authServiceProvider.overrideWithValue(_FakeAuth()),
    ]);
    await container.read(authProvider.notifier).restore();
    addTearDown(() {
      container.dispose();
      db.close();
    });

    await tester.pumpWidget(UncontrolledProviderScope(
      container: container,
      child: const MaterialApp(home: ChildrenListScreen()),
    ));
    await tester.pumpAndSettle();

    expect(find.text('Mine'), findsOneWidget);
    expect(find.text('Theirs'), findsNothing);
  });
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd flutter_app && flutter test test/children_list_screen_test.dart`
Expected: FAIL (current screen uses API, not local owner-scoped stream)

- [ ] **Step 4: Rewrite children_list_screen to use local owner stream + actions**

Replace `flutter_app/lib/screens/children/children_list_screen.dart` with a `ConsumerWidget` that watches `childDaoProvider.watchForOwner(currentUserId)`, shows a FAB to `/children/new`, and a long-press/swipe to archive via `setArchived`. Preserve the existing `AppScaffold` usage if the original used it (check Step 1). Reference implementation:
```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../database/database.dart';
import '../../providers/auth_provider.dart';
import '../../providers/database_provider.dart';

class ChildrenListScreen extends ConsumerWidget {
  const ChildrenListScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userId = ref.watch(authProvider).user?.id;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Children'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            tooltip: 'Log out',
            onPressed: () => ref.read(authProvider.notifier).logout(),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => context.go('/children/new'),
        icon: const Icon(Icons.add),
        label: const Text('New child'),
      ),
      body: userId == null
          ? const Center(child: Text('Not signed in'))
          : StreamBuilder<List<ChildrenData>>(
              stream: ref.watch(childDaoProvider).watchForOwner(userId),
              builder: (context, snapshot) {
                if (!snapshot.hasData) {
                  return const Center(child: CircularProgressIndicator());
                }
                final children = snapshot.data!;
                if (children.isEmpty) {
                  return const Center(child: Text('No children yet. Tap "New child" to add one.'));
                }
                return ListView.separated(
                  itemCount: children.length,
                  separatorBuilder: (_, __) => const Divider(height: 1),
                  itemBuilder: (context, i) {
                    final c = children[i];
                    return Dismissible(
                      key: ValueKey('child_${c.id}'),
                      direction: DismissDirection.endToStart,
                      background: Container(
                        color: Colors.red,
                        alignment: Alignment.centerRight,
                        padding: const EdgeInsets.only(right: 16),
                        child: const Icon(Icons.archive, color: Colors.white),
                      ),
                      confirmDismiss: (_) async {
                        return await showDialog<bool>(
                              context: context,
                              builder: (ctx) => AlertDialog(
                                title: const Text('Archive child?'),
                                content: Text('Archive ${c.name}? This hides them from your list.'),
                                actions: [
                                  TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
                                  FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Archive')),
                                ],
                              ),
                            ) ??
                            false;
                      },
                      onDismissed: (_) => ref.read(childDaoProvider).setArchived(c.id, true),
                      child: ListTile(
                        leading: const CircleAvatar(child: Icon(Icons.person)),
                        title: Text(c.name),
                        subtitle: Text('DOB ${c.dateOfBirth} · ${c.sex == "M" ? "Male" : "Female"}'),
                        trailing: const Icon(Icons.chevron_right),
                        onTap: () => context.go('/children/${c.id}'),
                      ),
                    );
                  },
                );
              },
            ),
    );
  }
}
```
(If the original used `AppScaffold` for the bottom nav, wrap the body in it instead of a bare `Scaffold`, matching the original. Confirm via Step 1.)

- [ ] **Step 5: Add detail-screen action buttons**

In `flutter_app/lib/screens/children/child_detail_screen.dart`, add two actions that navigate to the new routes. In the screen's `AppBar.actions` (or as buttons in the body if there's no AppBar), add:
```dart
          IconButton(
            icon: const Icon(Icons.edit),
            tooltip: 'Edit profile',
            onPressed: () => context.go('/children/${widget.childId}/edit'),
          ),
          IconButton(
            icon: const Icon(Icons.add_chart),
            tooltip: 'Add measurement',
            onPressed: () => context.go('/children/${widget.childId}/measure'),
          ),
```
Ensure `import 'package:go_router/go_router.dart';` is present. If the screen is a `ConsumerWidget` without `widget.childId`, use the local `childId` field name as defined in that file (check Step 1).

- [ ] **Step 6: Send new fields in sync_service**

In `flutter_app/lib/services/sync_service.dart`, add `entry_method` and `is_archived` to the `req.fields.addAll({...})` map (after the `'visit_date'` line):
```dart
        'entry_method': pair.visit.entryMethod,
        'is_archived': child.isArchived.toString(),
```
And send the child photo if present, after the back-image block (after sync_service.dart:139):
```dart
      if (child.photoPath != null && await File(child.photoPath!).exists()) {
        req.files.add(await http.MultipartFile.fromPath('photo', child.photoPath!));
      }
```

> Manual visits have `imagePath == null`. The backend still requires `image`. Guard: if `pair.visit.imagePath == null`, attach a tiny placeholder. After the existing image block (sync_service.dart:126-129), change it to handle null:
```dart
      if (pair.visit.imagePath != null && await File(pair.visit.imagePath!).exists()) {
        req.files.add(await http.MultipartFile.fromPath('image', pair.visit.imagePath!));
      } else {
        // Manual entry has no photo; backend requires the 'image' part.
        req.files.add(http.MultipartFile.fromBytes(
            'image', _placeholderJpeg, filename: 'manual.jpg'));
      }
```
Add this constant at the top of the `SyncService` class (after `static const _maxRetries = 5;`):
```dart
  // Minimal 1x1 JPEG so manual (image-less) visits satisfy the backend's required 'image' field.
  static final List<int> _placeholderJpeg = [
    0xFF, 0xD8, 0xFF, 0xD9
  ];
```
> Note: the existing image block reads `pair.visit.imagePath` as non-null; after Task 15 it's nullable, so this null-guard is required for the code to compile.

- [ ] **Step 7: Run tests + analyze**

Run: `cd flutter_app && flutter test test/children_list_screen_test.dart`
Expected: All tests passed!
Run: `cd flutter_app && flutter analyze`
Expected: No issues

- [ ] **Step 8: Commit**

```bash
git add flutter_app/lib/screens/children/ flutter_app/lib/services/sync_service.dart flutter_app/test/children_list_screen_test.dart
git commit -m "feat(flutter): owner-scoped children list with new/archive + detail actions + sync new fields"
```

---

## PHASE 7 — VERIFICATION

### Task 19: Full backend suite + Flutter suite + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the complete backend test suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: all tests pass. If any pre-existing test broke due to the auth requirement on `/children` or `/sync`, update that test to pass a token (do not weaken auth).

- [ ] **Step 2: Run the complete Flutter test suite**

Run: `cd flutter_app && flutter test`
Expected: All tests passed!

- [ ] **Step 3: Flutter analyze (whole project)**

Run: `cd flutter_app && flutter analyze`
Expected: No issues found

- [ ] **Step 4: Backend smoke — seed an admin + worker, start server**

Run:
```bash
PYTHONPATH=. .venv/bin/python scripts/create_user.py --username admin --full-name "Site Admin" --role admin
```
(enter a password when prompted)
Expected: `Created admin 'admin' (id=1)`

Run: `PYTHONPATH=. .venv/bin/python -c "import main; print('wired ok')"`
Expected: `wired ok`

- [ ] **Step 5: Manual end-to-end checklist (document results in the commit/PR)**

Confirm by reasoning over the code + tests (no device needed for backend):
- `/admin/login` rejects non-admins; admin can create a worker (covered by `tests/test_admin_web.py`).
- `/api/v1/auth/login` returns a token; `/api/v1/children` is 401 without it and owner-scoped with it (covered by tests).
- `/api/v1/sync` requires a token and stamps `user_id` + `entry_method` (covered by tests).
- Migration adds columns idempotently to a legacy DB (covered by `tests/test_migration.py`).

- [ ] **Step 6: Final commit (docs / any cleanup)**

```bash
git add -A
git commit -m "test: verify login + child management end-to-end"
```

---

## Self-Review Notes

- **Spec coverage:** auth model (T1-T7, T11-T14), per-worker ownership (T4, T8, T9, T16), admin-created accounts + web UI + CLI seed (T7, T10), profile+photo CRUD (T16, child_form), manual monthly entry with notes+backfill date (T17), growth timeline entry points (T18), archive/delete (T16 setArchived, T18 UI, T9/T18 sync field), offline auth persistence (T12 restore, T14 secure storage + 30-day token), migration safety (T5 backend, T15 Flutter). All spec sections map to tasks.
- **Safety rules preserved:** manual entry routes through NutritionService/MuacService (T17 Steps 1 & 7 enforce no z-score bypass; "no silent failures" via surfaced errors). ML untouched. Manual measurements stored in manual_* columns (priority preserved by existing chart/priority logic).
- **Type consistency:** `watchForOwner`, `createChild`, `updateChild`, `setArchived` (child_dao) used consistently across T16/T18. `createManualVisit` signature identical in T17 DAO + test. `authToken`/`authHeaders` (T14) consistent. `entryMethod`/`isArchived`/`ownerUserId`/`photoPath` column names consistent backend↔Flutter.
- **Known adaptation point:** T17 Step 7 requires reading the real NutritionService/MuacService API and wiring actual calls — flagged explicitly with fallback instructions (instantiate as assessment_service.dart does). This is the one place the executor must verify names against the codebase rather than copy verbatim.
