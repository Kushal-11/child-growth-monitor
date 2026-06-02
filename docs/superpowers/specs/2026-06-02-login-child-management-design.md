# Login + Child Management System — Design

**Date:** 2026-06-02
**Status:** Approved (design phase)
**Author:** Brainstormed with user

## Summary

Add two capabilities to the Child Growth Monitor app:

1. **Authentication** — JWT-based, admin-provisioned accounts. Health workers log
   in once while online; credentials are cached securely on-device for indefinite
   offline use. Every child and visit is owned by the worker who created it.
2. **Child Management** — a dedicated CRUD module, separate from the camera/ML
   assessment flow: create/edit child profiles with photos, manually enter monthly
   measurements (height / weight / MUAC + notes + backfillable date), view a
   per-child growth timeline, and archive/delete children.

Both layer onto the existing offline-first architecture (Drift local DB → idempotent
`/api/v1/sync`). The camera/ML assessment pipeline and the ML SAM-recall floor are
untouched.

## Decisions (from brainstorming)

| Topic | Decision |
|-------|----------|
| Auth model | Local-first + server sync. JWT issued by backend, cached on-device. |
| Management scope | All four: profile+photo CRUD, manual monthly entry, growth timeline, archive/delete. |
| Manual entry fields | Core (height, weight, MUAC) + visit notes + editable/backfillable visit date. No head circumference, no edema. |
| Data ownership | Per-worker. Workers see/sync only their own children. |
| Registration | Admin-created accounts only (no public signup). |
| Admin tooling | Web admin page (in existing Jinja2 UI) + CLI seed script for first admin. |
| Offline auth | Stay logged in after first online login (30-day token). Re-login prompted only when backend rejects an expired token during sync — never strands a worker offline. |

## Architecture

Two new concerns layered on the existing system:

- **Backend**: new `User` model + `auth_service` (bcrypt hashing, JWT). `user_id`
  FK added to `children` and `visits`. Auth routes (`/api/v1/auth/login`,
  `/api/v1/auth/me`). Existing `/api/v1/sync` becomes auth-protected and stamps
  records with the authenticated user. Admin web UI section (`/admin/...`) + CLI
  seed script.
- **Flutter**: new `auth` feature (login screen, auth provider, secure token
  storage via `flutter_secure_storage`). New `child_management` screens. API
  service attaches the bearer token to all calls. Router gates everything behind a
  login check. Drift gets profile-photo path + owner column + `entry_method`.

**Safety preservation:** Manual measurements still flow through WHO z-score
validation (`NutritionService` → HAZ/WHZ) and MUAC WHO thresholds. Manual
measurements take priority per the existing weight-priority rule. No assessment
path bypasses z-score computation. "No silent failures" — a record that fails
z-score computation surfaces the error and is not saved partially.

## Data Model Changes

### Backend (SQLAlchemy)

**New — `User`** (`app/models/user.py`):
```python
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="worker")   # "worker" | "admin"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    children = relationship("Child", back_populates="owner")
```

**Modified — `Child`** (add):
```python
user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
photo_path = Column(String(500), nullable=True)
is_archived = Column(Boolean, default=False)
owner = relationship("User", back_populates="children")
```

**Modified — `Visit`** (add):
```python
user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
entry_method = Column(String(20), default="assessment")  # "assessment" | "manual"
```

`user_id` is nullable so existing rows do not break (legacy rows = unowned,
visible to admins only). `entry_method` distinguishes camera/ML visits from
manually-typed ones.

### Flutter (Drift)

Mirror the backend: add `photoPath`, `ownerUserId`, `isArchived` to `Children`;
add `ownerUserId`, `entryMethod` to `Visits`. Requires a Drift **schema version
bump + migration** (DB currently has no migration strategy; one will be added).
Auth token/user is **not** in Drift — it lives in `flutter_secure_storage`.

### Sync contract (`POST /api/v1/sync`)

- Requires `Authorization: Bearer <token>`; the endpoint derives `user_id` from
  the token, not the request body.
- New optional multipart fields: `photo` (file), `entry_method`, `is_archived`.
- Archived/deleted children propagate so the backend can mark them archived.
- Remains idempotent by `local_uuid`. Backward-compatible: a request without the
  new fields still works.

### Migration safety

Plain SQLite, no Alembic. A one-time idempotent startup check runs `ALTER TABLE`
to add the new columns if missing. Existing children/visits get `user_id = NULL`.

## Authentication Flow

### Backend
- `auth_service.py` — bcrypt via `passlib`, JWT via `python-jose`. 30-day token
  expiry. A `get_current_user` FastAPI dependency decodes the bearer token, loads
  the active user, raises 401 if invalid/inactive.
- Routes: `POST /api/v1/auth/login` (username+password → `{access_token, user}`),
  `GET /api/v1/auth/me`. No public signup.
- All data routes (`/children`, `/sync`, `/assess`) gain `get_current_user` and
  filter/stamp by `user_id`.

### Flutter
- `auth_provider.dart` — `AuthState` (unauthenticated / authenticated / loading).
  On app start, reads token from secure storage; if present, proceeds without
  blocking on `/me` when offline.
- `login_screen.dart` — username + password, friendly errors.
- Router gate: no cached token → `/login`. Token present → all routes unlock and
  persist across restarts.
- `api_service.dart` — bearer token on every request. A `401` during sync flips
  auth state to "needs re-login," surfaced next time online (never strands a
  worker offline).

### Admin tooling
- CLI: `scripts/create_user.py --username … --role admin` — seeds the first admin.
- Web UI: login-protected `/admin/users` in the existing Jinja2 app — list,
  create (worker/admin), deactivate. Reuses `base.html` styling. Web admin login
  uses a session cookie (separate from the mobile JWT flow).

## Child Management UI & Flow (Flutter)

Feature-first under `lib/screens/child_management/` + matching providers.

1. **Child profile form** (`child_form_screen.dart`) — create/edit: name, DOB,
   sex, guardian, location, profile photo (`image_picker` → `ImageStorageService`,
   synced as `photo` field). Save → Drift + sync queue entry, stamped with
   current `user_id`.
2. **Manual measurement form** (`manual_measurement_screen.dart`) — for an
   existing child: editable visit date (defaults today, allows backfill), height
   (cm), weight (kg), MUAC (cm), notes. On save runs through existing on-device
   pipeline: `WhoDataService` + `NutritionService` → HAZ/WHZ + status;
   `MuacService` → MUAC status. Stored as a `Visit` with `entry_method="manual"`,
   manual measurements taking priority. No camera, no ML.
3. **Enhanced child detail / growth timeline** (extend `child_detail_screen.dart`)
   — profile header with photo, existing growth chart, chronological visit list
   (manual + assessment) badged by entry method, each tappable. Entry points to
   "Add monthly measurement" and "Edit profile."
4. **Children list** (extend `children_list_screen.dart`) — filtered to the
   logged-in worker, "+ New child" button, archived filter, swipe/long-press to
   archive/delete with confirmation; deletion propagates on next sync.

**Reuse, not duplication:** manual entry reuses `WhoDataService` /
`NutritionService` / `MuacService` and existing Drift DAOs + sync queue. The only
genuinely new service code is auth and profile-photo handling.

## Error Handling

- z-score computation failure (e.g., age out of WHO range) → form surfaces the
  error, refuses to save a partial record.
- Sync failures stay in the retry queue, visible in the existing sync-status UI.
- 401 on sync → "needs re-login" surfaced when online; offline access preserved.

## Testing

- **Backend (pytest + TestClient):** auth login, token expiry, ownership
  filtering, admin user CRUD, sync stamps user_id, migration idempotency.
- **Flutter:** widget tests for login + forms; unit tests for auth provider and
  manual-measurement provider (z-scores compute, manual priority holds).
- **ML:** untouched — manual entry does not run the ML model; SAM-recall floor
  unaffected.

## Out of Scope (YAGNI)

- Public self-service signup, email verification.
- Head circumference, edema, and other extra measurement fields.
- Role hierarchies beyond worker/admin.
- Password reset flow (admin can deactivate + recreate for now).
