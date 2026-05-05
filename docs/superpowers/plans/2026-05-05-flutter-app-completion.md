# Flutter App Completion — Offline-First MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Flutter app so a field worker can run a full WHO-grade child growth assessment fully offline, see the result immediately, and have it auto-sync to the FastAPI backend when connectivity returns. Ship as a sideloadable Android APK.

**Architecture:** Wire the existing on-device pose/WHO/nutrition services into the assessment screen, add the missing `MeasurementService` + `MlInferenceService` + `AssessmentService` + `SyncService`, switch list/detail screens from API to local Drift DB, bundle TFLite models, and add a thin idempotent `POST /api/v1/sync` backend endpoint with `local_uuid` deduplication.

**Tech Stack:** Flutter 3.3+, Dart 3, Riverpod, Drift (SQLite), `tflite_flutter`, `google_mlkit_pose_detection`, `connectivity_plus`, `uuid`, `path_provider`. Backend: FastAPI, SQLAlchemy.

**Spec:** `docs/superpowers/specs/2026-05-05-flutter-app-completion-design.md`

---

## File Structure

### New backend files
- `app/api/sync.py` — `POST /api/v1/sync` route handler (idempotent by `local_uuid`)
- `tests/test_sync.py` — pytest covering happy path, dedup, validation
- `scripts/migrate_add_local_uuid.py` — one-shot SQLite migration adding `local_uuid` column

### Modified backend files
- `app/models/visit.py` — add `local_uuid` column
- `main.py` — register the new sync router

### New Flutter files
- `flutter_app/lib/services/measurement_service.dart`
- `flutter_app/lib/services/ml_inference_service.dart`
- `flutter_app/lib/services/assessment_service.dart`
- `flutter_app/lib/services/image_storage_service.dart`
- `flutter_app/lib/services/sync_service.dart`
- `flutter_app/lib/providers/database_provider.dart`
- `flutter_app/lib/providers/assessment_service_provider.dart`
- `flutter_app/lib/providers/sync_provider.dart`
- `flutter_app/lib/models/body_measurements.dart` — extend existing file with `BodyMeasurements` class (the typed result of `MeasurementService`)
- `flutter_app/test/services/measurement_service_test.dart`
- `flutter_app/test/services/ml_inference_service_test.dart`
- `flutter_app/test/services/assessment_service_test.dart`
- `flutter_app/test/services/image_storage_service_test.dart`
- `flutter_app/test/services/sync_service_test.dart`
- `flutter_app/scripts/export_scaler.py` — one-shot Python script that converts `data/models/feature_scaler.pkl` → JSON
- `flutter_app/assets/models/weight_estimator.tflite` — copied from `data/models/`
- `flutter_app/assets/models/wasting_classifier.tflite` — copied from `data/models/`
- `flutter_app/assets/models/feature_scaler.json` — generated from the pkl

### Modified Flutter files
- `flutter_app/pubspec.yaml` — add `uuid` package
- `flutter_app/lib/database/tables/visits_table.dart` — add `localUuid`
- `flutter_app/lib/database/database.dart` — bump `schemaVersion` to 2 + migration
- `flutter_app/lib/database/daos/visit_dao.dart` — generate UUID at insert
- `flutter_app/lib/main.dart` — initialize DB + start SyncService listener
- `flutter_app/lib/screens/assessment/assessment_screen.dart` — call local AssessmentService
- `flutter_app/lib/screens/children/children_list_screen.dart` — Drift stream
- `flutter_app/lib/screens/children/child_detail_screen.dart` — Drift stream
- `flutter_app/lib/screens/settings/settings_screen.dart` — Storage row + Sync Now
- `flutter_app/lib/screens/shared/app_scaffold.dart` — sync status icon + badge
- `flutter_app/lib/screens/assessment/result_screen.dart` — fallback indicator
- `flutter_app/lib/providers/api_provider.dart` — keep, only used by SyncService now
- `flutter_app/lib/providers/children_provider.dart` — switch to DAO streams
- `flutter_app/lib/l10n/translations.dart` — add new strings

---

## Task 1: Backend — add `local_uuid` column to `visits`

**Files:**
- Modify: `app/models/visit.py`
- Create: `scripts/migrate_add_local_uuid.py`
- Test: `tests/test_visit_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_visit_model.py`:

```python
"""Visit model schema tests."""
from datetime import datetime

from app.models.database import Base, engine
from app.models.child import Child
from app.models.visit import Visit


def setup_module(_module):
    Base.metadata.create_all(bind=engine)


def test_visit_has_local_uuid_column():
    columns = {c.name for c in Visit.__table__.columns}
    assert "local_uuid" in columns


def test_local_uuid_is_unique_nullable():
    col = Visit.__table__.columns["local_uuid"]
    assert col.unique is True
    assert col.nullable is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_visit_model.py -v
```
Expected: FAIL with `assert 'local_uuid' in columns`.

- [ ] **Step 3: Add the column to the model**

Edit `app/models/visit.py` — add the `local_uuid` column after `notes`:

```python
"""Visit model representing a single assessment visit."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.models.database import Base


class Visit(Base):
    __tablename__ = "visits"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("children.id"), nullable=False)
    visit_date = Column(DateTime, default=datetime.utcnow)
    age_months = Column(Float, nullable=False)
    image_path = Column(String(500), nullable=True)
    notes = Column(Text, nullable=True)
    local_uuid = Column(String(36), unique=True, nullable=True, index=True)

    child = relationship("Child", back_populates="visits")
    measurement = relationship(
        "MeasurementResult",
        back_populates="visit",
        uselist=False,
        cascade="all, delete-orphan",
    )
```

- [ ] **Step 4: Write the migration script**

Create `scripts/migrate_add_local_uuid.py`:

```python
"""One-shot migration: add local_uuid TEXT UNIQUE column to visits table.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/migrate_add_local_uuid.py
"""
import sqlite3
from pathlib import Path

from config import DATABASE_URL


def main() -> None:
    if not DATABASE_URL.startswith("sqlite:///"):
        raise SystemExit(f"Only sqlite databases supported: {DATABASE_URL}")
    db_path = Path(DATABASE_URL.replace("sqlite:///", "", 1))
    if not db_path.exists():
        print(f"Database does not exist yet at {db_path}; nothing to migrate.")
        return

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(visits)")
        cols = {row[1] for row in cur.fetchall()}
        if "local_uuid" in cols:
            print("local_uuid already exists; nothing to do.")
            return
        cur.execute("ALTER TABLE visits ADD COLUMN local_uuid TEXT")
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_visits_local_uuid "
            "ON visits(local_uuid) WHERE local_uuid IS NOT NULL"
        )
        conn.commit()
        print("Added local_uuid column and unique index.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run model test to verify it passes**

Delete the auto-created test DB so the new schema picks up cleanly, then re-run:

```bash
rm -f child_growth.db
PYTHONPATH=. .venv/bin/python -m pytest tests/test_visit_model.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Verify the full backend test suite still passes**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```
Expected: all tests pass (the new column is nullable so existing assertions stay valid).

- [ ] **Step 7: Commit**

```bash
git add app/models/visit.py scripts/migrate_add_local_uuid.py tests/test_visit_model.py
git commit -m "feat(backend): add local_uuid column to visits for sync dedup"
```

---

## Task 2: Backend — `POST /api/v1/sync` endpoint

**Files:**
- Create: `app/api/sync.py`
- Modify: `main.py` (register the router)
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write the failing happy-path test**

Create `tests/test_sync.py`:

```python
"""Tests for POST /api/v1/sync — idempotent ingestion of mobile assessments."""
import io
import uuid

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _payload():
    return {
        "local_uuid": str(uuid.uuid4()),
        "child_name": "Test Child",
        "date_of_birth": "2024-01-01",
        "sex": "M",
        "age_months": "16.0",
        "visit_date": "2026-05-05T10:00:00",
        "predicted_height_cm": "78.0",
        "predicted_weight_kg": "9.5",
        "haz_zscore": "-1.0",
        "whz_zscore": "-0.5",
        "haz_status": "Normal",
        "whz_status": "Normal",
        "muac_cm": "14.0",
        "muac_status": "Normal",
        "muac_method": "estimated_from_whz",
        "ml_wasting_status": "Normal",
        "ml_estimated_weight_kg": "9.4",
        "confidence_score": "0.85",
        "body_build": "average",
        "side_view_used": "false",
        "sam_probability": "0.02",
        "mam_probability": "0.10",
        "normal_probability": "0.85",
        "risk_probability": "0.02",
        "overweight_probability": "0.01",
    }


def _file():
    return ("front.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")


def test_sync_happy_path_returns_synced():
    body = _payload()
    response = client.post(
        "/api/v1/sync",
        data=body,
        files={"image": _file()},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "synced"
    assert isinstance(data["server_visit_id"], int)


def test_sync_same_local_uuid_twice_is_idempotent():
    body = _payload()
    first = client.post("/api/v1/sync", data=body, files={"image": _file()})
    assert first.status_code == 200
    first_id = first.json()["server_visit_id"]

    second = client.post("/api/v1/sync", data=body, files={"image": _file()})
    assert second.status_code == 200
    assert second.json()["status"] == "already_synced"
    assert second.json()["server_visit_id"] == first_id


def test_sync_missing_required_field_returns_422():
    body = _payload()
    del body["local_uuid"]
    response = client.post("/api/v1/sync", data=body, files={"image": _file()})
    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_sync.py -v
```
Expected: FAIL with 404 (endpoint does not exist).

- [ ] **Step 3: Implement the sync route**

Create `app/api/sync.py`:

```python
"""POST /api/v1/sync — idempotent ingestion of mobile-computed assessments.

Mobile clients run the full assessment on-device, then upload the result here.
The server skips ML, dedups by local_uuid, and stores the image + measurement.
"""
import shutil
import uuid as uuid_lib
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.database import get_db
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from config import UPLOAD_DIR

router = APIRouter(prefix="/api/v1", tags=["Sync"])


def _save_upload(upload: UploadFile) -> str:
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = f"{uuid_lib.uuid4().hex}_{upload.filename or 'image.jpg'}"
    path = UPLOAD_DIR / filename
    with open(path, "wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return str(path)


@router.post("/sync")
async def sync_assessment(
    image: UploadFile = File(...),
    image_side: Optional[UploadFile] = File(None),
    image_back: Optional[UploadFile] = File(None),
    local_uuid: str = Form(...),
    child_name: str = Form(...),
    date_of_birth: str = Form(...),
    sex: str = Form(...),
    age_months: float = Form(...),
    visit_date: str = Form(...),
    predicted_height_cm: Optional[float] = Form(None),
    predicted_weight_kg: Optional[float] = Form(None),
    manual_height_cm: Optional[float] = Form(None),
    manual_weight_kg: Optional[float] = Form(None),
    haz_zscore: Optional[float] = Form(None),
    whz_zscore: Optional[float] = Form(None),
    haz_status: Optional[str] = Form(None),
    whz_status: Optional[str] = Form(None),
    confidence_score: Optional[float] = Form(None),
    body_build: Optional[str] = Form(None),
    side_view_used: str = Form("false"),
    chest_depth_cm: Optional[float] = Form(None),
    abd_depth_cm: Optional[float] = Form(None),
    ml_estimated_weight_kg: Optional[float] = Form(None),
    ml_wasting_status: Optional[str] = Form(None),
    sam_probability: Optional[float] = Form(None),
    mam_probability: Optional[float] = Form(None),
    normal_probability: Optional[float] = Form(None),
    risk_probability: Optional[float] = Form(None),
    overweight_probability: Optional[float] = Form(None),
    muac_cm: Optional[float] = Form(None),
    muac_status: Optional[str] = Form(None),
    muac_method: Optional[str] = Form(None),
    guardian_name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    if sex not in ("M", "F"):
        raise HTTPException(400, "sex must be 'M' or 'F'")

    try:
        dob = date.fromisoformat(date_of_birth)
    except ValueError:
        raise HTTPException(400, "date_of_birth must be ISO format (YYYY-MM-DD)")

    try:
        visit_dt = datetime.fromisoformat(visit_date)
    except ValueError:
        raise HTTPException(400, "visit_date must be ISO format")

    existing = db.query(Visit).filter(Visit.local_uuid == local_uuid).first()
    if existing is not None:
        return {"server_visit_id": existing.id, "status": "already_synced"}

    image_path = _save_upload(image)
    if image_side is not None:
        _save_upload(image_side)
    if image_back is not None:
        _save_upload(image_back)

    child = (
        db.query(Child)
        .filter(
            Child.name == child_name,
            Child.date_of_birth == dob,
            Child.sex == sex,
        )
        .first()
    )
    if child is None:
        child = Child(
            name=child_name,
            date_of_birth=dob,
            sex=sex,
            guardian_name=guardian_name,
            location=location,
        )
        db.add(child)
        db.flush()

    visit = Visit(
        child_id=child.id,
        visit_date=visit_dt,
        age_months=age_months,
        image_path=image_path,
        local_uuid=local_uuid,
    )
    db.add(visit)
    db.flush()

    measurement = MeasurementResult(
        visit_id=visit.id,
        predicted_height_cm=predicted_height_cm,
        predicted_weight_kg=predicted_weight_kg,
        manual_height_cm=manual_height_cm,
        manual_weight_kg=manual_weight_kg,
        haz_zscore=haz_zscore,
        whz_zscore=whz_zscore,
        haz_status=haz_status,
        whz_status=whz_status,
        confidence_score=confidence_score,
    )
    db.add(measurement)
    db.commit()

    return {"server_visit_id": visit.id, "status": "synced"}
```

- [ ] **Step 4: Register the router**

Edit `main.py` — add the import and `include_router` call. The diff is two lines:

```python
from app.api.routes import router as api_router
from app.api.sync import router as sync_router  # NEW
...
    app.include_router(api_router)
    app.include_router(sync_router)  # NEW
    app.include_router(web_router)
```

- [ ] **Step 5: Run the sync tests to verify they pass**

```bash
rm -f child_growth.db
PYTHONPATH=. .venv/bin/python -m pytest tests/test_sync.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Verify full backend test suite still passes**

```bash
rm -f child_growth.db
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/api/sync.py main.py tests/test_sync.py
git commit -m "feat(backend): add idempotent POST /api/v1/sync endpoint for mobile uploads"
```

---

## Task 3: Flutter — add `localUuid` to `Visits` table

**Files:**
- Modify: `flutter_app/pubspec.yaml`
- Modify: `flutter_app/lib/database/tables/visits_table.dart`
- Modify: `flutter_app/lib/database/database.dart`
- Modify: `flutter_app/lib/database/daos/visit_dao.dart`
- Test: `flutter_app/test/database/daos/visit_dao_test.dart`

- [ ] **Step 1: Add `uuid` package**

Edit `flutter_app/pubspec.yaml` — add `uuid: ^4.5.1` to dependencies (after `path: ^1.9.0`):

```yaml
  path: ^1.9.0
  uuid: ^4.5.1
```

Then:

```bash
cd flutter_app && flutter pub get
```

- [ ] **Step 2: Write the failing test**

Create `flutter_app/test/database/daos/visit_dao_test.dart`:

```dart
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';

void main() {
  late AppDatabase db;
  late ChildDao childDao;
  late VisitDao visitDao;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    childDao = ChildDao(db);
    visitDao = VisitDao(db);
  });

  tearDown(() async => db.close());

  test('createWithMeasurement assigns a non-empty localUuid', () async {
    final child = await childDao.findOrCreate(
      name: 'Test',
      dateOfBirth: '2024-01-01',
      sex: 'M',
    );
    final visitId = await visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 16,
      imagePath: '/tmp/front.jpg',
      measurement: const MeasurementsCompanion(),
    );
    final row = await visitDao.getById(visitId);
    expect(row, isNotNull);
    expect(row!.visit.localUuid.isNotEmpty, isTrue);
    expect(row.visit.localUuid.length, 36);
  });

  test('two visits get distinct localUuids', () async {
    final child = await childDao.findOrCreate(
      name: 'Test',
      dateOfBirth: '2024-01-01',
      sex: 'M',
    );
    final v1 = await visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 16,
      imagePath: '/tmp/a.jpg',
      measurement: const MeasurementsCompanion(),
    );
    final v2 = await visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 16,
      imagePath: '/tmp/b.jpg',
      measurement: const MeasurementsCompanion(),
    );
    final r1 = await visitDao.getById(v1);
    final r2 = await visitDao.getById(v2);
    expect(r1!.visit.localUuid, isNot(equals(r2!.visit.localUuid)));
  });
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd flutter_app && flutter test test/database/daos/visit_dao_test.dart
```
Expected: FAIL — `localUuid` does not exist on `Visit`.

- [ ] **Step 4: Add `localUuid` column to the table**

Edit `flutter_app/lib/database/tables/visits_table.dart`:

```dart
import 'package:drift/drift.dart';
import 'children_table.dart';

class Visits extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get childId => integer().references(Children, #id)();
  TextColumn get localUuid => text().withLength(min: 36, max: 36).unique()();
  DateTimeColumn get visitDate => dateTime().withDefault(currentDateAndTime)();
  RealColumn get ageMonths => real()();
  TextColumn get imagePath => text()();
  TextColumn get sideImagePath => text().nullable()();
  TextColumn get backImagePath => text().nullable()();
  TextColumn get notes => text().nullable()();
}
```

- [ ] **Step 5: Bump schema version + add migration**

Edit `flutter_app/lib/database/database.dart`:

```dart
import 'dart:io';
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import 'tables/children_table.dart';
import 'tables/visits_table.dart';
import 'tables/measurements_table.dart';
import 'tables/sync_queue_table.dart';

part 'database.g.dart';

@DriftDatabase(tables: [Children, Visits, Measurements, SyncQueue])
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_openConnection());

  /// For testing with in-memory database
  AppDatabase.forTesting(super.e);

  @override
  int get schemaVersion => 2;

  @override
  MigrationStrategy get migration => MigrationStrategy(
        onUpgrade: (migrator, from, to) async {
          if (from < 2) {
            // No production users yet — destructive recreate is acceptable.
            await migrator.deleteTable('visits');
            await migrator.deleteTable('measurements');
            await migrator.deleteTable('sync_queue');
            await migrator.createTable(visits);
            await migrator.createTable(measurements);
            await migrator.createTable(syncQueue);
          }
        },
      );
}

LazyDatabase _openConnection() {
  return LazyDatabase(() async {
    final dbFolder = await getApplicationDocumentsDirectory();
    final file = File(p.join(dbFolder.path, 'child_growth_monitor.sqlite'));
    return NativeDatabase.createInBackground(file);
  });
}
```

- [ ] **Step 6: Update `VisitDao.createWithMeasurement` to generate the UUID**

Edit `flutter_app/lib/database/daos/visit_dao.dart`:

```dart
import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';
import '../database.dart';

class VisitDao {
  final AppDatabase _db;
  VisitDao(this._db);

  static const _uuid = Uuid();

  Future<int> createWithMeasurement({
    required int childId,
    required double ageMonths,
    required String imagePath,
    String? sideImagePath,
    String? backImagePath,
    required MeasurementsCompanion measurement,
  }) async {
    return _db.transaction(() async {
      final visitId = await _db.into(_db.visits).insert(
        VisitsCompanion.insert(
          childId: childId,
          localUuid: _uuid.v4(),
          ageMonths: ageMonths,
          imagePath: imagePath,
          sideImagePath: Value(sideImagePath),
          backImagePath: Value(backImagePath),
        ),
      );
      await _db.into(_db.measurements).insert(
        measurement.copyWith(visitId: Value(visitId)),
      );
      return visitId;
    });
  }

  Stream<List<({Visit visit, Measurement? measurement})>> watchByChildId(int childId) {
    final query = _db.select(_db.visits).join([
      leftOuterJoin(_db.measurements, _db.measurements.visitId.equalsExp(_db.visits.id)),
    ])
      ..where(_db.visits.childId.equals(childId))
      ..orderBy([OrderingTerm.desc(_db.visits.visitDate)]);

    return query.watch().map((rows) => rows.map((row) => (
      visit: row.readTable(_db.visits),
      measurement: row.readTableOrNull(_db.measurements),
    )).toList());
  }

  Future<({Visit visit, Measurement? measurement})?> getById(int visitId) async {
    final query = _db.select(_db.visits).join([
      leftOuterJoin(_db.measurements, _db.measurements.visitId.equalsExp(_db.visits.id)),
    ])..where(_db.visits.id.equals(visitId));
    final row = await query.getSingleOrNull();
    if (row == null) return null;
    return (visit: row.readTable(_db.visits), measurement: row.readTableOrNull(_db.measurements));
  }
}
```

- [ ] **Step 7: Regenerate Drift code**

```bash
cd flutter_app && dart run build_runner build --delete-conflicting-outputs
```
Expected: `database.g.dart` updated with the new column.

- [ ] **Step 8: Run the test to verify it passes**

```bash
cd flutter_app && flutter test test/database/daos/visit_dao_test.dart
```
Expected: 2 passed.

- [ ] **Step 9: Run the full Flutter test suite**

```bash
cd flutter_app && flutter test
```
Expected: all existing tests still pass.

- [ ] **Step 10: Commit**

```bash
git add flutter_app/pubspec.yaml flutter_app/pubspec.lock \
        flutter_app/lib/database/tables/visits_table.dart \
        flutter_app/lib/database/database.dart \
        flutter_app/lib/database/database.g.dart \
        flutter_app/lib/database/daos/visit_dao.dart \
        flutter_app/test/database/daos/visit_dao_test.dart
git commit -m "feat(flutter): add localUuid to Visits table for sync dedup"
```

---

## Task 4: Bundle TFLite models + scaler params

**Files:**
- Create: `flutter_app/scripts/export_scaler.py`
- Create: `flutter_app/assets/models/feature_scaler.json` (generated)
- Copy: `data/models/weight_estimator.tflite` → `flutter_app/assets/models/weight_estimator.tflite`
- Copy: `data/models/wasting_classifier.tflite` → `flutter_app/assets/models/wasting_classifier.tflite`

> **Security note:** The export script reads `feature_scaler.pkl` (a pickle file generated by the local training pipeline). Pickle deserialization can execute arbitrary code; the script should only be run against pickles produced by this repo's `ml/train.py`. The Flutter app itself never touches pickle — it only reads the resulting JSON.

- [ ] **Step 1: Write the scaler-export script**

Create `flutter_app/scripts/export_scaler.py` with the contents below. The script uses `pickle.load` to read the local-training-pipeline output; this is acceptable here because the input file is generated and trusted by the same developer running the script.

```python
"""Convert sklearn StandardScaler pickle into JSON for the Flutter app.

Reads:  data/models/feature_scaler.pkl
Writes: flutter_app/assets/models/feature_scaler.json

Run from repo root:
    PYTHONPATH=. .venv/bin/python flutter_app/scripts/export_scaler.py

NOTE: pickle.load is used to read a file produced by this repo's own
ml/train.py. Do not run this script against pickle files from
untrusted sources.
"""
import json
import pickle  # nosec B403 - reading repo-local training artifact only
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKL = REPO_ROOT / "data" / "models" / "feature_scaler.pkl"
OUT = REPO_ROOT / "flutter_app" / "assets" / "models" / "feature_scaler.json"

FEATURE_NAMES = [
    "age_months",
    "sex_binary",
    "height_cm",
    "shoulder_width_cm",
    "hip_width_cm",
    "torso_length_cm",
    "upper_arm_length_cm",
    "shoulder_height_ratio",
    "hip_height_ratio",
    "body_build_score",
    "chest_depth_cm",
    "abd_depth_cm",
    "chest_depth_ratio",
    "abd_depth_ratio",
]


def main() -> None:
    with PKL.open("rb") as fh:
        scaler = pickle.load(fh)  # nosec B301 - trusted local artifact
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    if len(mean) != 14 or len(scale) != 14:
        raise SystemExit(
            f"Expected 14 features in scaler, got mean={len(mean)} scale={len(scale)}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(
            {"mean": mean, "scale": scale, "feature_names": FEATURE_NAMES},
            fh,
            indent=2,
        )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the export script**

```bash
PYTHONPATH=. .venv/bin/python flutter_app/scripts/export_scaler.py
```
Expected: `Wrote .../flutter_app/assets/models/feature_scaler.json`. Verify the file is now > 200 bytes (not the 46-byte placeholder).

- [ ] **Step 3: Copy the TFLite models into assets**

```bash
cp data/models/weight_estimator.tflite flutter_app/assets/models/weight_estimator.tflite
cp data/models/wasting_classifier.tflite flutter_app/assets/models/wasting_classifier.tflite
ls -la flutter_app/assets/models/
```
Expected: 3 files — `feature_scaler.json` (~700 B), `weight_estimator.tflite` (~8 KB), `wasting_classifier.tflite` (~18 KB).

- [ ] **Step 4: Verify pubspec already declares `tflite_flutter`**

```bash
grep tflite_flutter flutter_app/pubspec.yaml
```
Expected: `tflite_flutter: ^0.11.0` already present.

- [ ] **Step 5: Sanity-check assets load via Flutter test**

Create `flutter_app/test/assets/model_assets_test.dart`:

```dart
import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('feature_scaler.json has 14-feature mean and scale arrays', () async {
    final jsonStr = await rootBundle.loadString('assets/models/feature_scaler.json');
    final data = jsonDecode(jsonStr) as Map<String, dynamic>;
    expect((data['mean'] as List).length, 14);
    expect((data['scale'] as List).length, 14);
  });

  test('weight_estimator.tflite is bundled and non-trivial', () async {
    final bytes = await rootBundle.load('assets/models/weight_estimator.tflite');
    expect(bytes.lengthInBytes, greaterThan(2000));
  });

  test('wasting_classifier.tflite is bundled and non-trivial', () async {
    final bytes = await rootBundle.load('assets/models/wasting_classifier.tflite');
    expect(bytes.lengthInBytes, greaterThan(10000));
  });
}
```

```bash
cd flutter_app && flutter test test/assets/model_assets_test.dart
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add flutter_app/scripts/export_scaler.py \
        flutter_app/assets/models/feature_scaler.json \
        flutter_app/assets/models/weight_estimator.tflite \
        flutter_app/assets/models/wasting_classifier.tflite \
        flutter_app/test/assets/model_assets_test.dart
git commit -m "feat(flutter): bundle TFLite models and scaler params as assets"
```

---

## Task 5: `MeasurementService` + tests

**Files:**
- Modify: `flutter_app/lib/models/body_measurements.dart` (append `BodyMeasurements` class)
- Create: `flutter_app/lib/services/measurement_service.dart`
- Create: `flutter_app/test/fixtures/who_test_data.dart`
- Test: `flutter_app/test/services/measurement_service_test.dart`

- [ ] **Step 1: Read existing `body_measurements.dart` to confirm what exists**

```bash
cat flutter_app/lib/models/body_measurements.dart
```
Verify it contains `BodySegments` and `SideViewSegments` classes — the new `BodyMeasurements` class will be added below them.

- [ ] **Step 2: Add the `BodyMeasurements` class**

Append to `flutter_app/lib/models/body_measurements.dart` (do not modify existing classes):

```dart
/// Final measurement output: cm-scaled segments + height + body build.
/// Produced by MeasurementService from BodySegments (pixels) + WHO data.
class BodyMeasurements {
  final double effectiveHeightCm;
  final double shoulderWidthCm;
  final double hipWidthCm;
  final double torsoLengthCm;
  final double upperArmLengthCm;
  final double? chestDepthCm;
  final double? abdDepthCm;
  final String bodyBuild; // "slender" | "average" | "stocky"
  final int bodyBuildScore; // -1 | 0 | 1
  final double confidence; // 0.0 - 1.0
  final String estimationMethod; // "manual" | "who_statistical"
  final bool sideViewUsed;

  const BodyMeasurements({
    required this.effectiveHeightCm,
    required this.shoulderWidthCm,
    required this.hipWidthCm,
    required this.torsoLengthCm,
    required this.upperArmLengthCm,
    this.chestDepthCm,
    this.abdDepthCm,
    required this.bodyBuild,
    required this.bodyBuildScore,
    required this.confidence,
    required this.estimationMethod,
    required this.sideViewUsed,
  });
}
```

- [ ] **Step 3: Create the WHO fixture loader**

Create `flutter_app/test/fixtures/who_test_data.dart`:

```dart
import 'dart:io';

import 'package:child_growth_monitor_app/services/who_data_service.dart';

/// Loads bundled WHO fixtures for tests (rootBundle is unavailable in unit tests).
Future<void> loadWhoForTests(WhoDataService who) async {
  final base = Directory.current.path.endsWith('flutter_app')
      ? 'test/fixtures'
      : 'flutter_app/test/fixtures';
  await who.loadFromFiles(
    hazCsvPath: '$base/who_haz_0_59m.csv',
    wflBoysPath: '$base/who_wfl_boys_0_2.xlsx',
    wflGirlsPath: '$base/who_wfl_girls_0_2.xlsx',
    wfhBoysPath: '$base/who_wfh_boys_2_5.xlsx',
    wfhGirlsPath: '$base/who_wfh_girls_2_5.xlsx',
  );
}
```

- [ ] **Step 4: Write the failing tests**

Create `flutter_app/test/services/measurement_service_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/models/body_measurements.dart';
import 'package:child_growth_monitor_app/services/measurement_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

import '../fixtures/who_test_data.dart';

void main() {
  late WhoDataService who;
  late MeasurementService measurement;

  setUpAll(() async {
    who = WhoDataService();
    await loadWhoForTests(who);
    measurement = MeasurementService(who);
  });

  BodySegments segs({
    double totalHeightPx = 800,
    double shoulderWidthPx = 160,
    double hipWidthPx = 140,
    double torsoLengthPx = 240,
    double upperArmLengthPx = 120,
  }) =>
      BodySegments(
        headHeightPx: 100,
        torsoLengthPx: torsoLengthPx,
        legLengthPx: 380,
        shoulderWidthPx: shoulderWidthPx,
        hipWidthPx: hipWidthPx,
        upperArmLengthPx: upperArmLengthPx,
        totalHeightPx: totalHeightPx,
        headTopY: 0,
        chinY: 100,
        shoulderMidpointY: 200,
        hipMidpointY: 440,
        heelY: 800,
        headConfidence: 1.0,
        torsoConfidence: 1.0,
        legConfidence: 1.0,
        hipConfidence: 1.0,
        armConfidence: 1.0,
      );

  test('manual height takes priority over WHO median', () {
    final m = measurement.compute(
      segments: segs(),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 90.0,
      poseConfidence: 0.9,
    );
    expect(m.effectiveHeightCm, 90.0);
    expect(m.estimationMethod, 'manual');
  });

  test('falls back to WHO median when manual height absent', () {
    final m = measurement.compute(
      segments: segs(),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: null,
      poseConfidence: 0.9,
    );
    expect(m.estimationMethod, 'who_statistical');
    expect(m.effectiveHeightCm, greaterThan(70));
    expect(m.effectiveHeightCm, lessThan(100));
  });

  test('shoulder width converts to cm using height as scale', () {
    // totalHeightPx 800, height 80cm -> scale = 0.1 cm/px
    // shoulder 160 px -> 16.0 cm
    final m = measurement.compute(
      segments: segs(totalHeightPx: 800, shoulderWidthPx: 160),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.shoulderWidthCm, closeTo(16.0, 0.01));
  });

  test('body build classified as slender below threshold', () {
    // expected ratio at 24mo = 0.210; threshold 0.02 -> slender if < 0.190
    // height 80, slender shoulder = 80 * 0.18 = 14.4cm -> shoulder 144 px
    final m = measurement.compute(
      segments: segs(totalHeightPx: 800, shoulderWidthPx: 144),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.bodyBuild, 'slender');
    expect(m.bodyBuildScore, -1);
  });

  test('body build classified as stocky above threshold', () {
    // stocky if ratio > 0.230. shoulder 0.24 * 80 = 19.2cm -> 192 px
    final m = measurement.compute(
      segments: segs(totalHeightPx: 800, shoulderWidthPx: 192),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.bodyBuild, 'stocky');
    expect(m.bodyBuildScore, 1);
  });

  test('side-view depths populated when SideViewSegments provided', () {
    final m = measurement.compute(
      segments: segs(),
      sideSegments: SideViewSegments(
        chestDepthPx: 60,
        abdDepthPx: 70,
        totalHeightPx: 800,
        chestConfidence: 1.0,
        abdConfidence: 1.0,
      ),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.sideViewUsed, true);
    expect(m.chestDepthCm, closeTo(6.0, 0.01));
    expect(m.abdDepthCm, closeTo(7.0, 0.01));
  });

  test('chest/abd depth null when no side view', () {
    final m = measurement.compute(
      segments: segs(),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.sideViewUsed, false);
    expect(m.chestDepthCm, isNull);
    expect(m.abdDepthCm, isNull);
  });
}
```

- [ ] **Step 5: Run tests to verify they fail**

```bash
cd flutter_app && flutter test test/services/measurement_service_test.dart
```
Expected: FAIL — `MeasurementService` does not exist.

- [ ] **Step 6: Implement `MeasurementService`**

Create `flutter_app/lib/services/measurement_service.dart`:

```dart
import '../constants/config.dart';
import '../models/body_measurements.dart';
import 'who_data_service.dart';

/// Converts pixel-space body segments into cm measurements + body build.
/// No I/O — pure logic. Port of measurement_service.py height resolution
/// and body build classification.
class MeasurementService {
  final WhoDataService _who;
  MeasurementService(this._who);

  BodyMeasurements compute({
    required BodySegments segments,
    SideViewSegments? sideSegments,
    required double ageMonths,
    required String sex,
    double? manualHeightCm,
    required double poseConfidence,
  }) {
    final effectiveHeightCm = _resolveHeight(
      manualHeightCm: manualHeightCm,
      ageMonths: ageMonths,
      sex: sex,
    );
    final method = manualHeightCm != null ? 'manual' : 'who_statistical';

    final scale = _scale(segments, effectiveHeightCm);

    final shoulderCm = (segments.shoulderWidthPx ??
            _imputeShoulderPx(effectiveHeightCm, ageMonths)) *
        scale;
    final hipCm = (segments.hipWidthPx ?? shoulderCm * 0.88 / scale) * scale;
    final torsoCm =
        (segments.torsoLengthPx ?? effectiveHeightCm * 0.30 / scale) * scale;
    final armCm =
        (segments.upperArmLengthPx ?? _imputeArmPx(effectiveHeightCm, ageMonths)) *
            scale;

    double? chestCm;
    double? abdCm;
    bool sideUsed = false;
    if (sideSegments != null && sideSegments.totalHeightPx != null) {
      final sideScale = effectiveHeightCm / sideSegments.totalHeightPx!;
      if (sideSegments.chestDepthPx != null) {
        chestCm = sideSegments.chestDepthPx! * sideScale;
      }
      if (sideSegments.abdDepthPx != null) {
        abdCm = sideSegments.abdDepthPx! * sideScale;
      }
      sideUsed = chestCm != null || abdCm != null;
    }

    final ratio = shoulderCm / effectiveHeightCm;
    final expected = expectedShoulderRatio(ageMonths);
    String build;
    int buildScore;
    if (ratio < expected - bodyBuildThresholdMl) {
      build = 'slender';
      buildScore = -1;
    } else if (ratio > expected + bodyBuildThresholdMl) {
      build = 'stocky';
      buildScore = 1;
    } else {
      build = 'average';
      buildScore = 0;
    }

    return BodyMeasurements(
      effectiveHeightCm: effectiveHeightCm,
      shoulderWidthCm: shoulderCm,
      hipWidthCm: hipCm,
      torsoLengthCm: torsoCm,
      upperArmLengthCm: armCm,
      chestDepthCm: chestCm,
      abdDepthCm: abdCm,
      bodyBuild: build,
      bodyBuildScore: buildScore,
      confidence: poseConfidence,
      estimationMethod: method,
      sideViewUsed: sideUsed,
    );
  }

  double _resolveHeight({
    required double? manualHeightCm,
    required double ageMonths,
    required String sex,
  }) {
    if (manualHeightCm != null && manualHeightCm > 0) return manualHeightCm;
    final median = _who.getMedianHeightForAge(sex, ageMonths.round());
    if (median != null) return median;
    // Final fallback: WHO 24-month median to avoid /0; surfaces as low confidence upstream.
    return 87.1;
  }

  double _scale(BodySegments segments, double heightCm) {
    final px = segments.totalHeightPx;
    if (px != null && px > 0) return heightCm / px;
    return 1.0;
  }

  double _imputeShoulderPx(double heightCm, double ageMonths) {
    if (ageMonths < 24) return heightCm * 0.200;
    if (ageMonths < 48) return heightCm * 0.210;
    return heightCm * 0.218;
  }

  double _imputeArmPx(double heightCm, double ageMonths) {
    if (ageMonths < 24) return heightCm * 0.150;
    if (ageMonths < 48) return heightCm * 0.158;
    return heightCm * 0.165;
  }
}
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd flutter_app && flutter test test/services/measurement_service_test.dart
```
Expected: 7 passed.

- [ ] **Step 8: Commit**

```bash
git add flutter_app/lib/models/body_measurements.dart \
        flutter_app/lib/services/measurement_service.dart \
        flutter_app/test/services/measurement_service_test.dart \
        flutter_app/test/fixtures/who_test_data.dart
git commit -m "feat(flutter): add MeasurementService for pixel→cm conversion + body build"
```

---

## Task 6: `MlInferenceService` + tests

**Files:**
- Create: `flutter_app/lib/services/ml_inference_service.dart`
- Test: `flutter_app/test/services/ml_inference_service_test.dart`

- [ ] **Step 1: Write the failing tests**

Create `flutter_app/test/services/ml_inference_service_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/models/wasting_features.dart';
import 'package:child_growth_monitor_app/services/ml_inference_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('predicts wasting class for typical 24-month boy features', () async {
    final svc = MlInferenceService();
    await svc.load();
    final features = const WastingFeatures(
      ageMonths: 24,
      sexBinary: 1,
      heightCm: 87.1,
      shoulderWidthCm: 18.0,
      hipWidthCm: 15.5,
      torsoLengthCm: 26.5,
      upperArmLengthCm: 13.7,
      shoulderHeightRatio: 0.207,
      hipHeightRatio: 0.178,
      bodyBuildScore: 0,
    );
    final prediction = svc.predict(features);
    expect(prediction.estimatedWeightKg, isNotNull);
    expect(prediction.estimatedWeightKg!, inInclusiveRange(2.0, 30.0));
    expect(
      ['SAM', 'MAM', 'Normal', 'Risk_Overweight', 'Overweight']
          .contains(prediction.wastingStatus),
      isTrue,
    );
    final probSum = prediction.samProbability +
        prediction.mamProbability +
        prediction.normalProbability +
        prediction.riskProbability +
        prediction.overweightProbability;
    expect(probSum, closeTo(1.0, 0.01));
    svc.dispose();
  });

  test('weight bound check rejects values outside 45–180% of WHO median', () async {
    final svc = MlInferenceService();
    await svc.load();
    expect(svc.weightWithinBounds(predictedKg: 12.0, whoMedianKg: 12.0), isTrue);
    expect(svc.weightWithinBounds(predictedKg: 4.0, whoMedianKg: 12.0), isFalse);
    expect(svc.weightWithinBounds(predictedKg: 25.0, whoMedianKg: 12.0), isFalse);
    expect(svc.weightWithinBounds(predictedKg: 21.6, whoMedianKg: 12.0), isTrue);
    svc.dispose();
  });

  test('throws StateError when predict called before load', () {
    final svc = MlInferenceService();
    expect(
      () => svc.predict(const WastingFeatures(
        ageMonths: 24, sexBinary: 1, heightCm: 87.1,
        shoulderWidthCm: 18, hipWidthCm: 15.5, torsoLengthCm: 26.5,
        upperArmLengthCm: 13.7, shoulderHeightRatio: 0.207,
        hipHeightRatio: 0.178, bodyBuildScore: 0,
      )),
      throwsStateError,
    );
  });
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd flutter_app && flutter test test/services/ml_inference_service_test.dart
```
Expected: FAIL — `MlInferenceService` does not exist.

- [ ] **Step 3: Implement `MlInferenceService`**

Create `flutter_app/lib/services/ml_inference_service.dart`:

```dart
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../constants/config.dart';
import '../models/wasting_features.dart';

/// Runs the on-device weight estimator + wasting classifier.
/// Fails loudly if assets are missing — caller must catch and trigger fallback.
class MlInferenceService {
  Interpreter? _weight;
  Interpreter? _classifier;
  List<double>? _mean;
  List<double>? _scale;

  static const _weightAsset = 'assets/models/weight_estimator.tflite';
  static const _classifierAsset = 'assets/models/wasting_classifier.tflite';
  static const _scalerAsset = 'assets/models/feature_scaler.json';

  static const double _lowerBound = mlWeightLowerBound; // 0.45
  static const double _upperBound = mlWeightUpperBound; // 1.80

  bool get isLoaded => _weight != null && _classifier != null && _mean != null;

  Future<void> load() async {
    _weight = await Interpreter.fromAsset(_weightAsset);
    _classifier = await Interpreter.fromAsset(_classifierAsset);

    final scalerJson = await rootBundle.loadString(_scalerAsset);
    final data = jsonDecode(scalerJson) as Map<String, dynamic>;
    _mean = (data['mean'] as List).map((v) => (v as num).toDouble()).toList();
    _scale = (data['scale'] as List).map((v) => (v as num).toDouble()).toList();
    if (_mean!.length != 14 || _scale!.length != 14) {
      throw StateError(
        'feature_scaler.json must contain 14-element mean and scale arrays',
      );
    }

    final wOut = _weight!.getOutputTensor(0).shape;
    final cOut = _classifier!.getOutputTensor(0).shape;
    if (wOut.length != 2 || wOut[1] != 1) {
      throw StateError('weight_estimator output shape must be [1,1], got $wOut');
    }
    if (cOut.length != 2 || cOut[1] != 5) {
      throw StateError('wasting_classifier output shape must be [1,5], got $cOut');
    }
  }

  WastingPrediction predict(WastingFeatures features) {
    if (!isLoaded) {
      throw StateError('MlInferenceService.predict called before load()');
    }
    final scaled = _scale14(features.toArray());

    final weightOut = List.filled(1, List<double>.filled(1, 0.0));
    _weight!.run([scaled], weightOut);
    final weightKg = weightOut[0][0];

    final probsOut = List.filled(1, List<double>.filled(5, 0.0));
    _classifier!.run([scaled], probsOut);
    final probs = probsOut[0];

    int argmax = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[argmax]) argmax = i;
    }
    final label = wastingLabels[argmax];

    return WastingPrediction(
      estimatedWeightKg: weightKg,
      samProbability: probs[wastingLabels.indexOf('SAM')],
      mamProbability: probs[wastingLabels.indexOf('MAM')],
      normalProbability: probs[wastingLabels.indexOf('Normal')],
      riskProbability: probs[wastingLabels.indexOf('Risk_Overweight')],
      overweightProbability: probs[wastingLabels.indexOf('Overweight')],
      wastingStatus: label,
    );
  }

  bool weightWithinBounds({
    required double predictedKg,
    required double whoMedianKg,
  }) {
    if (whoMedianKg <= 0) return false;
    final ratio = predictedKg / whoMedianKg;
    return ratio >= _lowerBound && ratio <= _upperBound;
  }

  List<double> _scale14(Float32List raw) {
    final out = List<double>.filled(14, 0);
    for (var i = 0; i < 14; i++) {
      out[i] = (raw[i] - _mean![i]) / _scale![i];
    }
    return out;
  }

  void dispose() {
    _weight?.close();
    _classifier?.close();
    _weight = null;
    _classifier = null;
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd flutter_app && flutter test test/services/ml_inference_service_test.dart
```
Expected: 3 passed. If `Interpreter.fromAsset` fails in the host test environment due to a missing native library, mark the first test with `skip: 'requires device'` and rely on the bound-check + load-error tests for unit coverage; the integration test in Task 14 will exercise full inference on a real device.

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/services/ml_inference_service.dart \
        flutter_app/test/services/ml_inference_service_test.dart
git commit -m "feat(flutter): add MlInferenceService for on-device TFLite inference"
```

---

## Task 7: `ImageStorageService` + tests

**Files:**
- Create: `flutter_app/lib/services/image_storage_service.dart`
- Test: `flutter_app/test/services/image_storage_service_test.dart`

- [ ] **Step 1: Write the failing tests**

Create `flutter_app/test/services/image_storage_service_test.dart`:

```dart
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/services/image_storage_service.dart';

void main() {
  late Directory tempRoot;
  late ImageStorageService svc;

  setUp(() async {
    tempRoot = await Directory.systemTemp.createTemp('cgm_test_');
    svc = ImageStorageService(rootOverride: tempRoot);
  });

  tearDown(() async {
    if (await tempRoot.exists()) {
      await tempRoot.delete(recursive: true);
    }
  });

  test('persist copies file into images dir and returns new path', () async {
    final src = File('${tempRoot.path}/src.jpg');
    await src.writeAsBytes(List.filled(2048, 0xAB));

    final newPath = await svc.persist(src.path);
    expect(File(newPath).existsSync(), isTrue);
    expect(newPath, contains('${tempRoot.path}/images/'));
    expect((await File(newPath).readAsBytes()).length, 2048);
  });

  test('totalUsedBytes returns sum of all image bytes', () async {
    final a = File('${tempRoot.path}/a.jpg')..writeAsBytesSync(List.filled(1000, 1));
    final b = File('${tempRoot.path}/b.jpg')..writeAsBytesSync(List.filled(500, 2));
    await svc.persist(a.path);
    await svc.persist(b.path);
    expect(await svc.totalUsedBytes(), 1500);
  });

  test('clearAll removes every file in the images dir', () async {
    final src = File('${tempRoot.path}/x.jpg')..writeAsBytesSync([1, 2, 3]);
    await svc.persist(src.path);
    expect(await svc.totalUsedBytes(), 3);
    await svc.clearAll();
    expect(await svc.totalUsedBytes(), 0);
  });
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd flutter_app && flutter test test/services/image_storage_service_test.dart
```
Expected: FAIL — `ImageStorageService` does not exist.

- [ ] **Step 3: Implement `ImageStorageService`**

Create `flutter_app/lib/services/image_storage_service.dart`:

```dart
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

/// Manages the lifecycle of captured images on device storage.
///
/// All images live under `<app documents>/images/`. The service never
/// auto-deletes — callers (or the user via Settings) trigger `clearAll`.
class ImageStorageService {
  ImageStorageService({Directory? rootOverride}) : _rootOverride = rootOverride;

  final Directory? _rootOverride;
  static const _uuid = Uuid();

  Future<Directory> _imagesDir() async {
    final base = _rootOverride ?? await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(base.path, 'images'));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  /// Copies [tempPath] into the persistent images directory and returns the
  /// new absolute path.
  Future<String> persist(String tempPath) async {
    final src = File(tempPath);
    if (!await src.exists()) {
      throw FileSystemException('Source image not found', tempPath);
    }
    final ext = p.extension(tempPath).isEmpty ? '.jpg' : p.extension(tempPath);
    final dir = await _imagesDir();
    final dst = File(p.join(dir.path, '${_uuid.v4()}$ext'));
    await src.copy(dst.path);
    return dst.path;
  }

  /// Sum of bytes for every file under the images directory.
  Future<int> totalUsedBytes() async {
    final dir = await _imagesDir();
    var total = 0;
    await for (final entity in dir.list(recursive: true, followLinks: false)) {
      if (entity is File) {
        total += await entity.length();
      }
    }
    return total;
  }

  /// Deletes every file in the images directory. The directory itself remains.
  Future<void> clearAll() async {
    final dir = await _imagesDir();
    await for (final entity in dir.list()) {
      if (entity is File) await entity.delete();
    }
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd flutter_app && flutter test test/services/image_storage_service_test.dart
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/services/image_storage_service.dart \
        flutter_app/test/services/image_storage_service_test.dart
git commit -m "feat(flutter): add ImageStorageService for app-documents image lifecycle"
```

---

## Task 8: `AssessmentService` + tests

**Files:**
- Create: `flutter_app/lib/services/assessment_service.dart`
- Test: `flutter_app/test/services/assessment_service_test.dart`

- [ ] **Step 1: Write the failing test**

Create `flutter_app/test/services/assessment_service_test.dart`:

```dart
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/sync_queue_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';
import 'package:child_growth_monitor_app/models/body_measurements.dart';
import 'package:child_growth_monitor_app/models/wasting_features.dart';
import 'package:child_growth_monitor_app/services/assessment_service.dart';
import 'package:child_growth_monitor_app/services/measurement_service.dart';
import 'package:child_growth_monitor_app/services/ml_inference_service.dart';
import 'package:child_growth_monitor_app/services/muac_service.dart';
import 'package:child_growth_monitor_app/services/nutrition_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

import '../fixtures/who_test_data.dart';

class _StubPose {
  BodySegments segmentsFor(String _) => BodySegments(
        headHeightPx: 100,
        torsoLengthPx: 240,
        legLengthPx: 380,
        shoulderWidthPx: 160,
        hipWidthPx: 140,
        upperArmLengthPx: 120,
        totalHeightPx: 800,
        headTopY: 0,
        chinY: 100,
        shoulderMidpointY: 200,
        hipMidpointY: 440,
        heelY: 800,
        headConfidence: 1,
        torsoConfidence: 1,
        legConfidence: 1,
        hipConfidence: 1,
        armConfidence: 1,
      );
  SideViewSegments? sideSegmentsFor(String _) => null;
  double confidenceFor(String _) => 0.9;
}

class _StubMl extends MlInferenceService {
  WastingPrediction? canned;
  Object? throwOnPredict;

  @override
  Future<void> load() async {}

  @override
  WastingPrediction predict(WastingFeatures features) {
    if (throwOnPredict != null) throw throwOnPredict!;
    return canned ??
        const WastingPrediction(
          estimatedWeightKg: 11.0,
          samProbability: 0.02,
          mamProbability: 0.05,
          normalProbability: 0.90,
          riskProbability: 0.02,
          overweightProbability: 0.01,
          wastingStatus: 'Normal',
        );
  }

  @override
  bool weightWithinBounds({
    required double predictedKg,
    required double whoMedianKg,
  }) =>
      true;
}

void main() {
  late AppDatabase db;
  late AssessmentService svc;
  late _StubMl ml;

  setUp(() async {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    final who = WhoDataService();
    await loadWhoForTests(who);
    ml = _StubMl();
    svc = AssessmentService(
      db: db,
      childDao: ChildDao(db),
      visitDao: VisitDao(db),
      syncQueueDao: SyncQueueDao(db),
      pose: _StubPose() as dynamic,
      measurement: MeasurementService(who),
      nutrition: NutritionService(who),
      who: who,
      ml: ml,
      persistImage: (path) async => path,
    );
  });

  tearDown(() async => db.close());

  test('happy path returns Normal result and enqueues a sync row', () async {
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Aisha',
      dateOfBirth: '2024-01-01',
      sex: 'F',
    );

    expect(result.nutrition.whzStatus, isNotNull);
    expect(result.mlPrediction, isNotNull);
    expect(result.mlPrediction!.wastingStatus, 'Normal');

    final pending = await db.select(db.syncQueue).get();
    expect(pending.length, 1);
    expect(pending.first.status, 'pending');

    final visits = await db.select(db.visits).get();
    expect(visits.length, 1);
    expect(visits.first.localUuid.length, 36);
  });

  test('ML failure produces a result labelled who_fallback', () async {
    ml.throwOnPredict = StateError('boom');
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Bilal',
      dateOfBirth: '2024-06-01',
      sex: 'M',
    );

    expect(result.mlPrediction, isNull);
    expect(result.measurement.estimationMethod, isNotNull);
    expect(result.muac, isNotNull);
    final stored = await db.select(db.measurements).get();
    expect(stored.length, 1);
    expect(stored.first.whzStatus, isNotNull);
    expect(stored.first.wastingStatus, 'who_fallback');
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd flutter_app && flutter test test/services/assessment_service_test.dart
```
Expected: FAIL — `AssessmentService` does not exist.

- [ ] **Step 3: Implement `AssessmentService`**

Create `flutter_app/lib/services/assessment_service.dart`:

```dart
import 'package:drift/drift.dart';

import '../constants/config.dart';
import '../database/daos/child_dao.dart';
import '../database/daos/sync_queue_dao.dart';
import '../database/daos/visit_dao.dart';
import '../database/database.dart';
import '../models/assessment_result.dart';
import '../models/body_measurements.dart';
import '../models/wasting_features.dart';
import 'measurement_service.dart';
import 'ml_inference_service.dart';
import 'muac_service.dart';
import 'nutrition_service.dart';
import 'pose_service.dart';
import 'who_data_service.dart';

/// Function signature for moving an image into permanent storage.
/// Real impl: `ImageStorageService.persist`. Tests can pass an identity fn.
typedef ImagePersister = Future<String> Function(String tempPath);

class AssessmentService {
  AssessmentService({
    required AppDatabase db,
    required ChildDao childDao,
    required VisitDao visitDao,
    required SyncQueueDao syncQueueDao,
    required dynamic pose, // PoseService at runtime; stubs in tests
    required MeasurementService measurement,
    required NutritionService nutrition,
    required WhoDataService who,
    required MlInferenceService ml,
    required ImagePersister persistImage,
  })  : _db = db,
        _childDao = childDao,
        _visitDao = visitDao,
        _syncQueueDao = syncQueueDao,
        _pose = pose,
        _measurement = measurement,
        _nutrition = nutrition,
        _who = who,
        _ml = ml,
        _persistImage = persistImage;

  final AppDatabase _db;
  final ChildDao _childDao;
  final VisitDao _visitDao;
  final SyncQueueDao _syncQueueDao;
  final dynamic _pose;
  final MeasurementService _measurement;
  final NutritionService _nutrition;
  final WhoDataService _who;
  final MlInferenceService _ml;
  final ImagePersister _persistImage;

  Future<AssessmentResult> runAssessment({
    required String frontImagePath,
    String? sideImagePath,
    String? backImagePath,
    required String childName,
    required String dateOfBirth,
    required String sex,
    double? manualWeightKg,
    double? manualHeightCm,
    double? manualMuacCm,
    String? guardianName,
    String? location,
  }) async {
    final dob = DateTime.parse(dateOfBirth);
    final ageMonths =
        DateTime.now().difference(dob).inDays / daysPerMonth;

    final frontPath = await _persistImage(frontImagePath);
    final sidePath =
        sideImagePath != null ? await _persistImage(sideImagePath) : null;
    final backPath =
        backImagePath != null ? await _persistImage(backImagePath) : null;

    final segments = await _detectFront(frontPath);
    final sideSegments = sidePath != null ? await _detectSide(sidePath) : null;
    final poseConfidence = _confidenceFor(frontPath);

    final m = _measurement.compute(
      segments: segments,
      sideSegments: sideSegments,
      ageMonths: ageMonths,
      sex: sex,
      manualHeightCm: manualHeightCm,
      poseConfidence: poseConfidence,
    );

    WastingPrediction? prediction;
    try {
      final features = WastingFeatures(
        ageMonths: ageMonths,
        sexBinary: sex.toUpperCase() == 'M' ? 1 : 0,
        heightCm: m.effectiveHeightCm,
        shoulderWidthCm: m.shoulderWidthCm,
        hipWidthCm: m.hipWidthCm,
        torsoLengthCm: m.torsoLengthCm,
        upperArmLengthCm: m.upperArmLengthCm,
        shoulderHeightRatio: m.shoulderWidthCm / m.effectiveHeightCm,
        hipHeightRatio: m.hipWidthCm / m.effectiveHeightCm,
        bodyBuildScore: m.bodyBuildScore,
        chestDepthCm: m.chestDepthCm,
        abdDepthCm: m.abdDepthCm,
      );
      prediction = _ml.predict(features);
    } catch (_) {
      prediction = null; // fallback path
    }

    final whoMedianWeight = _who.getMedianWeightForHeight(
      sex,
      m.effectiveHeightCm,
      ageMonths: ageMonths,
    );
    final effectiveWeight = _resolveWeight(
      manualWeightKg: manualWeightKg,
      ml: prediction,
      whoMedianKg: whoMedianWeight,
      build: m.bodyBuild,
    );

    final haz =
        _nutrition.computeHaz(sex, ageMonths.round(), m.effectiveHeightCm);
    final whz = effectiveWeight != null
        ? _nutrition.computeWhz(
            sex, ageMonths, m.effectiveHeightCm, effectiveWeight)
        : null;

    final muac = MuacService.estimate(
      ageMonths: ageMonths,
      sex: sex,
      whz: whz,
      manualMuacCm: manualMuacCm,
    );

    final hazStatus = haz != null ? classifyHaz(haz) : null;
    final whzStatus = whz != null ? classifyWhz(whz) : null;

    final child = await _childDao.findOrCreate(
      name: childName,
      dateOfBirth: dateOfBirth,
      sex: sex,
      guardianName: guardianName,
      location: location,
    );

    final visitId = await _visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: ageMonths,
      imagePath: frontPath,
      sideImagePath: sidePath,
      backImagePath: backPath,
      measurement: MeasurementsCompanion(
        predictedHeightCm: Value(m.effectiveHeightCm),
        predictedWeightKg: Value(effectiveWeight),
        manualHeightCm: Value(manualHeightCm),
        manualWeightKg: Value(manualWeightKg),
        hazZscore: Value(haz),
        whzZscore: Value(whz),
        hazStatus: Value(hazStatus),
        whzStatus: Value(whzStatus),
        confidenceScore: Value(poseConfidence),
        bodyBuild: Value(m.bodyBuild),
        estimationMethod: Value(m.estimationMethod),
        sideViewUsed: Value(m.sideViewUsed),
        chestDepthCm: Value(m.chestDepthCm),
        abdDepthCm: Value(m.abdDepthCm),
        mlEstimatedWeightKg: Value(prediction?.estimatedWeightKg),
        samProbability: Value(prediction?.samProbability),
        mamProbability: Value(prediction?.mamProbability),
        normalProbability: Value(prediction?.normalProbability),
        riskOverweightProbability: Value(prediction?.riskProbability),
        overweightProbability: Value(prediction?.overweightProbability),
        wastingStatus:
            Value(prediction?.wastingStatus ?? 'who_fallback'),
        muacCm: Value(muac.muacCm),
        muacStatus: Value(muac.muacStatus),
        muacMethod: Value(muac.muacMethod),
      ),
    );
    await _syncQueueDao.enqueue(visitId);

    return AssessmentResult(
      childName: childName,
      sex: sex,
      ageMonths: ageMonths,
      measurement: Measurement(
        predictedHeightCm: m.effectiveHeightCm,
        predictedWeightKg: effectiveWeight,
        manualHeightCm: manualHeightCm,
        manualWeightKg: manualWeightKg,
        confidenceScore: poseConfidence,
        estimationMethod: m.estimationMethod,
        bodyBuild: m.bodyBuild,
        sideViewUsed: m.sideViewUsed,
        chestDepthCm: m.chestDepthCm,
        abdDepthCm: m.abdDepthCm,
      ),
      nutrition: Nutrition(
        hazZscore: haz,
        whzZscore: whz,
        hazStatus: hazStatus,
        whzStatus: whzStatus,
        ageMonths: ageMonths,
      ),
      mlPrediction: prediction == null
          ? null
          : MlPrediction(
              estimatedWeightKg: prediction.estimatedWeightKg,
              samProbability: prediction.samProbability,
              mamProbability: prediction.mamProbability,
              normalProbability: prediction.normalProbability,
              riskProbability: prediction.riskProbability,
              overweightProbability: prediction.overweightProbability,
              wastingStatus: prediction.wastingStatus,
              wastingMethod: 'ml_classifier',
            ),
      muac: MuacDetail(
        muacCm: muac.muacCm,
        muacStatus: muac.muacStatus,
        muacMethod: muac.muacMethod,
        ageInRange: muac.ageInRange,
      ),
    );
  }

  // --- Helpers ----------------------------------------------------------

  Future<BodySegments> _detectFront(String path) async {
    if (_pose is PoseService) {
      final landmarks = await (_pose as PoseService).detectPose(path);
      return (_pose as PoseService).extractSegments(landmarks, 1.0, 1.0);
    }
    return _pose.segmentsFor(path) as BodySegments;
  }

  Future<SideViewSegments?> _detectSide(String path) async {
    if (_pose is PoseService) {
      final landmarks = await (_pose as PoseService).detectPose(path);
      return (_pose as PoseService).extractSideSegments(landmarks, 1.0);
    }
    return _pose.sideSegmentsFor(path) as SideViewSegments?;
  }

  double _confidenceFor(String path) {
    if (_pose is PoseService) return 0.85; // pose service does its own scoring
    return _pose.confidenceFor(path) as double;
  }

  double? _resolveWeight({
    required double? manualWeightKg,
    required WastingPrediction? ml,
    required double? whoMedianKg,
    required String build,
  }) {
    if (manualWeightKg != null && manualWeightKg > 0) return manualWeightKg;
    if (ml?.estimatedWeightKg != null && whoMedianKg != null) {
      final ok = _ml.weightWithinBounds(
        predictedKg: ml!.estimatedWeightKg!,
        whoMedianKg: whoMedianKg,
      );
      if (ok) return ml.estimatedWeightKg;
    }
    if (whoMedianKg == null) return null;
    final adjustment = build == 'slender'
        ? 0.95
        : build == 'stocky'
            ? 1.05
            : 1.0;
    return whoMedianKg * adjustment;
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd flutter_app && flutter test test/services/assessment_service_test.dart
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/services/assessment_service.dart \
        flutter_app/test/services/assessment_service_test.dart
git commit -m "feat(flutter): add AssessmentService orchestrator with ML fallback path"
```

---

## Task 9: Database & service providers

**Files:**
- Create: `flutter_app/lib/providers/database_provider.dart`
- Create: `flutter_app/lib/providers/assessment_service_provider.dart`
- Modify: `flutter_app/lib/providers/children_provider.dart`

- [ ] **Step 1: Create the database provider**

Create `flutter_app/lib/providers/database_provider.dart`:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../database/database.dart';
import '../database/daos/child_dao.dart';
import '../database/daos/sync_queue_dao.dart';
import '../database/daos/visit_dao.dart';

final databaseProvider = Provider<AppDatabase>((ref) {
  final db = AppDatabase();
  ref.onDispose(db.close);
  return db;
});

final childDaoProvider =
    Provider<ChildDao>((ref) => ChildDao(ref.watch(databaseProvider)));

final visitDaoProvider =
    Provider<VisitDao>((ref) => VisitDao(ref.watch(databaseProvider)));

final syncQueueDaoProvider =
    Provider<SyncQueueDao>((ref) => SyncQueueDao(ref.watch(databaseProvider)));
```

- [ ] **Step 2: Create the assessment-service provider**

Create `flutter_app/lib/providers/assessment_service_provider.dart`:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/assessment_service.dart';
import '../services/image_storage_service.dart';
import '../services/measurement_service.dart';
import '../services/ml_inference_service.dart';
import '../services/nutrition_service.dart';
import '../services/pose_service.dart';
import '../services/who_data_service.dart';
import 'database_provider.dart';

final whoDataServiceProvider = FutureProvider<WhoDataService>((ref) async {
  final who = WhoDataService();
  await who.loadFromAssets();
  return who;
});

final poseServiceProvider = Provider<PoseService>((ref) {
  final svc = PoseService();
  ref.onDispose(svc.dispose);
  return svc;
});

final mlInferenceServiceProvider =
    FutureProvider<MlInferenceService>((ref) async {
  final svc = MlInferenceService();
  await svc.load();
  ref.onDispose(svc.dispose);
  return svc;
});

final imageStorageProvider =
    Provider<ImageStorageService>((ref) => ImageStorageService());

final assessmentServiceProvider =
    FutureProvider<AssessmentService>((ref) async {
  final who = await ref.watch(whoDataServiceProvider.future);
  final ml = await ref.watch(mlInferenceServiceProvider.future);
  final storage = ref.watch(imageStorageProvider);

  return AssessmentService(
    db: ref.watch(databaseProvider),
    childDao: ref.watch(childDaoProvider),
    visitDao: ref.watch(visitDaoProvider),
    syncQueueDao: ref.watch(syncQueueDaoProvider),
    pose: ref.watch(poseServiceProvider),
    measurement: MeasurementService(who),
    nutrition: NutritionService(who),
    who: who,
    ml: ml,
    persistImage: storage.persist,
  );
});
```

- [ ] **Step 3: Switch children providers to local DB streams**

Replace `flutter_app/lib/providers/children_provider.dart` entirely:

```dart
import 'package:drift/drift.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/child.dart';
import '../models/child_detail.dart';
import 'database_provider.dart';

/// Watches all children from the local DB, with visit counts joined in.
final childrenProvider = StreamProvider<List<ChildSummary>>((ref) {
  final db = ref.watch(databaseProvider);
  return db.select(db.children).watch().asyncMap((rows) async {
    return Future.wait(rows.map((c) async {
      final visitCount = await (db.selectOnly(db.visits)
            ..addColumns([db.visits.id.count()])
            ..where(db.visits.childId.equals(c.id)))
          .map((row) => row.read(db.visits.id.count()) ?? 0)
          .getSingle();
      return ChildSummary(
        id: c.id,
        name: c.name,
        dateOfBirth: c.dateOfBirth,
        sex: c.sex,
        visitCount: visitCount,
      );
    }));
  });
});

/// Watches a single child + their visit history.
final childDetailProvider =
    StreamProvider.family<ChildDetail, int>((ref, childId) {
  final db = ref.watch(databaseProvider);
  return db.select(db.children).watch().asyncMap((rows) async {
    final child = rows.firstWhere((c) => c.id == childId);
    final visitRows = await (db.select(db.visits).join([
      leftOuterJoin(
        db.measurements,
        db.measurements.visitId.equalsExp(db.visits.id),
      ),
    ])
          ..where(db.visits.childId.equals(childId))
          ..orderBy([OrderingTerm.desc(db.visits.visitDate)]))
        .get();

    final visits = visitRows.map((row) {
      final v = row.readTable(db.visits);
      final m = row.readTableOrNull(db.measurements);
      return ChildVisit(
        visitId: v.id,
        visitDate: v.visitDate.toIso8601String(),
        ageMonths: v.ageMonths,
        measurement: m == null
            ? null
            : VisitMeasurement(
                predictedHeightCm: m.predictedHeightCm,
                predictedWeightKg: m.predictedWeightKg,
                manualWeightKg: m.manualWeightKg,
                hazZscore: m.hazZscore,
                whzZscore: m.whzZscore,
                hazStatus: m.hazStatus,
                whzStatus: m.whzStatus,
                confidenceScore: m.confidenceScore,
              ),
      );
    }).toList();

    return ChildDetail(
      id: child.id,
      name: child.name,
      dateOfBirth: child.dateOfBirth,
      sex: child.sex,
      guardianName: child.guardianName,
      location: child.location,
      visits: visits,
    );
  });
});
```

> **Note for engineer:** This task assumes the existing `ChildSummary`, `ChildDetail`, `ChildVisit`, and `VisitMeasurement` constructors accept these named args. If signatures differ, adapt at the call site only — do **not** change the model classes (the result/detail screens depend on them).

- [ ] **Step 4: Run flutter analyze**

```bash
cd flutter_app && flutter analyze
```
Expected: clean. If `ChildSummary`/`ChildDetail`/`ChildVisit` constructors need different field names, fix this file (not the models).

- [ ] **Step 5: Run the test suite**

```bash
cd flutter_app && flutter test
```
Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add flutter_app/lib/providers/database_provider.dart \
        flutter_app/lib/providers/assessment_service_provider.dart \
        flutter_app/lib/providers/children_provider.dart
git commit -m "feat(flutter): wire database + assessment-service providers, switch children to local DB"
```

---

## Task 10: Wire AssessmentScreen to local pipeline

**Files:**
- Modify: `flutter_app/lib/screens/assessment/assessment_screen.dart`

- [ ] **Step 1: Add the import + replace `_submit`**

In `flutter_app/lib/screens/assessment/assessment_screen.dart`, add this import alongside the others:

```dart
import '../../providers/assessment_service_provider.dart';
```

Replace the entire `try/catch/finally` block in `_submit` with:

```dart
    try {
      final svc = await ref.read(assessmentServiceProvider.future);
      final result = await svc.runAssessment(
        frontImagePath: _frontImage!.path,
        sideImagePath: _sideImage?.path,
        backImagePath: _backImage?.path,
        childName: _childNameController.text.trim(),
        dateOfBirth: _resolvedDob(),
        sex: _sex,
        manualWeightKg: double.tryParse(_weightController.text.trim()),
        manualHeightCm: heightCm,
        manualMuacCm: double.tryParse(_muacController.text.trim()),
        guardianName: _guardianController.text.trim().isEmpty
            ? null
            : _guardianController.text.trim(),
        location: _locationController.text.trim().isEmpty
            ? null
            : _locationController.text.trim(),
      );
      if (!mounted) return;
      ref.read(assessmentResultProvider.notifier).state = result;
      ref.invalidate(childrenProvider);
      context.go('/result');
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
```

- [ ] **Step 2: Run analyze + tests**

```bash
cd flutter_app && flutter analyze && flutter test
```
Expected: clean and all tests pass.

- [ ] **Step 3: Commit**

```bash
git add flutter_app/lib/screens/assessment/assessment_screen.dart
git commit -m "feat(flutter): assessment screen calls local AssessmentService"
```

---

## Task 11: Result-screen fallback indicator

**Files:**
- Modify: `flutter_app/lib/screens/assessment/result_screen.dart`
- Modify: `flutter_app/lib/l10n/translations.dart`

- [ ] **Step 1: Add the translation key**

Open `flutter_app/lib/l10n/translations.dart` and add to the EN map:

```dart
'fallback_used': 'WHO median fallback used (on-device ML unavailable)',
```

For MR:

```dart
'fallback_used': 'WHO सरासरी वापरली (ऑन-डिव्हाइस ML अनुपलब्ध)',
```

- [ ] **Step 2: Show the badge in `result_screen.dart`**

In `_statusBanner` of `flutter_app/lib/screens/assessment/result_screen.dart`, after the `Text(message)` line at the end of the `Column`'s children list, add:

```dart
          if (result.mlPrediction == null) ...[
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.amber.shade100,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                t('fallback_used', ref),
                style: const TextStyle(fontSize: 11),
              ),
            ),
          ],
```

- [ ] **Step 3: Run analyze**

```bash
cd flutter_app && flutter analyze
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add flutter_app/lib/screens/assessment/result_screen.dart \
        flutter_app/lib/l10n/translations.dart
git commit -m "feat(flutter): show WHO-fallback badge on result when ML unavailable"
```

---

## Task 12: `SyncService` + tests

**Files:**
- Create: `flutter_app/lib/services/sync_service.dart`
- Create: `flutter_app/lib/providers/sync_provider.dart`
- Test: `flutter_app/test/services/sync_service_test.dart`

- [ ] **Step 1: Write the failing test**

Create `flutter_app/test/services/sync_service_test.dart`:

```dart
import 'dart:io';

import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/sync_queue_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';
import 'package:child_growth_monitor_app/services/sync_service.dart';

Future<int> _seedVisit(AppDatabase db) async {
  final childDao = ChildDao(db);
  final visitDao = VisitDao(db);
  final syncDao = SyncQueueDao(db);
  final child = await childDao.findOrCreate(
    name: 'A', dateOfBirth: '2024-01-01', sex: 'F',
  );
  final tmp = File(
      '${Directory.systemTemp.path}/sync_${DateTime.now().microsecondsSinceEpoch}.jpg')
    ..writeAsBytesSync([1, 2, 3]);
  final visitId = await visitDao.createWithMeasurement(
    childId: child.id,
    ageMonths: 12,
    imagePath: tmp.path,
    measurement: const MeasurementsCompanion(),
  );
  return syncDao.enqueue(visitId);
}

void main() {
  late AppDatabase db;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
  });

  tearDown(() async => db.close());

  test('drains pending queue on success', () async {
    final queueId = await _seedVisit(db);
    final mockClient = MockClient((_) async {
      return http.Response(
          '{"server_visit_id": 7, "status": "synced"}', 200);
    });
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: mockClient,
    );

    await svc.runOnce();

    final entry = await (db.select(db.syncQueue)
          ..where((s) => s.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'synced');
    expect(entry.serverVisitId, 7);
  });

  test('treats already_synced as success', () async {
    final queueId = await _seedVisit(db);
    final mockClient = MockClient((_) async {
      return http.Response(
          '{"server_visit_id": 9, "status": "already_synced"}', 200);
    });
    final svc = SyncService(
      db: db, visitDao: VisitDao(db), childDao: ChildDao(db),
      syncDao: SyncQueueDao(db), baseUrl: 'http://example.test',
      httpClient: mockClient,
    );
    await svc.runOnce();
    final entry = await (db.select(db.syncQueue)
          ..where((s) => s.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'synced');
  });

  test('marks failed and increments retry on 500', () async {
    final queueId = await _seedVisit(db);
    final mockClient = MockClient((_) async => http.Response('boom', 500));
    final svc = SyncService(
      db: db, visitDao: VisitDao(db), childDao: ChildDao(db),
      syncDao: SyncQueueDao(db), baseUrl: 'http://example.test',
      httpClient: mockClient,
    );
    await svc.runOnce();
    final entry = await (db.select(db.syncQueue)
          ..where((s) => s.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'failed');
    expect(entry.retryCount, 1);
    expect(entry.errorMessage, contains('500'));
  });

  test('skips entries past 5 retries', () async {
    await _seedVisit(db);
    await db.update(db.syncQueue).write(
        const SyncQueueCompanion(retryCount: Value(5)));
    var calls = 0;
    final mockClient = MockClient((_) async {
      calls++;
      return http.Response('{}', 200);
    });
    final svc = SyncService(
      db: db, visitDao: VisitDao(db), childDao: ChildDao(db),
      syncDao: SyncQueueDao(db), baseUrl: 'http://example.test',
      httpClient: mockClient,
    );
    await svc.runOnce();
    expect(calls, 0);
  });
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd flutter_app && flutter test test/services/sync_service_test.dart
```
Expected: FAIL — `SyncService` does not exist.

- [ ] **Step 3: Implement `SyncService`**

Create `flutter_app/lib/services/sync_service.dart`:

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:drift/drift.dart';
import 'package:http/http.dart' as http;

import '../database/daos/child_dao.dart';
import '../database/daos/sync_queue_dao.dart';
import '../database/daos/visit_dao.dart';
import '../database/database.dart';

class SyncService {
  SyncService({
    required AppDatabase db,
    required VisitDao visitDao,
    required ChildDao childDao,
    required SyncQueueDao syncDao,
    required String baseUrl,
    http.Client? httpClient,
  })  : _db = db,
        _visitDao = visitDao,
        _childDao = childDao,
        _syncDao = syncDao,
        _baseUrl = baseUrl,
        _client = httpClient ?? http.Client();

  final AppDatabase _db;
  final VisitDao _visitDao;
  final ChildDao _childDao;
  final SyncQueueDao _syncDao;
  final String _baseUrl;
  final http.Client _client;

  static const _maxRetries = 5;

  Future<void> runOnce() async {
    final entries = await (_db.select(_db.syncQueue)
          ..where((s) =>
              (s.status.equals('pending') | s.status.equals('failed')) &
              s.retryCount.isSmallerThanValue(_maxRetries))
          ..orderBy([(s) => OrderingTerm.asc(s.createdAt)]))
        .get();

    for (final entry in entries) {
      await _syncOne(entry);
    }
  }

  Future<void> _syncOne(SyncQueueData entry) async {
    await _syncDao.markSyncing(entry.id);
    try {
      final pair = await _visitDao.getById(entry.visitId);
      if (pair == null) {
        await _syncDao.markFailed(entry.id, 'Visit not found');
        return;
      }
      final child = await _childDao.getById(pair.visit.childId);
      if (child == null) {
        await _syncDao.markFailed(entry.id, 'Child not found');
        return;
      }
      final m = pair.measurement;

      final uri = Uri.parse('$_baseUrl/api/v1/sync');
      final req = http.MultipartRequest('POST', uri);
      req.fields.addAll({
        'local_uuid': pair.visit.localUuid,
        'child_name': child.name,
        'date_of_birth': child.dateOfBirth,
        'sex': child.sex,
        'age_months': pair.visit.ageMonths.toString(),
        'visit_date': pair.visit.visitDate.toIso8601String(),
        if (m?.predictedHeightCm != null)
          'predicted_height_cm': m!.predictedHeightCm.toString(),
        if (m?.predictedWeightKg != null)
          'predicted_weight_kg': m!.predictedWeightKg.toString(),
        if (m?.manualHeightCm != null)
          'manual_height_cm': m!.manualHeightCm.toString(),
        if (m?.manualWeightKg != null)
          'manual_weight_kg': m!.manualWeightKg.toString(),
        if (m?.hazZscore != null) 'haz_zscore': m!.hazZscore.toString(),
        if (m?.whzZscore != null) 'whz_zscore': m!.whzZscore.toString(),
        if (m?.hazStatus != null) 'haz_status': m!.hazStatus!,
        if (m?.whzStatus != null) 'whz_status': m!.whzStatus!,
        if (m?.confidenceScore != null)
          'confidence_score': m!.confidenceScore.toString(),
        if (m?.bodyBuild != null) 'body_build': m!.bodyBuild!,
        'side_view_used': (m?.sideViewUsed ?? false).toString(),
        if (m?.chestDepthCm != null)
          'chest_depth_cm': m!.chestDepthCm.toString(),
        if (m?.abdDepthCm != null) 'abd_depth_cm': m!.abdDepthCm.toString(),
        if (m?.mlEstimatedWeightKg != null)
          'ml_estimated_weight_kg': m!.mlEstimatedWeightKg.toString(),
        if (m?.wastingStatus != null) 'ml_wasting_status': m!.wastingStatus!,
        if (m?.samProbability != null)
          'sam_probability': m!.samProbability.toString(),
        if (m?.mamProbability != null)
          'mam_probability': m!.mamProbability.toString(),
        if (m?.normalProbability != null)
          'normal_probability': m!.normalProbability.toString(),
        if (m?.riskOverweightProbability != null)
          'risk_probability': m!.riskOverweightProbability.toString(),
        if (m?.overweightProbability != null)
          'overweight_probability': m!.overweightProbability.toString(),
        if (m?.muacCm != null) 'muac_cm': m!.muacCm.toString(),
        if (m?.muacStatus != null) 'muac_status': m!.muacStatus!,
        if (m?.muacMethod != null) 'muac_method': m!.muacMethod!,
        if (child.guardianName != null) 'guardian_name': child.guardianName!,
        if (child.location != null) 'location': child.location!,
      });

      if (await File(pair.visit.imagePath).exists()) {
        req.files.add(
            await http.MultipartFile.fromPath('image', pair.visit.imagePath));
      }
      if (pair.visit.sideImagePath != null &&
          await File(pair.visit.sideImagePath!).exists()) {
        req.files.add(await http.MultipartFile.fromPath(
            'image_side', pair.visit.sideImagePath!));
      }
      if (pair.visit.backImagePath != null &&
          await File(pair.visit.backImagePath!).exists()) {
        req.files.add(await http.MultipartFile.fromPath(
            'image_back', pair.visit.backImagePath!));
      }

      final streamed =
          await _client.send(req).timeout(const Duration(seconds: 60));
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        await _syncDao.markSynced(entry.id,
            serverVisitId: body['server_visit_id'] as int?);
      } else {
        await _syncDao.markFailed(
            entry.id, 'HTTP ${response.statusCode}: ${response.body}');
      }
    } catch (e) {
      await _syncDao.markFailed(entry.id, e.toString());
    }
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd flutter_app && flutter test test/services/sync_service_test.dart
```
Expected: 4 passed.

- [ ] **Step 5: Create the sync provider**

Create `flutter_app/lib/providers/sync_provider.dart`:

```dart
import 'dart:async';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/sync_service.dart';
import 'api_provider.dart';
import 'database_provider.dart';

final syncServiceProvider = Provider<SyncService>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  return SyncService(
    db: ref.watch(databaseProvider),
    visitDao: ref.watch(visitDaoProvider),
    childDao: ref.watch(childDaoProvider),
    syncDao: ref.watch(syncQueueDaoProvider),
    baseUrl: effectiveBaseUrl(baseUrl),
  );
});

/// Live count of pending/failed visits awaiting sync.
final pendingSyncCountProvider = StreamProvider<int>((ref) {
  return ref.watch(syncQueueDaoProvider).watchPendingCount();
});

/// Long-lived listener: triggers sync on connectivity changes.
/// Started by main.dart via `ref.read(syncTriggerProvider)`.
final syncTriggerProvider = Provider<StreamSubscription>((ref) {
  final svc = ref.watch(syncServiceProvider);
  final sub = Connectivity().onConnectivityChanged.listen((results) {
    final online = results.any((r) =>
        r == ConnectivityResult.wifi ||
        r == ConnectivityResult.mobile ||
        r == ConnectivityResult.ethernet);
    if (online) {
      svc.runOnce();
    }
  });
  ref.onDispose(sub.cancel);
  return sub;
});
```

- [ ] **Step 6: Run analyze**

```bash
cd flutter_app && flutter analyze
```
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add flutter_app/lib/services/sync_service.dart \
        flutter_app/lib/providers/sync_provider.dart \
        flutter_app/test/services/sync_service_test.dart
git commit -m "feat(flutter): add SyncService that drains queue to /api/v1/sync"
```

---

## Task 13: Wire SyncService into app startup + UI

**Files:**
- Modify: `flutter_app/lib/main.dart`
- Modify: `flutter_app/lib/screens/shared/app_scaffold.dart`
- Modify: `flutter_app/lib/screens/settings/settings_screen.dart`
- Modify: `flutter_app/lib/l10n/translations.dart`

- [ ] **Step 1: Start the sync trigger in `main.dart`**

Replace `flutter_app/lib/main.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/api_provider.dart';
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
    ref.read(syncTriggerProvider);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'SNEH Growth Monitor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      routerConfig: appRouter,
    );
  }
}
```

- [ ] **Step 2: Add sync icon + badge to `AppScaffold`**

Replace `flutter_app/lib/screens/shared/app_scaffold.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../l10n/l10n_provider.dart';
import '../../providers/sync_provider.dart';

class AppScaffold extends ConsumerWidget {
  const AppScaffold({
    super.key,
    required this.child,
    required this.currentIndex,
  });

  final Widget child;
  final int currentIndex;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final pending = ref.watch(pendingSyncCountProvider).value ?? 0;
    return Scaffold(
      appBar: AppBar(
        title: Text(t('app_title', ref)),
        actions: [
          IconButton(
            tooltip: t('sync_status', ref),
            onPressed: () => context.go('/settings'),
            icon: Stack(
              clipBehavior: Clip.none,
              children: [
                Icon(pending == 0 ? Icons.cloud_done : Icons.cloud_upload),
                if (pending > 0)
                  Positioned(
                    right: -4,
                    top: -4,
                    child: Container(
                      padding: const EdgeInsets.all(2),
                      decoration: const BoxDecoration(
                        color: Colors.red,
                        shape: BoxShape.circle,
                      ),
                      constraints:
                          const BoxConstraints(minWidth: 16, minHeight: 16),
                      child: Text(
                        '$pending',
                        style:
                            const TextStyle(color: Colors.white, fontSize: 10),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
              ],
            ),
          ),
          TextButton(
            onPressed: () => ref.read(localeProvider.notifier).toggle(),
            child: Text(
              ref.watch(localeProvider) == 'en'
                  ? t('lang_mr', ref)
                  : t('lang_en', ref),
              style: TextStyle(
                color: Theme.of(context).colorScheme.onPrimary,
              ),
            ),
          ),
        ],
      ),
      body: child,
      bottomNavigationBar: NavigationBar(
        selectedIndex: currentIndex,
        onDestinationSelected: (index) {
          switch (index) {
            case 0:
              context.go('/');
            case 1:
              context.go('/children');
            case 2:
              context.go('/settings');
          }
        },
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.assessment),
            label: t('nav_assess', ref),
          ),
          NavigationDestination(
            icon: const Icon(Icons.people),
            label: t('nav_children', ref),
          ),
          NavigationDestination(
            icon: const Icon(Icons.settings),
            label: t('nav_settings', ref),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 3: Add Sync Now + Storage cards to `SettingsScreen`**

Edit `flutter_app/lib/screens/settings/settings_screen.dart`. Add the imports at the top:

```dart
import '../../providers/sync_provider.dart';
import '../../services/image_storage_service.dart';
```

Add these methods and state to `_SettingsScreenState`:

```dart
  bool _syncing = false;
  int? _bytesUsed;

  Future<void> _refreshStorage() async {
    final used = await ImageStorageService().totalUsedBytes();
    if (!mounted) return;
    setState(() => _bytesUsed = used);
  }

  Future<void> _syncNow() async {
    setState(() => _syncing = true);
    try {
      await ref.read(syncServiceProvider).runOnce();
    } finally {
      if (mounted) setState(() => _syncing = false);
    }
  }

  Future<void> _clearImages() async {
    await ImageStorageService().clearAll();
    await _refreshStorage();
  }

  String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';
  }
```

Call `_refreshStorage()` from `initState()` after `_loadUrl()`. In `build()`, after the Server Connection `Card`, add:

```dart
          const SizedBox(height: 16),

          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(t('sync_status', ref),
                      style: theme.textTheme.titleMedium),
                  const SizedBox(height: 12),
                  Consumer(builder: (context, ref, _) {
                    final pending =
                        ref.watch(pendingSyncCountProvider).value ?? 0;
                    return Text(
                      pending == 0
                          ? t('sync_all_synced', ref)
                          : '${t('sync_pending', ref)}: $pending',
                    );
                  }),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: _syncing ? null : _syncNow,
                    icon: _syncing
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.sync),
                    label: Text(t('sync_now', ref)),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(t('storage_title', ref),
                      style: theme.textTheme.titleMedium),
                  const SizedBox(height: 8),
                  Text(
                    _bytesUsed == null
                        ? '...'
                        : '${t('storage_used', ref)}: ${_formatBytes(_bytesUsed!)}',
                  ),
                  const SizedBox(height: 12),
                  OutlinedButton.icon(
                    onPressed: _clearImages,
                    icon: const Icon(Icons.delete_outline),
                    label: Text(t('storage_clear', ref)),
                  ),
                ],
              ),
            ),
          ),
```

- [ ] **Step 4: Add the new translation keys**

Edit `flutter_app/lib/l10n/translations.dart`. Add to both EN and MR maps:

EN:
```dart
'sync_status': 'Sync status',
'sync_now': 'Sync now',
'sync_pending': 'Pending',
'sync_all_synced': 'All synced',
'storage_title': 'Storage',
'storage_used': 'Used',
'storage_clear': 'Clear all images',
```

MR:
```dart
'sync_status': 'सिंक स्थिती',
'sync_now': 'आता सिंक करा',
'sync_pending': 'प्रलंबित',
'sync_all_synced': 'सर्व सिंक केले',
'storage_title': 'स्टोरेज',
'storage_used': 'वापरले',
'storage_clear': 'सर्व प्रतिमा हटवा',
```

- [ ] **Step 5: Run analyze + tests**

```bash
cd flutter_app && flutter analyze && flutter test
```
Expected: clean and all tests pass.

- [ ] **Step 6: Commit**

```bash
git add flutter_app/lib/main.dart \
        flutter_app/lib/screens/shared/app_scaffold.dart \
        flutter_app/lib/screens/settings/settings_screen.dart \
        flutter_app/lib/l10n/translations.dart
git commit -m "feat(flutter): wire sync trigger + add sync/storage UI in settings"
```

---

## Task 14: Manual validation on a real Android device

This task is procedural — no code, no commits. Execute against a connected Android device or emulator.

- [ ] **Step 1: Install the debug build**

```bash
cd flutter_app && flutter run -d <device-id>
```

- [ ] **Step 2: Test 1 — Offline assessment**

Enable airplane mode on the device. Capture or pick a front-view photo. Fill in the form. Run assessment. Confirm the Result screen shows status banner + metric cards. Result must appear without any network call.

- [ ] **Step 3: Test 2 — Compare against Python backend**

With the Python backend running locally, submit the same image + child info via the web UI (`http://localhost:8000`). Compare HAZ z-score, WHZ z-score, and wasting class against the Flutter result. Both must agree within ±0.1 on z-scores and exactly on class.

- [ ] **Step 4: Test 3 — ML failure path**

Edit `flutter_app/pubspec.yaml` to temporarily comment out the `assets/models/` line, then `flutter run` again. Run an assessment. Result must complete and show the "WHO median fallback used" badge from Task 11. Restore `pubspec.yaml` afterwards.

- [ ] **Step 5: Test 4 — Sync drain**

Re-enable airplane mode. Run 3 assessments. Confirm app-bar badge shows `3`. Disable airplane mode. Within ~10s, badge drops to 0. Backend `GET /api/v1/children` returns the 3 children with their visits.

- [ ] **Step 6: Test 5 — Sync resilience**

Re-enable airplane mode. Run 1 assessment. Force-stop the app. Reopen. Disable airplane mode. The pending visit syncs, no duplicate created server-side.

- [ ] **Step 7: Test 6 — Storage**

Settings → Storage card shows non-zero MB. Tap "Clear all images". Counter goes to 0. Existing visits in Children list still show (only the image files are gone — DB rows survive).

If any of these fail, file an issue and fix before proceeding.

---

## Task 15: Build the release APK

- [ ] **Step 1: Run the existing release build script**

```bash
cd flutter_app && API_BASE_URL=http://<your-server-or-domain>:8000 ./scripts/build_android_release.sh
```

- [ ] **Step 2: Verify the APK is at the expected location**

```bash
ls -la flutter_app/build/app/outputs/flutter-apk/app-release.apk
```
Expected: file exists, > 30 MB.

- [ ] **Step 3: Smoke install on a test device**

```bash
adb install -r flutter_app/build/app/outputs/flutter-apk/app-release.apk
```

Open the app, run a single full assessment on the device. Confirm result displays. The MVP is now sharable.

- [ ] **Step 4: Tag the release**

```bash
git tag -a flutter-mvp-v0.3.0 -m "Offline-first MVP: on-device assessment + sync"
```

(Do not push the tag until the user confirms.)

---

## Self-Review

**1. Spec coverage** — every section of `2026-05-05-flutter-app-completion-design.md` maps to at least one task:
- §1 scope/safety → Tasks 1–13 collectively; ML fallback in 8 + 11
- §3.1 MeasurementService → Task 5
- §3.2 MlInferenceService → Task 6
- §3.3 AssessmentService → Task 8
- §3.4 SyncService → Task 12
- §3.5 ImageStorageService → Task 7
- §3.6 schema additions → Task 3 (Flutter), Task 1 (backend)
- §3.7 provider rewiring → Task 9
- §3.8 backend /sync → Task 2
- §3.9 unchanged components → preserved across all tasks
- §4 testing → unit tests baked into each service task; manual gates → Task 14
- §5 file-by-file → mapped 1:1 in the File Structure section above
- §6 order of work → matches Task numbering 1→15
- §7 risks → addressed (TFLite shape assertion in Task 6; destructive Drift migration documented in Task 3; storage-fill mitigation in Task 13)

**2. Placeholder scan** — no TBD/TODO/"add appropriate" patterns; every code-bearing step contains the actual code.

**3. Type consistency** — `BodyMeasurements` (Task 5) used in `AssessmentService` (Task 8); `WastingPrediction` from `wasting_features.dart` used by `MlInferenceService` (Task 6) and `AssessmentService` (Task 8); `SyncQueueData` used in Task 12 matches the existing DAO; `localUuid` field name consistent across tables, DAOs, and SyncService.

---

## Plan complete

Plan complete and saved to `docs/superpowers/plans/2026-05-05-flutter-app-completion.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
