# Flutter App Completion — Offline-First MVP

**Date:** 2026-05-05
**Status:** Draft
**Scope:** Ship-ready offline-first Flutter MVP for internal field testing (sideloaded APK).

---

## 1. Goal & Scope

A field worker must be able to complete a full WHO-grade child growth assessment with no internet, see the result immediately, and have it sync to the FastAPI backend automatically when connectivity returns.

### In scope
- Wire the existing on-device pose / WHO / nutrition services into the Assessment screen (currently the screen still calls the backend).
- Build the three missing service-layer pieces: `MeasurementService`, `MlInferenceService`, `AssessmentService`.
- Bundle the trained TFLite models + scaler params as Flutter assets.
- Switch the Children list and detail screens to read from the local Drift DB.
- Build a `SyncService` that drains `SyncQueue` to a new `POST /api/v1/sync` endpoint.
- Add the `/api/v1/sync` endpoint to FastAPI with `local_uuid` dedup, no server-side ML.
- Persist images to app documents directory; add a Settings storage row + manual clear button.
- Graceful WHO-median fallback when on-device ML inference fails, labelled in the result UI.
- Surface sync status in the app bar (cloud icon + pending count badge).
- Build a release APK via the existing `scripts/build_android_release.sh` for sideload distribution.

### Out of scope (explicit)
- Switching pose-detection model — stay on `google_mlkit_pose_detection` `accurate` tier. Bundling MediaPipe `pose_landmarker_heavy.task` is a bounded follow-up if field accuracy proves to be the bottleneck.
- Play Store distribution (sideload only for MVP).
- Server-side rewrites of `/api/v1/assess` (untouched — still serves the web UI).
- New features: GPS, photo annotation overlay, growth-trend predictions, supervisor approval workflow, multi-user accounts.
- Conflict resolution beyond last-write-wins by `local_uuid`.
- Auto-deletion of synced images (kept forever per user choice; manual clear button only).

### Safety invariants (preserved from the Python backend)
- HAZ + WHZ always computed when their inputs are available; never silently skipped.
- Weight priority: manual > ML estimate (within 45–180% of WHO median for height) > WHO median × body-build adjustment.
- MUAC thresholds fixed by WHO: <11.5 SAM, 11.5–12.5 MAM, ≥12.5 Normal.
- ML failure → WHO fallback, never a hard fail; the UI must mark the result as fallback.
- `local_uuid` is generated at visit creation and ensures sync retries never create duplicates server-side.

---

## 2. Architecture

```
User captures image(s)
    ↓
PoseService → 33 landmarks (google_mlkit_pose_detection)
    ↓
MeasurementService → body segments in cm, height, body build, confidence
    ↓
WhoDataService → WHO medians, LMS parameters (already implemented)
    ↓
MlInferenceService → weight estimate + 5-class wasting probabilities (TFLite)
    ↓
NutritionService → HAZ/WHZ z-scores; MuacService → MUAC + classification
    ↓
AssessmentService → orchestrates above, persists to Drift DB, enqueues SyncQueue
    ↓
Result displayed immediately (no network needed)
    ↓
SyncService (background) → POST /api/v1/sync when online; dedup by local_uuid
```

**Key principle**: assessment never requires the network. Sync is best-effort and idempotent.

---

## 3. Component Design

### 3.1 New: `MeasurementService` (`lib/services/measurement_service.dart`)

Pure-Dart, no I/O. Takes the outputs of `PoseService` + child age/sex, returns measured body segments in cm and a height estimate.

**Inputs**: `BodySegments` (pixels), `SideViewSegments?` (pixels), `ageMonths`, `sex`, optional `manualHeightCm`.

**Outputs**: `BodyMeasurements` { `effectiveHeightCm`, `shoulderWidthCm`, `hipWidthCm`, `torsoLengthCm`, `upperArmLengthCm`, `chestDepthCm?`, `abdDepthCm?`, `bodyBuild` (slender/average/stocky), `bodyBuildScore` (-1/0/+1), `confidence`, `estimationMethod` (manual/who_statistical), `sideViewUsed` }.

**Logic** (port of Python `measurement_service.py` + `assessment_service.py` height resolution):
1. **Effective height**: `manualHeightCm` if provided, else WHO median for age/sex from `WhoDataService.getMedianHeightForAge`.
2. **Pixels → cm**: `scale = effectiveHeightCm / totalHeightPx`. If `totalHeightPx` is null, fall back per the Snyder ratios in `WastingFeatures.toArray()`.
3. **Side-view depths**: if `SideViewSegments` provided and valid (already gated in PoseService), `chestDepthCm = chestDepthPx × sideScale`. Else null (downstream imputes from shoulder/hip widths).
4. **Body build**: compare `shoulderWidthCm / heightCm` to `expectedShoulderRatio(ageMonths)` from `config.dart`, classify with `bodyBuildThresholdMl`.

### 3.2 New: `MlInferenceService` (`lib/services/ml_inference_service.dart`)

**Inputs**: `WastingFeatures` (already exists in `models/wasting_features.dart`).

**Outputs**: `WastingPrediction` { weight kg, 5 class probabilities, status string }.

**Logic**:
1. Lazy-load both TFLite models from `assets/models/*.tflite` via `tflite_flutter` on first use.
2. Lazy-load the 14-feature scaler from `assets/models/feature_scaler.json` (`{mean: [...], scale: [...]}`).
3. Apply `(x - mean) / scale` to the 14-feature vector.
4. Run `weight_estimator.tflite` → kg.
5. Run `wasting_classifier.tflite` → softmax over `[MAM, Normal, Overweight, Risk_Overweight, SAM]`. Take argmax for status.
6. Validate weight: if outside `[0.45, 1.80] × WHO median for height`, return null weight (caller falls back).
7. Errors thrown if model/scaler missing or output shape mismatched — caller catches and triggers WHO-median fallback.
8. Output tensor shapes are read from `interpreter.getOutputTensors()` at load time and asserted against expected shapes; mismatch → throw at load, not at inference.

**Model assets**: must replace the placeholder `assets/models/feature_scaler.json` (currently 46 bytes) and ship the two real `.tflite` files (`weight_estimator.tflite`, `wasting_classifier.tflite`) — they exist in `data/models/` but aren't in `flutter_app/assets/models/` yet. The scaler JSON gets generated by a one-shot Python script (lifted from `README.md`'s "Export the StandardScaler parameters" snippet).

### 3.3 New: `AssessmentService` (`lib/services/assessment_service.dart`)

The pipeline orchestrator. **One method**, called by the UI:

```dart
Future<AssessmentResult> runAssessment({
  required String frontImagePath, String? sideImagePath,
  required String childName, required String dateOfBirth, required String sex,
  double? manualWeightKg, double? manualHeightCm, double? manualMuacCm,
  String? guardianName, String? location,
})
```

**Steps** (port of Python `assessment_service.py`):
1. Compute `ageMonths` from DOB.
2. Move images from temp → app documents dir via `ImageStorageService`, get permanent paths.
3. `PoseService.detectPose(front)` → landmarks; `extractSegments` → `BodySegments`. If side image given, repeat for it.
4. `MeasurementService` → `BodyMeasurements`.
5. Build `WastingFeatures` (14-vector); call `MlInferenceService` (catch errors → fallback path with `wastingMethod = "who_fallback"`).
6. **Weight resolution**: manual > ML (if in 45–180% bounds) > WHO median × body-build adjustment.
7. `NutritionService.computeHaz` + `computeWhz`; classify via `config.dart`.
8. `MuacService.estimate`.
9. Persist via `ChildDao.findOrCreate`, `VisitDao.createWithMeasurement` (which generates the `localUuid`), `SyncQueueDao.enqueue`.
10. Return `AssessmentResult` shaped exactly like the existing API response (so `result_screen.dart` works unchanged).

This is the only place that knows about the full pipeline; nothing else orchestrates.

### 3.4 New: `SyncService` (`lib/services/sync_service.dart`)

Background uploader. Owned by a Riverpod provider, started in `main.dart` after DB init.

**Triggers**:
- `connectivity_plus` stream → fires when connectivity changes to wifi/mobile.
- Periodic timer: every 15 min while app is foregrounded.
- Manual "Sync Now" button on Settings.

**Loop** (per pending visit, oldest first):
1. `SyncQueueDao.markSyncing(id)`.
2. Load visit + measurement + image bytes.
3. Build multipart `POST /api/v1/sync` with `local_uuid`.
4. Server returns `{ server_visit_id, status: "synced" }` or `{ server_visit_id, status: "already_synced" }` — both treated as success.
5. Success → `markSynced(serverVisitId)`. Failure → `markFailed(error)`, increment retry count. Stop after 5 retries.
6. Backoff between retries: `min(2^retry × 30s, 15min)`.

### 3.5 New: `ImageStorageService` (`lib/services/image_storage_service.dart`)

Owns image lifecycle:
- `persist(tempPath) → permanentPath`: copy temp file into `getApplicationDocumentsDirectory()/images/` with a UUID filename, return path.
- `totalUsedBytes() → int`: sum of all bytes under the images dir.
- `clearSyncedImages()`: delete files referenced by visits whose `SyncQueue.status == 'synced'`. Other images untouched.

Per Q4 user choice: never auto-deletes. The Settings screen exposes total usage + a manual clear button.

### 3.6 Schema additions

- **`Visits` table**: add `localUuid TEXT NOT NULL UNIQUE` (UUID v4, generated at insert via the `uuid` package). Bumps `schemaVersion` from 1 → 2 with a Drift migration.
- **`Measurements` table**: already has all fields needed.
- **No production users yet**, so the v1→v2 migration is destructive (drop + recreate) — acceptable. If this assumption is wrong, swap for a backfill migration that generates UUIDs for existing rows.

Backend DB:
- Add `local_uuid TEXT UNIQUE NULLABLE` column to the server's `visits` table via Alembic-style migration in the FastAPI app. Existing server rows have `NULL` (only sync-ingested rows get a value).

### 3.7 Provider rewiring

| Provider | Before | After |
|---|---|---|
| `assessmentResultProvider` | Set after API response | Set after `AssessmentService.runAssessment` |
| `childrenProvider` | `GET /api/v1/children` | `ChildDao.watchAll()` (Stream → Riverpod) |
| `childDetailProvider` | `GET /api/v1/children/{id}` | `ChildDao.watchById(id)` joined with `VisitDao.watchByChildId` |
| `apiProvider` | Used by all screens | Used only by `SyncService` |
| `syncStatusProvider` (new) | — | `SyncQueueDao.watchPendingCount()` |
| `databaseProvider` (new) | — | Singleton `AppDatabase` |
| `assessmentServiceProvider` (new) | — | Wires `PoseService` + `WhoDataService` + DAOs |

### 3.8 Backend additions (`POST /api/v1/sync`)

Single new route in FastAPI:

```
POST /api/v1/sync
  Content-Type: multipart/form-data
  fields: local_uuid, child_name, date_of_birth, sex,
          age_months, visit_date, predicted_height_cm,
          predicted_weight_kg, haz_zscore, whz_zscore,
          haz_status, whz_status, muac_cm, muac_status,
          muac_method, ml_wasting_status, ml_estimated_weight_kg,
          confidence_score, body_build, side_view_used,
          chest_depth_cm, abd_depth_cm,
          sam_probability, mam_probability, normal_probability,
          risk_probability, overweight_probability,
          guardian_name?, location?
  files: image (front), image_side?, image_back?

  → 200 { server_visit_id, status: "synced" }
  → 200 { server_visit_id, status: "already_synced" }   (idempotent retry)
  → 400 invalid payload
```

No server ML runs. Just `findOrCreate` child, insert visit (skip if `local_uuid` already exists — return `already_synced` with the existing `server_visit_id`), insert measurement, store images.

### 3.9 What stays the same

- `AssessmentResult` and all model classes — the on-device pipeline produces the same shape so `result_screen.dart` and `child_detail_screen.dart` work unchanged.
- `WhoDataService`, `NutritionService`, `MuacService`, `PoseService` — already correct, just need to be invoked.
- The screens themselves (Assessment form, Children list, Settings) — only their data sources change, not their layout or interactions.
- `ApiService` — now used only for `/api/v1/sync`, with the existing `/health` and `/children` endpoints untouched.

---

## 4. Testing & Verification

### 4.1 Test strategy

| Component | Test type | What we cover | Why |
|---|---|---|---|
| `MeasurementService` | Unit (pure Dart, no Flutter) | Body-build classification at thresholds; pixel→cm scale; height resolution priority (manual > WHO median); side-view vs imputed depth | Pure logic; cheap to test; safety-critical (wrong scale = wrong WHZ) |
| `MlInferenceService` | Unit + integration | Scaler load + apply; weight-bound rejection (45–180% of WHO median); model output shape; **error path triggers null prediction** | The fallback path is the safety net — it must work |
| `AssessmentService` | Integration (mocked PoseService + real Drift in-memory) | Full pipeline on a fixture: pose → measurements → ML → nutrition → DB persist → SyncQueue enqueue. ML-failure path produces a labelled fallback result | This is where everything plugs together; one bad wire breaks the whole assessment |
| `SyncService` | Integration (in-memory Drift + mock HTTP) | Drains pending queue; `already_synced` treated as success (idempotent); retry backoff increments; stops at 5 retries; survives connectivity drop mid-batch | Sync correctness — duplicates here = real-world data corruption |
| `ImageStorageService` | Unit | `persist` copies files; `totalUsedBytes` returns correct sum; `clearSyncedImages` only deletes synced | Storage is user-visible |
| Backend `/api/v1/sync` | Pytest with TestClient | Happy path inserts; same `local_uuid` twice returns `already_synced` with the same `server_visit_id`; missing required fields → 400; image files saved to disk | Backend dedup is the second line of defense against duplicates |
| Drift migration v1→v2 | Unit | Migration runs cleanly; new schema accepts inserts | Migration risk |
| `result_screen.dart` | Widget test | "ML fallback" badge shows when `wastingMethod == "who_fallback"` | UX safety |

**Skipped on purpose** (YAGNI for MVP):
- Widget tests for screens that didn't change layout.
- E2E device tests — the unit + integration combo covers the logic; manual field testing covers the rest.
- Performance benchmarks — measure if it becomes a problem.

### 4.2 Manual validation gates (before sharing the APK)

Run these on a real Android device before declaring "done":

1. **Airplane mode → full assessment → result displays.** Confirms offline path.
2. **Compare a known-input result against the Python backend.** Same image + same child info → HAZ/WHZ z-scores match within ±0.1, wasting class identical. Catches porting drift.
3. **Trigger ML failure** (rename the bundled model). Confirms WHO-fallback path completes and is visibly labelled.
4. **Offline assess 3 visits → enable wifi → all 3 sync.** Confirms drain.
5. **Kill app mid-sync → reopen.** Pending items resume. No duplicates server-side.
6. **Storage settings shows used MB.** Clear button removes synced images.

---

## 5. File-by-File Change List

### New files (Flutter)
- `lib/services/measurement_service.dart`
- `lib/services/ml_inference_service.dart`
- `lib/services/assessment_service.dart`
- `lib/services/sync_service.dart`
- `lib/services/image_storage_service.dart`
- `lib/providers/database_provider.dart`
- `lib/providers/sync_provider.dart`
- `lib/providers/assessment_service_provider.dart`
- `lib/models/body_measurements.dart` (typed result of MeasurementService — note: a stub `body_measurements.dart` already exists for `BodySegments`/`SideViewSegments`; add the new `BodyMeasurements` class to the same file or split as preferred during implementation)
- `test/services/measurement_service_test.dart`
- `test/services/ml_inference_service_test.dart`
- `test/services/assessment_service_test.dart`
- `test/services/sync_service_test.dart`
- `test/services/image_storage_service_test.dart`

### Modified files (Flutter)
- `lib/main.dart` — initialize DB, kick off SyncService listener
- `lib/screens/assessment/assessment_screen.dart` — replace `apiProvider.submitAssessment` call with `assessmentServiceProvider.runAssessment`
- `lib/screens/children/children_list_screen.dart` — switch to local DAO stream
- `lib/screens/children/child_detail_screen.dart` — switch to local DAO stream
- `lib/screens/settings/settings_screen.dart` — add Storage section + Sync Now button + pending count
- `lib/screens/shared/app_scaffold.dart` — add cloud icon + pending-count badge
- `lib/screens/assessment/result_screen.dart` — add "WHO fallback used" indicator
- `lib/providers/api_provider.dart` — keep, now used only by SyncService
- `lib/providers/children_provider.dart` — switch to DAO streams
- `lib/database/tables/visits_table.dart` — add `localUuid`
- `lib/database/database.dart` — bump `schemaVersion` to 2 + migration
- `lib/l10n/translations.dart` — strings for sync status, storage, fallback label
- `pubspec.yaml` — add `uuid` package

### New assets
- `flutter_app/assets/models/weight_estimator.tflite` — copied from `data/models/`
- `flutter_app/assets/models/wasting_classifier.tflite` — copied from `data/models/`
- `flutter_app/assets/models/feature_scaler.json` — overwrite the 46-byte placeholder with real scaler params
- `flutter_app/scripts/export_scaler.py` — one-shot script that reads `data/models/feature_scaler.pkl` and writes the JSON

### New backend files
- `app/api/sync.py` — the `/api/v1/sync` route handler
- `tests/test_sync.py` — pytest covering happy path + dedup + missing fields
- Alembic-style migration adding `local_uuid` column to `visits` table

### Modified backend files
- `app/main.py` (or `app/api/__init__.py`) — register the new router
- `app/models/visit.py` — add `local_uuid` column

---

## 6. Order of Work

1. **Backend `/api/v1/sync` first.** Mobile sync work will need an endpoint to call against; doing it first unblocks parallel local dev.
2. **Drift schema migration + `localUuid` field.** Foundation for both assessment and sync.
3. **Asset bundling** (TFLite models + real scaler JSON via the export script).
4. **`MeasurementService` + tests.**
5. **`MlInferenceService` + tests.**
6. **`AssessmentService` + tests** (this is when the full local pipeline becomes runnable).
7. **`ImageStorageService` + tests.**
8. **Assessment screen rewiring** — first time you can run an assessment fully offline.
9. **Provider rewiring for Children list/detail** — local DB becomes the source of truth.
10. **`SyncService` + tests.**
11. **Settings + app-bar UI for sync/storage status.**
12. **Manual validation pass** on a real Android device.
13. **Build the release APK** via the existing `scripts/build_android_release.sh` and share.

---

## 7. Risks & Known Unknowns

- **TFLite output shape might differ from spec.** The spec says weight-estimator `[1,1]` and classifier `[1,5]`, but if the real `.tflite` files were re-exported they could have different shapes. Mitigation: `MlInferenceService` reads `interpreter.getOutputTensors()` shape at load time and asserts before any inference.
- **`google_mlkit_pose_detection` accuracy on small landmarks.** We chose to defer this risk — if field testing shows height estimates are systematically off, switching to bundled MediaPipe heavy is a bounded follow-up project.
- **Drift migration on existing local DBs.** No production users yet, so a destructive migration (drop + recreate) is acceptable. If this assumption is wrong, swap for a backfill migration that generates UUIDs for existing rows.
- **Image storage growth without auto-cleanup.** By design — workers need to be told to use the manual clear button. Mitigation is a visible MB indicator on Settings.
- **Sync against a stale backend schema.** If the `local_uuid` migration isn't applied on the server, `/sync` returns 500. Mitigation: backend startup validates schema; sync requests fail loudly (not silently).

---

## 8. References

- Existing offline-first design: `docs/superpowers/specs/2026-04-09-offline-first-flutter-design.md` (this spec is its completion)
- Mobile spec: `MOBILE_APP_SPEC.md`
- Project conventions: `CLAUDE.md`
- Python services being ported: `app/services/{assessment_service,measurement_service,nutrition_service,who_data_service,ml_service,muac_service}.py`
