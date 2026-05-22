# Offline-First Flutter App with Local Processing

**Date:** 2026-04-09
**Status:** Draft

## Context

The Flutter app is currently a thin client — all assessment processing (MediaPipe pose detection, height/weight estimation, WHO z-score computation, ML wasting classification) runs on the backend FastAPI server. The app requires internet connectivity for every assessment, making it unusable in field conditions where health workers operate without reliable connectivity.

**Goal:** Make the app fully standalone and offline-capable. All processing runs locally on-device. Data syncs to the backend when internet becomes available.

## Architecture Overview

**Approach:** Mirror backend services as standalone Dart service classes. Each Python service (`assessment_service.py`, `measurement_service.py`, `who_data_service.py`, `ml_service.py`) gets a Dart equivalent. This preserves the well-designed separation of concerns and enables independent testing/validation against the Python implementation.

```
User captures image(s)
    ↓
PoseService → 33 landmarks (google_mlkit_pose_detection)
    ↓
MeasurementService → body segments, height, body build, confidence
    ↓
WhoDataService → WHO medians, LMS parameters
    ↓
MlInferenceService → weight estimate + wasting probabilities (TFLite)
    ↓
NutritionService → HAZ/WHZ z-scores, MUAC, classification
    ↓
AssessmentService → orchestrate, persist to Drift DB, queue for sync
    ↓
Result displayed immediately (no network needed)
```

## 1. Local Database (Drift)

### Tables

**Children**
| Column | Type | Notes |
|--------|------|-------|
| id | int (auto PK) | |
| name | text | |
| dateOfBirth | text | ISO 8601 |
| sex | text | M or F |
| guardianName | text? | |
| location | text? | |
| createdAt | dateTime | |
| updatedAt | dateTime | |

**Visits**
| Column | Type | Notes |
|--------|------|-------|
| id | int (auto PK) | |
| childId | int (FK → Children) | |
| visitDate | dateTime | |
| ageMonths | real | |
| imagePath | text | Local file path (front image) |
| sideImagePath | text? | |
| backImagePath | text? | |
| notes | text? | |

**Measurements**
| Column | Type | Notes |
|--------|------|-------|
| id | int (auto PK) | |
| visitId | int (FK → Visits, unique) | |
| predictedHeightCm | real? | |
| predictedWeightKg | real? | |
| manualHeightCm | real? | |
| manualWeightKg | real? | |
| hazZscore | real? | |
| whzZscore | real? | |
| hazStatus | text? | e.g., "Normal", "Stunted" |
| whzStatus | text? | e.g., "SAM", "MAM", "Normal" |
| confidenceScore | real? | 0.0–1.0 |
| bodyBuild | text? | slender/average/stocky |
| estimationMethod | text? | |
| sideViewUsed | bool | default false |
| chestDepthCm | real? | From side view |
| abdDepthCm | real? | From side view |
| mlEstimatedWeightKg | real? | |
| samProbability | real? | |
| mamProbability | real? | |
| normalProbability | real? | |
| riskOverweightProbability | real? | |
| overweightProbability | real? | |
| wastingStatus | text? | |
| muacCm | real? | |
| muacStatus | text? | |
| muacMethod | text? | manual/estimated |

**SyncQueue**
| Column | Type | Notes |
|--------|------|-------|
| id | int (auto PK) | |
| visitId | int (FK → Visits) | |
| status | text | pending / syncing / synced / failed |
| retryCount | int | default 0 |
| createdAt | dateTime | |
| lastAttemptAt | dateTime? | |
| serverVisitId | int? | Returned after successful sync |
| errorMessage | text? | |

### Data Access

- DAOs for each table with standard CRUD + reactive watch queries
- `ChildDao`: findOrCreate by (name, DOB, sex), watchAll with search, watchById with visits
- `VisitDao`: create with measurement, watch by childId
- `SyncQueueDao`: watchPending, markSyncing, markSynced, markFailed

### Provider Changes

Current providers switch from API calls to Drift queries:

| Provider | Before | After |
|----------|--------|-------|
| `childrenProvider` | `GET /api/v1/children` | `ChildDao.watchAll()` |
| `childDetailProvider` | `GET /api/v1/children/:id` | `ChildDao.watchById(id)` with visits + measurements |
| `assessmentResultProvider` | Holds API response | Writes to DB, holds local result |
| `syncProvider` (new) | — | Watches SyncQueue, exposes sync state |

## 2. Local Processing Services

### PoseService
**Wraps:** `google_mlkit_pose_detection`
**Input:** Image file path
**Output:** List of 33 `PoseLandmark` with (x, y, z, likelihood)

- Initialize `PoseDetector` with `PoseDetectorOptions(mode: PoseDetectionMode.single, model: PoseDetectionModel.accurate)`
- Convert `XFile` to `InputImage.fromFilePath()`
- Return raw landmarks; dispose detector after use
- Port the head-top estimation logic from Python's `measurement_service.py`: top of head = 2.5× nose-to-eye distance above nose

### MeasurementService
**Port of:** `measurement_service.py`
**Input:** Pose landmarks (33 points), image dimensions, age/sex
**Output:** `BodyMeasurements` (segments in pixels + cm, height estimate, body build, confidence)

Key logic to port:
1. **Body segment extraction** (lines ~100-200 of measurement_service.py):
   - Head height: top-of-head (estimated) to chin
   - Shoulder width: left shoulder to right shoulder
   - Torso length: shoulder midpoint to hip midpoint
   - Hip width: left hip to right hip
   - Leg length: hip to heel
   - Upper arm length: shoulder to elbow
   - Total height: top-of-head to heel

2. **Height estimation** — hybrid approach:
   - Primary: WHO statistical median for age/sex (from WhoDataService)
   - Supplementary: anthropometric ratios from body proportions
   - Pixels-to-cm conversion using height as scale factor

3. **Body build classification**:
   - Compute shoulder-to-height ratio
   - Compare to expected ratio (Snyder et al. 1975 references)
   - Classify: slender (ratio < expected - 0.03), stocky (ratio > expected + 0.03), average

4. **Confidence scoring**:
   - Base: average visibility of key landmarks
   - Penalties: poor posture, missing landmarks, low visibility
   - Range: 0.0–1.0

5. **Side-view processing** (if side image provided):
   - Run pose detection on side image
   - Extract anterior-posterior (AP) chest depth and abdominal depth
   - Validate: reject if AP depth > 65% of lateral width (frontal photo, not profile)

### WhoDataService
**Port of:** `who_data_service.py`
**Input:** Age, sex, height, weight
**Output:** Z-scores, medians, LMS parameters

Data files bundled as Flutter assets:
- `who_haz_0_59m.csv` — Height-for-Age z-score boundaries
- `who_wfl_boys_0_2.xlsx`, `who_wfl_girls_0_2.xlsx` — Weight-for-Length LMS (0-2 years)
- `who_wfh_boys_2_5.xlsx`, `who_wfh_girls_2_5.xlsx` — Weight-for-Height LMS (2-5 years)

Implementation:
- Parse CSV with Dart's built-in CSV support
- Parse Excel files using `excel` package
- Cache parsed data in memory (singleton or Riverpod provider)
- Load lazily on first assessment, not at app startup

Key functions:
- `getMedianHeightForAge(sex, ageMonths)` — boundary interpolation from HAZ CSV
- `getMedianWeightForHeight(sex, heightCm, ageMonths)` — LMS M parameter lookup
- `getWfhLms(sex, heightCm, ageMonths)` — returns (L, M, S) tuple
- `computeHaz(sex, ageMonths, heightCm)` — boundary interpolation method
- `computeWhz(sex, heightCm, weightKg, ageMonths)` — LMS formula: if L≠0: Z = ((W/M)^L - 1) / (L × S); if L=0: Z = ln(W/M) / S
- `getHeightRangeForAge(sex, ageMonths, sdCount)` — validation ranges

### MlInferenceService
**Port of:** `ml_service.py` + `inference.py`
**Input:** Body segments, age, sex, height
**Output:** Estimated weight, wasting probabilities (5-class)

Models (bundled as assets):
- `weight_estimator.tflite` (~64KB) — regression, output: predicted weight in kg
- `wasting_classifier.tflite` (~200KB) — 5-class softmax (SAM, MAM, Normal, Risk, Overweight)
- `feature_scaler.json` — min/max or mean/std for normalization

14-feature vector (order is critical — must match training):
```
0. age_months
1. sex_binary (1=M, 0=F)
2. height_cm
3. shoulder_width_cm
4. hip_width_cm
5. torso_length_cm
6. upper_arm_length_cm
7. shoulder_height_ratio
8. hip_height_ratio
9. body_build_score (-1=slender, 0=avg, 1=stocky)
10. chest_depth_cm (from side-view OR imputed: shoulder_width × 0.45)
11. abd_depth_cm (from side-view OR imputed: hip_width × 0.50)
12. chest_depth_ratio (chest_depth / height)
13. abd_depth_ratio (abd_depth / height)
```

Implementation:
- Use `tflite_flutter` to load models from assets
- Parse scaler JSON for feature normalization
- Weight estimate validation: must be 45-180% of WHO median for height (reject if out of bounds)
- Dispose interpreters when not in use

### NutritionService
**Port of:** Classification logic from `assessment_service.py` + `config.py`

**HAZ classification:**
- z < -3: Severely Stunted
- -3 ≤ z < -2: Stunted
- -2 ≤ z < 2: Normal
- z ≥ 2: Tall

**WHZ classification:**
- z < -3: SAM
- -3 ≤ z < -2: MAM
- -2 ≤ z < 1: Normal
- 1 ≤ z < 2: Possible Risk of Overweight
- 2 ≤ z < 3: Overweight
- z ≥ 3: Obese

**MUAC estimation** (ages 6-59 months only):
- If manual MUAC provided: use directly
- Otherwise estimate: `medianMuac × (1 + 0.087 × clamp(WHZ, -3, +3))`
- Thresholds: < 11.5 cm = SAM, 11.5–12.5 cm = MAM, ≥ 12.5 cm = Normal

**Weight priority:** manual > ML estimate (if in bounds) > WHO median (with body build adjustment)

### AssessmentService
**Port of:** `assessment_service.py` orchestration logic
**Orchestrates the full pipeline:**

1. Calculate age from DOB
2. Run PoseService on front image → landmarks
3. Run MeasurementService → body segments, height, build, confidence
4. If side image: run PoseService + MeasurementService for AP depth
5. Load WHO data → get medians
6. Run MlInferenceService → weight estimate + wasting probabilities
7. Determine final weight (manual > ML > WHO median with adjustment)
8. Compute HAZ and WHZ via WhoDataService
9. Classify via NutritionService
10. Estimate MUAC
11. Persist Child (findOrCreate), Visit, Measurement to Drift DB
12. Add SyncQueue entry (status: pending)
13. Return `AssessmentResult` for UI display

## 3. Sync Engine

### SyncService

**Triggers:**
- `connectivity_plus` stream: sync when connectivity restored
- Manual "Sync Now" button
- Periodic timer: every 15 minutes when app is open

**Sync flow:**
1. Query SyncQueue for `status = pending` or `status = failed` (with retryCount < 5)
2. For each entry (oldest first):
   a. Mark `status = syncing`
   b. Load visit + measurement + images from local DB/filesystem
   c. Build multipart request matching current `ApiService.submitAssessment` format
   d. POST to `/api/v1/assess` with 60s timeout
   e. On success: mark `status = synced`, store `serverVisitId`
   f. On failure: mark `status = failed`, increment `retryCount`, store `errorMessage`
   g. Exponential backoff between retries: `min(2^retryCount × 30s, 15min)`
3. After batch completes, notify UI of sync results

**Image handling during sync:**
- Images stored in app documents directory (not temp)
- Path stored in Visit record
- Images included as multipart files during sync
- Retained locally after sync (user can manually clear)

### Sync UI
- Sync status indicator in app bar (icon: cloud_done / cloud_off / sync)
- Badge showing count of pending syncs
- Sync settings screen: base URL config (kept from current app), manual sync button, sync history

## 4. Dependencies

### New dependencies (pubspec.yaml additions)
```yaml
# Local processing
google_mlkit_pose_detection: ^0.12.0   # MediaPipe pose detection via ML Kit
tflite_flutter: ^0.11.0                # TFLite model inference
excel: ^4.0.0                          # Parse WHO Excel data files

# Local database
drift: ^2.22.0                         # Type-safe SQLite ORM
sqlite3_flutter_libs: ^0.5.0           # Native SQLite libraries
path_provider: ^2.1.0                  # App document directory paths

# Connectivity
connectivity_plus: ^6.1.0              # Network state detection
```

### New dev dependencies
```yaml
drift_dev: ^2.22.0                     # Drift code generation
build_runner: ^2.4.0                   # Code generation runner
```

### Existing dependencies (retained)
- `http` — used by SyncService for upload
- `image_picker` — camera/gallery capture
- `intl` — date formatting
- `shared_preferences` — language pref, base URL
- `flutter_riverpod` — state management
- `go_router` — routing
- `fl_chart` — growth charts

## 5. File Structure

```
flutter_app/lib/
├── main.dart
├── router.dart
├── database/
│   ├── database.dart                  # @DriftDatabase definition
│   ├── database.g.dart                # Generated
│   ├── tables/
│   │   ├── children_table.dart
│   │   ├── visits_table.dart
│   │   ├── measurements_table.dart
│   │   └── sync_queue_table.dart
│   └── daos/
│       ├── child_dao.dart
│       ├── visit_dao.dart
│       └── sync_queue_dao.dart
├── services/
│   ├── api_service.dart               # Retained for sync uploads
│   ├── pose_service.dart              # NEW: MediaPipe wrapper
│   ├── measurement_service.dart       # NEW: body segment extraction
│   ├── who_data_service.dart          # NEW: WHO growth standard lookups
│   ├── ml_inference_service.dart      # NEW: TFLite inference
│   ├── nutrition_service.dart         # NEW: z-score classification + MUAC
│   ├── assessment_service.dart        # NEW: pipeline orchestrator
│   └── sync_service.dart              # NEW: queue-based upload
├── models/
│   ├── child.dart                     # Keep for UI compatibility
│   ├── child_detail.dart              # Keep
│   ├── assessment_result.dart         # Keep
│   ├── body_measurements.dart         # NEW: segments, build, confidence
│   └── wasting_features.dart          # NEW: 14-feature vector
├── providers/
│   ├── database_provider.dart         # NEW: Drift DB + DAO providers
│   ├── api_provider.dart              # Kept for sync
│   ├── assessment_provider.dart       # Modified: writes to local DB
│   ├── children_provider.dart         # Modified: reads from local DB
│   └── sync_provider.dart             # NEW: sync state + triggers
├── screens/                           # Minimal changes
│   ├── assessment/
│   ├── children/
│   └── shared/
└── l10n/
```

### Bundled assets (flutter_app/assets/)
```
assets/
├── models/
│   ├── weight_estimator.tflite
│   ├── wasting_classifier.tflite
│   └── feature_scaler.json
└── who_data/
    ├── who_haz_0_59m.csv
    ├── who_wfl_boys_0_2.xlsx
    ├── who_wfl_girls_0_2.xlsx
    ├── who_wfh_boys_2_5.xlsx
    └── who_wfh_girls_2_5.xlsx
```

Register in `pubspec.yaml`:
```yaml
flutter:
  assets:
    - assets/models/
    - assets/who_data/
```

## 6. Safety Rules (Preserved from Backend)

These safety constraints from the Python backend must be enforced identically on-device:

- **WHO z-score validation:** Every assessment must compute HAZ and WHZ — never skip
- **Manual measurement priority:** Manual > ML estimate > WHO median
- **ML weight bounds:** ML estimates must fall within 45-180% of WHO median to be accepted
- **MUAC thresholds (WHO fixed):** <11.5 SAM, 11.5-12.5 MAM, ≥12.5 Normal
- **No silent failures:** Surface all errors to the user (pose detection failure, missing landmarks, model load errors)
- **14-feature order:** Feature vector order must match training exactly — any change requires retraining + scaler update

## 7. Verification Plan

### Unit tests
- `WhoDataService`: compare z-score outputs against known WHO reference values
- `NutritionService`: verify all classification thresholds
- `MeasurementService`: test body build classification with mock landmarks
- `MlInferenceService`: load TFLite model, run inference on known feature vector, check output bounds
- `SyncQueueDao`: test state transitions (pending → syncing → synced/failed)

### Integration tests
- Full pipeline: provide a test image → run PoseService → MeasurementService → WhoDataService → MlInferenceService → NutritionService → verify AssessmentResult structure
- Sync roundtrip: create local assessment → trigger sync → verify backend receives correct data
- Offline→online: assess while offline → enable connectivity → verify auto-sync

### Manual validation
- Compare assessment results between Flutter app (offline) and backend (online) for the same input image — results should match within acceptable tolerance
- Test with no network: airplane mode → full assessment → verify result displayed
- Test sync recovery: kill app mid-sync → reopen → verify pending items retry

### Flutter commands
```bash
cd flutter_app && flutter test                    # Unit + widget tests
cd flutter_app && flutter analyze                 # Static analysis
cd flutter_app && dart run build_runner build     # Generate Drift code
```
