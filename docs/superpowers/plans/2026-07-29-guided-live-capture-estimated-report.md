# Guided Live Capture, Child Visits, and Estimated Reports Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task-by-task. Work in an isolated feature worktree, keep each task reviewable, and stop if a safety assertion or specified verification fails.

**Goal:** Add a profile-first, offline-first Flutter workflow that creates a dated visit before capture, guides required front and side still-image capture, stores an immutable non-clinical camera estimate, and later attaches audited measured details to that same visit without overwriting the estimate.

**Architecture:** Keep `Visit` as the aggregate root. Add dedicated capture-asset and camera-result records on both Flutter and FastAPI, keep the existing measurement table as the current measurement-based report, and add immutable measured-detail revisions. Replace visit-wide synchronization for this feature with a typed outbox and owner-scoped idempotent server endpoints. The guided workflow runs on-device first and remains behind `LIVE_CAPTURE`; the server accepts camera output only as non-clinical provenance and independently recomputes every measurement-based WHO/Poshan result.

**Tech Stack:** Flutter, Riverpod, Drift, `camera`, Google ML Kit Pose Detection, TFLite, `image`, `sensors_plus`, FastAPI, Pydantic, SQLAlchemy, SQLite, pytest.

**Design:** `docs/superpowers/specs/2026-07-29-guided-live-capture-estimated-report-design.md`

---

## Safety and delivery constraints

- Do not implement feature code directly on `main`. The current main worktree contains unrelated field-pipeline changes; create an isolated worktree from the intended base commit.
- Preserve the 14-feature TFLite interface. This plan does not train, replace, or promote a model.
- Camera-derived values remain `non_clinical=true` and never populate measured HAZ, measured WHZ, tape MUAC, oedema, or `poshan_setu_v1` inputs.
- WHO z-scores for the measurement-based report must use authoritative Excel LMS data. The repository currently uses Excel LMS for WFL/WFH but still reads HAZ boundaries from `who_haz_0_59m.csv`; Task 1A is therefore a blocking prerequisite for this feature.
- Manual values override estimates only in the measurement-based report. They never mutate a `CameraResult`.
- Marker-free pixels are not centimetres. Every displayed centimetre estimate must retain its component method and model version.
- A missing camera component is omitted. It must never be converted into a fabricated `Normal` result.
- A camera-only visit must not show the clinical `Indeterminate` headline. `Indeterminate` remains valid internally for ineligible measured/Poshan computation.
- Required media is retained locally until the server acknowledges each asset UUID. A visit-level response is not sufficient to delete local media.
- Every owner-scoped API query must filter by the authenticated user. Do not rely on child or visit identifiers alone.
- Any new threshold must be a named, versioned constant, not a widget-local magic number.
- Run `flutter analyze` and the full relevant Flutter test suite before considering Flutter work complete.
- Physical Android camera evidence is required before enabling the release flag.

---

## Target domain and contracts

### Visit aggregate

Extend `Visit` with:

- `capture_state`: `draft_capture`, `incomplete_capture`, `processing`, `estimated_report`, `processing_failed`, or `measured_report`;
- `capture_started_at` and nullable `capture_completed_at`;
- `device_metadata_json`;
- consent version, consent timestamp, and consent operator identifier;
- nullable `media_deleted_at`;
- existing child ID, owner ID, local UUID, visit date, and age-at-visit.

`visit_date` remains the clinical date. Any measured details entered for a different date create a different visit.

### Capture asset

One row per retained still:

- local asset UUID and visit UUID/foreign key;
- role: `front`, `side`, `back`, `arm_front`, or `arm_side`;
- local path, nullable server object ID, capture timestamp, and selected rank;
- pose, coverage, orientation, sharpness, lighting, and overall scores;
- quality verdict and rejection reason;
- image dimensions and EXIF/display orientation;
- device/camera metadata JSON;
- local sync state and server acknowledgement timestamp.

Front and side are required. Optional roles never block report generation.

### Camera result

One immutable versioned snapshot per inference run:

- local result UUID and visit UUID/foreign key;
- version number and `supersedes_result_uuid`, when reprocessed;
- estimated height/weight plus separate component source names;
- estimated HAZ/WHZ and stunting/wasting statuses;
- experimental overall category and probability map when the classifier supplies one;
- body-proportion feature JSON and capture-quality summary JSON;
- method `camera_screening_v1`;
- model version, manifest checksum, training-data label;
- `non_clinical=true` and creation timestamp.

No update API is provided for a camera result. Reprocessing inserts a new version.

### Current measured report and revisions

Retain `Measurements`/`MeasurementResult` as the current measurement-based report, but use its manual fields only for measured details on guided visits. Add:

- `measurement_mode`: `standing_height` or `recumbent_length`;
- `oedema`: `yes`, `no`, or `not_checked`;
- `measured_at`, `editor_user_id`, and notes;
- a separately named WHO acute-malnutrition result and triggers.

Add `MeasuredDetailRevision` rows containing:

- revision UUID, visit ID, revision number;
- complete before/after JSON snapshots;
- editor user ID, timestamp, and optional reason.

Measurement saves are atomic: append the revision and replace the current measured report in one transaction. Oedema participates in the WHO acute-malnutrition result but not Poshan Setu v1.

### Typed sync outbox

Add a new outbox for `visit`, `capture_asset`, `camera_result`, `measured_revision`, and `media_deletion` operations. Each entry has its own entity UUID, dependency, retry state, and acknowledgement payload. Keep the existing `SyncQueue` path for legacy assessments until migration is explicitly completed.

---

## PHASE 0 — ISOLATED BASELINE

### Task 1: Create the feature worktree and record the baseline

**Files:**
- No feature files modified
- Record results in the implementation handoff, not in generated source

- [ ] **Step 1: Inspect both divergence directions and the dirty tree**

Run from the current repository:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git rev-list --left-right --count main...origin/main
```

Expected: unrelated dirty field-pipeline files are identified and left untouched.

- [ ] **Step 2: Create an isolated worktree**

Choose an explicit sibling path and feature branch:

```bash
git worktree add -b feat/guided-live-capture \
  /storage/projects/child-growth-monitor/child-growth-monitor-guided-capture \
  main
```

If `main` is not the intended base after the divergence check, stop and resolve the base with the user. Do not copy the dirty working tree into the feature worktree.

- [ ] **Step 3: Confirm the isolated tree is clean**

```bash
git -C /storage/projects/child-growth-monitor/child-growth-monitor-guided-capture status --short
```

Expected: no output.

- [ ] **Step 4: Run baseline tests**

From the feature worktree:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
cd flutter_app
flutter analyze
flutter test
```

Record pre-existing failures exactly. Do not weaken new assertions to accommodate a real baseline failure.

---

### Task 1A: Replace the HAZ CSV runtime dependency with authoritative HFA LMS workbooks

**Blocking prerequisite:** Do not implement the measured-report tasks until this task passes. The current backend and Flutter `WhoDataService` implementations load HAZ boundaries from CSV, while the project requires Excel LMS inputs to be authoritative.

**Files:**
- Add verified official WHO HFA LMS workbooks under: `data/`
- Add the same verified runtime assets under: `flutter_app/assets/who_data/`
- Create: `data/who_reference_manifest.json`
- Modify: `config.py`
- Modify: `app/services/who_data_service.py`
- Modify: `app/services/nutrition_service.py`
- Modify: `flutter_app/lib/services/who_data_service.dart`
- Modify: `flutter_app/lib/services/nutrition_service.dart`
- Test: `tests/test_who_hfa_excel_parity.py`
- Test: `flutter_app/test/services/who_hfa_excel_parity_test.dart`
- Update existing WHO service tests to use real workbooks

**Interfaces:**
- HFA lookup returns sex-specific `(L, M, S)` by completed age in months.
- HAZ uses the LMS formula in both Python and Dart.
- Runtime fails closed when a workbook or checksum is missing/mismatched.
- The CSV may remain only as a non-authoritative migration/parity fixture; it is not a runtime fallback.

- [ ] **Step 1: Obtain and verify the official WHO height/length-for-age LMS workbooks**

Record the source URL, retrieval date, file size, and SHA-256 in `data/who_reference_manifest.json`. If the official files are unavailable or their format cannot be verified, stop and request the files rather than using the existing CSV as a fallback.

- [ ] **Step 2: Write failing Python and Dart parity tests**

Using the real workbooks, cover both sexes and boundary ages 0, 24, and 60 months plus representative known points. Assert Python/Dart HAZ parity to a documented tolerance.

- [ ] **Step 3: Parse HFA LMS data in both runtimes**

Keep WFL/WFH selection unchanged. Add a distinct HFA LMS lookup keyed by sex and completed age month.

- [ ] **Step 4: Switch HAZ computation to LMS**

Remove runtime calls that derive HAZ from CSV boundary interpolation. Missing or invalid Excel data must return a data error, not a Normal classification.

- [ ] **Step 5: Verify checksums and parity**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_who_hfa_excel_parity.py \
  tests/test_who_data_service.py -v
cd flutter_app
flutter test \
  test/services/who_hfa_excel_parity_test.dart \
  test/services/who_data_service_test.dart
```

---

## PHASE 1 — SHARED DOMAIN CONTRACTS

### Task 2: Add canonical visit, role, provenance, and oedema values

**Files:**
- Create: `flutter_app/lib/features/guided_capture/domain/capture_models.dart`
- Create: `app/schemas/guided_capture.py`
- Create: `app/services/guided_capture_contract.py`
- Create: `docs/contracts/guided_capture_v1.json`
- Test: `flutter_app/test/features/guided_capture/domain/capture_models_test.dart`
- Test: `tests/test_guided_capture_contract.py`

**Interfaces:**
- Dart enums serialize to the exact snake-case values in the target-domain section.
- Pydantic models reject unknown states, roles, measurement modes, and oedema values.
- `docs/contracts/guided_capture_v1.json` is the language-neutral compatibility fixture used by both test suites.

- [ ] **Step 1: Write failing Dart enum/serialization tests**

Cover:

- all six visit-state values;
- all five asset roles;
- required roles are exactly front and side;
- all three oedema values;
- unknown wire values fail closed instead of mapping to a safe-looking default.

Run:

```bash
cd flutter_app
flutter test test/features/guided_capture/domain/capture_models_test.dart
```

Expected: fail because the domain file does not exist.

- [ ] **Step 2: Write failing Python contract tests**

Test Pydantic validation, JSON field names, `non_clinical` fixed to true for camera submissions, and finite numeric constraints.

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_guided_capture_contract.py -v
```

Expected: fail because the schema does not exist.

- [ ] **Step 3: Implement the canonical contracts**

Keep transition logic out of widgets and route handlers. Add a pure transition function that allows only:

```text
draft_capture -> incomplete_capture
draft_capture -> processing
incomplete_capture -> draft_capture
processing -> estimated_report
processing -> processing_failed
processing_failed -> processing
estimated_report -> measured_report
measured_report -> measured_report
```

Reprocessing a camera result does not move a measured visit backwards.

- [ ] **Step 4: Add cross-language fixture verification**

Both test suites must load `docs/contracts/guided_capture_v1.json` and assert identical wire values.

- [ ] **Step 5: Verify**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_guided_capture_contract.py -v
cd flutter_app
flutter test test/features/guided_capture/domain/capture_models_test.dart
```

---

## PHASE 2 — BACKEND PERSISTENCE

### Task 3: Add backend capture assets, camera results, measured revisions, and visit state

**Files:**
- Modify: `app/models/visit.py`
- Modify: `app/models/measurement.py`
- Modify: `app/models/__init__.py`
- Modify: `app/models/database.py`
- Create: `app/models/capture_asset.py`
- Create: `app/models/camera_result.py`
- Create: `app/models/measured_detail_revision.py`
- Test: `tests/test_guided_capture_models.py`
- Test: `tests/test_guided_capture_migration.py`

**Interfaces:**
- `Visit.capture_assets` is one-to-many.
- `Visit.camera_results` is ordered and one-to-many.
- `Visit.measured_revisions` is ordered and one-to-many.
- Asset UUID, result UUID, and revision UUID are globally unique.
- Camera result rows are immutable through the service layer.

- [ ] **Step 1: Write failing model tests**

Cover relationships, defaults, unique constraints, required `non_clinical=true`, and independent deletion semantics:

- deleting media metadata must not delete the visit or measurement;
- deleting a visit cascades its asset/result/revision children;
- two camera result versions can coexist;
- duplicate entity UUIDs fail.

- [ ] **Step 2: Write failing migration tests**

Create a database at the current schema, run `run_migrations()`, and assert:

- all new visit and measurement columns exist;
- all three new tables and indexes exist;
- migration is idempotent;
- existing rows retain their values;
- legacy visits receive a safe state such as `measured_report` only when manual measurements exist, otherwise a documented legacy state mapping. Do not label legacy estimates as measured.

- [ ] **Step 3: Implement SQLAlchemy models**

Use typed constants from `guided_capture_contract.py`. Store score maps and feature maps as JSON. Add database checks where SQLite supports them and service-level validation everywhere.

- [ ] **Step 4: Extend the idempotent SQLite migration**

`run_migrations()` must add columns without dropping data and must create missing tables through registered metadata. Add explicit indexes for:

- owner plus local visit UUID;
- visit plus role;
- visit plus result version;
- visit plus revision number.

- [ ] **Step 5: Verify**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_guided_capture_models.py \
  tests/test_guided_capture_migration.py -v
```

---

### Task 4: Add backend service-layer validation and measured-report recomputation

**Files:**
- Create: `app/services/guided_visit_service.py`
- Create: `app/services/acute_malnutrition_service.py`
- Modify: `app/services/poshan_setu_service.py` only if a reusable public helper is required
- Test: `tests/test_guided_visit_service.py`
- Test: `tests/test_acute_malnutrition_service.py`

**Interfaces:**
- `create_draft_visit(...)`
- `transition_visit(...)`
- `append_capture_asset(...)`
- `append_camera_result(...)`
- `save_measured_details(...)`
- `delete_visit_media(...)`

- [ ] **Step 1: Write failing transition and immutability tests**

Assert that:

- required views must have accepted assets before `processing`;
- a camera result can be appended only while processing/reprocessing;
- appending a result moves processing to estimated report;
- camera results cannot be updated;
- measured saves preserve all camera-result rows;
- invalid saves leave the previous measurement and revision history unchanged.

- [ ] **Step 2: Write failing partial-measurement tests**

Use real WHO Excel files for integration cases:

- height/length only produces HAZ/stunting;
- height plus weight produces WHZ/wasting;
- tape MUAC is eligible only from 6 through 59 completed months;
- oedema `yes` independently makes WHO acute malnutrition actionable as SAM;
- missing components stay null and render as `Not measured`;
- Poshan Setu completeness and severity remain unchanged;
- oedema does not change Poshan Setu v1.

- [ ] **Step 3: Implement acute-malnutrition aggregation**

Return separately named:

- WHO HAZ stunting;
- WHO acute malnutrition from eligible WHZ, tape MUAC, and oedema;
- Poshan Setu v1 from its existing eligible BMI and tape-MUAC contract.

The aggregate must include trigger names and eligibility/missing reasons.

- [ ] **Step 4: Implement transactional measured saves**

Within one database transaction:

1. owner-scope the child and visit;
2. verify measurement date equals the visit’s clinical date;
3. validate finite plausible inputs and measurement mode;
4. compute age from DOB and visit date;
5. append the before/after revision;
6. recompute the current measurement-based report;
7. transition to `measured_report`;
8. commit.

- [ ] **Step 5: Verify**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_guided_visit_service.py \
  tests/test_acute_malnutrition_service.py -v
```

---

## PHASE 3 — FLUTTER PERSISTENCE

### Task 5: Add Drift tables and a non-destructive schema migration

**Files:**
- Modify: `flutter_app/lib/database/tables/visits_table.dart`
- Modify: `flutter_app/lib/database/tables/measurements_table.dart`
- Modify: `flutter_app/lib/database/database.dart`
- Create: `flutter_app/lib/database/tables/capture_assets_table.dart`
- Create: `flutter_app/lib/database/tables/camera_results_table.dart`
- Create: `flutter_app/lib/database/tables/measured_detail_revisions_table.dart`
- Create: `flutter_app/lib/database/tables/sync_outbox_table.dart`
- Regenerate: `flutter_app/lib/database/database.g.dart`
- Test: `flutter_app/test/database/guided_capture_migration_test.dart`
- Test: `flutter_app/test/database/guided_capture_constraints_test.dart`

**Interfaces:**
- New tables mirror backend wire fields.
- New guided visits use the typed outbox.
- Existing `SyncQueue` remains readable for legacy assessments.

- [ ] **Step 1: Write failing migration tests**

Build a schema-v4 fixture, insert a representative child/visit/measurement/sync row, upgrade, and assert all old data remains intact.

- [ ] **Step 2: Add table declarations and bump the schema version**

Use nullable additions for legacy data. Add uniqueness and foreign-key constraints in Drift declarations.

- [ ] **Step 3: Implement explicit migration steps**

Do not destructively recreate visits or measurements. Create the four new tables and add only the new nullable/defaulted columns to existing tables.

- [ ] **Step 4: Regenerate Drift code**

```bash
cd flutter_app
dart run build_runner build --delete-conflicting-outputs
```

- [ ] **Step 5: Verify migration and constraints**

```bash
flutter test \
  test/database/guided_capture_migration_test.dart \
  test/database/guided_capture_constraints_test.dart
```

---

### Task 6: Add visit-aggregate DAOs and typed outbox dependencies

**Files:**
- Create: `flutter_app/lib/database/daos/guided_visit_dao.dart`
- Create: `flutter_app/lib/database/daos/capture_asset_dao.dart`
- Create: `flutter_app/lib/database/daos/camera_result_dao.dart`
- Create: `flutter_app/lib/database/daos/measured_detail_revision_dao.dart`
- Create: `flutter_app/lib/database/daos/sync_outbox_dao.dart`
- Modify: `flutter_app/lib/providers/database_provider.dart`
- Test: `flutter_app/test/database/daos/guided_visit_dao_test.dart`
- Test: `flutter_app/test/database/daos/sync_outbox_dao_test.dart`

**Interfaces:**
- `createDraft(...)` persists the visit before opening the camera.
- `saveAcceptedAssets(...)` is atomic.
- `appendCameraResult(...)` rejects in-place replacement.
- `saveMeasuredReport(...)` appends revision and updates current report atomically.
- Outbox operations drain only after their declared dependency is acknowledged.

- [ ] **Step 1: Write failing aggregate transaction tests**

Include rollback injection at each write boundary and assert no partial aggregate is visible.

- [ ] **Step 2: Write failing outbox-order tests**

Required order:

```text
visit -> capture assets -> camera result
visit -> measured revision
asset acknowledgements -> media deletion
```

Retries must retain the original entity UUID and payload checksum.

- [ ] **Step 3: Implement the DAOs**

All queries must include `ownerUserId`. Use the visit’s stable UUID as the aggregate key and entity UUIDs as idempotency keys.

- [ ] **Step 4: Verify**

```bash
cd flutter_app
flutter test \
  test/database/daos/guided_visit_dao_test.dart \
  test/database/daos/sync_outbox_dao_test.dart
```

---

## PHASE 4 — GUIDED QUALITY AND BURST CAPTURE

### Task 7: Extend the live quality gate with role-aware, deterministic scoring

**Files:**
- Modify: `flutter_app/lib/services/capture_quality.dart`
- Create: `flutter_app/lib/features/guided_capture/services/frame_quality_service.dart`
- Create: `flutter_app/lib/features/guided_capture/services/burst_frame_ranker.dart`
- Create: `flutter_app/lib/features/guided_capture/domain/capture_thresholds.dart`
- Modify: `flutter_app/pubspec.yaml`
- Test: `flutter_app/test/services/capture_quality_test.dart`
- Test: `flutter_app/test/features/guided_capture/services/frame_quality_service_test.dart`
- Test: `flutter_app/test/features/guided_capture/services/burst_frame_ranker_test.dart`

**Dependencies:**
- Add `image` for deterministic still-image luminance, contrast, and sharpness analysis.
- Add `sensors_plus` for optional phone-tilt input. Missing sensor data must degrade gracefully.

- [ ] **Step 1: Expand failing quality-gate tests**

Cover, in priority order:

- zero and multiple detected poses;
- wrong front/side orientation;
- head/heel cropping;
- missing role-specific joints;
- body coverage and centring;
- landmark confidence;
- excessive tilt when sensor data exists;
- one actionable instruction only.

The evaluator must accept `poseCount`, role, landmarks, frame dimensions, and nullable tilt. Do not discard all but the first pose before quality evaluation.

- [ ] **Step 2: Add failing still-quality tests**

Use small checked-in synthetic fixtures for:

- dark, overexposed, low-contrast, and acceptably lit images;
- blurred and sharp edge patterns;
- deterministic normalized score ranges.

- [ ] **Step 3: Add failing burst-ranking tests**

Ranking must be deterministic with an explicit tie-breaker and must reject all frames when no frame passes minimum quality.

- [ ] **Step 4: Implement versioned thresholds**

Create `captureThresholdVersion = 'guided_capture_quality_v1'` and named constants for every threshold. Persist the version with every asset/result.

- [ ] **Step 5: Implement role-aware evaluation and burst ranking**

Live evaluation handles pose/orientation/coverage. Post-capture still evaluation handles sharpness, lighting, and final ranking.

- [ ] **Step 6: Verify**

```bash
cd flutter_app
flutter test \
  test/services/capture_quality_test.dart \
  test/features/guided_capture/services/frame_quality_service_test.dart \
  test/features/guided_capture/services/burst_frame_ranker_test.dart
```

---

### Task 8: Build an injectable burst camera controller

**Files:**
- Create: `flutter_app/lib/features/guided_capture/services/guided_camera_controller.dart`
- Create: `flutter_app/lib/features/guided_capture/services/device_metadata_service.dart`
- Refactor: `flutter_app/lib/screens/assessment/capture_screen.dart`
- Test: `flutter_app/test/features/guided_capture/services/guided_camera_controller_test.dart`
- Test: `flutter_app/test/screens/assessment/capture_screen_test.dart`

**Interfaces:**
- Camera plugin access sits behind an injectable gateway.
- A stable quality streak triggers a short still burst.
- Only accepted ranked frames are returned to the workflow.
- Temporary rejected frames are deleted only after accepted files have been durably copied into visit-owned storage.

- [ ] **Step 1: Write failing controller tests with a fake camera gateway**

Cover lifecycle pause/resume, rotation, camera init error, burst failure, retake, and cancellation.

- [ ] **Step 2: Refactor plugin calls behind the gateway**

Keep the existing accurate static pose path for final features. The live base detector remains a gate only.

- [ ] **Step 3: Capture and rank a short still burst**

Persist capture timestamp, dimensions, orientation, camera/lens identifiers, quality components, selected rank, and threshold version with each retained frame.

- [ ] **Step 4: Preserve a controlled fallback**

If live camera initialization fails, the system-camera fallback may collect a still only if the same post-capture role/quality validation passes. It must not bypass the quality gate.

- [ ] **Step 5: Verify**

```bash
cd flutter_app
flutter test \
  test/features/guided_capture/services/guided_camera_controller_test.dart \
  test/screens/assessment/capture_screen_test.dart
```

---

## PHASE 5 — PROFILE-FIRST CAPTURE WORKFLOW

### Task 9: Create the draft visit, consent gate, and resumable capture state

**Files:**
- Create: `flutter_app/lib/features/guided_capture/providers/guided_capture_provider.dart`
- Create: `flutter_app/lib/features/guided_capture/screens/capture_consent_screen.dart`
- Create: `flutter_app/lib/features/guided_capture/screens/guided_capture_flow_screen.dart`
- Create: `flutter_app/lib/features/guided_capture/screens/capture_review_screen.dart`
- Create: `flutter_app/lib/features/guided_capture/widgets/capture_role_card.dart`
- Modify: `flutter_app/lib/router.dart`
- Modify: `flutter_app/lib/screens/children/child_detail_screen.dart`
- Test: `flutter_app/test/features/guided_capture/providers/guided_capture_provider_test.dart`
- Test: `flutter_app/test/features/guided_capture/screens/guided_capture_flow_screen_test.dart`

**Routes:**
- `/children/:id/photo-assessment/consent`
- `/visits/:visitUuid/capture`
- `/visits/:visitUuid/capture/review`

- [ ] **Step 1: Write failing provider state-machine tests**

Assert:

- a valid owner-scoped child is required;
- consent is recorded before camera navigation;
- `createDraft` runs before the first camera screen opens;
- required roles are front and side;
- back and arm roles are skippable;
- interrupted drafts resume from the first missing required role;
- repeated required-role failure can save `incomplete_capture`;
- no height, weight, or MUAC field exists in the capture state.

- [ ] **Step 2: Write failing widget tests**

Cover the profile-to-consent-to-front-to-side-to-review flow using fake camera results.

- [ ] **Step 3: Implement the consent screen**

Display the approved purpose text and persist consent version, timestamp, and operator. Declining returns to the child profile without creating media.

- [ ] **Step 4: Implement the resumable workflow**

Persist every accepted asset immediately through the aggregate DAO. Never keep the only copy in Riverpod memory.

- [ ] **Step 5: Add the profile entry action**

Add **New photo assessment** to child detail only when `FeatureFlags.liveCaptureEnabled` is true. Keep the existing monthly measurement action separate and keep normal release builds gated.

- [ ] **Step 6: Verify**

```bash
cd flutter_app
flutter test \
  test/features/guided_capture/providers/guided_capture_provider_test.dart \
  test/features/guided_capture/screens/guided_capture_flow_screen_test.dart
```

---

## PHASE 6 — IMMUTABLE CAMERA INFERENCE

### Task 10: Split camera screening from measured classification

**Files:**
- Create: `flutter_app/lib/features/guided_capture/services/camera_screening_service.dart`
- Create: `flutter_app/lib/features/guided_capture/domain/camera_screening_result.dart`
- Modify: `flutter_app/lib/services/ml_inference_service.dart`
- Modify: `flutter_app/lib/services/measurement_service.dart` only through reusable pure adapters
- Modify: `flutter_app/lib/services/nutrition_service.dart` only if a clearly named estimated helper is needed
- Modify: `flutter_app/lib/providers/assessment_service_provider.dart`
- Test: `flutter_app/test/features/guided_capture/services/camera_screening_service_test.dart`
- Test: `flutter_app/test/features/guided_capture/domain/camera_screening_result_test.dart`

**Interfaces:**
- `run(visit, acceptedAssets) -> CameraScreeningResult`
- Result method is `camera_screening_v1`.
- Estimated HAZ/WHZ fields are distinct from measured HAZ/WHZ.
- The manifest checksum is computed from the manifest bytes and stored.

- [ ] **Step 1: Write failing provenance-isolation tests**

Assert that camera inference:

- requires accepted front and side assets;
- never writes `Measurements.manual*`, measured HAZ/WHZ, tape MUAC, oedema, or Poshan fields;
- stores component-specific source names for height and weight;
- carries model version, manifest checksum, training label, quality summary, and `nonClinical=true`;
- stores classifier category/probabilities only when valid output exists.

- [ ] **Step 2: Write failing partial-output and failure tests**

Cover:

- height estimate but no weight;
- weight estimate but no category;
- no valid ML output;
- WHO-statistical fallback explicitly labelled as such;
- non-finite model output;
- repeated retry creates version 2 and does not alter version 1.

- [ ] **Step 3: Implement the screening adapter**

Reuse pose/measurement/WHO helpers but return a camera-specific result. Estimated HAZ/WHZ may be computed from estimated inputs for display only; their names and persistence must remain explicitly estimated.

- [ ] **Step 4: Remove silent fallback semantics from this path**

If a WHO median/body-build fallback is used, record that exact component method. Do not create a camera `Normal` category unless the classifier actually produced and passed validation for it.

- [ ] **Step 5: Persist the result and state atomically**

Transition:

```text
draft_capture -> processing -> estimated_report
```

On inference failure:

```text
processing -> processing_failed
```

Keep all accepted assets and expose retry.

- [ ] **Step 6: Verify**

```bash
cd flutter_app
flutter test \
  test/features/guided_capture/services/camera_screening_service_test.dart \
  test/features/guided_capture/domain/camera_screening_result_test.dart
```

---

## PHASE 7 — ESTIMATED REPORT UI

### Task 11: Add the camera-only estimated report and retry states

**Files:**
- Create: `flutter_app/lib/features/reports/screens/visit_report_screen.dart`
- Create: `flutter_app/lib/features/reports/widgets/estimated_report_view.dart`
- Create: `flutter_app/lib/features/reports/widgets/report_metric_card.dart`
- Create: `flutter_app/lib/features/reports/widgets/estimate_provenance_card.dart`
- Create: `flutter_app/lib/features/reports/providers/visit_report_provider.dart`
- Modify: `flutter_app/lib/router.dart`
- Modify: `flutter_app/lib/l10n/translations.dart`
- Test: `flutter_app/test/features/reports/estimated_report_view_test.dart`
- Test: `flutter_app/test/features/reports/visit_report_screen_test.dart`

**Route:**
- `/visits/:visitUuid/report`

- [ ] **Step 1: Write failing rendering tests**

Assert:

- title is **Estimated Growth Screening Report**;
- the estimate notice is present verbatim from the design;
- method, model version, confidence, quality, and used views are visible;
- missing components say they could not be estimated;
- a successful camera report contains no visible `Indeterminate`;
- `Normal` is shown only when supplied by the camera result;
- **Add Measured Details** navigates with the visit UUID and visit date;
- processing failure shows retry without deleting media.

- [ ] **Step 2: Implement report provider and screen**

Read the persisted visit aggregate, not the transient latest-assessment provider. Refreshing or restarting the app must reproduce the report.

- [ ] **Step 3: Keep legacy result behavior isolated**

Do not weaken clinical messaging in the existing `ResultScreen`. Route guided visits to the new visit report.

- [ ] **Step 4: Verify**

```bash
cd flutter_app
flutter test \
  test/features/reports/estimated_report_view_test.dart \
  test/features/reports/visit_report_screen_test.dart
```

---

## PHASE 8 — SAME-VISIT MEASURED DETAILS

### Task 12: Add partial measured details, oedema, and immutable revisions on Flutter

**Files:**
- Create: `flutter_app/lib/features/measured_details/screens/add_measured_details_screen.dart`
- Create: `flutter_app/lib/features/measured_details/providers/measured_details_provider.dart`
- Create: `flutter_app/lib/features/measured_details/services/measured_report_service.dart`
- Create: `flutter_app/lib/features/measured_details/domain/measured_details.dart`
- Modify: `flutter_app/lib/router.dart`
- Refactor shared validation from: `flutter_app/lib/screens/child_management/manual_measurement_screen.dart`
- Test: `flutter_app/test/features/measured_details/measured_report_service_test.dart`
- Test: `flutter_app/test/features/measured_details/add_measured_details_screen_test.dart`

**Route:**
- `/visits/:visitUuid/measured-details`

- [ ] **Step 1: Write failing validation tests**

Cover finite/plausible optional values, measurement mode, oedema, MUAC age eligibility, visit-date lock, and all-empty rejection.

Height and weight must be independently optional for same-visit follow-up.

- [ ] **Step 2: Write failing classification tests using real WHO assets**

Mirror backend Task 4 cases and assert Dart/Python output compatibility.

- [ ] **Step 3: Implement the form**

Show the immutable visit date prominently. If the worker changes the date, offer to create a new visit rather than attaching it here.

- [ ] **Step 4: Implement atomic local save**

Append a complete before/after revision, update the current measured report, set manual/tape provenance, transition to `measured_report`, and enqueue the revision.

- [ ] **Step 5: Verify**

```bash
cd flutter_app
flutter test \
  test/features/measured_details/measured_report_service_test.dart \
  test/features/measured_details/add_measured_details_screen_test.dart
```

---

### Task 13: Make the measurement-based report primary and compare with estimate

**Files:**
- Create: `flutter_app/lib/features/reports/widgets/measured_report_view.dart`
- Create: `flutter_app/lib/features/reports/widgets/estimate_comparison_view.dart`
- Modify: `flutter_app/lib/features/reports/screens/visit_report_screen.dart`
- Modify: `flutter_app/lib/features/reports/providers/visit_report_provider.dart`
- Test: `flutter_app/test/features/reports/measured_report_view_test.dart`
- Test: `flutter_app/test/features/reports/estimate_comparison_view_test.dart`

- [ ] **Step 1: Write failing measurement-report tests**

Assert three separately named sections:

- WHO HAZ stunting;
- WHO acute malnutrition;
- Poshan Setu v1.

Missing manual components display **Not measured**, never a fabricated Normal value and never a user-facing `Indeterminate` placeholder.

- [ ] **Step 2: Write failing comparison tests**

For authorized users only, show:

- estimated and measured values;
- signed and absolute differences;
- classification agreement;
- camera model/result version.

Do not compare missing components.

- [ ] **Step 3: Implement primary/secondary report selection**

When a measured report exists, it is primary and the original camera result is under **Compare with estimate**.

- [ ] **Step 4: Verify**

```bash
cd flutter_app
flutter test \
  test/features/reports/measured_report_view_test.dart \
  test/features/reports/estimate_comparison_view_test.dart
```

---

## PHASE 9 — CHILD TIMELINE

### Task 14: Label visit states and attach actions to the exact visit

**Files:**
- Modify: `flutter_app/lib/models/child_detail.dart`
- Modify: `flutter_app/lib/providers/children_provider.dart`
- Modify: `flutter_app/lib/screens/children/child_detail_screen.dart`
- Modify: `flutter_app/lib/services/api_service.dart`
- Modify: `app/api/routes.py`
- Test: `flutter_app/test/child_detail_screen_test.dart`
- Test: `flutter_app/test/models/child_detail_test.dart`
- Test: `tests/test_guided_child_detail.py`

- [ ] **Step 1: Write failing API and Dart parsing tests**

Child detail must include visit local UUID, capture state, camera result summary/version, measured report presence, required asset acknowledgement state, and media deletion state.

- [ ] **Step 2: Add backend response schemas**

Replace hand-built untyped guided response fragments with Pydantic response models. Keep legacy fields compatible.

- [ ] **Step 3: Update timeline labels/actions**

Map exact UI labels:

- **Incomplete capture**
- **Processing estimate**
- **Estimated report**
- **Estimate failed — retry**
- **Measured report added**

Camera-only visits expose **Add Measured Details** for that visit UUID.

- [ ] **Step 4: Verify**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_guided_child_detail.py -v
cd flutter_app
flutter test \
  test/models/child_detail_test.dart \
  test/child_detail_screen_test.dart
```

---

## PHASE 10 — IDEMPOTENT SERVER SYNC

### Task 15: Add owner-scoped guided sync endpoints and per-entity acknowledgements

**Files:**
- Create: `app/api/guided_sync.py`
- Create: `app/schemas/guided_sync.py`
- Create: `app/services/guided_sync_service.py`
- Modify: `main.py`
- Keep compatible: `app/api/sync.py`
- Test: `tests/test_guided_sync.py`
- Test: `tests/test_guided_sync_ownership.py`
- Test: `tests/test_guided_sync_recovery.py`

**Endpoints:**
- `PUT /api/v1/sync/guided/visits/{visit_uuid}`
- `PUT /api/v1/sync/guided/visits/{visit_uuid}/assets/{asset_uuid}`
- `PUT /api/v1/sync/guided/visits/{visit_uuid}/camera-results/{result_uuid}`
- `PUT /api/v1/sync/guided/visits/{visit_uuid}/measured-revisions/{revision_uuid}`
- `DELETE /api/v1/sync/guided/visits/{visit_uuid}/media/{asset_uuid}`

Each success response identifies the accepted entity UUID, server ID/object ID, checksum where applicable, and status `accepted` or `already_accepted`.

- [ ] **Step 1: Write failing auth and ownership tests**

Every endpoint rejects missing auth and cross-owner child/visit/entity access.

- [ ] **Step 2: Write failing idempotency tests**

Repeat each request with the same UUID and payload. It must return the same server identity without duplicating rows or files.

Same UUID with a different immutable payload checksum returns 409.

- [ ] **Step 3: Write failing partial-recovery tests**

Cover:

- visit accepted, only front asset accepted, reconnect, side continues;
- asset bytes saved but transaction interrupted;
- result sent before required assets are acknowledged;
- measured revision arrives before or after camera result;
- duplicate revision number with a different UUID;
- media deletion repeated.

- [ ] **Step 4: Implement Pydantic endpoints through the service layer**

Validate provenance server-side:

- camera fields are always stored non-clinically;
- submitted estimated values cannot become manual values;
- required asset metadata and content checksum must match;
- measured revisions trigger authoritative recomputation;
- client-submitted measured classifications are ignored.

- [ ] **Step 5: Preserve legacy sync**

Do not silently change `/api/v1/sync` behavior for existing installations. Add explicit compatibility tests.

- [ ] **Step 6: Verify**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_guided_sync.py \
  tests/test_guided_sync_ownership.py \
  tests/test_guided_sync_recovery.py \
  tests/test_sync.py -v
```

---

### Task 16: Drain the typed Flutter outbox and retain media until asset acknowledgement

**Files:**
- Create: `flutter_app/lib/features/guided_capture/services/guided_sync_service.dart`
- Modify: `flutter_app/lib/providers/sync_provider.dart`
- Modify: `flutter_app/lib/services/image_storage_service.dart`
- Modify: `flutter_app/lib/screens/settings/settings_screen.dart`
- Test: `flutter_app/test/features/guided_capture/services/guided_sync_service_test.dart`
- Test: `flutter_app/test/services/image_storage_service_test.dart`

- [ ] **Step 1: Write failing request-shape and order tests**

Use a fake HTTP client to assert Dart payloads match Pydantic schemas and dependency order.

- [ ] **Step 2: Write failing retry/acknowledgement tests**

Cover offline, timeout, 401, 409 checksum conflict, partial acknowledgement, process death while syncing, and retry exhaustion.

- [ ] **Step 3: Implement outbox draining**

Only mark an entity synced when the response acknowledges that exact UUID. Store server IDs/object IDs on their local rows.

- [ ] **Step 4: Enforce local-media retention**

Normal cleanup may delete an asset only after its own server acknowledgement. User-requested deletion creates a media-deletion outbox operation and keeps measurement history.

Replace any broad **clear all images** behavior with a guarded flow that distinguishes:

- acknowledged synced media;
- pending media;
- failed media;
- explicit deletion requested by the user.

- [ ] **Step 5: Verify**

```bash
cd flutter_app
flutter test \
  test/features/guided_capture/services/guided_sync_service_test.dart \
  test/services/image_storage_service_test.dart
```

---

## PHASE 11 — PRIVACY, DELETION, AND RESEARCH EXPORT

### Task 17: Add owner-scoped media controls and a de-identified export

**Files:**
- Create: `app/api/guided_media.py`
- Create: `app/services/guided_media_service.py`
- Create: `scripts/export_guided_capture_dataset.py`
- Modify: `main.py`
- Test: `tests/test_guided_media.py`
- Test: `tests/test_guided_capture_export.py`
- Create: `docs/guided_capture_data_dictionary.md`

- [ ] **Step 1: Write failing media-deletion tests**

Deleting media must:

- require authentication and ownership;
- delete or tombstone only the selected asset bytes/metadata;
- preserve child, visit, camera-result metadata, measured report, and revisions;
- be idempotent;
- be blocked or explicitly warned when a required asset is still pending upload.

- [ ] **Step 2: Write failing export tests**

The research export includes stable pseudonymous child ID, visit UUID, asset role/object IDs, quality/model metadata, and same-visit measured values.

It excludes:

- child name;
- guardian name;
- profile image;
- location free text;
- operator name/username;
- source filesystem paths containing identifiers.

Verify splits are generated by child, never by image.

- [ ] **Step 3: Implement media service and export CLI**

Require an explicit output directory. Refuse to overwrite a non-empty output directory. Emit a manifest with schema version and source model/quality versions.

- [ ] **Step 4: Document every exported field and provenance**

Mark each numeric field as measured, tape, camera-estimated, derived-estimated, or calculated-from-measured.

- [ ] **Step 5: Verify**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_guided_media.py \
  tests/test_guided_capture_export.py -v
```

---

## PHASE 12 — END-TO-END VALIDATION AND ROLLOUT

### Task 18: Add full workflow tests, performance checks, and the field-pilot runbook

**Files:**
- Create: `flutter_app/integration_test/guided_capture_flow_test.dart`
- Create: `flutter_app/test/features/guided_capture/guided_capture_widget_flow_test.dart`
- Create: `tests/test_guided_capture_e2e.py`
- Create: `docs/guided_capture_field_pilot.md`
- Modify: `flutter_app/test/constants/feature_flags_test.dart`

- [ ] **Step 1: Add widget-level full-flow tests**

Using fakes:

```text
select child
-> consent
-> draft visit
-> front burst
-> side burst
-> review
-> estimated report
-> add partial measured details
-> measurement-based report
-> compare with immutable estimate
```

Also cover optional roles, interruption/resume, incomplete capture, inference retry, offline queueing, and app restart.

- [ ] **Step 2: Add backend end-to-end tests**

Use real database transactions and real WHO Excel files. Verify API/Dart contract fixtures and owner isolation.

- [ ] **Step 3: Run complete automated verification**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
cd flutter_app
dart format --output=none --set-exit-if-changed lib test integration_test
flutter analyze
flutter test
```

- [ ] **Step 4: Build a gated Android artifact**

```bash
cd flutter_app
flutter build apk --debug --dart-define=LIVE_CAPTURE=true
```

The normal release build remains gated unless explicitly enabled.

- [ ] **Step 5: Validate on a physical Android device**

Record device model, Android version, build SHA, and observed outcome for:

- front and side orientation;
- camera rotation and lifecycle interruption;
- background person/multiple poses;
- low light, overexposure, blur, cropped head/feet;
- excessive phone tilt;
- low-memory behavior;
- offline capture, app restart, delayed synchronization;
- per-asset server acknowledgement;
- local retention before acknowledgement;
- estimate-to-measurement wording comprehension.

Host/widget tests do not satisfy this step.

- [ ] **Step 6: Write the controlled-pilot runbook**

Include:

- `LIVE_CAPTURE` build command;
- consent script and version;
- operator training;
- capture threshold version;
- paired same-day measurement procedure;
- failure/retry logging;
- media retention/deletion procedure;
- export procedure;
- stop conditions.

Promotion is blocked when SAM recall is below 0.80. Passing the floor alone does not make the output clinical.

- [ ] **Step 7: Final acceptance review**

Check all 11 acceptance criteria from the design one by one and link each to automated or device evidence.

---

## Recommended review checkpoints

Pause for review after each checkpoint:

1. **Contracts and migrations:** Tasks 2–6.
2. **Capture quality and resumable workflow:** Tasks 7–9.
3. **Camera inference and report UI:** Tasks 10–11.
4. **Measured details and comparison:** Tasks 12–14.
5. **Sync, privacy, and export:** Tasks 15–17.
6. **Device evidence and rollout:** Task 18.

Do not start the next checkpoint while the previous checkpoint has failing safety or migration tests.

---

## Final definition of done

- A clean install and a schema-v4 upgrade both work.
- The worker starts from an existing/created child profile and no measurement fields appear during photo capture.
- The draft visit exists before the first image.
- Front and side must pass versioned, role-specific quality gates.
- A successful capture persists an immutable `camera_screening_v1` result and displays no clinical `Indeterminate` headline.
- Missing camera components are explicit and never fabricated as Normal.
- Same-date partial measured details update the exact visit and append an audit revision.
- The measured report clearly separates WHO HAZ, WHO acute malnutrition, and Poshan Setu v1.
- The original estimate remains available for authorized comparison.
- Every new entity syncs idempotently and every asset receives its own acknowledgement.
- Local media remains until its own acknowledgement or an explicit user deletion workflow.
- Owner isolation, provenance enforcement, and de-identified export tests pass.
- Python tests, `flutter analyze`, Flutter tests, gated APK build, and physical-device validation pass.
- The feature remains disabled in normal release builds until controlled-pilot evidence is reviewed.
