# Offline-First Flutter App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Flutter app fully standalone — all assessment processing (pose detection, height/weight estimation, WHO z-scores, ML inference) runs on-device with data syncing to backend when internet is available.

**Architecture:** Mirror each backend Python service as a Dart service class. Local Drift database stores children/visits/measurements. Queue-based sync pushes completed assessments to backend when connectivity is restored.

**Tech Stack:** Flutter, Drift (SQLite), google_mlkit_pose_detection, tflite_flutter, Riverpod, connectivity_plus

**Spec:** `docs/superpowers/specs/2026-04-09-offline-first-flutter-design.md`

---

## File Structure

```
flutter_app/lib/
├── main.dart                              # Modified: init database
├── router.dart                            # No changes
├── constants/
│   └── config.dart                        # NEW: thresholds, ratios, MUAC medians
├── database/
│   ├── database.dart                      # NEW: @DriftDatabase definition
│   ├── tables/
│   │   ├── children_table.dart            # NEW
│   │   ├── visits_table.dart              # NEW
│   │   ├── measurements_table.dart        # NEW
│   │   └── sync_queue_table.dart          # NEW
│   └── daos/
│       ├── child_dao.dart                 # NEW
│       ├── visit_dao.dart                 # NEW
│       └── sync_queue_dao.dart            # NEW
├── models/
│   ├── child.dart                         # Keep
│   ├── child_detail.dart                  # Keep
│   ├── assessment_result.dart             # Keep
│   ├── body_measurements.dart             # NEW
│   └── wasting_features.dart              # NEW
├── services/
│   ├── api_service.dart                   # Keep (sync only)
│   ├── pose_service.dart                  # NEW
│   ├── measurement_service.dart           # NEW
│   ├── who_data_service.dart              # NEW
│   ├── nutrition_service.dart             # NEW
│   ├── ml_inference_service.dart          # NEW
│   ├── muac_service.dart                  # NEW
│   ├── assessment_service.dart            # NEW
│   └── sync_service.dart                  # NEW
├── providers/
│   ├── database_provider.dart             # NEW
│   ├── api_provider.dart                  # Keep (for sync)
│   ├── children_provider.dart             # Modified: reads from DB
│   ├── assessment_provider.dart           # Modified: writes to DB
│   └── sync_provider.dart                 # NEW
├── screens/
│   ├── assessment/
│   │   └── assessment_screen.dart         # Modified: local pipeline
│   └── shared/
│       └── app_scaffold.dart              # Modified: sync indicator
└── l10n/
```

### Bundled assets
```
flutter_app/assets/
├── models/
│   ├── weight_estimator.tflite            # Export from ml/train.py
│   ├── wasting_classifier.tflite          # Export from ml/train.py
│   └── feature_scaler.json               # Export from ml/ scaler pkl to JSON
└── who_data/
    ├── who_haz_0_59m.csv                  # Copy from data/
    ├── who_wfl_boys_0_2.xlsx              # Copy from data/
    ├── who_wfl_girls_0_2.xlsx             # Copy from data/
    ├── who_wfh_boys_2_5.xlsx              # Copy from data/
    └── who_wfh_girls_2_5.xlsx             # Copy from data/
```

---

## Task 1: Project Setup & Dependencies

**Files:**
- Modify: `flutter_app/pubspec.yaml`
- Create: `flutter_app/assets/models/` (directory)
- Create: `flutter_app/assets/who_data/` (directory)

- [ ] **Step 1: Add new dependencies to pubspec.yaml**

Add to `dependencies:` section:
```yaml
  # Local processing
  google_mlkit_pose_detection: ^0.12.0
  tflite_flutter: ^0.11.0
  excel: ^4.0.6
  csv: ^6.0.0

  # Local database
  drift: ^2.22.1
  sqlite3_flutter_libs: ^0.5.28
  path_provider: ^2.1.5

  # Connectivity
  connectivity_plus: ^6.1.1
```

Add to `dev_dependencies:` section:
```yaml
  drift_dev: ^2.22.1
  build_runner: ^2.4.14
```

Add asset declarations under `flutter:`:
```yaml
flutter:
  uses-material-design: true
  assets:
    - assets/models/
    - assets/who_data/
```

- [ ] **Step 2: Copy WHO data files to assets**

```bash
mkdir -p flutter_app/assets/who_data flutter_app/assets/models
cp data/who_haz_0_59m.csv flutter_app/assets/who_data/
cp data/who_wfl_boys_0_2.xlsx flutter_app/assets/who_data/
cp data/who_wfl_girls_0_2.xlsx flutter_app/assets/who_data/
cp data/who_wfh_boys_2_5.xlsx flutter_app/assets/who_data/
cp data/who_wfh_girls_2_5.xlsx flutter_app/assets/who_data/
```

- [ ] **Step 3: Export ML models to TFLite and scaler to JSON**

Run in project root with Python venv. This converts the sklearn scaler from the existing pickle format to a portable JSON format for the Dart app:
```bash
PYTHONPATH=. .venv/bin/python -c "
import json, pickle, numpy as np
from pathlib import Path

# Convert scaler pickle to JSON (safe serialization)
scaler_path = Path('models/feature_scaler.pkl')
if scaler_path.exists():
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    scaler_data = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'type': 'standard'
    }
    with open('flutter_app/assets/models/feature_scaler.json', 'w') as f:
        json.dump(scaler_data, f)
    print('Scaler exported')

# TFLite models should already exist from training
import shutil
for name in ['weight_estimator.tflite', 'wasting_classifier.tflite']:
    src = Path('models') / name
    if src.exists():
        shutil.copy(src, f'flutter_app/assets/models/{name}')
        print(f'Copied {name}')
    else:
        print(f'WARNING: {name} not found - run ml/train.py first')
"
```

If TFLite models don't exist, train them:
```bash
PYTHONPATH=. .venv/bin/python ml/train.py
```

- [ ] **Step 4: Install dependencies**

```bash
cd flutter_app && flutter pub get
```

- [ ] **Step 5: Commit**

```bash
git add flutter_app/pubspec.yaml flutter_app/assets/
git commit -m "feat: add offline processing dependencies and bundled assets"
```

---

## Task 2: Constants & Configuration

**Files:**
- Create: `flutter_app/lib/constants/config.dart`
- Test: `flutter_app/test/constants/config_test.dart`

- [ ] **Step 1: Write failing tests**

```dart
// flutter_app/test/constants/config_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/constants/config.dart';

void main() {
  group('getAnthropometricRatios', () {
    test('returns 0-12 month ratios for age 6', () {
      final r = getAnthropometricRatios(6);
      expect(r.headRatio, 0.28);
      expect(r.torsoRatio, 0.32);
      expect(r.legRatio, 0.40);
    });

    test('returns 12-24 month ratios for age 18', () {
      final r = getAnthropometricRatios(18);
      expect(r.headRatio, 0.25);
    });

    test('returns 48-60 month ratios for age 55', () {
      final r = getAnthropometricRatios(55);
      expect(r.headRatio, 0.20);
      expect(r.legRatio, 0.50);
    });
  });

  group('classifyHaz', () {
    test('z < -3 is Severely Stunted', () {
      expect(classifyHaz(-3.5), 'Severely Stunted');
    });
    test('z = -2.5 is Stunted', () {
      expect(classifyHaz(-2.5), 'Stunted');
    });
    test('z = 0 is Normal', () {
      expect(classifyHaz(0), 'Normal');
    });
    test('z = 2.5 is Tall', () {
      expect(classifyHaz(2.5), 'Tall');
    });
  });

  group('classifyWhz', () {
    test('z < -3 is SAM', () {
      expect(classifyWhz(-3.5), 'Severe Acute Malnutrition (SAM)');
    });
    test('z = -2.5 is MAM', () {
      expect(classifyWhz(-2.5), 'Moderate Acute Malnutrition (MAM)');
    });
    test('z = 0 is Normal', () {
      expect(classifyWhz(0), 'Normal');
    });
    test('z = 1.5 is Risk of Overweight', () {
      expect(classifyWhz(1.5), 'Possible Risk of Overweight');
    });
    test('z = 2.5 is Overweight', () {
      expect(classifyWhz(2.5), 'Overweight');
    });
    test('z = 3.5 is Obese', () {
      expect(classifyWhz(3.5), 'Obese');
    });
  });

  group('classifyMuac', () {
    test('< 11.5 is SAM when age in range', () {
      expect(classifyMuac(11.0, true), 'SAM');
    });
    test('11.5-12.5 is At Risk (MAM)', () {
      expect(classifyMuac(12.0, true), 'At Risk (MAM)');
    });
    test('>= 12.5 is Normal', () {
      expect(classifyMuac(13.0, true), 'Normal');
    });
    test('returns null when age not in range', () {
      expect(classifyMuac(11.0, false), isNull);
    });
  });
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd flutter_app && flutter test test/constants/config_test.dart
```
Expected: FAIL (file not found)

- [ ] **Step 3: Implement config.dart**

```dart
// flutter_app/lib/constants/config.dart

/// Anthropometric segment ratios by age (Snyder et al. 1975)
class AnthropometricRatios {
  final double headRatio;
  final double torsoRatio;
  final double legRatio;

  const AnthropometricRatios({
    required this.headRatio,
    required this.torsoRatio,
    required this.legRatio,
  });
}

const _ratios0to12 = AnthropometricRatios(headRatio: 0.28, torsoRatio: 0.32, legRatio: 0.40);
const _ratios12to24 = AnthropometricRatios(headRatio: 0.25, torsoRatio: 0.32, legRatio: 0.43);
const _ratios24to48 = AnthropometricRatios(headRatio: 0.22, torsoRatio: 0.30, legRatio: 0.48);
const _ratios48to60 = AnthropometricRatios(headRatio: 0.20, torsoRatio: 0.30, legRatio: 0.50);

AnthropometricRatios getAnthropometricRatios(double ageMonths) {
  if (ageMonths < 12) return _ratios0to12;
  if (ageMonths < 24) return _ratios12to24;
  if (ageMonths < 48) return _ratios24to48;
  return _ratios48to60;
}

/// Height validation: flag if > 3 SD from WHO median
const double heightValidationSd = 3.0;

/// Max 15% difference between segment-based estimates
const double segmentAgreementThreshold = 0.15;

/// Minimum pose confidence to use measurement
const double minConfidenceThreshold = 0.5;

/// ML weight must be 45-180% of WHO median
const double mlWeightLowerBound = 0.45;
const double mlWeightUpperBound = 1.80;

/// Days per month for age calculation
const double daysPerMonth = 30.4375;

/// Expected shoulder-to-height ratios by age (for body build classification)
double expectedShoulderRatio(double ageMonths) {
  if (ageMonths < 24) return 0.200;
  if (ageMonths < 48) return 0.210;
  return 0.218;
}

/// Body build deviation thresholds
/// measurement_service uses 0.03 for display, ml_service uses 0.02 for scoring
const double bodyBuildThresholdDisplay = 0.03;
const double bodyBuildThresholdMl = 0.02;

// --- Classification functions ---

String classifyHaz(double z) {
  if (z < -3) return 'Severely Stunted';
  if (z < -2) return 'Stunted';
  if (z < 2) return 'Normal';
  return 'Tall';
}

String classifyWhz(double z) {
  if (z < -3) return 'Severe Acute Malnutrition (SAM)';
  if (z < -2) return 'Moderate Acute Malnutrition (MAM)';
  if (z < 1) return 'Normal';
  if (z < 2) return 'Possible Risk of Overweight';
  if (z < 3) return 'Overweight';
  return 'Obese';
}

String? classifyMuac(double muacCm, bool ageInRange) {
  if (!ageInRange) return null;
  if (muacCm < 11.5) return 'SAM';
  if (muacCm < 12.5) return 'At Risk (MAM)';
  return 'Normal';
}

// --- MUAC WHO medians (age_months, median_cm) ---

const List<(int, double)> muacBoys = [
  (3, 12.5), (6, 14.0), (9, 14.8), (12, 15.2), (18, 15.5), (24, 15.7),
  (30, 15.8), (36, 15.9), (42, 16.0), (48, 16.1), (54, 16.1), (60, 16.2),
];

const List<(int, double)> muacGirls = [
  (3, 12.3), (6, 13.8), (9, 14.6), (12, 14.9), (18, 15.2), (24, 15.4),
  (30, 15.5), (36, 15.6), (42, 15.7), (48, 15.7), (54, 15.8), (60, 15.8),
];

/// Wasting classifier labels (alphabetical, matching training order)
const List<String> wastingLabels = ['MAM', 'Normal', 'Overweight', 'Risk_Overweight', 'SAM'];

/// 14-feature names in exact order
const List<String> featureNames = [
  'age_months', 'sex_binary', 'height_cm', 'shoulder_width_cm',
  'hip_width_cm', 'torso_length_cm', 'upper_arm_length_cm',
  'shoulder_height_ratio', 'hip_height_ratio', 'body_build_score',
  'chest_depth_cm', 'abd_depth_cm', 'chest_depth_ratio', 'abd_depth_ratio',
];
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd flutter_app && flutter test test/constants/config_test.dart
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/constants/config.dart flutter_app/test/constants/config_test.dart
git commit -m "feat: add config constants ported from Python config.py"
```

---

## Task 3: Data Models (BodyMeasurements, WastingFeatures)

**Files:**
- Create: `flutter_app/lib/models/body_measurements.dart`
- Create: `flutter_app/lib/models/wasting_features.dart`
- Test: `flutter_app/test/models/wasting_features_test.dart`

- [ ] **Step 1: Write failing test for WastingFeatures.toArray()**

```dart
// flutter_app/test/models/wasting_features_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/models/wasting_features.dart';

void main() {
  test('toArray returns 14-element vector in correct order', () {
    final f = WastingFeatures(
      ageMonths: 24.0,
      sexBinary: 1,
      heightCm: 85.0,
      shoulderWidthCm: 18.0,
      hipWidthCm: 15.0,
      torsoLengthCm: 25.5,
      upperArmLengthCm: 13.0,
      shoulderHeightRatio: 18.0 / 85.0,
      hipHeightRatio: 15.0 / 85.0,
      bodyBuildScore: 0,
      chestDepthCm: 10.0,
      abdDepthCm: 8.0,
    );
    final arr = f.toArray();
    expect(arr.length, 14);
    expect(arr[0], 24.0); // age_months
    expect(arr[1], 1.0);  // sex_binary
    expect(arr[2], 85.0); // height_cm
    expect(arr[10], 10.0); // chest_depth_cm (provided)
    expect(arr[12], closeTo(10.0 / 85.0, 0.001)); // chest_depth_ratio
  });

  test('toArray imputes AP depth when not provided', () {
    final f = WastingFeatures(
      ageMonths: 24.0,
      sexBinary: 0,
      heightCm: 85.0,
      shoulderWidthCm: 18.0,
      hipWidthCm: 15.0,
      torsoLengthCm: 25.5,
      upperArmLengthCm: 13.0,
      shoulderHeightRatio: 18.0 / 85.0,
      hipHeightRatio: 15.0 / 85.0,
      bodyBuildScore: -1,
    );
    final arr = f.toArray();
    expect(arr[10], closeTo(18.0 * 0.45, 0.01)); // chest = shoulder * 0.45
    expect(arr[11], closeTo(15.0 * 0.50, 0.01)); // abd = hip * 0.50
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd flutter_app && flutter test test/models/wasting_features_test.dart
```
Expected: FAIL

- [ ] **Step 3: Implement body_measurements.dart**

```dart
// flutter_app/lib/models/body_measurements.dart

/// Body segments measured from pose landmarks (in pixels)
class BodySegments {
  final double? headHeightPx;
  final double? torsoLengthPx;
  final double? legLengthPx;
  final double? shoulderWidthPx;
  final double? hipWidthPx;
  final double? upperArmLengthPx;
  final double? totalHeightPx;
  final double? headTopY;
  final double? chinY;
  final double? shoulderMidpointY;
  final double? hipMidpointY;
  final double? heelY;
  final double headConfidence;
  final double torsoConfidence;
  final double legConfidence;
  final double hipConfidence;
  final double armConfidence;

  const BodySegments({
    this.headHeightPx,
    this.torsoLengthPx,
    this.legLengthPx,
    this.shoulderWidthPx,
    this.hipWidthPx,
    this.upperArmLengthPx,
    this.totalHeightPx,
    this.headTopY,
    this.chinY,
    this.shoulderMidpointY,
    this.hipMidpointY,
    this.heelY,
    this.headConfidence = 0.0,
    this.torsoConfidence = 0.0,
    this.legConfidence = 0.0,
    this.hipConfidence = 0.0,
    this.armConfidence = 0.0,
  });
}

/// Side-view measurements
class SideViewSegments {
  final double? chestDepthPx;
  final double? abdDepthPx;
  final double? totalHeightPx;
  final double chestConfidence;
  final double abdConfidence;

  const SideViewSegments({
    this.chestDepthPx,
    this.abdDepthPx,
    this.totalHeightPx,
    this.chestConfidence = 0.0,
    this.abdConfidence = 0.0,
  });
}

/// Full measurement output from the processing pipeline
class MeasurementOutput {
  final double? predictedHeightCm;
  final BodySegments? bodySegments;
  final String? bodyBuild;
  final double weightAdjustment;
  final double confidenceScore;
  final String estimationMethod;

  const MeasurementOutput({
    this.predictedHeightCm,
    this.bodySegments,
    this.bodyBuild,
    this.weightAdjustment = 1.0,
    this.confidenceScore = 0.0,
    this.estimationMethod = 'unknown',
  });
}
```

- [ ] **Step 4: Implement wasting_features.dart**

```dart
// flutter_app/lib/models/wasting_features.dart
import 'dart:typed_data';

/// 14-feature vector for ML wasting prediction.
/// Feature order MUST match training - do not reorder.
class WastingFeatures {
  final double ageMonths;
  final int sexBinary; // 1 = Male, 0 = Female
  final double heightCm;
  final double shoulderWidthCm;
  final double hipWidthCm;
  final double torsoLengthCm;
  final double upperArmLengthCm;
  final double shoulderHeightRatio;
  final double hipHeightRatio;
  final int bodyBuildScore; // -1 = slender, 0 = average, 1 = stocky
  final double? chestDepthCm;
  final double? abdDepthCm;

  const WastingFeatures({
    required this.ageMonths,
    required this.sexBinary,
    required this.heightCm,
    required this.shoulderWidthCm,
    required this.hipWidthCm,
    required this.torsoLengthCm,
    required this.upperArmLengthCm,
    required this.shoulderHeightRatio,
    required this.hipHeightRatio,
    required this.bodyBuildScore,
    this.chestDepthCm,
    this.abdDepthCm,
  });

  /// Convert to 14-element Float32 array for TFLite inference.
  /// Imputes AP depth from lateral widths when side view unavailable (Snyder 1975).
  Float32List toArray() {
    final cd = chestDepthCm ?? shoulderWidthCm * 0.45;
    final ad = abdDepthCm ?? hipWidthCm * 0.50;
    final cdr = cd / heightCm;
    final adr = ad / heightCm;

    return Float32List.fromList([
      ageMonths,
      sexBinary.toDouble(),
      heightCm,
      shoulderWidthCm,
      hipWidthCm,
      torsoLengthCm,
      upperArmLengthCm,
      shoulderHeightRatio,
      hipHeightRatio,
      bodyBuildScore.toDouble(),
      cd,
      ad,
      cdr,
      adr,
    ]);
  }
}

/// ML prediction result
class WastingPrediction {
  final double? estimatedWeightKg;
  final double samProbability;
  final double mamProbability;
  final double normalProbability;
  final double riskProbability;
  final double overweightProbability;
  final String wastingStatus;

  const WastingPrediction({
    this.estimatedWeightKg,
    required this.samProbability,
    required this.mamProbability,
    required this.normalProbability,
    required this.riskProbability,
    required this.overweightProbability,
    required this.wastingStatus,
  });
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd flutter_app && flutter test test/models/wasting_features_test.dart
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add flutter_app/lib/models/body_measurements.dart flutter_app/lib/models/wasting_features.dart flutter_app/test/models/wasting_features_test.dart
git commit -m "feat: add BodyMeasurements and WastingFeatures models"
```

---

## Task 4: Drift Database Schema

**Files:**
- Create: `flutter_app/lib/database/tables/children_table.dart`
- Create: `flutter_app/lib/database/tables/visits_table.dart`
- Create: `flutter_app/lib/database/tables/measurements_table.dart`
- Create: `flutter_app/lib/database/tables/sync_queue_table.dart`
- Create: `flutter_app/lib/database/database.dart`

- [ ] **Step 1: Create table definitions**

```dart
// flutter_app/lib/database/tables/children_table.dart
import 'package:drift/drift.dart';

class Children extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get name => text()();
  TextColumn get dateOfBirth => text()(); // ISO 8601
  TextColumn get sex => text().withLength(min: 1, max: 1)(); // M or F
  TextColumn get guardianName => text().nullable()();
  TextColumn get location => text().nullable()();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  DateTimeColumn get updatedAt => dateTime().withDefault(currentDateAndTime)();
}
```

```dart
// flutter_app/lib/database/tables/visits_table.dart
import 'package:drift/drift.dart';
import 'children_table.dart';

class Visits extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get childId => integer().references(Children, #id)();
  DateTimeColumn get visitDate => dateTime().withDefault(currentDateAndTime)();
  RealColumn get ageMonths => real()();
  TextColumn get imagePath => text()();
  TextColumn get sideImagePath => text().nullable()();
  TextColumn get backImagePath => text().nullable()();
  TextColumn get notes => text().nullable()();
}
```

```dart
// flutter_app/lib/database/tables/measurements_table.dart
import 'package:drift/drift.dart';
import 'visits_table.dart';

class Measurements extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get visitId => integer().unique().references(Visits, #id)();
  RealColumn get predictedHeightCm => real().nullable()();
  RealColumn get predictedWeightKg => real().nullable()();
  RealColumn get manualHeightCm => real().nullable()();
  RealColumn get manualWeightKg => real().nullable()();
  RealColumn get hazZscore => real().nullable()();
  RealColumn get whzZscore => real().nullable()();
  TextColumn get hazStatus => text().nullable()();
  TextColumn get whzStatus => text().nullable()();
  RealColumn get confidenceScore => real().nullable()();
  TextColumn get bodyBuild => text().nullable()();
  TextColumn get estimationMethod => text().nullable()();
  BoolColumn get sideViewUsed => boolean().withDefault(const Constant(false))();
  RealColumn get chestDepthCm => real().nullable()();
  RealColumn get abdDepthCm => real().nullable()();
  RealColumn get mlEstimatedWeightKg => real().nullable()();
  RealColumn get samProbability => real().nullable()();
  RealColumn get mamProbability => real().nullable()();
  RealColumn get normalProbability => real().nullable()();
  RealColumn get riskOverweightProbability => real().nullable()();
  RealColumn get overweightProbability => real().nullable()();
  TextColumn get wastingStatus => text().nullable()();
  RealColumn get muacCm => real().nullable()();
  TextColumn get muacStatus => text().nullable()();
  TextColumn get muacMethod => text().nullable()();
}
```

```dart
// flutter_app/lib/database/tables/sync_queue_table.dart
import 'package:drift/drift.dart';
import 'visits_table.dart';

class SyncQueue extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get visitId => integer().references(Visits, #id)();
  TextColumn get status => text().withDefault(const Constant('pending'))();
  IntColumn get retryCount => integer().withDefault(const Constant(0))();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  DateTimeColumn get lastAttemptAt => dateTime().nullable()();
  IntColumn get serverVisitId => integer().nullable()();
  TextColumn get errorMessage => text().nullable()();
}
```

- [ ] **Step 2: Create database definition**

```dart
// flutter_app/lib/database/database.dart
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
  int get schemaVersion => 1;
}

LazyDatabase _openConnection() {
  return LazyDatabase(() async {
    final dbFolder = await getApplicationDocumentsDirectory();
    final file = File(p.join(dbFolder.path, 'child_growth_monitor.sqlite'));
    return NativeDatabase.createInBackground(file);
  });
}
```

- [ ] **Step 3: Run code generation**

```bash
cd flutter_app && dart run build_runner build --delete-conflicting-outputs
```
Expected: Generates `database.g.dart` successfully

- [ ] **Step 4: Verify it compiles**

```bash
cd flutter_app && flutter analyze
```
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/database/
git commit -m "feat: add Drift database schema with children, visits, measurements, sync_queue tables"
```

---

## Task 5: DAOs (Data Access Objects)

**Files:**
- Create: `flutter_app/lib/database/daos/child_dao.dart`
- Create: `flutter_app/lib/database/daos/visit_dao.dart`
- Create: `flutter_app/lib/database/daos/sync_queue_dao.dart`
- Test: `flutter_app/test/database/daos/child_dao_test.dart`
- Test: `flutter_app/test/database/daos/sync_queue_dao_test.dart`

- [ ] **Step 1: Write failing tests for ChildDao**

```dart
// flutter_app/test/database/daos/child_dao_test.dart
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

  test('findOrCreate creates new child', () async {
    final child = await dao.findOrCreate(
      name: 'Aarav', dateOfBirth: '2023-06-15', sex: 'M',
    );
    expect(child.id, greaterThan(0));
    expect(child.name, 'Aarav');
  });

  test('findOrCreate returns existing child', () async {
    final c1 = await dao.findOrCreate(
      name: 'Aarav', dateOfBirth: '2023-06-15', sex: 'M',
    );
    final c2 = await dao.findOrCreate(
      name: 'Aarav', dateOfBirth: '2023-06-15', sex: 'M',
    );
    expect(c1.id, c2.id);
  });

  test('watchAll returns all children', () async {
    await dao.findOrCreate(name: 'A', dateOfBirth: '2023-01-01', sex: 'M');
    await dao.findOrCreate(name: 'B', dateOfBirth: '2023-02-01', sex: 'F');
    final all = await dao.watchAll().first;
    expect(all.length, 2);
  });

  test('watchAll filters by search query', () async {
    await dao.findOrCreate(name: 'Aarav', dateOfBirth: '2023-01-01', sex: 'M');
    await dao.findOrCreate(name: 'Priya', dateOfBirth: '2023-02-01', sex: 'F');
    final filtered = await dao.watchAll(search: 'pri').first;
    expect(filtered.length, 1);
    expect(filtered.first.name, 'Priya');
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd flutter_app && flutter test test/database/daos/child_dao_test.dart
```
Expected: FAIL

- [ ] **Step 3: Implement all three DAOs**

```dart
// flutter_app/lib/database/daos/child_dao.dart
import 'package:drift/drift.dart';
import '../database.dart';

class ChildDao {
  final AppDatabase _db;
  ChildDao(this._db);

  Future<Child> findOrCreate({
    required String name,
    required String dateOfBirth,
    required String sex,
    String? guardianName,
    String? location,
  }) async {
    final existing = await (_db.select(_db.children)
          ..where((c) =>
              c.name.equals(name) &
              c.dateOfBirth.equals(dateOfBirth) &
              c.sex.equals(sex)))
        .getSingleOrNull();

    if (existing != null) return existing;

    final id = await _db.into(_db.children).insert(
      ChildrenCompanion.insert(
        name: name,
        dateOfBirth: dateOfBirth,
        sex: sex,
        guardianName: Value(guardianName),
        location: Value(location),
      ),
    );
    return (_db.select(_db.children)..where((c) => c.id.equals(id)))
        .getSingle();
  }

  Stream<List<Child>> watchAll({String? search}) {
    final query = _db.select(_db.children)
      ..orderBy([(c) => OrderingTerm.desc(c.updatedAt)]);
    if (search != null && search.isNotEmpty) {
      query.where((c) => c.name.like('%$search%'));
    }
    return query.watch();
  }

  Future<Child?> getById(int id) {
    return (_db.select(_db.children)..where((c) => c.id.equals(id)))
        .getSingleOrNull();
  }

  Stream<Child?> watchById(int id) {
    return (_db.select(_db.children)..where((c) => c.id.equals(id)))
        .watchSingleOrNull();
  }
}
```

```dart
// flutter_app/lib/database/daos/visit_dao.dart
import 'package:drift/drift.dart';
import '../database.dart';

class VisitDao {
  final AppDatabase _db;
  VisitDao(this._db);

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

    return query.watch().map((rows) => rows.map((row) {
      return (
        visit: row.readTable(_db.visits),
        measurement: row.readTableOrNull(_db.measurements),
      );
    }).toList());
  }

  Future<({Visit visit, Measurement? measurement})?> getById(int visitId) async {
    final query = _db.select(_db.visits).join([
      leftOuterJoin(_db.measurements, _db.measurements.visitId.equalsExp(_db.visits.id)),
    ])
      ..where(_db.visits.id.equals(visitId));

    final row = await query.getSingleOrNull();
    if (row == null) return null;
    return (
      visit: row.readTable(_db.visits),
      measurement: row.readTableOrNull(_db.measurements),
    );
  }
}
```

```dart
// flutter_app/lib/database/daos/sync_queue_dao.dart
import 'package:drift/drift.dart';
import '../database.dart';

class SyncQueueDao {
  final AppDatabase _db;
  SyncQueueDao(this._db);

  Future<int> enqueue(int visitId) {
    return _db.into(_db.syncQueue).insert(
      SyncQueueCompanion.insert(visitId: visitId),
    );
  }

  Stream<List<SyncQueueData>> watchPending() {
    return (_db.select(_db.syncQueue)
          ..where((s) =>
              (s.status.equals('pending') | s.status.equals('failed')) &
              s.retryCount.isSmallerThanValue(5))
          ..orderBy([(s) => OrderingTerm.asc(s.createdAt)]))
        .watch();
  }

  Stream<int> watchPendingCount() {
    final count = _db.syncQueue.id.count();
    final query = _db.selectOnly(_db.syncQueue)
      ..addColumns([count])
      ..where(
          (_db.syncQueue.status.equals('pending') | _db.syncQueue.status.equals('failed')) &
          _db.syncQueue.retryCount.isSmallerThanValue(5));
    return query.watchSingle().map((row) => row.read(count) ?? 0);
  }

  Future<void> markSyncing(int id) {
    return (_db.update(_db.syncQueue)..where((s) => s.id.equals(id))).write(
      SyncQueueCompanion(
        status: const Value('syncing'),
        lastAttemptAt: Value(DateTime.now()),
      ),
    );
  }

  Future<void> markSynced(int id, {int? serverVisitId}) {
    return (_db.update(_db.syncQueue)..where((s) => s.id.equals(id))).write(
      SyncQueueCompanion(
        status: const Value('synced'),
        serverVisitId: Value(serverVisitId),
        lastAttemptAt: Value(DateTime.now()),
      ),
    );
  }

  Future<void> markFailed(int id, String error) async {
    final entry = await (_db.select(_db.syncQueue)..where((s) => s.id.equals(id))).getSingle();
    await (_db.update(_db.syncQueue)..where((s) => s.id.equals(id))).write(
      SyncQueueCompanion(
        status: const Value('failed'),
        retryCount: Value(entry.retryCount + 1),
        errorMessage: Value(error),
        lastAttemptAt: Value(DateTime.now()),
      ),
    );
  }
}
```

- [ ] **Step 4: Regenerate and run tests**

```bash
cd flutter_app && dart run build_runner build --delete-conflicting-outputs
cd flutter_app && flutter test test/database/daos/child_dao_test.dart
```
Expected: ALL PASS

- [ ] **Step 5: Write and run SyncQueue DAO test**

```dart
// flutter_app/test/database/daos/sync_queue_dao_test.dart
import 'package:drift/native.dart';
import 'package:drift/drift.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';
import 'package:child_growth_monitor_app/database/daos/sync_queue_dao.dart';

void main() {
  late AppDatabase db;
  late SyncQueueDao syncDao;
  late ChildDao childDao;
  late VisitDao visitDao;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    syncDao = SyncQueueDao(db);
    childDao = ChildDao(db);
    visitDao = VisitDao(db);
  });

  tearDown(() => db.close());

  Future<int> createVisit() async {
    final child = await childDao.findOrCreate(
      name: 'Test', dateOfBirth: '2023-01-01', sex: 'M',
    );
    return visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 24.0,
      imagePath: '/test/image.jpg',
      measurement: const MeasurementsCompanion(),
    );
  }

  test('enqueue creates pending entry', () async {
    final visitId = await createVisit();
    await syncDao.enqueue(visitId);
    final pending = await syncDao.watchPending().first;
    expect(pending.length, 1);
    expect(pending.first.status, 'pending');
  });

  test('markSynced updates status', () async {
    final visitId = await createVisit();
    await syncDao.enqueue(visitId);
    final entries = await syncDao.watchPending().first;
    await syncDao.markSynced(entries.first.id, serverVisitId: 42);
    final updated = await syncDao.watchPending().first;
    expect(updated, isEmpty);
  });

  test('markFailed increments retryCount', () async {
    final visitId = await createVisit();
    await syncDao.enqueue(visitId);
    final entries = await syncDao.watchPending().first;
    await syncDao.markFailed(entries.first.id, 'Network error');
    final updated = await syncDao.watchPending().first;
    expect(updated.first.retryCount, 1);
    expect(updated.first.errorMessage, 'Network error');
  });
}
```

```bash
cd flutter_app && flutter test test/database/daos/sync_queue_dao_test.dart
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add flutter_app/lib/database/ flutter_app/test/database/
git commit -m "feat: add Drift DAOs for children, visits, measurements, and sync queue"
```

---

## Task 6: WHO Data Service

**Files:**
- Create: `flutter_app/lib/services/who_data_service.dart`
- Test: `flutter_app/test/services/who_data_service_test.dart`

This is the most critical service to port correctly.

- [ ] **Step 1: Copy WHO fixture files for testing**

```bash
mkdir -p flutter_app/test/fixtures
cp data/who_haz_0_59m.csv flutter_app/test/fixtures/
cp data/who_wfl_boys_0_2.xlsx flutter_app/test/fixtures/
cp data/who_wfl_girls_0_2.xlsx flutter_app/test/fixtures/
cp data/who_wfh_boys_2_5.xlsx flutter_app/test/fixtures/
cp data/who_wfh_girls_2_5.xlsx flutter_app/test/fixtures/
```

- [ ] **Step 2: Write failing tests**

```dart
// flutter_app/test/services/who_data_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

void main() {
  late WhoDataService svc;

  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    svc = WhoDataService();
    await svc.loadFromFiles(
      hazCsvPath: 'test/fixtures/who_haz_0_59m.csv',
      wflBoysPath: 'test/fixtures/who_wfl_boys_0_2.xlsx',
      wflGirlsPath: 'test/fixtures/who_wfl_girls_0_2.xlsx',
      wfhBoysPath: 'test/fixtures/who_wfh_boys_2_5.xlsx',
      wfhGirlsPath: 'test/fixtures/who_wfh_girls_2_5.xlsx',
    );
  });

  group('getMedianHeightForAge', () {
    test('returns median for 24-month-old boy', () {
      final h = svc.getMedianHeightForAge('M', 24);
      expect(h, isNotNull);
      expect(h!, closeTo(87.1, 1.0));
    });

    test('returns null for invalid age', () {
      expect(svc.getMedianHeightForAge('M', -1), isNull);
    });
  });

  group('lmsZscore', () {
    test('LMS formula for L != 0', () {
      final z = WhoDataService.lmsZscore(10.0, 0.5, 9.5, 0.08);
      expect(z, isNotNull);
      expect(z, isFinite);
    });

    test('LMS formula for L = 0 uses log', () {
      final z = WhoDataService.lmsZscore(10.0, 0.0, 9.5, 0.08);
      expect(z, closeTo(0.6454, 0.01));
    });
  });

  group('getHazBoundaries', () {
    test('returns 7 z-score boundaries for valid age/sex', () {
      final b = svc.getHazBoundaries('M', 24);
      expect(b, isNotNull);
      expect(b!.length, 7);
      expect(b[-3]!, lessThan(b[-2]!));
      expect(b[0]!, closeTo(87.1, 1.0));
    });
  });
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd flutter_app && flutter test test/services/who_data_service_test.dart
```
Expected: FAIL

- [ ] **Step 4: Implement WhoDataService**

Port of `who_data_service.py`. The service loads CSV and Excel WHO reference data, caches in memory, and provides lookup functions for z-score computation.

Key implementation details:
- Parse HAZ CSV: columns are sex, age, measure, z_minus_3 through z_plus_3
- Parse Excel LMS files: columns are index_value (height/length), L, M, S
- `getWfhLms()`: exact match within 0.05 cm tolerance, then linear interpolation between nearest entries
- `lmsZscore()`: static formula — if L near 0 use `ln(W/M)/S`, else `((W/M)^L - 1)/(L*S)`
- Two loading paths: `loadFromAssets()` for production, `loadFromFiles()` for tests

The full implementation code is provided in the spec exploration above (services/who_data_service.py). Port all functions exactly, preserving the interpolation logic and dataset selection (WFL for age < 24mo, WFH for age >= 24mo).

- [ ] **Step 5: Run tests**

```bash
cd flutter_app && flutter test test/services/who_data_service_test.dart
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add flutter_app/lib/services/who_data_service.dart flutter_app/test/services/who_data_service_test.dart flutter_app/test/fixtures/
git commit -m "feat: add WhoDataService with HAZ boundary interpolation and LMS z-score computation"
```

---

## Task 7: Nutrition Service

**Files:**
- Create: `flutter_app/lib/services/nutrition_service.dart`
- Test: `flutter_app/test/services/nutrition_service_test.dart`

- [ ] **Step 1: Write failing tests**

```dart
// flutter_app/test/services/nutrition_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/nutrition_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

void main() {
  late WhoDataService who;
  late NutritionService svc;

  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    who = WhoDataService();
    await who.loadFromFiles(
      hazCsvPath: 'test/fixtures/who_haz_0_59m.csv',
      wflBoysPath: 'test/fixtures/who_wfl_boys_0_2.xlsx',
      wflGirlsPath: 'test/fixtures/who_wfl_girls_0_2.xlsx',
      wfhBoysPath: 'test/fixtures/who_wfh_boys_2_5.xlsx',
      wfhGirlsPath: 'test/fixtures/who_wfh_girls_2_5.xlsx',
    );
    svc = NutritionService(who);
  });

  test('computeHaz returns z near 0 for median height', () {
    final median = who.getMedianHeightForAge('M', 24);
    final z = svc.computeHaz('M', 24, median!);
    expect(z, isNotNull);
    expect(z!, closeTo(0.0, 0.1));
  });

  test('computeHaz returns -2 for z=-2 boundary height', () {
    final boundaries = who.getHazBoundaries('M', 24);
    final z = svc.computeHaz('M', 24, boundaries![-2]!);
    expect(z!, closeTo(-2.0, 0.1));
  });

  test('computeWhz returns z for known weight/height', () {
    final z = svc.computeWhz('M', 24.0, 87.0, 12.0);
    expect(z, isNotNull);
    expect(z!, closeTo(0.0, 1.0));
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd flutter_app && flutter test test/services/nutrition_service_test.dart
```

- [ ] **Step 3: Implement NutritionService**

Port of `nutrition_service.py`. Key logic:
- `computeHaz()`: builds 7 z-point list from HAZ boundaries, calls `_interpolateZscore()`
- `computeWhz()`: gets LMS from WhoDataService, calls `lmsZscore()`
- `_interpolateZscore()`: below lowest → extrapolate down; above highest → extrapolate up; between → linear interpolation. Exact port of Python logic.

- [ ] **Step 4: Run tests**

```bash
cd flutter_app && flutter test test/services/nutrition_service_test.dart
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/services/nutrition_service.dart flutter_app/test/services/nutrition_service_test.dart
git commit -m "feat: add NutritionService with HAZ/WHZ z-score computation"
```

---

## Task 8: MUAC Service

**Files:**
- Create: `flutter_app/lib/services/muac_service.dart`
- Test: `flutter_app/test/services/muac_service_test.dart`

- [ ] **Step 1: Write failing tests**

```dart
// flutter_app/test/services/muac_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/muac_service.dart';

void main() {
  test('manual MUAC takes priority', () {
    final r = MuacService.estimate(
      ageMonths: 24, sex: 'M', whz: -1.0, manualMuacCm: 13.5,
    );
    expect(r.muacCm, 13.5);
    expect(r.muacMethod, 'manual');
    expect(r.muacStatus, 'Normal');
  });

  test('estimates from WHZ for boy age 24', () {
    final r = MuacService.estimate(ageMonths: 24, sex: 'M', whz: 0.0);
    // median 24mo boy = 15.7, whz=0: 15.7 * (1 + 0.087*0) = 15.7
    expect(r.muacCm!, closeTo(15.7, 0.1));
    expect(r.muacStatus, 'Normal');
  });

  test('age out of range returns null status', () {
    final r = MuacService.estimate(ageMonths: 3, sex: 'M', whz: 0.0);
    expect(r.ageInRange, false);
    expect(r.muacStatus, isNull);
  });

  test('null whz returns null muac', () {
    final r = MuacService.estimate(ageMonths: 24, sex: 'M', whz: null);
    expect(r.muacCm, isNull);
  });

  test('median interpolates between table entries', () {
    // Age 15 between 12 (15.2) and 18 (15.5) for boys
    final m = MuacService.medianForAge(15, 'M');
    expect(m, closeTo(15.35, 0.01));
  });
}
```

- [ ] **Step 2: Run test, implement, run test**

Implement `MuacService` with:
- `estimate()`: manual priority, then WHZ-based formula `median * (1 + 0.087 * clamp(WHZ, -3, 3))`
- `medianForAge()`: linear interpolation between WHO MUAC reference table entries
- `_classify()`: `<11.5 SAM`, `<12.5 MAM`, else `Normal`; null when age not in 6-59.9mo range

- [ ] **Step 3: Run tests, commit**

```bash
cd flutter_app && flutter test test/services/muac_service_test.dart
git add flutter_app/lib/services/muac_service.dart flutter_app/test/services/muac_service_test.dart
git commit -m "feat: add MuacService with WHO median interpolation and WHZ-based estimation"
```

---

## Task 9: Pose Service

**Files:**
- Create: `flutter_app/lib/services/pose_service.dart`
- Test: `flutter_app/test/services/pose_service_test.dart`

- [ ] **Step 1: Write failing tests for pure math functions**

```dart
// flutter_app/test/services/pose_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/pose_service.dart';

void main() {
  group('estimateHeadTop', () {
    test('computes head top from nose and eyes', () {
      // nose y=100, eyes y=90, nose_to_eye=10, head_top = 100 - 10*2.5 = 75
      final y = PoseService.estimateHeadTopY(
        noseY: 100, leftEyeY: 90, rightEyeY: 90,
        leftEarY: null, rightEarY: null,
      );
      expect(y, closeTo(75.0, 0.1));
    });

    test('averages with ear method when ears visible', () {
      // method1: 100 - 10*2.5 = 75
      // method2: 88 - 10*3.0 = 58
      // avg: 66.5
      final y = PoseService.estimateHeadTopY(
        noseY: 100, leftEyeY: 90, rightEyeY: 90,
        leftEarY: 88, rightEarY: 88,
      );
      expect(y, closeTo(66.5, 0.1));
    });
  });

  group('estimateChinY', () {
    test('estimates from nose without mouth', () {
      // chin = 100 + 10*1.5 = 115
      expect(PoseService.estimateChinY(noseY: 100, noseToEye: 10, mouthY: null), closeTo(115, 0.1));
    });

    test('uses mouth when available', () {
      // chin = 110 + 10*0.5 = 115
      expect(PoseService.estimateChinY(noseY: 100, noseToEye: 10, mouthY: 110), closeTo(115, 0.1));
    });
  });
}
```

- [ ] **Step 2: Run test, implement, run test**

Implement `PoseService` wrapping `google_mlkit_pose_detection`:
- `detectPose(imagePath)` → list of 33 `PoseLandmark`
- `extractSegments(landmarks, w, h)` → `BodySegments` with all pixel measurements
- `extractSideSegments(landmarks, heightCm)` → `SideViewSegments` with AP depth
- `computeConfidence(landmarks)` → average visibility of 7 key landmarks
- Static helpers: `estimateHeadTopY()`, `estimateChinY()`

Port the exact landmark indices and math from Python `measurement_service.py`.

- [ ] **Step 3: Run tests, commit**

```bash
cd flutter_app && flutter test test/services/pose_service_test.dart
git add flutter_app/lib/services/pose_service.dart flutter_app/test/services/pose_service_test.dart
git commit -m "feat: add PoseService with MediaPipe pose detection and body segment extraction"
```

---

## Task 10: Measurement Service

**Files:**
- Create: `flutter_app/lib/services/measurement_service.dart`
- Test: `flutter_app/test/services/measurement_service_test.dart`

- [ ] **Step 1: Write failing tests for body build and height estimation**

```dart
// flutter_app/test/services/measurement_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/measurement_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

void main() {
  late WhoDataService who;

  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    who = WhoDataService();
    await who.loadFromFiles(
      hazCsvPath: 'test/fixtures/who_haz_0_59m.csv',
      wflBoysPath: 'test/fixtures/who_wfl_boys_0_2.xlsx',
      wflGirlsPath: 'test/fixtures/who_wfl_girls_0_2.xlsx',
      wfhBoysPath: 'test/fixtures/who_wfh_boys_2_5.xlsx',
      wfhGirlsPath: 'test/fixtures/who_wfh_girls_2_5.xlsx',
    );
  });

  test('average build for normal shoulder ratio', () {
    final r = MeasurementService.estimateBodyBuild(
      shoulderWidthPx: 100, totalHeightPx: 500, ageMonths: 24,
    );
    expect(r.bodyBuild, 'average');
    expect(r.weightAdjustment, 1.0);
  });

  test('slender build for low ratio', () {
    final r = MeasurementService.estimateBodyBuild(
      shoulderWidthPx: 80, totalHeightPx: 500, ageMonths: 24,
    );
    expect(r.bodyBuild, 'slender');
    expect(r.weightAdjustment, 0.95);
  });

  test('stocky build for high ratio', () {
    final r = MeasurementService.estimateBodyBuild(
      shoulderWidthPx: 130, totalHeightPx: 500, ageMonths: 24,
    );
    expect(r.bodyBuild, 'stocky');
    expect(r.weightAdjustment, 1.05);
  });

  test('WHO height estimation returns median', () {
    final h = MeasurementService.estimateHeightFromWho(
      ageMonths: 24, sex: 'M', who: who,
    );
    expect(h, isNotNull);
    expect(h!, closeTo(87.1, 1.0));
  });
}
```

- [ ] **Step 2: Run test, implement, run test**

Implement `MeasurementService`:
- `processSegments()` — full pipeline: WHO height → body build → confidence
- Static `estimateBodyBuild()` — shoulder/height ratio vs expected ± 0.03
- Static `estimateHeightFromWho()` — uses median from WHO HAZ data
- Confidence: base from pose × proportion agreement score

- [ ] **Step 3: Run tests, commit**

```bash
cd flutter_app && flutter test test/services/measurement_service_test.dart
git add flutter_app/lib/services/measurement_service.dart flutter_app/test/services/measurement_service_test.dart
git commit -m "feat: add MeasurementService with WHO height estimation and body build classification"
```

---

## Task 11: ML Inference Service

**Files:**
- Create: `flutter_app/lib/services/ml_inference_service.dart`
- Test: `flutter_app/test/services/ml_inference_service_test.dart`

- [ ] **Step 1: Write failing tests for feature extraction**

```dart
// flutter_app/test/services/ml_inference_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/ml_inference_service.dart';
import 'package:child_growth_monitor_app/models/body_measurements.dart';

void main() {
  test('extracts features from body segments', () {
    final segments = BodySegments(
      shoulderWidthPx: 100, hipWidthPx: 88, torsoLengthPx: 150,
      upperArmLengthPx: 75, totalHeightPx: 500,
    );
    final features = MlInferenceService.extractFeatures(
      segments: segments, ageMonths: 24.0, sex: 'M',
      heightCm: 87.0, sideSegments: null,
    );
    expect(features, isNotNull);
    expect(features!.ageMonths, 24.0);
    expect(features.sexBinary, 1);
    // scale = 87.0/500 = 0.174
    expect(features.shoulderWidthCm, closeTo(17.4, 0.1));
    expect(features.chestDepthCm, isNull); // no side view
  });

  test('imputes shoulder from age ratio when missing', () {
    final segments = BodySegments(totalHeightPx: 500);
    final f = MlInferenceService.extractFeatures(
      segments: segments, ageMonths: 24.0, sex: 'F',
      heightCm: 87.0, sideSegments: null,
    );
    expect(f!.shoulderWidthCm, closeTo(87.0 * 0.210, 0.1));
  });

  test('bodyBuildScore uses 0.02 threshold', () {
    expect(MlInferenceService.bodyBuildScore(18.0, 87.0, 24.0), 0);
  });
}
```

- [ ] **Step 2: Run test, implement, run test**

Implement `MlInferenceService`:
- `loadModels()` — load TFLite interpreters + scaler JSON from assets
- `predict(features)` — normalize → weight estimator → wasting classifier → return prediction
- Static `extractFeatures()` — port of `ml_service.py extract_features()`: scale px→cm, impute missing, validate side-view AP depth (15-65% of lateral width)
- Static `bodyBuildScore()` — uses ±0.02 ML threshold

- [ ] **Step 3: Run tests, commit**

```bash
cd flutter_app && flutter test test/services/ml_inference_service_test.dart
git add flutter_app/lib/services/ml_inference_service.dart flutter_app/test/services/ml_inference_service_test.dart
git commit -m "feat: add MlInferenceService with TFLite inference and feature extraction"
```

---

## Task 12: Assessment Service (Pipeline Orchestrator)

**Files:**
- Create: `flutter_app/lib/services/assessment_service.dart`
- Test: `flutter_app/test/services/assessment_service_test.dart`

- [ ] **Step 1: Write failing tests for weight priority and age computation**

```dart
// flutter_app/test/services/assessment_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/assessment_service.dart';

void main() {
  test('manual weight takes priority', () {
    final r = AssessmentService.determineWeight(
      manualWeightKg: 12.0, mlWeightKg: 11.5,
      whoMedianWeightKg: 12.2, weightAdjustment: 1.0,
    );
    expect(r.effectiveWeight, 12.0);
    expect(r.source, 'manual');
  });

  test('ML weight used when in bounds', () {
    final r = AssessmentService.determineWeight(
      manualWeightKg: null, mlWeightKg: 11.5,
      whoMedianWeightKg: 12.0, weightAdjustment: 1.0,
    );
    expect(r.effectiveWeight, 11.5);
    expect(r.source, 'ml_estimated');
  });

  test('ML weight rejected when out of bounds', () {
    final r = AssessmentService.determineWeight(
      manualWeightKg: null, mlWeightKg: 3.0,
      whoMedianWeightKg: 12.0, weightAdjustment: 1.0,
    );
    expect(r.effectiveWeight, 12.0);
    expect(r.source, 'who_median_estimated');
  });

  test('WHO median adjusted by body build', () {
    final r = AssessmentService.determineWeight(
      manualWeightKg: null, mlWeightKg: null,
      whoMedianWeightKg: 12.0, weightAdjustment: 1.05,
    );
    expect(r.effectiveWeight, closeTo(12.6, 0.01));
  });

  test('age computation is correct', () {
    final age = AssessmentService.computeAgeMonths(
      DateTime(2024, 1, 1), DateTime(2026, 1, 1),
    );
    expect(age, closeTo(24.0, 0.5));
  });
}
```

- [ ] **Step 2: Run test, implement, run test**

Implement `AssessmentService.assess()` — the full pipeline:
1. Compute age from DOB
2. Pose detection on front image
3. Extract body segments
4. Process measurements (height, build, confidence)
5. Side-view processing (if provided)
6. ML feature extraction + inference
7. Weight determination (manual > ML > WHO median)
8. Z-score computation (HAZ + WHZ)
9. MUAC estimation
10. Persist to Drift DB (child, visit, measurement)
11. Queue for sync
12. Return `AssessmentResult`

- [ ] **Step 3: Run tests, commit**

```bash
cd flutter_app && flutter test test/services/assessment_service_test.dart
git add flutter_app/lib/services/assessment_service.dart flutter_app/test/services/assessment_service_test.dart
git commit -m "feat: add AssessmentService pipeline orchestrator with weight priority logic"
```

---

## Task 13: Riverpod Providers (Database + Services)

**Files:**
- Create: `flutter_app/lib/providers/database_provider.dart`
- Modify: `flutter_app/lib/providers/children_provider.dart`
- Create: `flutter_app/lib/providers/sync_provider.dart`
- Modify: `flutter_app/lib/main.dart`

- [ ] **Step 1: Create database_provider.dart with all service providers**

Providers to create:
- `databaseProvider` — singleton `AppDatabase`
- `childDaoProvider`, `visitDaoProvider`, `syncQueueDaoProvider` — DAOs
- `whoDataProvider`, `nutritionProvider`, `measurementProvider` — processing services
- `poseProvider`, `mlInferenceProvider` — ML services (with dispose)
- `assessmentServiceProvider` — orchestrator wiring all services together
- `initializeServices()` — async function to load WHO data and ML models

- [ ] **Step 2: Update children_provider.dart**

Switch from `FutureProvider` calling API to `StreamProvider` watching Drift DB:
- `childrenProvider` → `StreamProvider<List<Child>>` from `ChildDao.watchAll()`
- `childDetailProvider` → `StreamProvider.family` from `VisitDao.watchByChildId()`

- [ ] **Step 3: Create sync_provider.dart**

- `pendingSyncCountProvider` → `StreamProvider<int>` from `SyncQueueDao.watchPendingCount()`

- [ ] **Step 4: Update main.dart**

Initialize services before running app:
```dart
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final container = ProviderContainer();
  await initializeServices(container);
  runApp(UncontrolledProviderScope(container: container, child: const ChildGrowthMonitorApp()));
}
```

- [ ] **Step 5: Verify compilation, commit**

```bash
cd flutter_app && flutter analyze
git add flutter_app/lib/providers/ flutter_app/lib/main.dart
git commit -m "feat: add database and service providers, switch children to local DB"
```

---

## Task 14: Sync Service

**Files:**
- Create: `flutter_app/lib/services/sync_service.dart`
- Test: `flutter_app/test/services/sync_service_test.dart`

- [ ] **Step 1: Write failing test for backoff logic**

```dart
// flutter_app/test/services/sync_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/sync_service.dart';

void main() {
  test('backoff durations are correct', () {
    expect(SyncService.backoffDuration(0), const Duration(seconds: 30));
    expect(SyncService.backoffDuration(1), const Duration(seconds: 60));
    expect(SyncService.backoffDuration(2), const Duration(seconds: 120));
    expect(SyncService.backoffDuration(10), const Duration(minutes: 15)); // capped
  });
}
```

- [ ] **Step 2: Run test, implement, run test**

Implement `SyncService`:
- `start()` — listen to `connectivity_plus` stream + 15-min periodic timer
- `syncAll()` — query pending/failed entries (retryCount < 5), sync each one
- `_syncOne()` — mark syncing → build multipart request → POST → mark synced/failed
- Static `backoffDuration()` — `min(2^retryCount * 30s, 15min)`
- `dispose()` — cancel subscriptions and timers

- [ ] **Step 3: Run tests, commit**

```bash
cd flutter_app && flutter test test/services/sync_service_test.dart
git add flutter_app/lib/services/sync_service.dart flutter_app/test/services/sync_service_test.dart
git commit -m "feat: add SyncService with connectivity-triggered queue-based upload"
```

---

## Task 15: Update Assessment Screen for Local Pipeline

**Files:**
- Modify: `flutter_app/lib/screens/assessment/assessment_screen.dart`

- [ ] **Step 1: Replace API submission with local pipeline**

In `_submitAssessment()`:
1. Copy images from temp to `getApplicationDocumentsDirectory()/images/`
2. Call `ref.read(assessmentServiceProvider).assess(...)` instead of `ApiService.submitAssessment()`
3. Set `assessmentResultProvider` from the local result
4. Remove the health check dependency on API connectivity

Key change — replace this pattern:
```dart
// Old: final result = await api.submitAssessment(...)
// New:
final assessmentSvc = ref.read(assessmentServiceProvider);
final result = await assessmentSvc.assess(
  imagePath: frontPath,
  sideImagePath: sidePath,
  childName: _nameController.text.trim(),
  dateOfBirth: _selectedDOB,
  sex: _selectedSex,
  // ... other fields
);
ref.read(assessmentResultProvider.notifier).state = result;
```

- [ ] **Step 2: Add path_provider import, copy images to persistent storage**

Images must be stored in app documents directory (not temp) so they survive for sync.

- [ ] **Step 3: Verify compilation, commit**

```bash
cd flutter_app && flutter analyze
git add flutter_app/lib/screens/assessment/assessment_screen.dart
git commit -m "feat: switch assessment screen to local processing pipeline"
```

---

## Task 16: Sync Status UI

**Files:**
- Modify: `flutter_app/lib/screens/shared/app_scaffold.dart`

- [ ] **Step 1: Add sync status indicator to app bar**

Add a `Consumer` widget in the AppBar actions that watches `pendingSyncCountProvider`:
- `count > 0`: show cloud_upload icon with badge count, tap triggers manual sync
- `count == 0`: show cloud_done icon (green)
- Error state: show cloud_off icon (red)

- [ ] **Step 2: Verify compilation, commit**

```bash
cd flutter_app && flutter analyze
git add flutter_app/lib/screens/shared/app_scaffold.dart
git commit -m "feat: add sync status indicator with pending count badge"
```

---

## Task 17: Update Children Screens for DB

**Files:**
- Modify: `flutter_app/lib/screens/children/children_list_screen.dart`
- Modify: `flutter_app/lib/screens/children/child_detail_screen.dart`

- [ ] **Step 1: Update children_list_screen.dart**

Switch from `FutureProvider<List<ChildSummary>>` to `StreamProvider<List<Child>>`:
- Use `ref.watch(childrenProvider)` (now a stream)
- Map `Child` (Drift generated) fields to display: `child.name`, `child.dateOfBirth`, `child.sex`
- Visit count: query from visits table or show last visit date

- [ ] **Step 2: Update child_detail_screen.dart**

Switch to stream-based visit provider:
- Use `ref.watch(childDetailProvider(childId))` which returns visits with measurements
- Map Drift `Measurement` fields to growth chart data
- Map visit records to visit history list

- [ ] **Step 3: Run full test suite**

```bash
cd flutter_app && flutter test
```
Expected: ALL PASS

- [ ] **Step 4: Run static analysis**

```bash
cd flutter_app && flutter analyze
```
Expected: No issues

- [ ] **Step 5: Commit**

```bash
git add flutter_app/
git commit -m "feat: complete offline-first migration - children and detail screens use local DB"
```

---

## Verification Plan

After all tasks are complete:

### Automated
```bash
cd flutter_app && flutter test                      # All unit + widget tests
cd flutter_app && flutter analyze                   # Static analysis clean
cd flutter_app && dart run build_runner build        # Drift codegen succeeds
```

### Manual Testing
1. **Offline assessment:** Enable airplane mode -> capture photo -> submit -> verify result screen shows z-scores and classification
2. **Data persistence:** Close and reopen app -> verify children list shows previously assessed children
3. **Sync recovery:** Disable airplane mode -> verify sync indicator changes from pending to done
4. **Side view improvement:** Assess with front + side photo -> verify chest/abd depth populated
5. **Weight priority:** Submit with manual weight -> verify it takes priority over ML estimate
