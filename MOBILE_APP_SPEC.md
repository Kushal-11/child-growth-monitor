# Child Growth Monitor — Mobile App Specification

> Technical specification for building an offline-first Flutter app that performs
> on-device malnutrition screening for children aged 0–60 months.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [On-Device ML Pipeline](#on-device-ml-pipeline)
3. [Model Files & Formats](#model-files--formats)
4. [Feature Extraction](#feature-extraction)
5. [WHO Data Tables](#who-data-tables)
6. [Z-Score Computation](#z-score-computation)
7. [MUAC Estimation](#muac-estimation)
8. [Assessment Flow](#assessment-flow)
9. [API Endpoints (Sync)](#api-endpoints-sync)
10. [Schemas & Data Types](#schemas--data-types)
11. [Database Schema (Local)](#database-schema-local)
12. [Constants & Thresholds](#constants--thresholds)
13. [Recommended Flutter Packages](#recommended-flutter-packages)
14. [App Bundle Size Budget](#app-bundle-size-budget)

---

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│               PHONE APP (offline)            │
│                                              │
│  Camera ──► MediaPipe PoseLandmarker         │
│               (on-device, GPU delegate)      │
│                    │                         │
│                    ▼                         │
│           Body Segment Extraction            │
│               (pixel measurements)           │
│                    │                         │
│                    ▼                         │
│         TFLite Weight Estimator (7 KB)       │
│         TFLite Wasting Classifier (17 KB)    │
│                    │                         │
│                    ▼                         │
│         WHO Z-Score + MUAC Computation       │
│               (bundled JSON tables)          │
│                    │                         │
│                    ▼                         │
│           Local SQLite Storage               │
│                                              │
│  ──── When online ────────────────────────►  │
│         Sync to FastAPI backend              │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│            FASTAPI BACKEND                   │
│  POST /api/v1/sync    ← batch upload        │
│  GET  /api/v1/children                       │
│  GET  /api/v1/children/{id}                  │
│  GET  /api/v1/health                         │
└──────────────────────────────────────────────┘
```

**Key principle**: All assessment logic runs on-device. The server is used only
for data sync, dashboards, and supervisor review. No internet is required to
complete a screening.

---

## On-Device ML Pipeline

### Step 1: Pose Detection (MediaPipe)

| Property | Value |
|---|---|
| Model file | `pose_landmarker_heavy.task` |
| Model size | 29.2 MB |
| Landmarks | 33 body keypoints |
| Input | RGB image (any resolution, internally resized) |
| Min detection confidence | 0.5 |
| Min presence confidence | 0.5 |
| Max poses | 1 |
| Mobile acceleration | GPU delegate (Metal on iOS, OpenGL ES on Android) |

#### MediaPipe Landmark Indices

```
 0: NOSE                  11: LEFT_SHOULDER      23: LEFT_HIP
 1: LEFT_EYE_INNER        12: RIGHT_SHOULDER     24: RIGHT_HIP
 2: LEFT_EYE              13: LEFT_ELBOW         25: LEFT_KNEE
 3: LEFT_EYE_OUTER        14: RIGHT_ELBOW        26: RIGHT_KNEE
 4: RIGHT_EYE_INNER       15: LEFT_WRIST         27: LEFT_ANKLE
 5: RIGHT_EYE             16: RIGHT_WRIST        28: RIGHT_ANKLE
 6: RIGHT_EYE_OUTER       17: LEFT_PINKY         29: LEFT_HEEL
 7: LEFT_EAR              18: RIGHT_PINKY        30: RIGHT_HEEL
 8: RIGHT_EAR             19: LEFT_INDEX         31: LEFT_FOOT_INDEX
 9: MOUTH_LEFT            20: RIGHT_INDEX        32: RIGHT_FOOT_INDEX
10: MOUTH_RIGHT           21: LEFT_THUMB
                           22: RIGHT_THUMB
```

#### Landmarks Used for Measurements

| Measurement | Landmarks | Formula |
|---|---|---|
| **Head top** (estimated) | 0 (NOSE), 2 (L_EYE), 5 (R_EYE) | `nose_y - (nose_y - eye_midpoint_y) × 2.5` |
| **Shoulder midpoint** | 11, 12 | `(L_SHOULDER + R_SHOULDER) / 2` |
| **Hip midpoint** | 23, 24 | `(L_HIP + R_HIP) / 2` |
| **Heel (floor)** | 29, 30 | `min(L_HEEL.y, R_HEEL.y)` — lowest point |
| **Shoulder width** | 11, 12 | `abs(L_SHOULDER.x - R_SHOULDER.x)` (horizontal only) |
| **Hip width** | 23, 24 | `abs(L_HIP.x - R_HIP.x)` |
| **Upper arm length** | 11, 13 | Euclidean 2D distance (shoulder → elbow) |
| **Total height** | head_top, heel | `heel_y - head_top_y` (in pixels) |

### Step 2: Body Segment Extraction

Extract these pixel measurements from pose landmarks:

```dart
class BodySegments {
  double? headHeightPx;       // head_top to chin
  double? torsoLengthPx;      // shoulder_midpoint to hip_midpoint
  double? legLengthPx;        // hip_midpoint to heel
  double? shoulderWidthPx;    // L_SHOULDER to R_SHOULDER (x-axis)
  double? hipWidthPx;         // L_HIP to R_HIP (x-axis)
  double? upperArmLengthPx;   // shoulder to elbow (2D euclidean)
  double? totalHeightPx;      // head_top to heel

  double headConfidence;      // fraction of head landmarks visible
  double torsoConfidence;
  double legConfidence;
  double hipConfidence;
  double armConfidence;
}
```

### Step 3: Feature Computation

Convert pixel segments to centimeters and compute the 14-feature vector.

**Scale factor**: `scale = height_cm / totalHeightPx` (cm per pixel)

```dart
class WastingFeatures {
  double ageMonths;            // 0–60
  int    sexBinary;            // 1 = Male, 0 = Female
  double heightCm;             // child's height
  double shoulderWidthCm;     // px × scale
  double hipWidthCm;          // px × scale
  double torsoLengthCm;       // px × scale
  double upperArmLengthCm;    // px × scale
  double shoulderHeightRatio; // shoulderWidthCm / heightCm
  double hipHeightRatio;      // hipWidthCm / heightCm
  int    bodyBuildScore;      // -1 (slender), 0 (average), 1 (stocky)
  double chestDepthCm;        // from side view or imputed
  double abdDepthCm;          // from side view or imputed
  double chestDepthRatio;     // chestDepthCm / heightCm
  double abdDepthRatio;       // abdDepthCm / heightCm
}
```

#### Imputation Rules (When Side View Unavailable)

```
chestDepthCm  = shoulderWidthCm × 0.45   (Snyder et al. 1975)
abdDepthCm    = hipWidthCm × 0.50        (Snyder et al. 1975)
chestDepthRatio = chestDepthCm / heightCm
abdDepthRatio   = abdDepthCm / heightCm
```

#### Fallback Imputation (When Landmark Missing)

```
Shoulder width:
  < 24 months:  heightCm × 0.200
  24–48 months: heightCm × 0.210
  ≥ 48 months:  heightCm × 0.218

Hip width:
  shoulderWidthCm × 0.88

Upper arm length:
  < 24 months:  heightCm × 0.150
  24–48 months: heightCm × 0.158
  ≥ 48 months:  heightCm × 0.165

Torso length:
  heightCm × 0.30
```

### Step 4: StandardScaler Normalization

Apply `(x - mean) / stddev` to each of the 14 features before model inference.

**Bundle these as constants** (extracted from `feature_scaler.pkl`):

| Index | Feature | Mean | Std Dev |
|---|---|---|---|
| 0 | age_months | 29.56 | 17.31 |
| 1 | sex_binary | 0.498 | 0.500 |
| 2 | height_cm | 88.30 | 15.56 |
| 3 | shoulder_width_cm | 18.07 | 3.88 |
| 4 | hip_width_cm | 15.90 | 3.41 |
| 5 | torso_length_cm | 26.92 | 4.28 |
| 6 | upper_arm_length_cm | 13.85 | 2.93 |
| 7 | shoulder_height_ratio | 0.203 | 0.013 |
| 8 | hip_height_ratio | 0.179 | 0.011 |
| 9 | body_build_score | -0.026 | 0.210 |
| 10 | chest_depth_cm | 7.96 | 1.87 |
| 11 | abd_depth_cm | 7.78 | 1.83 |
| 12 | chest_depth_ratio | 0.090 | 0.010 |
| 13 | abd_depth_ratio | 0.088 | 0.010 |

### Step 5: TFLite Inference

**Weight Estimator:**
- Input: `Float32List` of shape `[1, 14]` (scaled features)
- Output: `Float32List` of shape `[1, 1]` → predicted weight in kg

**Wasting Classifier:**
- Input: `Float32List` of shape `[1, 14]` (scaled features)
- Output: `Float32List` of shape `[1, 5]` → softmax probabilities

**Label mapping** (alphabetical order, from `label_encoder.pkl`):

| Output Index | Class |
|---|---|
| 0 | MAM |
| 1 | Normal |
| 2 | Overweight |
| 3 | Risk_Overweight |
| 4 | SAM |

---

## Model Files & Formats

Bundle these files with the app:

| File | Size | Format | Purpose |
|---|---|---|---|
| `pose_landmarker_heavy.task` | 29.2 MB | MediaPipe Task | Pose detection (33 landmarks) |
| `weight_estimator.tflite` | 8.3 KB | TensorFlow Lite | Weight regression |
| `wasting_classifier.tflite` | 18.5 KB | TensorFlow Lite | 5-class wasting classification |

**Not bundled** (convert to constants instead):
- `feature_scaler.pkl` → hardcode mean/stddev table above
- `label_encoder.pkl` → hardcode label mapping above

### Model Architecture (for reference)

```
Weight Estimator:
  Input(14) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)

Wasting Classifier:
  Input(14) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(5, Softmax)
```

---

## WHO Data Tables

Bundle as JSON assets. Total size: ~500 KB.

### HAZ Table: Height-for-Age Z-Scores

**Source file**: `who_haz_0_59m.csv` (125 rows)

```json
[
  {
    "sex": "M",
    "measure": "length",
    "age_months": 0,
    "z_minus_3": 44.2,
    "z_minus_2": 46.1,
    "z_minus_1": 48.0,
    "z_0": 49.9,
    "z_plus_1": 51.8,
    "z_plus_2": 53.7,
    "z_plus_3": 55.6
  },
  ...
]
```

- **sex**: `"M"` or `"F"`
- **measure**: `"length"` (0–24 months) or `"height"` (24–59 months)
- **age_months**: 0–59 (integer months)
- **z_minus_3 to z_plus_3**: height in cm at each Z-score boundary

### WFH LMS Tables: Weight-for-Height (Authoritative for WHZ)

**Source files** (4 Excel files → convert to JSON):

| File | Population | Height Range |
|---|---|---|
| `wfl_boys_0-to-2-years_zscores.xlsx` | Boys 0–24 months | 45.0–120.0 cm |
| `wfl_girls_0-to-2-years_zscores.xlsx` | Girls 0–24 months | 45.0–120.0 cm |
| `wfh_boys_2-to-5-years_zscores.xlsx` | Boys 24–60 months | 65.0–120.0 cm |
| `wfh_girls_2-to-5-years_zscores.xlsx` | Girls 24–60 months | 65.0–120.0 cm |

```json
[
  {
    "height_cm": 45.0,
    "L": -0.3521,
    "M": 2.4410,
    "S": 0.09182
  },
  ...
]
```

- **L, M, S**: Box-Cox transformation parameters for the LMS method
- Height/length values at 0.1 cm increments

### WHZ Reference (Quick Lookup Fallback)

**Source file**: `who_whz_reference.csv` (303 rows)

```json
[
  {
    "sex": "M",
    "height_cm": 45.0,
    "minus2sd_kg": 1.9,
    "minus3sd_kg": 1.7,
    "median_kg": 2.4
  },
  ...
]
```

Used to quickly get the median weight for a given height when full LMS
computation is not needed (e.g., weight imputation fallback).

---

## Z-Score Computation

### HAZ (Height-for-Age)

```
Input: sex, age_months (integer), height_cm

1. Look up the row in HAZ table matching sex and age_months
2. Get 7 boundary values: z_minus_3 ... z_plus_3
3. Build interpolation table:
     [(-3, z_minus_3), (-2, z_minus_2), (-1, z_minus_1),
      (0, z_0), (1, z_plus_1), (2, z_plus_2), (3, z_plus_3)]
4. Linear interpolation: find where height_cm falls, interpolate z-score
5. Extrapolate linearly if outside ±3 SD
```

**HAZ Classification:**

| Z-Score Range | Status |
|---|---|
| < -3 | Severely Stunted |
| -3 to -2 | Stunted |
| -2 to +2 | Normal |
| > +2 | Tall |

### WHZ (Weight-for-Height)

```
Input: sex, age_months, height_cm, weight_kg

1. Select correct LMS table based on sex and age:
     age < 24 months → wfl_{sex}_0_2
     age ≥ 24 months → wfh_{sex}_2_5

2. Interpolate L, M, S for the given height_cm
   (linear interpolation between nearest 0.1 cm rows)

3. Apply Box-Cox formula:
     IF L ≠ 0:
       Z = [((weight_kg / M) ^ L) - 1] / (L × S)
     IF L ≈ 0:
       Z = ln(weight_kg / M) / S
```

**WHZ Classification:**

| Z-Score Range | Status |
|---|---|
| < -3 | Severe Acute Malnutrition (SAM) |
| -3 to -2 | Moderate Acute Malnutrition (MAM) |
| -2 to +1 | Normal |
| +1 to +2 | Possible Risk of Overweight |
| +2 to +3 | Overweight |
| > +3 | Obese |

---

## MUAC Estimation

### MUAC-for-Age Medians (WHO 2006 Standards)

**Boys:**

| Age (months) | 3 | 6 | 9 | 12 | 18 | 24 | 30 | 36 | 42 | 48 | 54 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MUAC (cm) | 12.5 | 14.0 | 14.8 | 15.2 | 15.5 | 15.7 | 15.8 | 15.9 | 16.0 | 16.1 | 16.1 | 16.2 |

**Girls:**

| Age (months) | 3 | 6 | 9 | 12 | 18 | 24 | 30 | 36 | 42 | 48 | 54 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MUAC (cm) | 12.3 | 13.8 | 14.6 | 14.9 | 15.2 | 15.4 | 15.5 | 15.6 | 15.7 | 15.7 | 15.8 | 15.8 |

Interpolate linearly between table entries for intermediate ages.

### Estimation Formula

```
IF manual_muac_cm is provided:
  Use manual value directly.

ELSE IF age is 6–59 months AND WHZ is available:
  median = interpolate MUAC median from table above
  muac_cm = median × (1 + 0.087 × clamp(WHZ, -3, +3))
  Round to 1 decimal place.

ELSE:
  MUAC cannot be estimated (age out of range or no WHZ).
```

### MUAC Classification (Ages 6–59 Months Only)

| MUAC (cm) | Status |
|---|---|
| < 11.5 | SAM (Severe Acute Malnutrition) |
| 11.5 – 12.5 | At Risk (MAM) |
| ≥ 12.5 | Normal |

For children under 6 months or over 59 months, MUAC screening does not apply
(`age_in_range = false`).

---

## Assessment Flow

Complete on-device assessment sequence:

```
INPUT: camera_image, child_name, date_of_birth, sex,
       optional: manual_weight_kg, manual_height_cm, manual_muac_cm,
       optional: side_view_image

STEP 1 — Compute age
  age_months = (today - date_of_birth).days / 30.4375

STEP 2 — Pose detection (MediaPipe)
  landmarks = PoseLandmarker.detect(camera_image)
  body_segments = extract_body_segments(landmarks)  // pixel values

STEP 3 — Height estimation
  Priority order:
    a) manual_height_cm (if provided by user)
    b) WHO statistical median for age/sex (from HAZ table, z_0 column)
  effective_height = selected value

STEP 4 — Convert pixels to centimeters
  scale = effective_height / body_segments.totalHeightPx
  shoulder_cm = body_segments.shoulderWidthPx × scale
  hip_cm = body_segments.hipWidthPx × scale
  torso_cm = body_segments.torsoLengthPx × scale
  arm_cm = body_segments.upperArmLengthPx × scale

STEP 5 — Side-view processing (optional)
  IF side_view_image provided:
    side_landmarks = PoseLandmarker.detect(side_view_image)
    side_height_px = nose_to_heel(side_landmarks)
    side_scale = effective_height / side_height_px
    chest_depth_cm = x_span(shoulder+elbow landmarks) × side_scale
    abd_depth_cm = x_span(hip+knee landmarks) × side_scale
    Validate: 2 cm < depth < 50 cm
  ELSE:
    chest_depth_cm = shoulder_cm × 0.45
    abd_depth_cm = hip_cm × 0.50

STEP 6 — Build 14-feature vector
  features = [age_months, sex_binary, height_cm,
              shoulder_cm, hip_cm, torso_cm, arm_cm,
              shoulder_cm/height_cm, hip_cm/height_cm,
              body_build_score,
              chest_depth_cm, abd_depth_cm,
              chest_depth_cm/height_cm, abd_depth_cm/height_cm]

STEP 7 — Normalize features
  FOR i IN 0..13:
    scaled[i] = (features[i] - SCALER_MEAN[i]) / SCALER_STD[i]

STEP 8 — ML inference
  weight_kg = WeightEstimator.run(scaled)     // single float
  probs = WastingClassifier.run(scaled)        // 5 floats (softmax)
  wasting_class = LABELS[argmax(probs)]

STEP 9 — Determine effective weight
  Priority:
    a) manual_weight_kg (if provided)
    b) ML estimated weight (if within 45%–180% of WHO median for height)
    c) WHO median weight for height × body_build_adjustment

STEP 10 — Compute Z-scores
  haz = compute_haz(sex, age_months, effective_height)
  whz = compute_whz(sex, age_months, effective_height, effective_weight)

STEP 11 — MUAC estimation
  IF manual_muac_cm: use it
  ELIF age 6–59 months AND whz available:
    muac_cm = median(age, sex) × (1 + 0.087 × clamp(whz, -3, +3))

STEP 12 — Save locally + queue for sync
  Insert into local SQLite: child, visit, measurement_result
  Set synced_at = NULL (pending sync)
```

### Body Build Classification

Used in Step 6 and Step 9:

```
Expected shoulder/height ratio by age:
  < 24 months:  0.200
  24–48 months: 0.210
  ≥ 48 months:  0.218

IF actual_ratio < (expected - 0.02):
  body_build = "slender",  score = -1,  weight_adjustment = 0.95
ELIF actual_ratio > (expected + 0.02):
  body_build = "stocky",   score = +1,  weight_adjustment = 1.05
ELSE:
  body_build = "average",  score =  0,  weight_adjustment = 1.00
```

---

## API Endpoints (Sync)

The backend is only used for syncing data when the device has connectivity.
The base URL is configurable (default: `http://<server>:8000`).

### POST `/api/v1/assess`

Full server-side assessment (can be used as fallback when on-device ML is
unavailable, e.g., older devices).

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | File | Yes | Frontal photo (PNG/JPG) |
| `image_side` | File | No | Side-view photo |
| `image_back` | File | No | Back view (reserved, unused) |
| `child_name` | String | Yes | 1–100 characters |
| `date_of_birth` | String | Yes | ISO format `yyyy-mm-dd` |
| `sex` | String | Yes | `"M"` or `"F"` |
| `weight_kg` | Float | No | Manual weight, 0–50 kg |
| `height_cm` | Float | No | Manual height, 0–200 cm |
| `height_unit` | String | No | `"cm"` (default) or `"inch"` |
| `muac_cm` | Float | No | Manual MUAC in cm |
| `guardian_name` | String | No | |
| `location` | String | No | |

**Response**: `AssessmentResponse` (see [Schemas](#schemas--data-types))

### GET `/api/v1/children`

Returns array of child summaries:

```json
[
  {
    "id": 1,
    "name": "Aisha",
    "date_of_birth": "2023-06-15",
    "sex": "F",
    "visit_count": 3
  }
]
```

### GET `/api/v1/children/{child_id}`

Returns full child record with visit history (see schemas below).

### GET `/api/v1/health`

```json
{ "status": "ok", "service": "child-growth-monitor" }
```

---

## Schemas & Data Types

### AssessmentResponse

```json
{
  "child_name": "string",
  "sex": "M",
  "age_months": 18.5,

  "measurement": {
    "predicted_height_cm": 80.2,
    "predicted_weight_kg": 10.5,
    "manual_height_cm": null,
    "manual_weight_kg": null,
    "reference_object_detected": false,
    "scale_factor": 0.142,
    "confidence_score": 0.82,
    "annotated_image": "annotated_abc123.jpg",
    "estimation_method": "who_statistical",
    "body_build": "average",
    "side_view_used": false,
    "chest_depth_cm": null,
    "abd_depth_cm": null
  },

  "nutrition": {
    "haz_zscore": -1.2,
    "whz_zscore": -0.8,
    "haz_status": "Normal",
    "whz_status": "Normal",
    "age_months": 18.5
  },

  "ml_prediction": {
    "estimated_weight_kg": 10.3,
    "sam_probability": 0.02,
    "mam_probability": 0.08,
    "normal_probability": 0.85,
    "risk_probability": 0.03,
    "overweight_probability": 0.02,
    "wasting_status": "Normal",
    "wasting_method": "ml_classifier"
  },

  "muac": {
    "muac_cm": 14.8,
    "muac_status": "Normal",
    "muac_method": "estimated_from_whz",
    "age_in_range": true
  },

  "summary": "Normal nutritional status. Height and weight within expected ranges."
}
```

### Estimation Method Values

| Value | Meaning |
|---|---|
| `"who_statistical"` | Height from WHO median for age/sex |
| `"reference_object"` | Height from detected reference card |
| `"manual"` | Height provided by user |
| `"none"` | No height could be determined |

### Wasting Status Values

| Value | Meaning |
|---|---|
| `"SAM"` | Severe Acute Malnutrition (urgent referral) |
| `"MAM"` | Moderate Acute Malnutrition |
| `"Normal"` | Normal nutritional status |
| `"Risk_Overweight"` | Possible risk of overweight |
| `"Overweight"` | Overweight |

---

## Database Schema (Local)

### `children` Table

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `name` | TEXT(100) | NOT NULL |
| `date_of_birth` | TEXT | NOT NULL, ISO date |
| `sex` | TEXT(1) | NOT NULL, "M" or "F" |
| `guardian_name` | TEXT(100) | NULLABLE |
| `location` | TEXT(200) | NULLABLE |
| `created_at` | TEXT | ISO datetime, default now |
| `updated_at` | TEXT | ISO datetime, default now |

### `visits` Table

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `child_id` | INTEGER | FK → children.id, NOT NULL |
| `local_uuid` | TEXT | UNIQUE, for sync deduplication |
| `visit_date` | TEXT | ISO datetime, default now |
| `age_months` | REAL | NOT NULL |
| `image_path` | TEXT(500) | Local file path |
| `notes` | TEXT | NULLABLE |
| `synced_at` | TEXT | NULL = pending sync |
| `device_id` | TEXT | Device identifier |

### `measurement_results` Table

| Column | Type | Constraints |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `visit_id` | INTEGER | FK → visits.id, UNIQUE, NOT NULL |
| `predicted_height_cm` | REAL | NULLABLE |
| `predicted_weight_kg` | REAL | NULLABLE |
| `manual_height_cm` | REAL | NULLABLE |
| `manual_weight_kg` | REAL | NULLABLE |
| `reference_object_detected` | TEXT(10) | Default "false" |
| `scale_factor` | REAL | NULLABLE |
| `haz_zscore` | REAL | NULLABLE |
| `whz_zscore` | REAL | NULLABLE |
| `haz_status` | TEXT(50) | NULLABLE |
| `whz_status` | TEXT(50) | NULLABLE |
| `confidence_score` | REAL | NULLABLE |
| `muac_cm` | REAL | NULLABLE |
| `muac_status` | TEXT(50) | NULLABLE |
| `muac_method` | TEXT(50) | NULLABLE |
| `ml_wasting_status` | TEXT(50) | NULLABLE |
| `ml_weight_kg` | REAL | NULLABLE |
| `created_at` | TEXT | ISO datetime, default now |

---

## Constants & Thresholds

### Height Validation Ranges (by age)

| Age Range (months) | Min Height (cm) | Max Height (cm) |
|---|---|---|
| 0–6 | 45 | 75 |
| 6–12 | 60 | 85 |
| 12–24 | 70 | 95 |
| 24–36 | 80 | 105 |
| 36–48 | 85 | 115 |
| 48–60 | 95 | 125 |

### Body Segment Ratios (Expected, by age)

| Age Range | Head % | Torso % | Legs % |
|---|---|---|---|
| 0–12 months | 28% | 32% | 40% |
| 12–24 months | 25% | 32% | 43% |
| 24–48 months | 22% | 30% | 48% |
| 48–60 months | 20% | 30% | 50% |

### ML Weight Validation

Accept ML-predicted weight only if:
```
0.45 × who_median_weight ≤ predicted_weight ≤ 1.80 × who_median_weight
```

Otherwise fall back to WHO median × body_build_adjustment.

### Reference Object (Optional Detection)

| Property | Value |
|---|---|
| Type | Yellow rectangular packet |
| Length | 12.7 cm |
| Width | 5.5 cm |

---

## Recommended Flutter Packages

| Package | Purpose | Notes |
|---|---|---|
| `google_mlkit_pose_detection` | MediaPipe pose detection | Native Android/iOS, GPU accelerated |
| `tflite_flutter` | TFLite model inference | Runs weight estimator + classifier |
| `camera` | Camera capture | Fine-grained control for photo quality |
| `sqflite` | Local SQLite | Offline data persistence |
| `dio` | HTTP client | For API sync with retry logic |
| `connectivity_plus` | Network detection | Trigger sync when online |
| `uuid` | UUID generation | Local visit IDs for deduplication |
| `path_provider` | File system paths | Store images locally |
| `image` | Image manipulation | Resize before inference if needed |
| `fl_chart` | Growth charts | Plot Z-score trends over time |

### Alternative: Native Android (Kotlin)

If targeting Android only:

| Library | Purpose |
|---|---|
| `com.google.mediapipe:tasks-vision` | MediaPipe PoseLandmarker |
| `org.tensorflow:tensorflow-lite` | TFLite runtime |
| `org.tensorflow:tensorflow-lite-gpu` | GPU delegate |
| `androidx.camera:camera-camera2` | CameraX for capture |
| `androidx.room:room-runtime` | SQLite ORM |
| `com.squareup.retrofit2:retrofit` | HTTP for sync |

---

## App Bundle Size Budget

| Component | Size |
|---|---|
| MediaPipe pose model | 29.2 MB |
| Weight estimator TFLite | 8.3 KB |
| Wasting classifier TFLite | 18.5 KB |
| WHO data (JSON) | ~500 KB |
| MUAC lookup table | ~1 KB (hardcoded) |
| Scaler constants | ~1 KB (hardcoded) |
| **Total ML/data assets** | **~30 MB** |
| Flutter framework | ~5 MB |
| App code + UI | ~2 MB |
| **Estimated total APK** | **~37 MB** |

---

## Sync Strategy

### Offline-First Design

1. All assessments are saved locally immediately
2. Each visit gets a `local_uuid` (UUID v4)
3. When connectivity is detected, sync pending visits:
   - Upload image to server
   - POST assessment data to `/api/v1/sync`
   - On success, set `synced_at = now()`
4. Server deduplicates on `local_uuid`
5. Conflict resolution: last-write-wins (field workers rarely edit)

### Sync Payload (Proposed)

```json
{
  "device_id": "device-uuid",
  "visits": [
    {
      "local_uuid": "visit-uuid",
      "child_name": "Aisha",
      "date_of_birth": "2023-06-15",
      "sex": "F",
      "visit_date": "2025-03-23T10:30:00",
      "age_months": 21.3,
      "measurement": { ... },
      "nutrition": { ... },
      "ml_prediction": { ... },
      "muac": { ... },
      "image_base64": "..."
    }
  ]
}
```

---

## Current Model Performance

For reference when evaluating on-device accuracy:

| Metric | Value | Notes |
|---|---|---|
| Validation accuracy (5-class) | 0.702 | On synthetic data |
| SAM recall | 0.886 | Key safety metric (≥ 0.80 target) |
| Weight MAE | 0.403 kg | On synthetic data |
| Height estimation error | ~7% | Relative to actual height |
| Feature count | 14 | 10 frontal + 4 AP depth |
| Training data | 60,000 synthetic | Generated from WHO LMS parameters |

---

## Notes for Developers

1. **The scaler mean/std and label mapping are hardcoded constants** — do NOT
   ship the `.pkl` files. They are Python-specific pickle format.

2. **WHO Excel files must be converted to JSON** before bundling. Use the
   conversion scripts in `scripts/`.

3. **MediaPipe's `pose_landmarker_heavy.task`** works identically on mobile
   and desktop — same file, same landmark indices.

4. **The LMS Z-score formula** requires careful handling of the `L ≈ 0` case.
   Use `abs(L) < 0.001` as the threshold for switching to the logarithmic form.

5. **Image preprocessing**: MediaPipe handles all resizing internally. Pass the
   raw camera frame (RGB format) directly.

6. **GPS coordinates**: Consider capturing GPS with each assessment for mapping
   malnutrition hotspots. The backend schema can be extended to store this.

7. **Multi-language support**: Field workers may speak local languages. Plan for
   i18n from the start — all status strings and UI labels should be localizable.

---

## Optional guided ARCore depth capture (Android research path)

The Android app may offer `arcore_guided_depth_v2` when Google Play Services
for AR reports a supported device, the app has a sufficient Android memory
class, and the native Depth API starts successfully. ARCore is optional: the
app and its existing guided RGB workflow remain available on devices without
ARCore or depth support.

The v2 capture is a bounded, guided multi-view scan. It:

- samples only unique raw-depth frames and pairs each one with ARCore's raw
  depth confidence image;
- rejects pixels below half confidence and accepts data only in ARCore's useful
  0.5-5.0 m depth range;
- uses depth-texture camera intrinsics to unproject samples into world space;
- identifies a tracked upward-facing floor plane and requires both feet to be
  visible near that plane;
- isolates the centered subject by target depth and a bounded 3D radius;
- rejects near-duplicate views, unstable floor estimates, low-confidence
  frames, insufficient points, and apparent subject movement;
- requires a multi-view arc and camera travel before producing a result;
- keeps at most 32 reduced keyframes and stops after 45 seconds; and
- closes every raw depth/confidence image immediately after reducing it to
  summary evidence.

No RGB image, raw depth image, point cloud, or mesh is retained. The result is
stored under `device_metadata.arcore_depth_scan` and queued with the visit sync
payload. It includes the method, experimental height, robust uncertainty,
accepted keyframes, valid-depth fraction, mean depth confidence, scan coverage,
camera travel, floor stability, captured point count, duration, quality score,
and depth mode. It MUST include both `raw_media_retained: false` and
`clinical_measurement_eligible: false`.

The result must not populate effective height, HAZ, WHZ, stunting, wasting,
BMI, or MUAC fields. It is available only when the child can stand safely
without support. Operators always continue the normal front/side guided-photo
workflow after the optional scan.

### Runtime fallback

| Condition | Behavior |
|---|---|
| ARCore available and Depth API starts | Offer guided full depth scan |
| Availability still transient | Use guided RGB capture |
| ARCore unavailable/unsupported | Use guided RGB capture |
| Android memory class below 256 MB | Use guided RGB capture |
| Depth mode unsupported | Cancel depth activity and use guided RGB capture |
| Child cannot stand safely | Skip depth and use guided RGB capture |
| Live quality gate/timeout failure | Discard summary and use guided RGB capture |
| Permission/startup/runtime failure | Show fallback message; guided RGB remains usable |

### Resource ceilings

| Resource | v2 ceiling |
|---|---|
| Accepted keyframes | 32 (20 target, 12 minimum) |
| Sampling interval | 250 ms minimum |
| Scan duration | 45 seconds |
| Sampled depth grid | 6,000 points per attempted keyframe |
| Retained raw RGB/depth | none |
| Retained point cloud/mesh | none |
| Raw depth range | 0.5–5.0 m |
| Minimum raw confidence | 128/255 |
| Candidate body-height range | 0.35–1.45 m above tracked floor |
| Minimum camera travel | 0.25 m |
| Minimum scan coverage | 20 degrees |
| Maximum floor instability | 5 cm |
| Maximum robust height uncertainty | 6 cm |

The AR estimate is research evidence only. Before any clinical eligibility
is considered it requires device-specific calibration and prospective agreement
validation against duplicate conventional anthropometry on real children.
