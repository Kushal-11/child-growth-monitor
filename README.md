# Child Growth Monitor

A WHO-standard child growth assessment system that uses computer vision and pose estimation to measure child height and detect malnutrition (SAM/MAM) — including a camera-based wasting detection model that works without a weighing scale.

## Overview

The system processes a frontal standing photo of a child to:
- Estimate height using MediaPipe pose landmarks + WHO growth reference data
- Classify stunting (Height-for-Age Z-score) against WHO standards
- Detect wasting / acute malnutrition (SAM/MAM) using an ML model trained on WHO-derived data, even when no weight measurement is available
- Track child growth over repeated visits

## Features

| Feature | Details |
|---------|---------|
| Height estimation | Hybrid: WHO statistical + anthropometric ratios — no reference objects needed |
| Stunting classification | Height-for-Age Z-score via WHO HAZ boundaries |
| Wasting detection (SAM/MAM) | ML weight estimator + 5-class classifier from body proportions |
| Weight estimation | ML-estimated from shoulder/hip widths when no scale available |
| WHO data | All thresholds from verified WHO Excel LMS files |
| Web UI | FastAPI + Bootstrap 5 |
| REST API | Full JSON API, Swagger UI at `/docs` |
| Mobile-ready | TFLite models (7 KB + 17 KB) for Android/iOS |
| Data persistence | SQLite — child records + visit history |

## Technology Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn
- **Computer Vision**: MediaPipe PoseLandmarker, OpenCV
- **Machine Learning**: TensorFlow 2.16+ (Keras MLP), scikit-learn
- **Database**: SQLAlchemy + SQLite
- **Frontend**: Bootstrap 5, Jinja2
- **Reference Data**: WHO Child Growth Standards (verified Excel/CSV)

---

## Running the Project

### System Requirements (Server / Laptop)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.12 |
| RAM | 2 GB | 4 GB |
| Disk space | 1.5 GB | 2 GB |
| OS | Linux / macOS / Windows | Ubuntu 22.04+ |
| CPU | Any x86-64 | 4+ cores |
| GPU | Not required | Optional (speeds up training only) |

### Step-by-step Setup

```bash
# 1. Clone the repository
git clone https://github.com/Kushal-11/child-growth-monitor.git
cd child-growth-monitor

# 2. Create a virtual environment (required — keeps dependencies isolated)
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows (Command Prompt)
.venv\Scripts\Activate.ps1         # Windows (PowerShell)

# 4. Install all dependencies (~5 min, downloads ~1.5 GB)
pip install -r requirements.txt

# 5. Train the wasting detection models (first run only — ~2 min on CPU)
PYTHONPATH=. .venv/bin/python ml/train.py

# 6. Start the server
PYTHONPATH=. .venv/bin/python main.py
```

> The server starts at **http://localhost:8000**
> API documentation: **http://localhost:8000/docs**

> **Step 5 only needs to be run once.** It generates 60K synthetic training samples from WHO data and trains both models. Output is saved to `data/models/` and is not committed to git.

### Verifying the Installation

```bash
# Run all tests (should show 62 passed)
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v

# Check model performance (SAM recall target ≥ 0.80)
PYTHONPATH=. .venv/bin/python ml/evaluate.py
```

### Restarting the Server

Every subsequent run only needs:
```bash
source .venv/bin/activate
PYTHONPATH=. .venv/bin/python main.py
```

---

## Android Deployment Guide

There are two deployment options for Android. Choose based on your use case:

| Option | Latency | Internet needed | Complexity | Best for |
|--------|---------|-----------------|------------|----------|
| **A. Web app** (phone browser → server) | ~2–5 s | Yes (WiFi/4G) | Low | Clinic with internet |
| **B. On-device TFLite** (native Android app) | < 0.5 s | No | High | Field / offline use |

---

### Option A — Use the Web App on Android (no app install needed)

The FastAPI server already serves a mobile-responsive web UI. You just need the phone and the server on the same network.

**On the server machine:**

```bash
# Find your local IP address
ip a | grep "inet " | grep -v 127        # Linux
ipconfig                                  # Windows

# Start the server accessible from other devices on the network
PYTHONPATH=. .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**On the Android phone:**

1. Connect to the **same WiFi network** as the server
2. Open Chrome and go to `http://<server-ip>:8000`
   Example: `http://192.168.1.5:8000`
3. For **camera access**, tap the file input → select camera → take photo directly
4. Optional: tap **⋮ → Add to Home screen** in Chrome to create an app icon

**Android requirements for Option A:**
- Any Android 6.0+ phone
- Chrome 80+ (or any modern browser)
- WiFi / mobile data to reach the server

---

### Option B — Native Android App (Offline / TFLite)

The two trained models are already exported as TFLite:
- `data/models/weight_estimator.tflite` — 8 KB
- `data/models/wasting_classifier.tflite` — 20 KB

The Android app would run MediaPipe pose detection on-device and feed features into these TFLite models — **no internet or server required**.

#### Recommended Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Pose detection | [MediaPipe Android SDK](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/android) | Same model as server (30 MB .task file) |
| TFLite inference | `org.tensorflow:tensorflow-lite:2.14.0` | Weight estimator + classifier |
| Camera | CameraX (`androidx.camera`) | Best for Android 5.0+ |
| UI | Kotlin + Jetpack Compose | Or Java + XML layouts |
| Pre-processing | Port `ml/inference.py` to Kotlin/Java | StandardScaler must be ported |

#### Minimum Android Specs

| Spec | Minimum | Recommended |
|------|---------|-------------|
| Android version | 8.0 (API 26) | 10.0+ (API 29+) |
| RAM | 2 GB | 3 GB+ |
| Storage | 150 MB free | 300 MB free |
| Camera | 5 MP, autofocus | 8 MP+ |
| CPU | Any ARMv8 (64-bit) | Snapdragon 660+ or equivalent |
| NPU/DSP | Not required | Speeds up TFLite inference |

> The heaviest component is MediaPipe's pose model (30 MB). TFLite ML models are tiny (28 KB total). Total app size would be ~80–120 MB installed.

#### Key Integration Steps

**1. Add dependencies to `build.gradle`:**
```gradle
dependencies {
    implementation 'com.google.mediapipe:tasks-vision:0.10.7'
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'androidx.camera:camera-view:1.3.0'
}
```

**2. Copy model files into `assets/`:**
```
app/src/main/assets/
├── pose_landmarker_heavy.task      # 30 MB — copy from data/
├── weight_estimator.tflite         # 8 KB  — copy from data/models/
├── wasting_classifier.tflite       # 20 KB — copy from data/models/
└── feature_scaler_params.json      # export scaler mean/std (see below)
```

**3. Export the StandardScaler parameters** (run once on the server):
```bash
PYTHONPATH=. .venv/bin/python -c "
import pickle, json
with open('data/models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
params = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
          'feature_names': ['age_months','sex_binary','height_cm',
                            'shoulder_width_cm','hip_width_cm','torso_length_cm',
                            'upper_arm_length_cm','shoulder_height_ratio',
                            'hip_height_ratio','body_build_score']}
print(json.dumps(params, indent=2))
" > android/app/src/main/assets/feature_scaler_params.json
```

**4. Inference flow in Kotlin (pseudocode):**
```kotlin
// 1. Run MediaPipe on camera frame → get 33 landmarks
val landmarks = poseLandmarker.detect(bitmap)

// 2. Extract the 10 features (same logic as ml_service.py)
val features = extractFeatures(landmarks, ageMonths, sex, heightCm)

// 3. Normalise using scaler params from JSON
val scaledFeatures = (features - scalerMean) / scalerScale

// 4. Run weight estimator TFLite
val estimatedWeight = runWeightEstimator(scaledFeatures)  // → Float

// 5. Run wasting classifier TFLite
val probabilities = runWastingClassifier(scaledFeatures)  // → FloatArray(5)
// Classes in order: MAM, Normal, Overweight, Risk_Overweight, SAM
val wastingLabel = WASTING_LABELS[probabilities.indexOfMax()]

// 6. Display result
```

**5. Label order for the classifier output** (must match `label_encoder.pkl`):
```kotlin
val WASTING_LABELS = arrayOf("MAM", "Normal", "Overweight", "Risk_Overweight", "SAM")
// Index 0 = MAM, 1 = Normal, 2 = Overweight, 3 = Risk_Overweight, 4 = SAM
```

#### Testing TFLite Models Before Building the App

Verify the TFLite models produce correct output on your machine:
```bash
PYTHONPATH=. .venv/bin/python -c "
import numpy as np
import tensorflow as tf

# Load TFLite weight estimator
interp = tf.lite.Interpreter('data/models/weight_estimator.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
print('Input shape:', inp[0]['shape'])    # expect (1, 10)
print('Output shape:', out[0]['shape'])   # expect (1, 1)

# Load TFLite classifier
interp2 = tf.lite.Interpreter('data/models/wasting_classifier.tflite')
interp2.allocate_tensors()
inp2 = interp2.get_input_details()
out2 = interp2.get_output_details()
print('Classifier input shape:', inp2[0]['shape'])   # expect (1, 10)
print('Classifier output shape:', out2[0]['shape'])  # expect (1, 5)
"
```

---

### HTTPS for Remote Access (optional)

To access the web app from outside your local network (or to use HTTPS, which is needed for camera access on some Android browsers):

```bash
# Quick option: expose with ngrok (for testing only)
ngrok http 8000
# → gives you a public https://xxxx.ngrok.io URL

# Production option: run behind nginx with a Let's Encrypt certificate
# See: https://fastapi.tiangolo.com/deployment/https/
```

> Chrome on Android requires HTTPS to access the camera from non-localhost URLs. If you host the server on a remote machine, you need HTTPS or use the `--host 0.0.0.0` option on a local network.

---

## Usage

### Web Interface

1. Open `http://localhost:8000`
2. **Upload a photo** — full body, standing upright, head to toe visible (see Photo Guidelines below)
3. **Child Information** (required):
   - Child name and sex
   - **Age in months** (e.g. `24`) — the default input. Or click "Or enter exact date of birth" to use a date picker instead.
4. **Optional Measurements** (improve accuracy when available):
   - **Weight (kg)** — from a weighing scale; if omitted, the ML model estimates it from the photo
   - **MUAC (cm)** — from a physical MUAC tape; if omitted, MUAC is estimated from WHZ
   - **Height** — manual fallback if image detection fails (accepts cm or inches)
   - Guardian name and location/clinic for record-keeping
5. Click **Run Assessment**

**Results page shows:**
- A coloured **status banner**: SAM (red) / MAM (orange) / Normal (green)
- **Three metric cards**: Height (+ stunting/HAZ), Weight (+ wasting/WHZ), MUAC (+ status)
- **ML confidence bars**: SAM, MAM, Normal probability from the camera-based model
- **Pose-annotated photo** (with tab to toggle to original)

### Processing Videos

Extract the best standing frame from a walking video before uploading:

```bash
PYTHONPATH=. .venv/bin/python scripts/extract_best_frame.py footage/child.mp4 photos/child.jpg

# Batch a whole folder:
for f in footage/*.mp4; do
    python scripts/extract_best_frame.py "$f" "photos/$(basename ${f%.mp4}).jpg"
done
```

### Batch Assessment

To process multiple photos and compare against known measurements:

```bash
# 1. Generate a blank ground-truth template
PYTHONPATH=. .venv/bin/python scripts/batch_assess.py --template
# → writes data/ground_truth_template.csv

# 2. Fill in your measurements (child name, DOB, sex, height, weight per photo)
# Save as data/ground_truth.csv

# 3. Run batch assessment
PYTHONPATH=. .venv/bin/python scripts/batch_assess.py \
    --images /path/to/photos/ \
    --ground-truth data/ground_truth.csv \
    --output data/batch_results.csv
```

Output CSV includes: predicted vs actual height/weight, Z-scores, wasting status, and pose features ready for model fine-tuning.

### Photo Guidelines

| Requirement | Details |
|-------------|---------|
| **Pose** | Standing upright, facing directly at the camera |
| **Coverage** | Full body in frame — top of head to soles of feet |
| **Distance** | 1–2 metres from camera |
| **Background** | Plain, contrasting background preferred |
| **Lighting** | Even lighting, avoid harsh shadows |
| **Arms** | Slightly away from body (so hip landmarks are visible) |
| **Clothing** | Fitted clothing gives better body proportion measurements |

### API

#### POST /api/v1/assess

Multipart form submission.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | ✓ | JPEG/PNG photo |
| `child_name` | string | ✓ | Child's name |
| `date_of_birth` | date | ✓ | YYYY-MM-DD |
| `sex` | string | ✓ | `M` or `F` |
| `weight_kg` | float | — | Weight in kg (ML-estimated if omitted) |
| `height_cm` | float | — | Height in cm (fallback if image detection fails) |
| `muac_cm` | float | — | MUAC tape measurement in cm (estimated from WHZ if omitted) |
| `guardian_name` | string | — | Guardian / caregiver name |
| `location` | string | — | Village / clinic name |

**Response:**
```json
{
  "child_name": "Priya",
  "sex": "F",
  "age_months": 30.2,
  "measurement": {
    "predicted_height_cm": 88.4,
    "predicted_weight_kg": 10.1,
    "confidence_score": 0.84,
    "estimation_method": "who_statistical",
    "body_build": "slender",
    "annotated_image": "uuid_priya_annotated.jpg"
  },
  "nutrition": {
    "haz_zscore": -1.2,
    "whz_zscore": -2.4,
    "haz_status": "Normal",
    "whz_status": "Moderate Acute Malnutrition (MAM)",
    "age_months": 30.2
  },
  "ml_prediction": {
    "estimated_weight_kg": 10.1,
    "sam_probability": 0.04,
    "mam_probability": 0.61,
    "normal_probability": 0.28,
    "risk_probability": 0.05,
    "overweight_probability": 0.02,
    "wasting_status": "MAM",
    "wasting_method": "ml_classifier"
  },
  "muac": {
    "muac_cm": 13.2,
    "muac_status": "At Risk (MAM)",
    "muac_method": "estimated_from_whz",
    "age_in_range": true
  },
  "summary": "..."
}
```

#### Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/children` | List all children |
| `GET` | `/api/v1/children/{id}` | Child detail + visit history |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

---

## How It Works

### Height Estimation (no scale, no reference object)

1. **MediaPipe PoseLandmarker** detects 33 body landmarks (nose, shoulders, hips, knees, ankles, etc.)
2. **Pixel measurements** are taken: head-top to heel (total height), shoulder width, hip width, torso length, upper arm length
3. **Scale factor** (cm/pixel) is computed using WHO median height for age/sex as the reference
4. **Validation** checks that the estimated height falls within ±3 SD of WHO norms for the age

### Wasting Detection (SAM/MAM)

**The problem**: WHZ (Weight-for-Height Z-score) requires weight. Without a scale, weight must be estimated from visual cues.

**The solution**: A two-model ML pipeline trained on WHO-derived data:

1. **Weight Estimator** (MLP regression) — predicts weight (kg) from 10 body proportion features; val MAE ≈ 0.54 kg
2. **Wasting Classifier** (5-class MLP) — directly classifies SAM / MAM / Normal / Risk / Overweight from pose-derived measurements; SAM recall ≥ 0.80

**Features used** (all derived from pose landmarks):

| Feature | Source |
|---------|--------|
| `age_months`, `sex` | Form input |
| `height_cm` | Pose estimation (existing) |
| `shoulder_width_cm` | LEFT_SHOULDER ↔ RIGHT_SHOULDER horizontal distance × scale |
| `hip_width_cm` | LEFT_HIP ↔ RIGHT_HIP horizontal distance × scale |
| `torso_length_cm` | Shoulder midpoint ↔ hip midpoint × scale |
| `upper_arm_length_cm` | Shoulder → elbow Euclidean distance × scale |
| `shoulder_height_ratio` | shoulder_width / height |
| `hip_height_ratio` | hip_width / height |
| `body_build_score` | –1 slender / 0 average / +1 stocky |

**Weight priority chain** (when manual weight is not entered):
```
ML-estimated weight  →  WHO median (with body build adjustment)
```

### WHO Data Provenance

All classification thresholds and training labels are derived from official WHO publications:

| File | Status | Use |
|------|--------|-----|
| `who_haz_0_59m.csv` | ✅ Verified | Height-for-Age Z-score boundaries 0–60 months |
| `wfl_*_zscores.xlsx` | ✅ Verified | WHO WFL LMS parameters (0–2 years) |
| `wfh_*_zscores.xlsx` | ✅ Verified | WHO WFH LMS parameters (2–5 years) |

The **body width proportion baseline** (shoulder/hip ratios used to generate synthetic training data) comes from Snyder RG et al. (1975) *Anthropometry of Infants, Children, and Youths to Age 18 for Product Safety Design* — this is **not** from WHO standards and is explicitly labeled as a physical approximation in the code. Model accuracy will improve with real labeled data.

### WHO Clinical Criteria — Malnutrition Diagnosis

This section documents the official WHO diagnostic thresholds and formulas used throughout the system.

#### Z-score Calculation — LMS (Box-Cox) Method

All anthropometric z-scores (HAZ, WHZ, WAZ) use the **WHO LMS method** (Cole & Green, 1992).

**General formula:**

$$
Z = \begin{cases}
\dfrac{\left(\dfrac{X}{M}\right)^{\!L} - 1}{L \cdot S} & \text{if } L \neq 0 \\[12pt]
\dfrac{\ln(X/M)}{S} & \text{if } L = 0
\end{cases}
$$

**Parameters:**

| Symbol | Meaning |
|--------|---------|
| $X$ | Observed measurement (height in cm, or weight in kg) |
| $L$ | Box-Cox power — corrects for skewness of the reference distribution |
| $M$ | Median reference value for the child's age, sex (and height for WHZ) |
| $S$ | Coefficient of variation of the reference population |

The $L$, $M$, $S$ parameters come from the **WHO 2006 Child Growth Standards** Excel tables:
- `data/wfl/wfl_boys_zscores.xlsx` / `wfl_girls_zscores.xlsx` — weight-for-length (< 2 years)
- `data/wfh/wfh_boys_zscores.xlsx` / `wfh_girls_zscores.xlsx` — weight-for-height (≥ 2 years)
- `data/who_haz_0_59m.csv` — pre-computed HAZ boundary values at $Z \in \{-3, -2, -1, 0, +1, +2, +3\}$

**Extreme-value correction** — for $|Z| > 3$, WHO recommends linear extrapolation to avoid over-compression:

$$
Z_{\text{adj}} = \begin{cases}
-3 + \dfrac{X - \text{SD3neg}}{\text{SD3neg} - \text{SD2neg}} & \text{if } Z < -3 \\[10pt]
+3 + \dfrac{X - \text{SD3pos}}{\text{SD4pos} - \text{SD3pos}} & \text{if } Z > +3
\end{cases}
$$

where $\text{SD}k\text{neg}$ and $\text{SD}k\text{pos}$ are the reference values at $-k$ and $+k$ SD for the child's age/sex. Applied in `nutrition_service.py`.

---

#### Acute Malnutrition — Wasting Criteria

**WHO / UNICEF / WFP / IASC 2023** define acute malnutrition by **any one** of the following independent indicators:

| Indicator | SAM threshold | MAM threshold | Age range |
|-----------|--------------|---------------|-----------|
| **WHZ** (Weight-for-Height Z-score) | $Z < -3$ | $-3 \leq Z < -2$ | 0–59 months |
| **MUAC** (Mid-Upper Arm Circumference) | $< 11.5\,\text{cm}$ | $11.5\text{–}12.5\,\text{cm}$ | 6–59 months |
| **Bilateral pitting oedema** | Present (any grade) | — | Any age |
| **WAZ** (Weight-for-Age Z-score) | $Z < -3$ | $-3 \leq Z < -2$ | 0–59 months (if height unavailable) |

> Each indicator is sufficient on its own for a SAM/MAM diagnosis. MUAC and WHZ identify overlapping but **not identical** populations — using both improves case detection.

**WHZ classification in detail:**

| WHZ Z-score | Status |
|-------------|--------|
| $Z < -3$ | **Severe Acute Malnutrition (SAM)** |
| $-3 \leq Z < -2$ | **Moderate Acute Malnutrition (MAM)** |
| $-2 \leq Z \leq +1$ | Normal |
| $+1 < Z \leq +2$ | Possible risk of overweight |
| $+2 < Z \leq +3$ | Overweight |
| $Z > +3$ | Obese |

**MUAC classification (6–59 months, absolute — not age-adjusted):**

| MUAC | Status |
|------|--------|
| $< 11.5\,\text{cm}$ | **Severe Acute Malnutrition (SAM)** |
| $11.5 \leq \text{MUAC} < 12.5\,\text{cm}$ | **At Risk / Moderate Acute Malnutrition (MAM)** |
| $\geq 12.5\,\text{cm}$ | Normal |

> MUAC thresholds are **absolute** (same for all ages 6–59 months). They are **not z-score based**. Source: WHO/UNICEF 2009 Joint Statement on MUAC.

---

#### Chronic Malnutrition — Stunting Criteria

Stunting reflects **cumulative long-term** nutritional deficiency, assessed via **Height-for-Age Z-score (HAZ)**:

| HAZ Z-score | Status |
|-------------|--------|
| $Z < -3$ | **Severely Stunted** |
| $-3 \leq Z < -2$ | **Stunted** |
| $-2 \leq Z \leq +3$ | Normal |
| $Z > +3$ | Tall (possible measurement error — verify) |

> Stunting threshold: $\text{HAZ} < -2\,\text{SD}$. Source: WHO 2006 Child Growth Standards.

---

#### Underweight — Weight-for-Age Criteria

| WAZ Z-score | Status |
|-------------|--------|
| $Z < -3$ | Severely Underweight |
| $-3 \leq Z < -2$ | Underweight |
| $-2 \leq Z \leq +2$ | Normal |
| $Z > +2$ | Possible overweight |

> Note: WAZ alone cannot distinguish wasting from stunting. WHO recommends WHZ as the primary wasting indicator.

---

#### MUAC Estimation Formula (when no tape is available)

When a physical MUAC tape measurement is not provided, MUAC is **estimated** using WHO MUAC-for-age medians and the child's WHZ:

$$
\widehat{\text{MUAC}} = \bar{\mu}(\text{age},\,\text{sex}) \;\times\; \bigl[1 + 0.087 \times \text{clamp}(\text{WHZ},\,-3,\,+3)\bigr]
$$

$$
\text{clamp}(x, a, b) = \max\!\bigl(a,\;\min(b,\,x)\bigr)
$$

**Parameters:**

| Symbol | Meaning |
|--------|---------|
| $\widehat{\text{MUAC}}$ | Estimated MUAC in cm |
| $\bar{\mu}(\text{age},\,\text{sex})$ | WHO 2006 MUAC-for-age median ($M$ column, $L=0$), linearly interpolated |
| $0.087$ | Calibration coefficient: at $\text{WHZ} = -3$, a 24-month boy maps to $\approx 11.6\,\text{cm}$ |
| $\text{clamp}$ | Prevents extrapolation beyond the SAM/obese extremes |

**Calibration check:**

$$
\bar{\mu}(24\,\text{mo},\,\text{M}) = 15.7\,\text{cm} \implies \widehat{\text{MUAC}}\big|_{\text{WHZ}=-3} = 15.7 \times (1 - 0.261) \approx 11.6\,\text{cm}
$$

This estimate is clearly labeled as "Estimated" in the UI. **Always confirm SAM/MAM with a physical tape measurement before clinical action.**

---

#### Summary of Indicators Used by This System

| Indicator | Source in this system | WHO Threshold for SAM |
|-----------|-----------------------|-----------------------|
| HAZ (stunting) | WHO LMS tables via `nutrition_service.py` | $Z < -3$ |
| WHZ (wasting) | WHO LMS tables via `nutrition_service.py` | $Z < -3$ |
| MUAC | Manual tape (preferred) or WHZ-estimated | $< 11.5\,\text{cm}$ |
| ML wasting classifier | Trained on WHZ labels from WHO LMS data | SAM recall $\geq 0.80$ |

> **Clinical decision rule**: For treatment referral, WHO recommends acting on **either** $\text{WHZ} < -3$ **or** $\text{MUAC} < 11.5\,\text{cm}$ — whichever flags the child first.

**References:**
- WHO 2006 Child Growth Standards — https://www.who.int/tools/child-growth-standards
- WHO/UNICEF 2009 Joint Statement on MUAC — https://www.who.int/publications/i/item/9789241598163
- WHO 2013 Updates on Management of SAM — https://www.who.int/publications/i/item/9789241506328
- UNICEF/WHO/World Bank 2023 Joint Malnutrition Estimates

---

## ML Model Training

### Train from scratch

```bash
# Regenerate the synthetic dataset (uses WHO Excel LMS files)
PYTHONPATH=. python ml/generate_synthetic_data.py

# Train and export models
PYTHONPATH=. python ml/train.py

# Evaluate (check SAM recall ≥ 0.80)
PYTHONPATH=. python ml/evaluate.py
```

### Fine-tune with real data

When you collect real labeled measurements (age, sex, height, weight, confirmed wasting status):

1. Run batch assessment to extract pose features alongside your ground-truth measurements:
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/batch_assess.py \
       --images /path/to/photos/ \
       --ground-truth data/ground_truth.csv \
       --output data/batch_results.csv
   ```

2. Open the fine-tuning notebook:
   ```bash
   .venv/bin/jupyter notebook notebooks/finetune_with_real_data.ipynb
   ```
   The notebook mixes your real samples (repeated 20× by default) with the 60K synthetic dataset and retrains both models. It shows a per-class confusion matrix and SAM recall before saving updated `.keras` and `.tflite` files.

Even 100–200 real labeled samples will significantly improve accuracy over the synthetic baseline.

### Model files

| File | Size | Use |
|------|------|-----|
| `data/models/weight_estimator.keras` | ~200 KB | Server inference |
| `data/models/wasting_classifier.keras` | ~500 KB | Server inference |
| `data/models/weight_estimator.tflite` | **7 KB** | Android/iOS on-device |
| `data/models/wasting_classifier.tflite` | **17 KB** | Android/iOS on-device |
| `data/models/feature_scaler.pkl` | <1 KB | Preprocessing (required) |
| `data/models/label_encoder.pkl` | <1 KB | Class name mapping |

---

## Project Structure

```
child-growth-monitor/
├── app/
│   ├── api/routes.py                 # REST API endpoints
│   ├── models/                       # SQLAlchemy ORM models
│   ├── schemas/assessment.py         # Pydantic request/response schemas
│   ├── services/
│   │   ├── assessment_service.py     # Main orchestrator
│   │   ├── measurement_service.py    # Pose + height estimation
│   │   ├── nutrition_service.py      # Z-score computation
│   │   ├── who_data_service.py       # WHO reference data loader
│   │   ├── ml_service.py            # ML wasting detection service
│   │   └── muac_service.py          # MUAC estimation (WHO medians + WHZ)
│   └── web/                          # Jinja2 templates + static assets
├── ml/
│   ├── generate_synthetic_data.py    # WHO-derived training data generator
│   ├── models.py                     # Keras model architectures
│   ├── train.py                      # Training + TFLite export
│   ├── evaluate.py                   # Metrics + SAM recall check
│   └── inference.py                  # Runtime predictor (singleton)
├── scripts/
│   ├── fix_who_data.py               # Regenerates CSV files from Excel LMS
│   ├── extract_best_frame.py         # Extract best upright frame from a video
│   └── batch_assess.py               # Batch assess photos + ground-truth comparison
├── data/
│   ├── who_haz_0_59m.csv             # Verified WHO HAZ data
│   ├── wfl_*/wfh_*_zscores.xlsx      # Verified WHO LMS data (authoritative)
│   ├── models/                       # Trained ML models
│   └── training_data/                # Generated synthetic dataset
├── tests/                            # 62 unit tests (100% passing)
├── notebooks/                        # Fine-tuning notebook
├── config.py                         # Centralised configuration
├── main.py                           # Application entry point
└── requirements.txt
```

## Testing

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

All 62 tests pass, covering WHO data loading, Z-score computation, height estimation, and API endpoints.

---

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `POSE_MODEL_PATH` | `data/pose_landmarker_heavy.task` | MediaPipe model |
| `UPLOAD_DIR` | `uploads/` | Image storage |
| `ML_MODELS_DIR` | `data/models/` | Trained ML model directory |
| `ANTHROPOMETRIC_RATIOS` | Age-specific | Head/torso/leg ratios for height estimation |
| `HEIGHT_VALIDATION_SD` | 3.0 | Reject estimates > 3 SD from WHO median |
| `SEGMENT_AGREEMENT_THRESHOLD` | 0.15 | Max 15% disagreement between segment estimates |

---

## Limitations

- **Camera-only accuracy**: Without a weighing scale, weight is estimated from body widths. The ML model is trained on synthetic data — real-world accuracy improves with labeled clinical data (see Fine-tuning section).
- **MUAC is estimated, not measured**: When no tape measurement is provided, MUAC is estimated from WHZ using WHO MUAC-for-age medians. This is clearly labeled in the UI. Always confirm SAM/MAM with a physical tape measurement for clinical decisions.
- **2D pose only**: MediaPipe gives 2D landmarks from a single camera. Body widths assume a frontal view with no perspective distortion.
- **Age range**: 0–59 months (per WHO reference data coverage). MUAC classification only applies to 6–59 months.
- **Pose requirements**: Child must be standing upright, full body visible, frontal view, 1–2 m from camera.

---

## References

- WHO Child Growth Standards: https://www.who.int/tools/child-growth-standards
- WHO Weight-for-Height LMS tables: https://www.who.int/tools/child-growth-standards/standards/weight-for-height
- WHO MUAC-for-Age standards: https://www.who.int/tools/child-growth-standards/standards/arm-circumference-for-age
- MediaPipe Pose Landmarker: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- Snyder RG et al. (1975) *Anthropometry of Infants, Children, and Youths to Age 18* — body proportion baseline used in ML synthetic data

## Flutter Mobile Scaffold (`flutter_app/`)

A starter Flutter client has been added in `flutter_app/` and wired to existing FastAPI endpoints:

- `GET /api/v1/health` for backend connectivity checks
- `GET /api/v1/children` to list registered children
- `GET /api/v1/children/{id}` for child-level visit history
- `POST /api/v1/assess` (multipart) for submitting front/side/back image assessments

Flutter app capabilities currently include:
- Camera/gallery image selection via `image_picker`
- Full assessment payload support (`height_value`, `height_unit`, `muac_cm`, optional side/back images)
- Persisted API base URL using `shared_preferences`
- Child list + child detail timeline view

### Run the Flutter app

```bash
cd flutter_app
flutter pub get
flutter run
```

Default API base URL in the app is `http://10.0.2.2:8000` (Android emulator mapping to localhost).
Change it in-app when targeting a physical device/server.

### Build a Downloadable Android APK

You can now build a downloadable Android APK either locally or via GitHub Actions.

#### Option 1: Local build

```bash
cd flutter_app
./scripts/build_android_release.sh
```

The release APK will be generated at:

```text
flutter_app/build/app/outputs/flutter-apk/app-release.apk
```

You can install it on an Android phone using:

```bash
adb install -r flutter_app/build/app/outputs/flutter-apk/app-release.apk
```

#### Option 2: GitHub Actions (recommended for sharing APK)

A workflow has been added at `.github/workflows/flutter-android-apk.yml`.

- Trigger the workflow from **Actions → Build Android APK → Run workflow**
- Download the artifact named `child-growth-monitor-release-apk`
- Share that APK file with field devices

To set production backend URL for release builds, set repository variable:

- `API_BASE_URL` (for example, `https://api.yourdomain.com`)
