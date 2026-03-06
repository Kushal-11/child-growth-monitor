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

## Installation

### Prerequisites

- Python 3.10 or higher
- ~1.5 GB disk space (MediaPipe model + TensorFlow)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Kushal-11/child-growth-monitor.git
cd child-growth-monitor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. (First run only) Train the wasting detection ML model
PYTHONPATH=. python ml/train.py

# 5. Start the server
PYTHONPATH=. python main.py
```

The server will start at `http://localhost:8000`.

> **Note**: Step 4 trains the ML models on a synthetic WHO-based dataset (60K samples). This takes 1–3 minutes on CPU and only needs to be run once. Trained models are saved to `data/models/`.

---

## Usage

### Web Interface

1. Open `http://localhost:8000`
2. Upload a full-body photo of the child standing upright against a plain background
3. Fill in:
   - Name, Date of Birth, Sex (required)
   - Weight in kg (optional — will be ML-estimated if omitted)
   - Height in cm (optional fallback if image detection fails)
4. Click **Run Assessment**
5. Results show:
   - Estimated height + weight
   - HAZ (stunting) and WHZ (wasting) Z-scores and classifications
   - ML wasting prediction with SAM/MAM probabilities
   - Annotated image with pose landmarks

### Photo Guidelines

For best results:
- Child stands upright, full body visible from head to toe
- Plain background, good lighting
- Camera at 1–2 m distance, roughly at child's height
- Frontal (front-facing) view
- Arms slightly away from body so shoulder and hip landmarks are visible

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
| `weight_kg` | float | — | Weight in kg (will be estimated if omitted) |
| `height_cm` | float | — | Height in cm (fallback if image detection fails) |
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
    "whz_status": "Moderate Acute Malnutrition (MAM)"
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

### Malnutrition Classification (WHO thresholds)

**Wasting (Weight-for-Height Z-score):**

| WHZ | Status |
|-----|--------|
| < −3 | **Severe Acute Malnutrition (SAM)** |
| −3 to −2 | **Moderate Acute Malnutrition (MAM)** |
| −2 to +1 | Normal |
| +1 to +2 | Possible Risk of Overweight |
| +2 to +3 | Overweight |
| ≥ +3 | Obese |

**Stunting (Height-for-Age Z-score):**

| HAZ | Status |
|-----|--------|
| < −3 | Severely Stunted |
| −3 to −2 | Stunted |
| −2 to +2 | Normal |
| > +2 | Tall |

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

When you collect real labeled measurements (age, sex, height, weight, confirmed wasting status), use the notebook:

```bash
.venv/bin/jupyter notebook notebooks/train_malnutrition_model.ipynb
```

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
│   │   └── ml_service.py            # ML wasting detection service
│   └── web/                          # Jinja2 templates + static assets
├── ml/
│   ├── generate_synthetic_data.py    # WHO-derived training data generator
│   ├── models.py                     # Keras model architectures
│   ├── train.py                      # Training + TFLite export
│   ├── evaluate.py                   # Metrics + SAM recall check
│   └── inference.py                  # Runtime predictor (singleton)
├── scripts/
│   └── fix_who_data.py               # Regenerates CSV files from Excel LMS
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

- **Camera-only accuracy**: Without a weighing scale, weight is estimated from body widths. The ML model is trained on synthetic data — real-world accuracy improves with clinical validation data.
- **2D pose only**: MediaPipe gives 2D landmarks. Arm circumference (MUAC) cannot be directly measured from a standard standing photo.
- **MUAC not yet integrated**: WHO MUAC-for-age tables are not yet in this project. Download from [WHO website](https://www.who.int/tools/child-growth-standards/standards/arm-circumference-for-age) to enable MUAC-based screening.
- **Age range**: 0–59 months (per WHO reference data coverage).
- **Pose requirements**: Child must be standing upright, full body visible, frontal view.

---

## References

- WHO Child Growth Standards: https://www.who.int/tools/child-growth-standards
- WHO Weight-for-Height LMS tables: https://www.who.int/tools/child-growth-standards/standards/weight-for-height
- WHO MUAC-for-Age standards: https://www.who.int/tools/child-growth-standards/standards/arm-circumference-for-age
- MediaPipe Pose Landmarker: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- Snyder RG et al. (1975) *Anthropometry of Infants, Children, and Youths to Age 18* — body proportion baseline used in ML synthetic data
