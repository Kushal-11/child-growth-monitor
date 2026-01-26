# Child Growth Monitor

A WHO standard-based child growth assessment system that uses computer vision and pose estimation to automatically measure child height and classify nutritional status (stunting and wasting).

## Overview

This system processes images of children to estimate height using MediaPipe pose detection and WHO growth reference data. It provides automated nutritional status classification based on WHO Z-score standards, eliminating the need for manual measurements and reference objects.

## Features

- **Automated Height Estimation**: Uses hybrid approach combining WHO statistical methods and anthropometric ratios - no reference objects required
- **Pose Detection**: MediaPipe-based pose landmark detection with visual annotations
- **WHO Z-Score Classification**: Automatic classification of stunting (Height-for-Age) and wasting (Weight-for-Height) using WHO growth standards
- **Web Interface**: User-friendly web UI for uploading images and viewing assessment results
- **REST API**: Full API for integration with mobile apps or other systems
- **Data Persistence**: SQLite database for storing child records and assessment history
- **Multi-Unit Support**: Height input supports both centimeters and inches

## Technology Stack

- **Backend**: FastAPI (Python)
- **Computer Vision**: MediaPipe, OpenCV
- **Database**: SQLAlchemy with SQLite
- **Frontend**: Bootstrap 5, Jinja2 templates
- **Data Processing**: Pandas, NumPy
- **Reference Data**: WHO Child Growth Standards (CSV and Excel files)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Kushal-11/child-growth-monitor.git
cd child-growth-monitor
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure WHO reference data files are in the `data/` directory:
   - `who_haz_0_59m.csv` - Height-for-Age Z-score boundaries
   - `who_wfh_0_59m.csv` - Weight-for-Height reference
   - `who_whz_reference.csv` - Weight-for-Height Z-score reference
   - `wfl_boys_0-to-2-years_zscores.xlsx` - Weight-for-Length (boys, 0-2 years)
   - `wfl_girls_0-to-2-years_zscores.xlsx` - Weight-for-Length (girls, 0-2 years)
   - `wfh_boys_2-to-5-years_zscores.xlsx` - Weight-for-Height (boys, 2-5 years)
   - `wfh_girls_2-to-5-years_zscores.xlsx` - Weight-for-Height (girls, 2-5 years)
   - `pose_landmarker_heavy.task` - MediaPipe pose detection model

5. Run the application:
```bash
python main.py
```

The server will start at `http://localhost:8000`

## Usage

### Web Interface

1. Open `http://localhost:8000` in your browser
2. Upload a photo of a child standing upright (full body visible)
3. Enter child information:
   - Name
   - Date of Birth
   - Sex (Male/Female)
   - Weight (optional - will be estimated if not provided)
   - Height (optional - used as fallback if image detection fails)
4. Click "Run Assessment"
5. View results including:
   - Estimated height and weight
   - Height-for-Age Z-score and classification (Stunting)
   - Weight-for-Height Z-score and classification (Wasting)
   - Annotated image with pose landmarks

### API Endpoints

#### POST /api/v1/assess
Main assessment endpoint. Accepts multipart form data with image and metadata.

**Request:**
- `image`: Image file (required)
- `child_name`: Child's name (required)
- `date_of_birth`: Date in YYYY-MM-DD format (required)
- `sex`: 'M' or 'F' (required)
- `weight_kg`: Weight in kilograms (optional)
- `height_cm`: Height in centimeters (optional)
- `height_value`: Height value (optional, used with height_unit)
- `height_unit`: 'cm' or 'inch' (optional, default: 'cm')
- `guardian_name`: Guardian name (optional)
- `location`: Location (optional)

**Response:**
```json
{
  "child_name": "John Doe",
  "sex": "M",
  "age_months": 36.4,
  "measurement": {
    "predicted_height_cm": 102.7,
    "predicted_weight_kg": 16.03,
    "confidence_score": 0.87,
    "estimation_method": "who_statistical",
    "annotated_image": "filename_annotated.jpg"
  },
  "nutrition": {
    "haz_zscore": 0.0,
    "whz_zscore": -0.0,
    "haz_status": "Normal",
    "whz_status": "Normal"
  },
  "summary": "..."
}
```

#### GET /api/v1/children
List all registered children.

#### GET /api/v1/children/{id}
Get child detail with full visit history.

#### GET /api/v1/health
Health check endpoint.

### API Documentation

Interactive API documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

## How It Works

### Height Estimation

The system uses a hybrid approach that doesn't require reference objects:

1. **WHO Statistical Estimation** (Primary): Uses WHO median height for the child's age and sex as the baseline estimate
2. **Anthropometric Ratios** (Supplementary): Measures body segments (head, torso, legs) and uses age-specific proportions to validate estimates
3. **Reference Object Detection** (Fallback): Optional detection of known-size objects for scale calibration

### Nutritional Classification

- **Height-for-Age (HAZ)**: Classifies stunting using WHO Z-score boundaries
  - Severely Stunted: Z < -3
  - Stunted: -3 ≤ Z < -2
  - Normal: -2 ≤ Z ≤ 2
  - Tall: Z > 2

- **Weight-for-Height (WHZ)**: Classifies wasting using LMS method
  - Severe Acute Malnutrition (SAM): Z < -3
  - Moderate Acute Malnutrition (MAM): -3 ≤ Z < -2
  - Normal: -2 ≤ Z ≤ 1
  - Possible Risk of Overweight: 1 ≤ Z < 2
  - Overweight: 2 ≤ Z < 3
  - Obese: Z ≥ 3

## Project Structure

```
child-growth-monitor/
├── app/
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── models/
│   │   ├── child.py           # Child database model
│   │   ├── visit.py           # Visit database model
│   │   ├── measurement.py     # Measurement result model
│   │   └── database.py        # Database configuration
│   ├── schemas/
│   │   └── assessment.py      # Pydantic request/response models
│   ├── services/
│   │   ├── assessment_service.py    # Main assessment orchestrator
│   │   ├── measurement_service.py   # Image processing & height estimation
│   │   ├── nutrition_service.py     # Z-score computation & classification
│   │   └── who_data_service.py      # WHO reference data loader
│   └── web/
│       ├── views.py           # Web UI routes
│       ├── templates/         # Jinja2 HTML templates
│       └── static/            # CSS and JavaScript
├── data/                      # WHO reference data files
├── tests/                     # Unit tests
├── uploads/                   # Uploaded images
├── config.py                  # Configuration
├── main.py                    # Application entry point
└── requirements.txt           # Python dependencies
```

## Testing

Run the test suite:
```bash
PYTHONPATH="$(pwd)" python -m pytest tests/ -v
```

All 62 tests should pass, covering:
- WHO data service functionality
- Nutrition service Z-score calculations
- Measurement service height estimation
- API endpoint validation

## Configuration

Key configuration options in `config.py`:
- `UPLOAD_DIR`: Directory for uploaded images
- `DB_PATH`: SQLite database file path
- `POSE_MODEL_PATH`: MediaPipe pose detection model
- `ANTHROPOMETRIC_RATIOS`: Age-specific body segment ratios
- `HEIGHT_VALIDATION_SD`: Standard deviation threshold for validation

## Limitations

- Height estimation accuracy depends on image quality and pose visibility
- Requires full body visibility in the image
- Works best with children standing upright
- Age range: 0-59 months (as per WHO data coverage)

## License

This project is provided as-is for research and development purposes.

## References

- WHO Child Growth Standards: https://www.who.int/tools/child-growth-standards
- MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
