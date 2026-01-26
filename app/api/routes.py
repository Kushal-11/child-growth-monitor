"""
FastAPI API route handlers.

Endpoints:
  POST /api/v1/assess       - Main assessment endpoint (multipart: image + metadata)
  GET  /api/v1/children      - List all children
  GET  /api/v1/children/{id} - Get child detail with visit history
  GET  /api/v1/health        - Health check
"""
import shutil
import uuid
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.database import get_db
from app.schemas.assessment import AssessmentResponse
from app.services.assessment_service import AssessmentService
from config import UPLOAD_DIR

router = APIRouter(prefix="/api/v1", tags=["API"])


def get_assessment_service() -> AssessmentService:
    """Placeholder; overridden at app startup in main.py."""
    raise NotImplementedError


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "child-growth-monitor"}


@router.post("/assess", response_model=AssessmentResponse)
async def assess_child(
    image: UploadFile = File(...),
    child_name: str = Form(...),
    date_of_birth: str = Form(...),  # YYYY-MM-DD
    sex: str = Form(...),  # 'M' or 'F'
    weight_kg: float = Form(None),
    height_cm: float = Form(None),
    guardian_name: str = Form(None),
    location: str = Form(None),
    db: Session = Depends(get_db),
    svc: AssessmentService = Depends(get_assessment_service),
):
    """Main assessment endpoint. Accepts multipart form with image + metadata."""
    if sex not in ("M", "F"):
        raise HTTPException(400, "sex must be 'M' or 'F'")

    try:
        dob = date.fromisoformat(date_of_birth)
    except ValueError:
        raise HTTPException(400, "date_of_birth must be ISO format (YYYY-MM-DD)")

    # Save uploaded image
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{image.filename}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    result = svc.assess(
        db=db,
        image_path=str(file_path),
        child_name=child_name,
        dob=dob,
        sex=sex,
        weight_kg=weight_kg,
        height_cm=height_cm,
        guardian_name=guardian_name,
        location=location,
    )
    return result


@router.get("/children")
def list_children(db: Session = Depends(get_db)):
    """List all registered children."""
    children = db.query(Child).order_by(Child.name).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "date_of_birth": c.date_of_birth.isoformat(),
            "sex": c.sex,
            "visit_count": len(c.visits),
        }
        for c in children
    ]


@router.get("/children/{child_id}")
def get_child(child_id: int, db: Session = Depends(get_db)):
    """Get child detail with full visit history."""
    child = db.query(Child).filter(Child.id == child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    visits = []
    for v in child.visits:
        visit_data = {
            "visit_id": v.id,
            "visit_date": v.visit_date.isoformat() if v.visit_date else None,
            "age_months": v.age_months,
        }
        if v.measurement:
            m = v.measurement
            visit_data["measurement"] = {
                "predicted_height_cm": m.predicted_height_cm,
                "predicted_weight_kg": m.predicted_weight_kg,
                "manual_weight_kg": m.manual_weight_kg,
                "haz_zscore": m.haz_zscore,
                "whz_zscore": m.whz_zscore,
                "haz_status": m.haz_status,
                "whz_status": m.whz_status,
                "confidence_score": m.confidence_score,
            }
        visits.append(visit_data)

    return {
        "id": child.id,
        "name": child.name,
        "date_of_birth": child.date_of_birth.isoformat(),
        "sex": child.sex,
        "guardian_name": child.guardian_name,
        "location": child.location,
        "visits": visits,
    }
