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
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.user import User
from app.services.auth_service import get_current_user
from app.models.database import get_db
from app.schemas.assessment import AssessmentResponse
from app.services.assessment_service import AssessmentService
from app.services.measurement_service import PoseRuntimeUnavailableError
from app.services.poshan_setu_service import normalize_muac_method
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
    image: Optional[UploadFile] = File(None),
    image_side: Optional[UploadFile] = File(None),
    image_back: Optional[UploadFile] = File(None),
    child_name: str = Form(...),
    date_of_birth: str = Form(...),  # yyyy-mm-dd (HTML5 date input)
    assessment_date: Optional[str] = Form(None),  # yyyy-mm-dd; defaults to today
    sex: str = Form(...),  # 'M' or 'F'
    weight_kg: float = Form(None),
    height_cm: float = Form(None),
    height_value: float = Form(None),  # Height value (from form)
    height_unit: str = Form("cm"),  # Height unit: "cm" or "inch"
    muac_cm: float = Form(None),
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

    # Convert height if provided with unit
    final_height_cm = height_cm
    if height_value is not None and height_cm is None:
        if height_unit not in ("cm", "inch"):
            raise HTTPException(400, "height_unit must be 'cm' or 'inch'")
        if height_unit == "inch":
            final_height_cm = height_value * 2.54
        else:
            final_height_cm = height_value

    if assessment_date:
        try:
            assessed_on = date.fromisoformat(assessment_date)
        except ValueError:
            raise HTTPException(
                400, "assessment_date must be ISO format (YYYY-MM-DD)"
            )
    else:
        assessed_on = date.today()

    if assessed_on < dob:
        raise HTTPException(
            400, "assessment_date must not be before date_of_birth"
        )
    if assessed_on > date.today():
        raise HTTPException(400, "assessment_date must not be in the future")

    if image is None and (final_height_cm is None or weight_kg is None):
        raise HTTPException(
            400,
            "An image or both manual height_cm and weight_kg are required",
        )

    age_months = svc._compute_age_months(dob, assessed_on)
    try:
        svc._validate_inputs(
            child_name=child_name,
            sex=sex,
            age_months=age_months,
            weight_kg=weight_kg,
            height_cm=final_height_cm,
            muac_cm=muac_cm,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Save a front image only when one was supplied. Manual measurements can
    # be assessed without loading the optional MediaPipe runtime.
    file_path: Optional[Path] = None
    if image is not None:
        UPLOAD_DIR.mkdir(exist_ok=True)
        filename = (
            f"{uuid.uuid4().hex}_"
            f"{Path(image.filename or 'image.jpg').name}"
        )
        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

    # Read side image bytes (kept in memory; not saved to disk)
    side_image_bytes = None
    if image_side is not None:
        side_image_bytes = await image_side.read()

    try:
        result = svc.assess(
            db=db,
            image_path=str(file_path) if file_path is not None else None,
            child_name=child_name,
            dob=dob,
            sex=sex,
            weight_kg=weight_kg,
            height_cm=final_height_cm,
            muac_cm=muac_cm,
            guardian_name=guardian_name,
            location=location,
            side_image=side_image_bytes,
            assessment_date=assessed_on,
        )
    except PoseRuntimeUnavailableError as exc:
        if file_path is not None:
            file_path.unlink(missing_ok=True)
        raise HTTPException(
            503,
            f"Pose measurement runtime is unavailable: {exc}",
        ) from exc
    return result


@router.get("/children")
def list_children(
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """List the authenticated worker's non-archived children."""
    # Owner-scoped: legacy rows with user_id=NULL (pre-auth data) are intentionally
    # excluded here — they are unowned and only reachable by an admin.
    children = (
        db.query(Child)
        .filter(Child.user_id == current.id, Child.is_archived == False)  # noqa: E712
        .order_by(Child.name)
        .all()
    )
    return [
        {
            "id": c.id,
            "name": c.name,
            "date_of_birth": c.date_of_birth.isoformat(),
            "sex": c.sex,
            "photo_path": c.photo_path,
            "visit_count": len(c.visits),
        }
        for c in children
    ]


@router.get("/children/{child_id}")
def get_child(
    child_id: int,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    """Get child detail with full visit history (owner-scoped)."""
    child = (
        db.query(Child)
        .filter(Child.id == child_id, Child.user_id == current.id)
        .first()
    )
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
            has_poshan_v1 = m.classification_method == "poshan_setu_v1"
            authoritative_height = (
                m.manual_height_cm is not None
                or m.height_source in ("manual", "reference_object")
            )
            authoritative_weight = (
                m.manual_weight_kg is not None
                or m.weight_source in ("manual", "reference_object")
            )
            canonical_muac_method = normalize_muac_method(m.muac_method)
            visit_data["measurement"] = {
                "predicted_height_cm": m.predicted_height_cm,
                "predicted_weight_kg": m.predicted_weight_kg,
                "manual_height_cm": m.manual_height_cm,
                "manual_weight_kg": m.manual_weight_kg,
                "effective_height_cm": m.effective_height_cm,
                "effective_weight_kg": m.effective_weight_kg,
                "height_source": m.height_source or "unavailable",
                "weight_source": m.weight_source or "unavailable",
                "haz_zscore": (
                    m.haz_zscore if authoritative_height else None
                ),
                "whz_zscore": (
                    m.whz_zscore
                    if authoritative_height and authoritative_weight
                    else None
                ),
                "haz_status": (
                    m.haz_status if authoritative_height else None
                ),
                "whz_status": (
                    m.whz_status
                    if authoritative_height and authoritative_weight
                    else None
                ),
                "muac_cm": (
                    m.muac_cm
                    if canonical_muac_method == "manual"
                    else None
                ),
                "muac_status": (
                    m.muac_status
                    if has_poshan_v1 and canonical_muac_method == "manual"
                    else "Indeterminate"
                ),
                "muac_method": (
                    canonical_muac_method
                    if canonical_muac_method == "manual"
                    else "unavailable"
                ),
                "bmi": m.bmi,
                "bmi_status": (
                    m.bmi_status if has_poshan_v1 else "Indeterminate"
                ),
                "poshan_status": (
                    m.poshan_status if has_poshan_v1 else "Indeterminate"
                ),
                "poshan_triggered_by": (
                    (m.poshan_triggered_by or []) if has_poshan_v1 else []
                ),
                "classification_method": (
                    m.classification_method or "unavailable"
                ),
                "classification_rationale": m.classification_rationale,
                "ml_model_version": m.ml_model_version,
                "ml_training_data": m.ml_training_data,
                "ml_non_clinical": m.ml_non_clinical,
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
