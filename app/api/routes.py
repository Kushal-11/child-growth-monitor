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
import json
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.user import User
from app.services.auth_service import get_current_user
from app.models.database import get_db
from app.schemas.assessment import AssessmentRequest, AssessmentResponse
from app.services.age_service import AgeService
from app.services.assessment_service import AssessmentService
from config import UPLOAD_DIR



router = APIRouter(prefix="/api/v1", tags=["API"])


def _decode_string_list(value: Optional[str]) -> list[str]:
    """Decode legacy JSON evidence without breaking child history."""
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, str)]


def get_assessment_service() -> AssessmentService:
    """Placeholder; overridden at app startup in main.py."""
    raise NotImplementedError


def get_age_service() -> AgeService:
    """Provide the shared stateless clinical-age service."""
    return AgeService()


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "child-growth-monitor"}


@router.post("/assess", response_model=AssessmentResponse)
async def assess_child(
    image: UploadFile = File(...),
    image_side: Optional[UploadFile] = File(None),
    image_back: Optional[UploadFile] = File(None),
    child_name: str = Form(...),
    date_of_birth: str = Form(...),  # yyyy-mm-dd (HTML5 date input)
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
    age_svc: AgeService = Depends(get_age_service),
):
    """Main assessment endpoint. Accepts multipart form with image + metadata."""
    if sex not in ("M", "F"):
        raise HTTPException(400, "sex must be 'M' or 'F'")

    try:
        dob = date.fromisoformat(date_of_birth)
    except ValueError:
        raise HTTPException(
            422, "date_of_birth must be ISO format (YYYY-MM-DD)"
        ) from None

    as_of = date.today()
    try:
        # Apply the same Pydantic contract used by JSON callers to multipart data.
        AssessmentRequest.model_validate(
            {
                "child_name": child_name,
                "date_of_birth": dob,
                "sex": sex,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "guardian_name": guardian_name,
                "location": location,
            },
            context={"age_service": age_svc, "as_of": as_of},
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    # Convert height if provided with unit
    final_height_cm = height_cm
    if height_value is not None and height_cm is None:
        if height_unit == "inch":
            final_height_cm = height_value * 2.54
        else:
            final_height_cm = height_value

    # Save front image
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{image.filename}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Read side image bytes (kept in memory; not saved to disk)
    side_image_bytes = None
    if image_side is not None:
        side_image_bytes = await image_side.read()

    result = svc.assess(
        db=db,
        image_path=str(file_path),
        child_name=child_name,
        dob=dob,
        sex=sex,
        weight_kg=weight_kg,
        height_cm=final_height_cm,
        muac_cm=muac_cm,
        guardian_name=guardian_name,
        location=location,
        side_image=side_image_bytes,
        as_of=as_of,
    )
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
            visit_data["measurement"] = {
                "predicted_height_cm": m.predicted_height_cm,
                "predicted_weight_kg": m.predicted_weight_kg,
                "manual_height_cm": m.manual_height_cm,
                "manual_weight_kg": m.manual_weight_kg,
                "reference_object_detected": m.reference_object_detected == "true",
                "scale_factor": m.scale_factor,
                "haz_zscore": m.haz_zscore,
                "whz_zscore": m.whz_zscore,
                "haz_status": m.haz_status,
                "whz_status": m.whz_status,
                "confidence_score": m.confidence_score,
                "effective_height_cm": m.effective_height_cm,
                "effective_weight_kg": m.effective_weight_kg,
                "height_method": m.height_method,
                "weight_method": m.weight_method,
                "estimation_method": m.estimation_method,
                "bmi": m.bmi,
                "bmi_status": m.bmi_status,
                "height_confidence": m.height_confidence,
                "weight_confidence": m.weight_confidence,
                "classification_confidence": m.classification_confidence,
                "body_build": m.body_build,
                "side_view_used": m.side_view_used,
                "chest_depth_cm": m.chest_depth_cm,
                "abd_depth_cm": m.abd_depth_cm,
                "ml_estimated_weight_kg": m.ml_estimated_weight_kg,
                "ml_wasting_status": m.ml_wasting_status,
                "ml_wasting_method": m.ml_wasting_method,
                "sam_probability": m.sam_probability,
                "mam_probability": m.mam_probability,
                "normal_probability": m.normal_probability,
                "risk_probability": m.risk_probability,
                "overweight_probability": m.overweight_probability,
                "muac_cm": m.muac_cm,
                "muac_status": m.muac_status,
                "muac_method": m.muac_method,
                "muac_age_in_range": m.muac_age_in_range,
                "muac_confidence": m.muac_confidence,
                "muac_uncertainty_lower_cm": m.muac_uncertainty_lower_cm,
                "muac_uncertainty_upper_cm": m.muac_uncertainty_upper_cm,
                "muac_model_version": m.muac_model_version,
                "muac_calibration_version": m.muac_calibration_version,
                "muac_is_direct_measurement": m.muac_is_direct_measurement,
                "muac_requires_confirmation": m.muac_requires_confirmation,
                "muac_referral_guidance": m.muac_referral_guidance,
                "combined_status": m.combined_status,
                "combined_triggered_by": _decode_string_list(
                    m.combined_triggered_by
                ),
                "combined_rationale": m.combined_rationale,
                "combined_method": m.combined_method,
                "combined_confidence_score": m.combined_confidence_score,
                "combined_protocol_version": m.combined_protocol_version,
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
