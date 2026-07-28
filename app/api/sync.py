"""POST /api/v1/sync — idempotent ingestion of mobile-computed assessments.

Mobile clients run the full assessment on-device, then upload the result here.
The server skips ML, dedups by local_uuid, and stores the image + measurement.
"""
import shutil
import json
import uuid as uuid_lib
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.database import get_db
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from app.models.user import User
from app.services.auth_service import get_current_user
from config import UPLOAD_DIR

router = APIRouter(prefix="/api/v1", tags=["Sync"])


def _save_upload(upload: UploadFile) -> str:
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = f"{uuid_lib.uuid4().hex}_{upload.filename or 'image.jpg'}"
    path = UPLOAD_DIR / filename
    with open(path, "wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return str(path)


@router.post("/sync")
async def sync_assessment(
    image: Optional[UploadFile] = File(None),
    image_side: Optional[UploadFile] = File(None),
    image_back: Optional[UploadFile] = File(None),
    photo: Optional[UploadFile] = File(None),
    local_uuid: str = Form(...),
    child_name: str = Form(...),
    date_of_birth: str = Form(...),
    sex: str = Form(...),
    age_months: float = Form(...),
    visit_date: str = Form(...),
    predicted_height_cm: Optional[float] = Form(None),
    predicted_weight_kg: Optional[float] = Form(None),
    manual_height_cm: Optional[float] = Form(None),
    manual_weight_kg: Optional[float] = Form(None),
    haz_zscore: Optional[float] = Form(None),
    whz_zscore: Optional[float] = Form(None),
    haz_status: Optional[str] = Form(None),
    whz_status: Optional[str] = Form(None),
    confidence_score: Optional[float] = Form(None),
    effective_height_cm: Optional[float] = Form(None),
    effective_weight_kg: Optional[float] = Form(None),
    height_method: Optional[str] = Form(None),
    weight_method: Optional[str] = Form(None),
    estimation_method: Optional[str] = Form(None),
    bmi: Optional[float] = Form(None),
    bmi_status: Optional[str] = Form(None),
    height_confidence: Optional[float] = Form(None),
    weight_confidence: Optional[float] = Form(None),
    classification_confidence: Optional[float] = Form(None),
    body_build: Optional[str] = Form(None),
    side_view_used: str = Form("false"),
    chest_depth_cm: Optional[float] = Form(None),
    abd_depth_cm: Optional[float] = Form(None),
    ml_estimated_weight_kg: Optional[float] = Form(None),
    ml_wasting_status: Optional[str] = Form(None),
    ml_wasting_method: Optional[str] = Form(None),
    sam_probability: Optional[float] = Form(None),
    mam_probability: Optional[float] = Form(None),
    normal_probability: Optional[float] = Form(None),
    risk_probability: Optional[float] = Form(None),
    overweight_probability: Optional[float] = Form(None),
    muac_cm: Optional[float] = Form(None),
    muac_status: Optional[str] = Form(None),
    muac_method: Optional[str] = Form(None),
    muac_age_in_range: str = Form("false"),
    muac_confidence: Optional[float] = Form(None),
    muac_uncertainty_lower_cm: Optional[float] = Form(None),
    muac_uncertainty_upper_cm: Optional[float] = Form(None),
    muac_model_version: Optional[str] = Form(None),
    muac_calibration_version: Optional[str] = Form(None),
    muac_is_direct_measurement: str = Form("false"),
    muac_requires_confirmation: str = Form("false"),
    muac_referral_guidance: Optional[str] = Form(None),
    combined_status: Optional[str] = Form(None),
    combined_triggered_by: str = Form("[]"),
    combined_rationale: Optional[str] = Form(None),
    combined_method: Optional[str] = Form(None),
    combined_confidence_score: Optional[float] = Form(None),
    combined_protocol_version: Optional[str] = Form(None),
    entry_method: str = Form("assessment"),
    is_archived: str = Form("false"),
    guardian_name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    if sex not in ("M", "F"):
        raise HTTPException(400, "sex must be 'M' or 'F'")

    if entry_method not in ("assessment", "manual"):
        raise HTTPException(400, "entry_method must be 'assessment' or 'manual'")

    if image is None and manual_height_cm is None and manual_weight_kg is None and predicted_height_cm is None:
        raise HTTPException(400, "Submission must include an image or at least one measurement")

    try:
        dob = date.fromisoformat(date_of_birth)
    except ValueError:
        raise HTTPException(400, "date_of_birth must be ISO format (YYYY-MM-DD)")

    try:
        visit_dt = datetime.fromisoformat(visit_date)
    except ValueError:
        raise HTTPException(400, "visit_date must be ISO format")

    # Dedup check BEFORE saving image — retried requests won't litter uploads dir
    existing = db.query(Visit).filter(Visit.local_uuid == local_uuid).first()
    if existing is not None:
        return {"server_visit_id": existing.id, "status": "already_synced"}

    image_path = _save_upload(image) if image is not None else None
    side_path = _save_upload(image_side) if image_side is not None else None
    back_path = _save_upload(image_back) if image_back is not None else None

    child = (
        db.query(Child)
        .filter(
            Child.name == child_name,
            Child.date_of_birth == dob,
            Child.sex == sex,
            Child.user_id == current.id,
        )
        .first()
    )
    archived = is_archived.lower() in ("true", "1", "yes")
    photo_path = _save_upload(photo) if photo is not None else None
    if child is None:
        child = Child(
            name=child_name,
            date_of_birth=dob,
            sex=sex,
            guardian_name=guardian_name,
            location=location,
            user_id=current.id,
            photo_path=photo_path,
            is_archived=archived,
        )
        db.add(child)
        db.flush()
    else:
        child.is_archived = archived
        if photo_path is not None:
            child.photo_path = photo_path

    visit = Visit(
        child_id=child.id,
        visit_date=visit_dt,
        age_months=age_months,
        image_path=image_path,
        side_image_path=side_path,
        back_image_path=back_path,
        local_uuid=local_uuid,
        user_id=current.id,
        entry_method=entry_method,
    )
    db.add(visit)
    db.flush()

    try:
        triggers = json.loads(combined_triggered_by)
        if not isinstance(triggers, list) or not all(isinstance(v, str) for v in triggers):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            400, "combined_triggered_by must be a JSON array of strings"
        )

    measurement = MeasurementResult(
        visit_id=visit.id,
        predicted_height_cm=predicted_height_cm,
        predicted_weight_kg=predicted_weight_kg,
        manual_height_cm=manual_height_cm,
        manual_weight_kg=manual_weight_kg,
        haz_zscore=haz_zscore,
        whz_zscore=whz_zscore,
        haz_status=haz_status,
        whz_status=whz_status,
        confidence_score=confidence_score,
        effective_height_cm=effective_height_cm,
        effective_weight_kg=effective_weight_kg,
        height_method=height_method,
        weight_method=weight_method,
        estimation_method=estimation_method,
        bmi=bmi,
        bmi_status=bmi_status,
        height_confidence=height_confidence,
        weight_confidence=weight_confidence,
        classification_confidence=classification_confidence,
        body_build=body_build,
        side_view_used=(side_view_used.lower() in ("true", "1", "yes")),
        chest_depth_cm=chest_depth_cm,
        abd_depth_cm=abd_depth_cm,
        ml_estimated_weight_kg=ml_estimated_weight_kg,
        ml_wasting_status=ml_wasting_status,
        ml_wasting_method=ml_wasting_method,
        sam_probability=sam_probability,
        mam_probability=mam_probability,
        normal_probability=normal_probability,
        risk_probability=risk_probability,
        overweight_probability=overweight_probability,
        muac_cm=muac_cm,
        muac_status=muac_status,
        muac_method=muac_method,
        muac_age_in_range=(muac_age_in_range.lower() in ("true", "1", "yes")),
        muac_confidence=muac_confidence,
        muac_uncertainty_lower_cm=muac_uncertainty_lower_cm,
        muac_uncertainty_upper_cm=muac_uncertainty_upper_cm,
        muac_model_version=muac_model_version,
        muac_calibration_version=muac_calibration_version,
        muac_is_direct_measurement=(
            muac_is_direct_measurement.lower() in ("true", "1", "yes")
        ),
        muac_requires_confirmation=(
            muac_requires_confirmation.lower() in ("true", "1", "yes")
        ),
        muac_referral_guidance=muac_referral_guidance,
        combined_status=combined_status,
        combined_triggered_by=json.dumps(triggers),
        combined_rationale=combined_rationale,
        combined_method=combined_method,
        combined_confidence_score=combined_confidence_score,
        combined_protocol_version=combined_protocol_version,
    )
    db.add(measurement)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.query(Visit).filter(Visit.local_uuid == local_uuid).first()
        if existing is not None:
            return {"server_visit_id": existing.id, "status": "already_synced"}
        # Truly unexpected integrity error — propagate
        raise

    return {"server_visit_id": visit.id, "status": "synced"}
