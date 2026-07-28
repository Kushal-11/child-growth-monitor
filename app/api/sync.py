"""POST /api/v1/sync — server-verified ingestion of mobile evidence."""
import json
import math
import shutil
import uuid as uuid_lib
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.child import Child
from app.models.database import get_db
from app.models.measurement import MeasurementResult
from app.models.user import User
from app.models.visit import Visit
from app.services.age_service import AgeService
from app.services.auth_service import get_current_user
from app.services.muac_service import MUACService
from app.services.nutrition_service import NutritionService
from app.services.poshan_setu_service import (
    ELIGIBLE_BMI_SOURCES,
    classify_poshan_setu,
    normalize_muac_method,
    normalize_source,
)
from app.services.who_data_service import WHODataService
from config import (
    POSHAN_MUAC_MAX_AGE_MONTHS,
    POSHAN_MUAC_MIN_AGE_MONTHS,
    UPLOAD_DIR,
    WastingStatus,
)

router = APIRouter(prefix="/api/v1", tags=["Sync"])


@lru_cache(maxsize=1)
def _get_sync_nutrition_service() -> NutritionService:
    who_data = WHODataService()
    who_data.load_all()
    return NutritionService(who_data)


def _validate_number(
    name: str, value: Optional[float], lower: float, upper: float
) -> None:
    if value is None:
        return
    if not math.isfinite(value) or not lower <= value <= upper:
        raise HTTPException(
            400, f"{name} must be finite and between {lower:g} and {upper:g}"
        )


def _as_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _save_upload(upload: UploadFile) -> str:
    UPLOAD_DIR.mkdir(exist_ok=True)
    safe_name = Path(upload.filename or "image.jpg").name
    path = UPLOAD_DIR / f"{uuid_lib.uuid4().hex}_{safe_name}"
    with open(path, "wb") as file_handle:
        shutil.copyfileobj(upload.file, file_handle)
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
    reference_object_detected: str = Form("false"),
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
    poshan_status: Optional[str] = Form(None),
    poshan_triggered_by: str = Form("[]"),
    classification_method: Optional[str] = Form(None),
    classification_rationale: Optional[str] = Form(None),
    poshan_complete: str = Form("false"),
    entry_method: str = Form("assessment"),
    is_archived: str = Form("false"),
    guardian_name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
):
    # Client z-scores and verdict fields are accepted for wire compatibility,
    # then ignored. The server derives all clinical/programme results below.
    del (
        haz_zscore,
        whz_zscore,
        haz_status,
        whz_status,
        bmi,
        bmi_status,
        muac_status,
        combined_status,
        combined_triggered_by,
        combined_rationale,
        combined_method,
        combined_confidence_score,
        combined_protocol_version,
        poshan_status,
        poshan_triggered_by,
        classification_method,
        classification_rationale,
        poshan_complete,
    )

    if sex not in ("M", "F"):
        raise HTTPException(400, "sex must be 'M' or 'F'")
    if entry_method not in ("assessment", "manual"):
        raise HTTPException(400, "entry_method must be 'assessment' or 'manual'")
    if not child_name.strip() or len(child_name) > 100:
        raise HTTPException(400, "child_name must contain 1 to 100 characters")
    if (
        image is None
        and manual_height_cm is None
        and manual_weight_kg is None
        and predicted_height_cm is None
        and predicted_weight_kg is None
        and effective_height_cm is None
        and effective_weight_kg is None
        and muac_cm is None
    ):
        raise HTTPException(
            400, "Submission must include an image or at least one measurement"
        )

    try:
        dob = date.fromisoformat(date_of_birth)
    except ValueError as exc:
        raise HTTPException(
            400, "date_of_birth must be ISO format (YYYY-MM-DD)"
        ) from exc
    try:
        visit_dt = datetime.fromisoformat(visit_date)
    except ValueError as exc:
        raise HTTPException(400, "visit_date must be ISO format") from exc
    if visit_dt.date() > datetime.now(timezone.utc).date():
        raise HTTPException(400, "visit_date must not be in the future")
    try:
        age = AgeService().validate_clinical_age(dob, visit_dt.date())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not math.isfinite(age_months) or abs(age_months - age.months) > 0.5:
        raise HTTPException(
            400, "age_months is inconsistent with date_of_birth and visit_date"
        )

    for name, value, lower, upper in (
        ("predicted_height_cm", predicted_height_cm, 30, 130),
        ("manual_height_cm", manual_height_cm, 30, 130),
        ("effective_height_cm", effective_height_cm, 30, 130),
        ("predicted_weight_kg", predicted_weight_kg, 0.5, 40),
        ("manual_weight_kg", manual_weight_kg, 0.5, 40),
        ("effective_weight_kg", effective_weight_kg, 0.5, 40),
        ("ml_estimated_weight_kg", ml_estimated_weight_kg, 0.5, 40),
        ("muac_cm", muac_cm, 5, 25),
        ("chest_depth_cm", chest_depth_cm, 0.1, 50),
        ("abd_depth_cm", abd_depth_cm, 0.1, 50),
        ("confidence_score", confidence_score, 0, 1),
        ("height_confidence", height_confidence, 0, 1),
        ("weight_confidence", weight_confidence, 0, 1),
        ("classification_confidence", classification_confidence, 0, 1),
        ("muac_confidence", muac_confidence, 0, 1),
        ("sam_probability", sam_probability, 0, 1),
        ("mam_probability", mam_probability, 0, 1),
        ("normal_probability", normal_probability, 0, 1),
        ("risk_probability", risk_probability, 0, 1),
        ("overweight_probability", overweight_probability, 0, 1),
    ):
        _validate_number(name, value, lower, upper)

    if manual_height_cm is not None:
        server_height = manual_height_cm
        server_height_source = "manual"
    else:
        server_height = effective_height_cm or predicted_height_cm
        validated_reference = (
            server_height is not None
            and normalize_source(estimation_method) == "reference_object"
            and _as_bool(reference_object_detected)
        )
        server_height_source = (
            "reference_object" if validated_reference else "unavailable"
        )

    if manual_weight_kg is not None:
        server_weight = manual_weight_kg
        server_weight_source = "manual"
    else:
        server_weight = (
            effective_weight_kg
            if effective_weight_kg is not None
            else ml_estimated_weight_kg
            if ml_estimated_weight_kg is not None
            else predicted_weight_kg
        )
        requested_weight_source = normalize_source(weight_method)
        server_weight_source = (
            requested_weight_source
            if requested_weight_source in {"ml_estimated", "who_statistical"}
            else "unavailable"
        )

    server_muac_method = normalize_muac_method(muac_method)
    direct_muac = (
        muac_cm
        if server_muac_method == "manual"
        and _as_bool(muac_is_direct_measurement)
        else None
    )
    poshan = classify_poshan_setu(
        sex=sex,
        age_months=age.months,
        weight_kg=server_weight,
        height_cm=server_height,
        weight_source=server_weight_source,
        height_source=server_height_source,
        muac_cm=direct_muac,
        muac_method=server_muac_method,
    )

    server_haz_z = None
    server_whz_z = None
    server_haz_status = None
    server_whz_status = None
    reliable_height = server_height_source in ELIGIBLE_BMI_SOURCES
    reliable_weight = server_weight_source in ELIGIBLE_BMI_SOURCES
    if reliable_height and server_height is not None:
        nutrition = _get_sync_nutrition_service()
        server_haz_z = nutrition.compute_haz(
            sex, age.completed_months, server_height
        )
        if server_haz_z is not None:
            server_haz_status = nutrition.classify_haz(server_haz_z)
        if reliable_weight and server_weight is not None:
            server_whz_z = nutrition.compute_whz(
                sex, age.months, server_height, server_weight
            )
            if server_whz_z is not None:
                server_whz_status = nutrition.classify_whz(server_whz_z)

    direct_muac_status = {
        "SAM": WastingStatus.SAM,
        "MAM": WastingStatus.MAM,
        "Normal": WastingStatus.NORMAL,
    }.get(poshan.muac_status)
    combined = MUACService.combine_with_whz_status(
        muac_status=direct_muac_status,
        whz_status=server_whz_status,
        muac_method=server_muac_method,
        is_direct_measurement=direct_muac is not None,
    )

    existing = (
        db.query(Visit)
        .filter(Visit.local_uuid == local_uuid, Visit.user_id == current.id)
        .first()
    )
    if existing is not None:
        return {"server_visit_id": existing.id, "status": "already_synced"}
    if db.query(Visit.id).filter(Visit.local_uuid == local_uuid).first() is not None:
        raise HTTPException(409, "local_uuid is already in use")

    image_path = _save_upload(image) if image is not None else None
    side_path = _save_upload(image_side) if image_side is not None else None
    back_path = _save_upload(image_back) if image_back is not None else None
    photo_path = _save_upload(photo) if photo is not None else None
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
    if child is None:
        child = Child(
            name=child_name,
            date_of_birth=dob,
            sex=sex,
            guardian_name=guardian_name,
            location=location,
            user_id=current.id,
            photo_path=photo_path,
            is_archived=_as_bool(is_archived),
        )
        db.add(child)
        db.flush()
    else:
        child.is_archived = _as_bool(is_archived)
        if photo_path is not None:
            child.photo_path = photo_path

    visit = Visit(
        child_id=child.id,
        visit_date=visit_dt,
        age_months=age.months,
        image_path=image_path,
        side_image_path=side_path,
        back_image_path=back_path,
        local_uuid=local_uuid,
        user_id=current.id,
        entry_method=entry_method,
    )
    db.add(visit)
    db.flush()

    measurement = MeasurementResult(
        visit_id=visit.id,
        predicted_height_cm=predicted_height_cm,
        predicted_weight_kg=predicted_weight_kg,
        manual_height_cm=manual_height_cm,
        manual_weight_kg=manual_weight_kg,
        effective_height_cm=server_height,
        effective_weight_kg=server_weight,
        height_method=server_height_source,
        weight_method=server_weight_source,
        estimation_method=estimation_method,
        bmi=poshan.bmi,
        bmi_status=poshan.bmi_status,
        reference_object_detected=str(_as_bool(reference_object_detected)).lower(),
        haz_zscore=server_haz_z,
        whz_zscore=server_whz_z,
        haz_status=server_haz_status,
        whz_status=server_whz_status.value if server_whz_status else None,
        confidence_score=confidence_score,
        height_confidence=height_confidence,
        weight_confidence=weight_confidence,
        classification_confidence=(
            1.0 if poshan.triggered_by and poshan.complete else None
        ),
        body_build=body_build,
        side_view_used=_as_bool(side_view_used),
        chest_depth_cm=chest_depth_cm,
        abd_depth_cm=abd_depth_cm,
        # Client ML output is retained as explicitly non-diagnostic evidence.
        ml_estimated_weight_kg=ml_estimated_weight_kg,
        ml_wasting_status=ml_wasting_status,
        ml_wasting_method=ml_wasting_method,
        sam_probability=sam_probability,
        mam_probability=mam_probability,
        normal_probability=normal_probability,
        risk_probability=risk_probability,
        overweight_probability=overweight_probability,
        muac_cm=direct_muac,
        muac_status=poshan.muac_status,
        muac_method=server_muac_method,
        muac_age_in_range=(
            POSHAN_MUAC_MIN_AGE_MONTHS
            <= age.months
            < POSHAN_MUAC_MAX_AGE_MONTHS
        ),
        muac_confidence=muac_confidence if direct_muac is not None else None,
        muac_uncertainty_lower_cm=(
            muac_uncertainty_lower_cm if direct_muac is not None else None
        ),
        muac_uncertainty_upper_cm=(
            muac_uncertainty_upper_cm if direct_muac is not None else None
        ),
        muac_model_version=muac_model_version,
        muac_calibration_version=muac_calibration_version,
        muac_is_direct_measurement=direct_muac is not None,
        muac_requires_confirmation=(
            False if direct_muac is not None else _as_bool(muac_requires_confirmation)
        ),
        muac_referral_guidance=muac_referral_guidance,
        combined_status=combined.status.value,
        combined_triggered_by=json.dumps(combined.triggered_by),
        combined_rationale=combined.rationale,
        combined_method="who_muac_whz_or_rule",
        combined_confidence_score=(
            1.0 if "muac" in combined.triggered_by else confidence_score
        ),
        combined_protocol_version="WHO-CMAM-OR-2009/2013-v1",
        poshan_status=poshan.final_status,
        poshan_triggered_by=json.dumps(poshan.triggered_by),
        classification_method=poshan.classification_method,
        classification_rationale=poshan.rationale,
        poshan_complete=poshan.complete,
    )
    db.add(measurement)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(Visit)
            .filter(Visit.local_uuid == local_uuid, Visit.user_id == current.id)
            .first()
        )
        if existing is not None:
            return {"server_visit_id": existing.id, "status": "already_synced"}
        if db.query(Visit.id).filter(Visit.local_uuid == local_uuid).first() is not None:
            raise HTTPException(409, "local_uuid is already in use")
        raise

    return {
        "server_visit_id": visit.id,
        "status": "synced",
        "poshan": {
            "bmi": poshan.bmi,
            "bmi_status": poshan.bmi_status,
            "muac_status": poshan.muac_status,
            "final_status": poshan.final_status,
            "triggered_by": list(poshan.triggered_by),
            "classification_method": poshan.classification_method,
            "rationale": poshan.rationale,
            "complete": poshan.complete,
        },
    }
