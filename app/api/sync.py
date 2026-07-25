"""POST /api/v1/sync — idempotent ingestion of mobile-computed assessments.

Mobile clients run screening on-device, then upload raw measurements here.
The server recomputes Poshan/WHO results from eligible evidence, discards
unverified client ML verdicts, and deduplicates within the authenticated owner.
"""
import shutil
import uuid as uuid_lib
from datetime import date, datetime, timezone
from functools import lru_cache
import math
from pathlib import Path
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
from app.services.age_service import age_months_at, completed_months
from app.services.nutrition_service import NutritionService
from app.services.poshan_setu_service import (
    ELIGIBLE_BMI_SOURCES,
    classify_poshan_setu,
    normalize_muac_method,
    normalize_source,
)
from app.services.who_data_service import WHODataService
from config import UPLOAD_DIR

router = APIRouter(prefix="/api/v1", tags=["Sync"])


@lru_cache(maxsize=1)
def _get_sync_nutrition_service() -> NutritionService:
    """Load authoritative WHO Excel tables once for server recomputation."""
    who_data = WHODataService()
    who_data.load_all()
    return NutritionService(who_data)


def _validate_number(
    name: str,
    value: Optional[float],
    lower: float,
    upper: float,
) -> None:
    if value is None:
        return
    if not math.isfinite(value) or not lower <= value <= upper:
        raise HTTPException(
            400,
            f"{name} must be finite and between {lower:g} and {upper:g}",
        )


def _save_upload(upload: UploadFile) -> str:
    UPLOAD_DIR.mkdir(exist_ok=True)
    safe_name = Path(upload.filename or "image.jpg").name
    filename = f"{uuid_lib.uuid4().hex}_{safe_name}"
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
    effective_height_cm: Optional[float] = Form(None),
    effective_weight_kg: Optional[float] = Form(None),
    height_source: Optional[str] = Form(None),
    weight_source: Optional[str] = Form(None),
    reference_object_detected: str = Form("false"),
    scale_factor: Optional[float] = Form(None),
    haz_zscore: Optional[float] = Form(None),
    whz_zscore: Optional[float] = Form(None),
    haz_status: Optional[str] = Form(None),
    whz_status: Optional[str] = Form(None),
    confidence_score: Optional[float] = Form(None),
    body_build: Optional[str] = Form(None),
    side_view_used: str = Form("false"),
    chest_depth_cm: Optional[float] = Form(None),
    abd_depth_cm: Optional[float] = Form(None),
    ml_estimated_weight_kg: Optional[float] = Form(None),
    ml_wasting_status: Optional[str] = Form(None),
    ml_model_version: Optional[str] = Form(None),
    ml_training_data: Optional[str] = Form(None),
    ml_non_clinical: str = Form("true"),
    sam_probability: Optional[float] = Form(None),
    mam_probability: Optional[float] = Form(None),
    normal_probability: Optional[float] = Form(None),
    risk_probability: Optional[float] = Form(None),
    overweight_probability: Optional[float] = Form(None),
    muac_cm: Optional[float] = Form(None),
    muac_status: Optional[str] = Form(None),
    muac_method: Optional[str] = Form(None),
    bmi: Optional[float] = Form(None),
    bmi_status: Optional[str] = Form(None),
    poshan_status: Optional[str] = Form(None),
    poshan_triggered_by: Optional[str] = Form(None),
    classification_method: Optional[str] = Form(None),
    classification_rationale: Optional[str] = Form(None),
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
        raise HTTPException(400, "Submission must include an image or at least one measurement")

    try:
        dob = date.fromisoformat(date_of_birth)
    except ValueError:
        raise HTTPException(400, "date_of_birth must be ISO format (YYYY-MM-DD)")

    try:
        visit_dt = datetime.fromisoformat(visit_date)
    except ValueError:
        raise HTTPException(400, "visit_date must be ISO format")

    if dob > visit_dt.date():
        raise HTTPException(400, "date_of_birth must not be after visit_date")
    if visit_dt.date() > datetime.now(timezone.utc).date():
        raise HTTPException(400, "visit_date must not be in the future")

    if not child_name or not child_name.strip():
        raise HTTPException(400, "child_name must not be empty")
    if len(child_name) > 100:
        raise HTTPException(400, "child_name must be at most 100 characters")

    computed_age_months = age_months_at(dob, visit_dt.date())
    if (
        not math.isfinite(age_months)
        or not 0.0 <= age_months < 60.0
        or not 0.0 <= computed_age_months < 60.0
    ):
        raise HTTPException(400, "assessment must be for a child under 60 months")
    if abs(age_months - computed_age_months) > 0.5:
        raise HTTPException(
            400,
            "age_months is inconsistent with date_of_birth and visit_date",
        )

    for name, value, lower, upper in (
        ("predicted_height_cm", predicted_height_cm, 30.0, 130.0),
        ("manual_height_cm", manual_height_cm, 30.0, 130.0),
        ("effective_height_cm", effective_height_cm, 30.0, 130.0),
        ("predicted_weight_kg", predicted_weight_kg, 0.5, 40.0),
        ("manual_weight_kg", manual_weight_kg, 0.5, 40.0),
        ("effective_weight_kg", effective_weight_kg, 0.5, 40.0),
        ("ml_estimated_weight_kg", ml_estimated_weight_kg, 0.5, 40.0),
        ("muac_cm", muac_cm, 5.0, 25.0),
        ("chest_depth_cm", chest_depth_cm, 0.1, 50.0),
        ("abd_depth_cm", abd_depth_cm, 0.1, 50.0),
        ("scale_factor", scale_factor, 0.0001, 10.0),
        ("haz_zscore", haz_zscore, -20.0, 20.0),
        ("whz_zscore", whz_zscore, -20.0, 20.0),
        ("confidence_score", confidence_score, 0.0, 1.0),
        ("sam_probability", sam_probability, 0.0, 1.0),
        ("mam_probability", mam_probability, 0.0, 1.0),
        ("normal_probability", normal_probability, 0.0, 1.0),
        ("risk_probability", risk_probability, 0.0, 1.0),
        ("overweight_probability", overweight_probability, 0.0, 1.0),
        ("bmi", bmi, 3.0, 60.0),
    ):
        _validate_number(name, value, lower, upper)

    # Resolve effective values from raw evidence.  Client classifications are
    # deliberately ignored; provenance cannot elevate a value to "manual"
    # unless the corresponding manual field is actually present.
    if manual_height_cm is not None:
        server_height_cm = manual_height_cm
        server_height_source = "manual"
    else:
        server_height_cm = (
            effective_height_cm
            if effective_height_cm is not None
            else predicted_height_cm
        )
        requested_height_source = normalize_source(height_source)
        if server_height_cm is None:
            server_height_source = "unavailable"
        elif requested_height_source in (
            "who_statistical",
            "landmark_estimated",
        ):
            server_height_source = requested_height_source
        elif height_source is not None:
            server_height_source = "unavailable"
        else:
            # Legacy predicted heights are screening estimates, not validated
            # measurements, so they remain ineligible for Poshan BMI.
            server_height_source = "who_statistical"

    if manual_weight_kg is not None:
        server_weight_kg = manual_weight_kg
        server_weight_source = "manual"
    else:
        server_weight_kg = effective_weight_kg
        if server_weight_kg is None:
            server_weight_kg = (
                ml_estimated_weight_kg
                if ml_estimated_weight_kg is not None
                else predicted_weight_kg
            )
        requested_weight_source = normalize_source(weight_source)
        if server_weight_kg is None:
            server_weight_source = "unavailable"
        elif requested_weight_source in ("ml_estimated", "who_statistical"):
            server_weight_source = requested_weight_source
        elif weight_source is not None:
            server_weight_source = "unavailable"
        elif (
            ml_estimated_weight_kg is not None
            and math.isclose(
                server_weight_kg,
                ml_estimated_weight_kg,
                rel_tol=0.0,
                abs_tol=0.05,
            )
        ):
            server_weight_source = "ml_estimated"
        else:
            server_weight_source = "who_statistical"

    server_muac_method = normalize_muac_method(muac_method)
    # The canonical MUAC column is a tape/manual measurement only. A client
    # may upload an estimated screening value for its own UI, but the server
    # must not store or classify that value as though a field worker measured
    # the arm.
    server_muac_cm = (
        muac_cm if server_muac_method == "manual" else None
    )
    poshan = classify_poshan_setu(
        sex=sex,
        age_months=computed_age_months,
        weight_kg=server_weight_kg,
        height_cm=server_height_cm,
        weight_source=server_weight_source,
        height_source=server_height_source,
        muac_cm=server_muac_cm,
        muac_method=server_muac_method,
    )

    # Recompute WHO results from authoritative server-resolved measurements.
    # Client labels/z-scores are ignored: statistically imputed or tampered
    # values must never be persisted as a clinical "Normal" assessment.
    server_haz_zscore = None
    server_whz_zscore = None
    server_haz_status = None
    server_whz_status = None
    reliable_height = server_height_source in ELIGIBLE_BMI_SOURCES
    reliable_weight = server_weight_source in ELIGIBLE_BMI_SOURCES
    if reliable_height and server_height_cm is not None:
        nutrition = _get_sync_nutrition_service()
        server_haz_zscore = nutrition.compute_haz(
            sex,
            completed_months(dob, visit_dt.date()),
            server_height_cm,
        )
        if server_haz_zscore is not None:
            server_haz_status = nutrition.classify_haz(server_haz_zscore)
        if reliable_weight and server_weight_kg is not None:
            server_whz_zscore = nutrition.compute_whz(
                sex,
                computed_age_months,
                server_height_cm,
                server_weight_kg,
            )
            if server_whz_zscore is not None:
                server_whz_status = nutrition.classify_whz(server_whz_zscore)

    # Dedup check BEFORE saving image — scoped to the authenticated owner so a
    # guessed UUID can never disclose another worker's server visit id.
    existing = (
        db.query(Visit)
        .filter(
            Visit.local_uuid == local_uuid,
            Visit.user_id == current.id,
        )
        .first()
    )
    if existing is not None:
        return {"server_visit_id": existing.id, "status": "already_synced"}
    uuid_collision = (
        db.query(Visit.id)
        .filter(Visit.local_uuid == local_uuid)
        .first()
    )
    if uuid_collision is not None:
        raise HTTPException(409, "local_uuid is already in use")

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
        age_months=computed_age_months,
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
        effective_height_cm=server_height_cm,
        effective_weight_kg=server_weight_kg,
        height_source=server_height_source,
        weight_source=server_weight_source,
        reference_object_detected="false",
        scale_factor=scale_factor,
        haz_zscore=server_haz_zscore,
        whz_zscore=server_whz_zscore,
        haz_status=server_haz_status,
        whz_status=server_whz_status,
        confidence_score=confidence_score,
        body_build=body_build,
        side_view_used=(side_view_used.lower() in ("true", "1", "yes")),
        chest_depth_cm=chest_depth_cm,
        abd_depth_cm=abd_depth_cm,
        # Mobile ML output is not re-run or hash-verified by this endpoint.
        # Discard its clinical-looking values and record the trust decision.
        ml_estimated_weight_kg=None,
        ml_wasting_status=None,
        ml_model_version=None,
        ml_training_data="client_sync_unverified_discarded",
        ml_non_clinical=True,
        sam_probability=None,
        mam_probability=None,
        normal_probability=None,
        risk_probability=None,
        overweight_probability=None,
        muac_cm=server_muac_cm,
        muac_status=poshan.muac_status,
        muac_method=server_muac_method,
        bmi=poshan.bmi,
        bmi_status=poshan.bmi_status,
        poshan_status=poshan.final_status,
        poshan_triggered_by=list(poshan.triggered_by),
        classification_method=poshan.classification_method,
        classification_rationale=poshan.rationale,
    )
    db.add(measurement)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(Visit)
            .filter(
                Visit.local_uuid == local_uuid,
                Visit.user_id == current.id,
            )
            .first()
        )
        if existing is not None:
            return {"server_visit_id": existing.id, "status": "already_synced"}
        collision = (
            db.query(Visit.id)
            .filter(Visit.local_uuid == local_uuid)
            .first()
        )
        if collision is not None:
            raise HTTPException(409, "local_uuid is already in use")
        # Truly unexpected integrity error — propagate.
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
