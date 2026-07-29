"""Transactional service layer for owner-scoped guided visits."""

import json
import math
from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.camera_result import CameraResult
from app.models.capture_asset import CaptureAsset
from app.models.child import Child
from app.models.measured_detail_revision import MeasuredDetailRevision
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from app.schemas.guided_capture import (
    CameraResultSubmission,
    CaptureAssetSubmission,
    MeasuredDetailsSubmission,
)
from app.services.acute_malnutrition_service import AcuteMalnutritionService
from app.services.age_service import AgeService
from app.services.guided_capture_contract import (
    REQUIRED_CAPTURE_ROLES,
    CaptureState,
    require_capture_transition,
)
from app.services.poshan_setu_service import classify_poshan_setu
from app.services.who_data_service import WHODataService
from config import (
    MEASURED_HEIGHT_MAX_CM,
    MEASURED_HEIGHT_MIN_CM,
    MEASURED_MUAC_MAX_CM,
    MEASURED_MUAC_MIN_CM,
    MEASURED_WEIGHT_MAX_KG,
    MEASURED_WEIGHT_MIN_KG,
)


class GuidedVisitService:
    """Maintain Visit as the aggregate root for guided capture."""

    def __init__(
        self,
        who_data: WHODataService,
        age_service: AgeService | None = None,
    ):
        self._age = age_service or AgeService()
        self._acute = AcuteMalnutritionService(who_data)

    @staticmethod
    def _scoped_visit(
        db: Session,
        owner_user_id: int,
        visit_uuid: UUID | str,
    ) -> Visit:
        visit = db.scalar(
            select(Visit).where(
                Visit.local_uuid == str(visit_uuid),
                Visit.user_id == owner_user_id,
            )
        )
        if visit is None:
            raise LookupError("Owner-scoped visit was not found")
        return visit

    def create_draft_visit(
        self,
        db: Session,
        *,
        owner_user_id: int,
        child_id: int,
        local_uuid: UUID | str,
        visit_date: date,
        device_metadata: dict[str, Any],
        consent_version: str,
        consent_timestamp: datetime,
        consent_operator_identifier: str,
    ) -> Visit:
        child = db.scalar(
            select(Child).where(
                Child.id == child_id,
                Child.user_id == owner_user_id,
            )
        )
        if child is None:
            raise LookupError("Owner-scoped child was not found")
        age = self._age.validate_clinical_age(child.date_of_birth, visit_date)
        existing = db.scalar(
            select(Visit).where(
                Visit.user_id == owner_user_id,
                Visit.local_uuid == str(local_uuid),
            )
        )
        if existing is not None:
            return existing

        visit = Visit(
            child_id=child.id,
            user_id=owner_user_id,
            local_uuid=str(local_uuid),
            visit_date=datetime.combine(visit_date, datetime.min.time()),
            age_months=age.months,
            entry_method="guided_capture",
            capture_state=CaptureState.DRAFT_CAPTURE.value,
            capture_started_at=datetime.utcnow(),
            device_metadata_json=device_metadata,
            consent_version=consent_version,
            consent_timestamp=consent_timestamp,
            consent_operator_identifier=consent_operator_identifier,
        )
        try:
            db.add(visit)
            db.commit()
            db.refresh(visit)
            return visit
        except Exception:
            db.rollback()
            raise

    def transition_visit(
        self,
        db: Session,
        owner_user_id: int,
        visit_uuid: UUID | str,
        target: CaptureState,
    ) -> Visit:
        visit = self._scoped_visit(db, owner_user_id, visit_uuid)
        current = CaptureState(visit.capture_state)
        target_state = CaptureState(target)
        require_capture_transition(current, target_state)
        if target_state == CaptureState.PROCESSING:
            accepted_roles = {
                asset.role
                for asset in visit.capture_assets
                if asset.quality_verdict == "accepted"
            }
            required_roles = {role.value for role in REQUIRED_CAPTURE_ROLES}
            if not required_roles.issubset(accepted_roles):
                raise ValueError(
                    "Processing requires accepted front and side capture assets"
                )
        visit.capture_state = target_state.value
        if target_state == CaptureState.PROCESSING:
            visit.capture_completed_at = datetime.utcnow()
        try:
            db.commit()
            db.refresh(visit)
            return visit
        except Exception:
            db.rollback()
            raise

    def append_capture_asset(
        self,
        db: Session,
        owner_user_id: int,
        submission: CaptureAssetSubmission,
    ) -> CaptureAsset:
        visit = self._scoped_visit(db, owner_user_id, submission.visit_uuid)
        existing = db.scalar(
            select(CaptureAsset).where(
                CaptureAsset.asset_uuid == str(submission.asset_uuid)
            )
        )
        if existing is not None:
            if existing.visit_id != visit.id or existing.role != submission.role.value:
                raise ValueError("Capture asset UUID already identifies another asset")
            return existing

        asset = CaptureAsset(
            asset_uuid=str(submission.asset_uuid),
            visit_id=visit.id,
            role=submission.role.value,
            local_path=submission.local_path,
            server_object_id=submission.server_object_id,
            captured_at=submission.captured_at,
            selected_rank=submission.selected_rank,
            pose_score=submission.pose_score,
            coverage_score=submission.coverage_score,
            orientation_score=submission.orientation_score,
            sharpness_score=submission.sharpness_score,
            lighting_score=submission.lighting_score,
            overall_score=submission.overall_score,
            quality_verdict=submission.quality_verdict,
            rejection_reason=submission.rejection_reason,
            image_width=submission.image_width,
            image_height=submission.image_height,
            display_orientation=submission.display_orientation,
            device_camera_metadata_json=submission.device_camera_metadata,
        )
        try:
            db.add(asset)
            db.commit()
            db.refresh(asset)
            return asset
        except Exception:
            db.rollback()
            raise

    def append_camera_result(
        self,
        db: Session,
        owner_user_id: int,
        submission: CameraResultSubmission,
    ) -> CameraResult:
        visit = self._scoped_visit(db, owner_user_id, submission.visit_uuid)
        existing = db.scalar(
            select(CameraResult).where(
                CameraResult.result_uuid == str(submission.result_uuid)
            )
        )
        if existing is not None:
            if (
                existing.visit_id == visit.id
                and existing.version == submission.version
                and existing.estimated_height_cm == submission.estimated_height_cm
                and existing.estimated_weight_kg == submission.estimated_weight_kg
            ):
                return existing
            raise ValueError(
                "Camera results are immutable; append a new version and UUID"
            )

        current_state = CaptureState(visit.capture_state)
        if current_state not in (
            CaptureState.PROCESSING,
            CaptureState.MEASURED_REPORT,
        ):
            raise ValueError(
                "A camera result can be appended only while processing or "
                "reprocessing a measured visit"
            )
        latest_version = db.scalar(
            select(func.max(CameraResult.version)).where(
                CameraResult.visit_id == visit.id
            )
        )
        expected_version = 1 if latest_version is None else latest_version + 1
        if submission.version != expected_version:
            raise ValueError(
                f"Camera result version must be {expected_version} for this visit"
            )

        result = CameraResult(
            result_uuid=str(submission.result_uuid),
            visit_id=visit.id,
            version=submission.version,
            supersedes_result_uuid=(
                str(submission.supersedes_result_uuid)
                if submission.supersedes_result_uuid
                else None
            ),
            estimated_height_cm=submission.estimated_height_cm,
            estimated_weight_kg=submission.estimated_weight_kg,
            height_source=submission.height_source,
            weight_source=submission.weight_source,
            estimated_haz=submission.estimated_haz,
            estimated_whz=submission.estimated_whz,
            estimated_stunting_status=submission.estimated_stunting_status,
            estimated_wasting_status=submission.estimated_wasting_status,
            experimental_overall_category=(
                submission.experimental_overall_category
            ),
            component_probabilities_json=submission.component_probabilities,
            body_proportion_features_json=submission.body_proportion_features,
            capture_quality_summary_json=submission.capture_quality_summary,
            method=submission.method,
            model_version=submission.model_version,
            manifest_checksum=submission.manifest_checksum,
            training_data_label=submission.training_data_label,
            non_clinical=True,
            created_at=submission.created_at,
        )
        if current_state == CaptureState.PROCESSING:
            visit.capture_state = CaptureState.ESTIMATED_REPORT.value
        try:
            db.add(result)
            db.commit()
            db.refresh(result)
            return result
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def _validate_measured_ranges(details: MeasuredDetailsSubmission) -> None:
        ranges = (
            (
                "height_cm",
                details.height_cm,
                MEASURED_HEIGHT_MIN_CM,
                MEASURED_HEIGHT_MAX_CM,
            ),
            (
                "weight_kg",
                details.weight_kg,
                MEASURED_WEIGHT_MIN_KG,
                MEASURED_WEIGHT_MAX_KG,
            ),
            (
                "muac_cm",
                details.muac_cm,
                MEASURED_MUAC_MIN_CM,
                MEASURED_MUAC_MAX_CM,
            ),
        )
        for name, value, minimum, maximum in ranges:
            if value is not None and (
                not math.isfinite(value) or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{name} must be between {minimum} and {maximum}"
                )

    @staticmethod
    def _measurement_snapshot(measurement: MeasurementResult | None) -> dict:
        if measurement is None:
            return {}
        return {
            "height_cm": measurement.manual_height_cm,
            "weight_kg": measurement.manual_weight_kg,
            "muac_cm": measurement.muac_cm,
            "measurement_mode": measurement.measurement_mode,
            "oedema": measurement.oedema,
            "measured_at": (
                measurement.measured_at.isoformat()
                if measurement.measured_at
                else None
            ),
            "notes": measurement.measured_notes,
            "haz_zscore": measurement.haz_zscore,
            "whz_zscore": measurement.whz_zscore,
            "haz_status": measurement.haz_status,
            "whz_status": measurement.whz_status,
            "who_acute_status": measurement.who_acute_status,
            "poshan_status": measurement.poshan_status,
        }

    def save_measured_details(
        self,
        db: Session,
        *,
        owner_user_id: int,
        visit_uuid: UUID | str,
        measurement_date: date,
        details: MeasuredDetailsSubmission,
        editor_user_id: int,
    ) -> MeasurementResult:
        visit = self._scoped_visit(db, owner_user_id, visit_uuid)
        if editor_user_id != owner_user_id:
            raise ValueError("Measured-detail editor must match the visit owner")
        if visit.visit_date.date() != measurement_date:
            raise ValueError(
                "The measurement date must match the visit clinical date"
            )
        child = db.scalar(
            select(Child).where(
                Child.id == visit.child_id,
                Child.user_id == owner_user_id,
            )
        )
        if child is None:
            raise LookupError("Owner-scoped child was not found")
        self._validate_measured_ranges(details)
        age = self._age.validate_clinical_age(
            child.date_of_birth,
            measurement_date,
        )
        current_state = CaptureState(visit.capture_state)
        if current_state not in (
            CaptureState.ESTIMATED_REPORT,
            CaptureState.MEASURED_REPORT,
        ):
            raise ValueError(
                "Measured details require an estimated or measured visit"
            )

        acute = self._acute.assess(
            sex=child.sex,
            completed_age_months=age.completed_months,
            age_months=age.months,
            height_cm=details.height_cm,
            weight_kg=details.weight_kg,
            tape_muac_cm=details.muac_cm,
            oedema=details.oedema.value,
        )
        poshan = classify_poshan_setu(
            sex=child.sex,
            age_months=age.months,
            weight_kg=details.weight_kg,
            height_cm=details.height_cm,
            weight_source="manual" if details.weight_kg is not None else None,
            height_source="manual" if details.height_cm is not None else None,
            muac_cm=details.muac_cm,
            muac_method="tape" if details.muac_cm is not None else None,
        )

        try:
            measurement = db.scalar(
                select(MeasurementResult).where(
                    MeasurementResult.visit_id == visit.id
                )
            )
            before = self._measurement_snapshot(measurement)
            if measurement is None:
                measurement = MeasurementResult(visit_id=visit.id)
                db.add(measurement)

            measurement.manual_height_cm = details.height_cm
            measurement.manual_weight_kg = details.weight_kg
            measurement.effective_height_cm = details.height_cm
            measurement.effective_weight_kg = details.weight_kg
            measurement.height_method = (
                "manual" if details.height_cm is not None else "unavailable"
            )
            measurement.weight_method = (
                "manual" if details.weight_kg is not None else "unavailable"
            )
            measurement.estimation_method = "manual"
            measurement.measurement_mode = details.measurement_mode.value
            measurement.oedema = details.oedema.value
            measurement.measured_at = details.measured_at
            measurement.editor_user_id = editor_user_id
            measurement.measured_notes = details.notes
            measurement.haz_zscore = acute.haz_zscore
            measurement.whz_zscore = acute.whz_zscore
            measurement.haz_status = acute.stunting_status
            measurement.whz_status = (
                acute.whz_status.value if acute.whz_status else None
            )
            measurement.muac_cm = details.muac_cm
            measurement.muac_status = (
                acute.muac_status.value if acute.muac_status else None
            )
            measurement.muac_method = (
                "manual" if details.muac_cm is not None else "unavailable"
            )
            measurement.muac_age_in_range = acute.muac_eligible
            measurement.muac_is_direct_measurement = details.muac_cm is not None
            measurement.who_acute_status = acute.acute_status.value
            measurement.who_acute_triggered_by = json.dumps(
                acute.triggered_by
            )
            measurement.who_acute_rationale = acute.rationale
            measurement.combined_status = acute.acute_status.value
            measurement.combined_triggered_by = json.dumps(acute.triggered_by)
            measurement.combined_rationale = acute.rationale
            measurement.combined_method = "who_measured_whz_muac_oedema_v1"
            measurement.poshan_status = poshan.final_status
            measurement.poshan_triggered_by = json.dumps(poshan.triggered_by)
            measurement.classification_method = poshan.classification_method
            measurement.classification_rationale = poshan.rationale
            measurement.poshan_complete = poshan.complete
            measurement.bmi = poshan.bmi
            measurement.bmi_status = poshan.bmi_status

            after = self._measurement_snapshot(measurement)
            latest_revision = db.scalar(
                select(func.max(MeasuredDetailRevision.revision_number)).where(
                    MeasuredDetailRevision.visit_id == visit.id
                )
            )
            revision = MeasuredDetailRevision(
                revision_uuid=str(uuid4()),
                visit_id=visit.id,
                revision_number=(latest_revision or 0) + 1,
                before_json=before,
                after_json=after,
                editor_user_id=editor_user_id,
                created_at=datetime.utcnow(),
                reason=details.reason,
            )
            db.add(revision)
            visit.capture_state = CaptureState.MEASURED_REPORT.value
            db.commit()
            db.refresh(measurement)
            return measurement
        except Exception:
            db.rollback()
            raise

    def delete_visit_media(
        self,
        db: Session,
        owner_user_id: int,
        visit_uuid: UUID | str,
    ) -> Visit:
        visit = self._scoped_visit(db, owner_user_id, visit_uuid)
        unacknowledged = [
            asset.asset_uuid
            for asset in visit.capture_assets
            if asset.server_acknowledged_at is None
        ]
        if unacknowledged:
            raise ValueError(
                "Cannot delete local media until every asset UUID is acknowledged"
            )
        try:
            for asset in visit.capture_assets:
                asset.local_path = None
            visit.media_deleted_at = datetime.utcnow()
            db.commit()
            db.refresh(visit)
            return visit
        except Exception:
            db.rollback()
            raise
