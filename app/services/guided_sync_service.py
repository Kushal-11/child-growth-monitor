"""Idempotent, owner-scoped ingestion for guided-capture entities."""

import base64
import binascii
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.camera_result import CameraResult
from app.models.capture_asset import CaptureAsset
from app.models.child import Child
from app.models.measured_detail_revision import MeasuredDetailRevision
from app.models.visit import Visit
from app.schemas.guided_capture import MeasuredDetailsSubmission
from app.schemas.guided_sync import (
    GuidedAssetSyncRequest,
    GuidedCameraResultSyncRequest,
    GuidedMeasuredRevisionSyncRequest,
    GuidedSyncAcknowledgement,
    GuidedVisitSyncRequest,
)
from app.services.age_service import AgeService
from app.services.guided_capture_contract import CaptureState
from app.services.guided_visit_service import GuidedVisitService
from app.services.who_data_service import WHODataService
from config import GUIDED_CAPTURE_MAX_ASSET_BYTES


class GuidedSyncNotFound(LookupError):
    """The owner-scoped aggregate or entity does not exist."""


class GuidedSyncConflict(ValueError):
    """An immutable UUID/checksum or ordering contract was violated."""


class GuidedSyncValidation(ValueError):
    """A request is structurally valid but violates aggregate rules."""


class GuidedSyncService:
    def __init__(self, *, media_root: Path, who_data: WHODataService):
        self._media_root = Path(media_root)
        self._visits = GuidedVisitService(who_data)
        self._age = AgeService()

    def asset_path(
        self,
        owner_user_id: int,
        visit_uuid: UUID | str,
        asset_uuid: UUID | str,
        content_type: str,
    ) -> Path:
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }[content_type]
        return (
            self._media_root
            / str(owner_user_id)
            / str(visit_uuid)
            / f"{asset_uuid}{extension}"
        )

    def sync_visit(
        self,
        db: Session,
        *,
        owner_user_id: int,
        visit_uuid: UUID,
        body: GuidedVisitSyncRequest,
    ) -> GuidedSyncAcknowledgement:
        self._require_uuid_match(visit_uuid, body.local_uuid, "visit")
        child = db.scalar(
            select(Child).where(
                Child.id == body.child_id,
                Child.user_id == owner_user_id,
            )
        )
        if child is None:
            raise GuidedSyncNotFound("Owner-scoped child was not found")
        try:
            age = self._age.validate_clinical_age(
                child.date_of_birth,
                body.visit_date.date(),
            )
        except ValueError as exc:
            raise GuidedSyncValidation(str(exc)) from exc
        if abs(age.months - body.age_months) > 0.5:
            raise GuidedSyncValidation(
                "age_months is inconsistent with child DOB and visit date"
            )

        existing = self._scoped_visit_or_none(
            db,
            owner_user_id,
            visit_uuid,
        )
        checksum = self._checksum(self._visit_immutable_from_body(body))
        if existing is not None:
            stored_checksum = self._checksum(
                self._visit_immutable_from_row(existing)
            )
            if stored_checksum != checksum:
                raise GuidedSyncConflict(
                    "Visit UUID already has a different immutable payload"
                )
            if (
                body.capture_state == CaptureState.INCOMPLETE_CAPTURE
                and existing.capture_state == CaptureState.DRAFT_CAPTURE.value
            ):
                existing.capture_state = CaptureState.INCOMPLETE_CAPTURE.value
                existing.capture_completed_at = self._naive_utc(
                    body.capture_completed_at
                )
                db.commit()
            return self._ack(
                "visit",
                visit_uuid,
                "already_accepted",
                server_id=existing.id,
                checksum=checksum,
            )

        initial_state = (
            CaptureState.INCOMPLETE_CAPTURE.value
            if body.capture_state == CaptureState.INCOMPLETE_CAPTURE
            else CaptureState.DRAFT_CAPTURE.value
        )
        visit = Visit(
            child_id=child.id,
            user_id=owner_user_id,
            local_uuid=str(visit_uuid),
            visit_date=self._naive_utc(body.visit_date),
            age_months=age.months,
            entry_method="guided_capture",
            capture_state=initial_state,
            capture_started_at=self._naive_utc(body.capture_started_at),
            capture_completed_at=self._naive_utc(body.capture_completed_at),
            device_metadata_json=body.device_metadata,
            consent_version=body.consent_version,
            consent_timestamp=self._naive_utc(body.consent_timestamp),
            consent_operator_identifier=body.consent_operator_identifier,
        )
        try:
            db.add(visit)
            db.commit()
            db.refresh(visit)
        except IntegrityError as exc:
            db.rollback()
            raise GuidedSyncConflict(
                "Visit UUID already identifies another immutable payload"
            ) from exc
        return self._ack(
            "visit",
            visit_uuid,
            "accepted",
            server_id=visit.id,
            checksum=checksum,
        )

    def sync_asset(
        self,
        db: Session,
        *,
        owner_user_id: int,
        visit_uuid: UUID,
        asset_uuid: UUID,
        body: GuidedAssetSyncRequest,
    ) -> GuidedSyncAcknowledgement:
        self._require_uuid_match(visit_uuid, body.visit_uuid, "visit")
        self._require_uuid_match(asset_uuid, body.asset_uuid, "asset")
        visit = self._scoped_visit(db, owner_user_id, visit_uuid)
        content = self._decode_content(body)
        path = self.asset_path(
            owner_user_id,
            visit_uuid,
            asset_uuid,
            body.content_type,
        )
        existing = db.scalar(
            select(CaptureAsset).where(
                CaptureAsset.asset_uuid == str(asset_uuid),
                CaptureAsset.visit_id == visit.id,
            )
        )
        request_checksum = self._checksum(
            self._asset_immutable_from_body(body)
        )
        if existing is not None:
            stored_checksum = self._checksum(
                self._asset_immutable_from_row(existing, body.content_type)
            )
            if stored_checksum != request_checksum:
                raise GuidedSyncConflict(
                    "Asset UUID already has a different immutable payload"
                )
            self._persist_asset_bytes(path, content, body.content_checksum)
            existing.local_path = str(path)
            existing.server_object_id = self._object_id(path)
            existing.sync_state = "synced"
            existing.server_acknowledged_at = (
                existing.server_acknowledged_at or self._now()
            )
            db.commit()
            return self._ack(
                "capture_asset",
                asset_uuid,
                "already_accepted",
                server_id=existing.id,
                server_object_id=existing.server_object_id,
                checksum=body.content_checksum,
            )

        self._persist_asset_bytes(path, content, body.content_checksum)
        asset = CaptureAsset(
            asset_uuid=str(asset_uuid),
            visit_id=visit.id,
            role=body.role.value,
            local_path=str(path),
            server_object_id=self._object_id(path),
            captured_at=self._naive_utc(body.captured_at),
            selected_rank=body.selected_rank,
            pose_score=body.quality.pose,
            coverage_score=body.quality.coverage,
            orientation_score=body.quality.orientation,
            sharpness_score=body.quality.sharpness,
            lighting_score=body.quality.lighting,
            overall_score=body.quality.overall,
            quality_verdict="accepted",
            quality_threshold_version=body.quality.threshold_version,
            image_width=body.image_width,
            image_height=body.image_height,
            exif_orientation=body.exif_orientation,
            display_orientation=body.display_orientation,
            device_camera_metadata_json=body.device_camera_metadata,
            sync_state="synced",
            server_acknowledged_at=self._now(),
        )
        try:
            db.add(asset)
            db.flush()
            if self._required_assets_acknowledged(visit):
                if visit.capture_state in {
                    CaptureState.DRAFT_CAPTURE.value,
                    CaptureState.INCOMPLETE_CAPTURE.value,
                    CaptureState.PROCESSING_FAILED.value,
                }:
                    visit.capture_state = CaptureState.PROCESSING.value
                    visit.capture_completed_at = visit.capture_completed_at or self._now()
            db.commit()
            db.refresh(asset)
        except IntegrityError as exc:
            db.rollback()
            raise GuidedSyncConflict(
                "Asset UUID or visit role already conflicts"
            ) from exc
        return self._ack(
            "capture_asset",
            asset_uuid,
            "accepted",
            server_id=asset.id,
            server_object_id=asset.server_object_id,
            checksum=body.content_checksum,
        )

    def sync_camera_result(
        self,
        db: Session,
        *,
        owner_user_id: int,
        visit_uuid: UUID,
        result_uuid: UUID,
        body: GuidedCameraResultSyncRequest,
    ) -> GuidedSyncAcknowledgement:
        self._require_uuid_match(visit_uuid, body.visit_uuid, "visit")
        self._require_uuid_match(result_uuid, body.result_uuid, "result")
        visit = self._scoped_visit(db, owner_user_id, visit_uuid)
        existing = db.scalar(
            select(CameraResult).where(
                CameraResult.result_uuid == str(result_uuid),
                CameraResult.visit_id == visit.id,
            )
        )
        request_checksum = self._checksum(
            self._camera_immutable_from_body(body)
        )
        if existing is not None:
            stored_checksum = self._checksum(
                self._camera_immutable_from_row(existing, visit_uuid)
            )
            if stored_checksum != request_checksum:
                raise GuidedSyncConflict(
                    "Camera result UUID already has a different immutable payload"
                )
            return self._ack(
                "camera_result",
                result_uuid,
                "already_accepted",
                server_id=existing.id,
                checksum=request_checksum,
            )
        if not self._required_assets_acknowledged(visit):
            raise GuidedSyncConflict(
                "Required front and side assets must be acknowledged first"
            )
        duplicate_version = db.scalar(
            select(CameraResult).where(
                CameraResult.visit_id == visit.id,
                CameraResult.version == body.version,
            )
        )
        if duplicate_version is not None:
            raise GuidedSyncConflict(
                "Camera result version already belongs to another UUID"
            )
        if visit.capture_state not in {
            CaptureState.PROCESSING.value,
            CaptureState.MEASURED_REPORT.value,
        }:
            visit.capture_state = CaptureState.PROCESSING.value
            db.flush()
        try:
            result = self._visits.append_camera_result(
                db,
                owner_user_id,
                body,
            )
        except (ValueError, IntegrityError) as exc:
            db.rollback()
            raise GuidedSyncConflict(str(exc)) from exc
        return self._ack(
            "camera_result",
            result_uuid,
            "accepted",
            server_id=result.id,
            checksum=request_checksum,
        )

    def sync_measured_revision(
        self,
        db: Session,
        *,
        owner_user_id: int,
        visit_uuid: UUID,
        revision_uuid: UUID,
        body: GuidedMeasuredRevisionSyncRequest,
    ) -> GuidedSyncAcknowledgement:
        self._require_uuid_match(visit_uuid, body.visit_uuid, "visit")
        self._require_uuid_match(revision_uuid, body.revision_uuid, "revision")
        visit = self._scoped_visit(db, owner_user_id, visit_uuid)
        request_inputs = self._measured_inputs_from_snapshot(body.after)
        request_checksum = self._checksum(
            {
                "revision_number": body.revision_number,
                "after": request_inputs,
                "reason": body.reason,
            }
        )
        existing = db.scalar(
            select(MeasuredDetailRevision).where(
                MeasuredDetailRevision.revision_uuid == str(revision_uuid),
                MeasuredDetailRevision.visit_id == visit.id,
            )
        )
        if existing is not None:
            stored_checksum = self._checksum(
                {
                    "revision_number": existing.revision_number,
                    "after": self._measured_inputs_from_mapping(
                        existing.after_json
                    ),
                    "reason": existing.reason,
                }
            )
            if stored_checksum != request_checksum:
                raise GuidedSyncConflict(
                    "Measured revision UUID has a different immutable payload"
                )
            return self._ack(
                "measured_revision",
                revision_uuid,
                "already_accepted",
                server_id=existing.id,
                checksum=request_checksum,
            )
        duplicate_number = db.scalar(
            select(MeasuredDetailRevision).where(
                MeasuredDetailRevision.visit_id == visit.id,
                MeasuredDetailRevision.revision_number == body.revision_number,
            )
        )
        if duplicate_number is not None:
            raise GuidedSyncConflict(
                "Measured revision number already belongs to another UUID"
            )
        after = body.after
        if (
            after.measurement_mode is None
            or after.oedema is None
            or after.measured_at is None
        ):
            raise GuidedSyncValidation(
                "Measured revision requires mode, oedema, and measured_at"
            )
        if (
            after.height_cm is None
            and after.weight_kg is None
            and after.muac_cm is None
            and after.oedema.value != "yes"
        ):
            raise GuidedSyncValidation(
                "Measured revision must contain a measurement or oedema"
            )
        details = MeasuredDetailsSubmission(
            measurement_mode=after.measurement_mode,
            oedema=after.oedema,
            height_cm=after.height_cm,
            weight_kg=after.weight_kg,
            muac_cm=after.muac_cm,
            measured_at=after.measured_at,
            notes=after.notes,
            reason=body.reason,
        )
        try:
            self._visits.save_measured_details(
                db,
                owner_user_id=owner_user_id,
                visit_uuid=visit_uuid,
                measurement_date=visit.visit_date.date(),
                details=details,
                editor_user_id=owner_user_id,
                revision_uuid=revision_uuid,
                revision_number=body.revision_number,
                revision_created_at=self._naive_utc(body.created_at),
                allow_without_estimate=True,
            )
        except ValueError as exc:
            db.rollback()
            message = str(exc)
            if "revision number" in message:
                raise GuidedSyncConflict(message) from exc
            raise GuidedSyncValidation(message) from exc
        revision = db.scalar(
            select(MeasuredDetailRevision).where(
                MeasuredDetailRevision.revision_uuid == str(revision_uuid)
            )
        )
        return self._ack(
            "measured_revision",
            revision_uuid,
            "accepted",
            server_id=revision.id,
            checksum=request_checksum,
        )

    def delete_media(
        self,
        db: Session,
        *,
        owner_user_id: int,
        visit_uuid: UUID,
        asset_uuid: UUID,
    ) -> GuidedSyncAcknowledgement:
        visit = self._scoped_visit(db, owner_user_id, visit_uuid)
        asset = db.scalar(
            select(CaptureAsset).where(
                CaptureAsset.asset_uuid == str(asset_uuid),
                CaptureAsset.visit_id == visit.id,
            )
        )
        if asset is None:
            raise GuidedSyncNotFound("Owner-scoped asset was not found")
        if asset.local_path is None:
            return self._ack(
                "media_deletion",
                asset_uuid,
                "already_accepted",
                server_id=asset.id,
                server_object_id=asset.server_object_id,
            )
        path = Path(asset.local_path)
        root = self._media_root.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise GuidedSyncConflict("Asset path is outside guided media storage")
        if path.exists():
            path.unlink()
        asset.local_path = None
        if all(
            candidate.local_path is None
            for candidate in visit.capture_assets
        ):
            visit.media_deleted_at = self._now()
        db.commit()
        return self._ack(
            "media_deletion",
            asset_uuid,
            "accepted",
            server_id=asset.id,
            server_object_id=asset.server_object_id,
        )

    def _scoped_visit(
        self,
        db: Session,
        owner_user_id: int,
        visit_uuid: UUID | str,
    ) -> Visit:
        visit = self._scoped_visit_or_none(db, owner_user_id, visit_uuid)
        if visit is None:
            raise GuidedSyncNotFound("Owner-scoped visit was not found")
        return visit

    @staticmethod
    def _scoped_visit_or_none(
        db: Session,
        owner_user_id: int,
        visit_uuid: UUID | str,
    ) -> Visit | None:
        return db.scalar(
            select(Visit).where(
                Visit.local_uuid == str(visit_uuid),
                Visit.user_id == owner_user_id,
            )
        )

    @staticmethod
    def _require_uuid_match(path_uuid: UUID, body_uuid: UUID, name: str) -> None:
        if path_uuid != body_uuid:
            raise GuidedSyncValidation(
                f"{name} UUID in path and payload must match"
            )

    @staticmethod
    def _decode_content(body: GuidedAssetSyncRequest) -> bytes:
        try:
            content = base64.b64decode(body.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise GuidedSyncValidation("Asset content is not valid base64") from exc
        if len(content) > GUIDED_CAPTURE_MAX_ASSET_BYTES:
            raise GuidedSyncValidation("Asset exceeds the configured size limit")
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != body.content_checksum:
            raise GuidedSyncConflict(
                "Asset content checksum does not match the submitted checksum"
            )
        return content

    def _persist_asset_bytes(
        self,
        path: Path,
        content: bytes,
        checksum: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing_checksum != checksum:
                raise GuidedSyncConflict(
                    "Stored asset bytes conflict with the submitted checksum"
                )
            return
        temporary = path.with_suffix(f"{path.suffix}.part")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    def _object_id(self, path: Path) -> str:
        return path.relative_to(self._media_root).as_posix()

    @staticmethod
    def _required_assets_acknowledged(visit: Visit) -> bool:
        acknowledged = {
            asset.role
            for asset in visit.capture_assets
            if asset.quality_verdict == "accepted"
            and asset.server_acknowledged_at is not None
        }
        return {"front", "side"}.issubset(acknowledged)

    @staticmethod
    def _visit_immutable_from_body(body: GuidedVisitSyncRequest) -> dict:
        return {
            "child_id": body.child_id,
            "visit_date": body.visit_date.date().isoformat(),
            "device_metadata": body.device_metadata,
            "consent_version": body.consent_version,
            "consent_timestamp": GuidedSyncService._iso_utc(
                body.consent_timestamp
            ),
            "consent_operator_identifier": body.consent_operator_identifier,
        }

    @staticmethod
    def _visit_immutable_from_row(visit: Visit) -> dict:
        return {
            "child_id": visit.child_id,
            "visit_date": visit.visit_date.date().isoformat(),
            "device_metadata": visit.device_metadata_json or {},
            "consent_version": visit.consent_version,
            "consent_timestamp": GuidedSyncService._iso_utc(
                visit.consent_timestamp
            ),
            "consent_operator_identifier": visit.consent_operator_identifier,
        }

    @staticmethod
    def _asset_immutable_from_body(body: GuidedAssetSyncRequest) -> dict:
        return {
            "role": body.role.value,
            "captured_at": GuidedSyncService._iso_utc(body.captured_at),
            "selected_rank": body.selected_rank,
            "quality": body.quality.model_dump(mode="json"),
            "image_width": body.image_width,
            "image_height": body.image_height,
            "exif_orientation": body.exif_orientation,
            "display_orientation": body.display_orientation,
            "device_camera_metadata": body.device_camera_metadata,
            "content_checksum": body.content_checksum,
        }

    @staticmethod
    def _asset_immutable_from_row(
        asset: CaptureAsset,
        content_type: str,
    ) -> dict:
        content_checksum = None
        if asset.local_path and Path(asset.local_path).exists():
            content_checksum = hashlib.sha256(
                Path(asset.local_path).read_bytes()
            ).hexdigest()
        return {
            "role": asset.role,
            "captured_at": GuidedSyncService._iso_utc(asset.captured_at),
            "selected_rank": asset.selected_rank,
            "quality": {
                "pose": asset.pose_score,
                "coverage": asset.coverage_score,
                "orientation": asset.orientation_score,
                "sharpness": asset.sharpness_score,
                "lighting": asset.lighting_score,
                "overall": asset.overall_score,
                "threshold_version": asset.quality_threshold_version,
            },
            "image_width": asset.image_width,
            "image_height": asset.image_height,
            "exif_orientation": asset.exif_orientation,
            "display_orientation": asset.display_orientation,
            "device_camera_metadata": asset.device_camera_metadata_json or {},
            "content_checksum": content_checksum,
        }

    @staticmethod
    def _camera_immutable_from_body(
        body: GuidedCameraResultSyncRequest,
    ) -> dict:
        payload = body.model_dump(mode="json")
        payload["result_uuid"] = str(body.result_uuid)
        payload["visit_uuid"] = str(body.visit_uuid)
        payload["supersedes_result_uuid"] = (
            str(body.supersedes_result_uuid)
            if body.supersedes_result_uuid is not None
            else None
        )
        payload["created_at"] = GuidedSyncService._iso_utc(body.created_at)
        return payload

    @staticmethod
    def _camera_immutable_from_row(
        result: CameraResult,
        visit_uuid: UUID,
    ) -> dict:
        return {
            "result_uuid": result.result_uuid,
            "visit_uuid": str(visit_uuid),
            "version": result.version,
            "supersedes_result_uuid": result.supersedes_result_uuid,
            "estimated_height_cm": result.estimated_height_cm,
            "estimated_weight_kg": result.estimated_weight_kg,
            "estimated_haz": result.estimated_haz,
            "estimated_whz": result.estimated_whz,
            "estimated_stunting_status": result.estimated_stunting_status,
            "estimated_wasting_status": result.estimated_wasting_status,
            "experimental_overall_category": (
                result.experimental_overall_category
            ),
            "height_source": result.height_source,
            "weight_source": result.weight_source,
            "component_probabilities": result.component_probabilities_json or {},
            "body_proportion_features": (
                result.body_proportion_features_json or {}
            ),
            "capture_quality_summary": (
                result.capture_quality_summary_json or {}
            ),
            "method": result.method,
            "model_version": result.model_version,
            "manifest_checksum": result.manifest_checksum,
            "training_data_label": result.training_data_label,
            "non_clinical": True,
            "created_at": GuidedSyncService._iso_utc(result.created_at),
        }

    @staticmethod
    def _measured_inputs_from_snapshot(snapshot) -> dict:
        return {
            "height_cm": snapshot.height_cm,
            "weight_kg": snapshot.weight_kg,
            "muac_cm": snapshot.muac_cm,
            "measurement_mode": (
                snapshot.measurement_mode.value
                if snapshot.measurement_mode is not None
                else None
            ),
            "oedema": snapshot.oedema.value if snapshot.oedema is not None else None,
            "measured_at": GuidedSyncService._iso_utc(snapshot.measured_at),
            "notes": snapshot.notes,
        }

    @staticmethod
    def _measured_inputs_from_mapping(snapshot: dict[str, Any]) -> dict:
        return {
            "height_cm": snapshot.get("height_cm"),
            "weight_kg": snapshot.get("weight_kg"),
            "muac_cm": snapshot.get("muac_cm"),
            "measurement_mode": snapshot.get("measurement_mode"),
            "oedema": snapshot.get("oedema"),
            "measured_at": GuidedSyncService._iso_utc(
                snapshot.get("measured_at")
            ),
            "notes": snapshot.get("notes"),
        }

    @staticmethod
    def _checksum(value: dict[str, Any]) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _naive_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _iso_utc(value: datetime | str | None) -> str | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
        normalized = GuidedSyncService._naive_utc(parsed)
        return normalized.isoformat()

    @staticmethod
    def _ack(
        entity_type: str,
        entity_uuid: UUID,
        status: str,
        *,
        server_id: int | None = None,
        server_object_id: str | None = None,
        checksum: str | None = None,
    ) -> GuidedSyncAcknowledgement:
        return GuidedSyncAcknowledgement(
            entity_type=entity_type,
            entity_uuid=entity_uuid,
            status=status,
            server_id=server_id,
            server_object_id=server_object_id,
            checksum=checksum,
            acknowledged_at=datetime.now(timezone.utc),
        )
