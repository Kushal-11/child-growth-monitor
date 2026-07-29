"""Owner-scoped, idempotent guided-capture synchronization endpoints."""

from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.user import User
from app.schemas.guided_sync import (
    GuidedAssetSyncRequest,
    GuidedCameraResultSyncRequest,
    GuidedMeasuredRevisionSyncRequest,
    GuidedSyncAcknowledgement,
    GuidedVisitSyncRequest,
)
from app.services.auth_service import get_current_user
from app.services.guided_sync_service import (
    GuidedSyncConflict,
    GuidedSyncNotFound,
    GuidedSyncService,
    GuidedSyncValidation,
)
from app.services.who_data_service import WHODataService
from config import GUIDED_CAPTURE_MEDIA_DIR


router = APIRouter(prefix="/api/v1/sync/guided", tags=["Guided Sync"])


@lru_cache(maxsize=1)
def get_guided_sync_service() -> GuidedSyncService:
    who = WHODataService()
    who.load_all()
    return GuidedSyncService(
        media_root=GUIDED_CAPTURE_MEDIA_DIR,
        who_data=who,
    )


def _call(operation):
    try:
        return operation()
    except GuidedSyncNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except GuidedSyncConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except GuidedSyncValidation as exc:
        raise HTTPException(422, str(exc)) from exc


@router.put(
    "/visits/{visit_uuid}",
    response_model=GuidedSyncAcknowledgement,
)
def sync_visit(
    visit_uuid: UUID,
    body: GuidedVisitSyncRequest,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    service: GuidedSyncService = Depends(get_guided_sync_service),
):
    return _call(
        lambda: service.sync_visit(
            db,
            owner_user_id=current.id,
            visit_uuid=visit_uuid,
            body=body,
        )
    )


@router.put(
    "/visits/{visit_uuid}/assets/{asset_uuid}",
    response_model=GuidedSyncAcknowledgement,
)
def sync_asset(
    visit_uuid: UUID,
    asset_uuid: UUID,
    body: GuidedAssetSyncRequest,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    service: GuidedSyncService = Depends(get_guided_sync_service),
):
    return _call(
        lambda: service.sync_asset(
            db,
            owner_user_id=current.id,
            visit_uuid=visit_uuid,
            asset_uuid=asset_uuid,
            body=body,
        )
    )


@router.put(
    "/visits/{visit_uuid}/camera-results/{result_uuid}",
    response_model=GuidedSyncAcknowledgement,
)
def sync_camera_result(
    visit_uuid: UUID,
    result_uuid: UUID,
    body: GuidedCameraResultSyncRequest,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    service: GuidedSyncService = Depends(get_guided_sync_service),
):
    return _call(
        lambda: service.sync_camera_result(
            db,
            owner_user_id=current.id,
            visit_uuid=visit_uuid,
            result_uuid=result_uuid,
            body=body,
        )
    )


@router.put(
    "/visits/{visit_uuid}/measured-revisions/{revision_uuid}",
    response_model=GuidedSyncAcknowledgement,
)
def sync_measured_revision(
    visit_uuid: UUID,
    revision_uuid: UUID,
    body: GuidedMeasuredRevisionSyncRequest,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    service: GuidedSyncService = Depends(get_guided_sync_service),
):
    return _call(
        lambda: service.sync_measured_revision(
            db,
            owner_user_id=current.id,
            visit_uuid=visit_uuid,
            revision_uuid=revision_uuid,
            body=body,
        )
    )


@router.delete(
    "/visits/{visit_uuid}/media/{asset_uuid}",
    response_model=GuidedSyncAcknowledgement,
)
def delete_media(
    visit_uuid: UUID,
    asset_uuid: UUID,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    service: GuidedSyncService = Depends(get_guided_sync_service),
):
    return _call(
        lambda: service.delete_media(
            db,
            owner_user_id=current.id,
            visit_uuid=visit_uuid,
            asset_uuid=asset_uuid,
        )
    )
