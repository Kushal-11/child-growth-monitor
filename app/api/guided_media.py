"""Authenticated media controls for guided-capture visits."""

from functools import lru_cache
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.guided_media_service import (
    GuidedMediaConflict,
    GuidedMediaNotFound,
    GuidedMediaService,
)
from config import GUIDED_CAPTURE_MEDIA_DIR


router = APIRouter(prefix="/api/v1/guided", tags=["Guided Media"])


class GuidedMediaDeletionResponse(BaseModel):
    asset_uuid: UUID
    status: Literal["deleted", "already_deleted"]
    server_id: int
    server_object_id: str | None = None
    metadata_tombstoned: bool = True
    history_preserved: bool = True


@lru_cache(maxsize=1)
def get_guided_media_service() -> GuidedMediaService:
    return GuidedMediaService(media_root=GUIDED_CAPTURE_MEDIA_DIR)


@router.delete(
    "/visits/{visit_uuid}/media/{asset_uuid}",
    response_model=GuidedMediaDeletionResponse,
)
def delete_guided_media(
    visit_uuid: UUID,
    asset_uuid: UUID,
    db: Session = Depends(get_db),
    current: User = Depends(get_current_user),
    service: GuidedMediaService = Depends(get_guided_media_service),
):
    try:
        result = service.delete_asset_media(
            db,
            owner_user_id=current.id,
            visit_uuid=visit_uuid,
            asset_uuid=asset_uuid,
        )
    except GuidedMediaNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except GuidedMediaConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    return GuidedMediaDeletionResponse(
        asset_uuid=result.asset_uuid,
        status=result.status,
        server_id=result.server_id,
        server_object_id=result.server_object_id,
    )
