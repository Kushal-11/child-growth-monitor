"""Owner-scoped lifecycle controls for retained guided-capture media."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.capture_asset import CaptureAsset
from app.models.visit import Visit


class GuidedMediaNotFound(LookupError):
    """The owner-scoped visit or media asset does not exist."""


class GuidedMediaConflict(ValueError):
    """Media cannot be deleted without violating retention guarantees."""


@dataclass(frozen=True)
class GuidedMediaDeletion:
    asset_uuid: UUID
    status: str
    server_id: int
    server_object_id: str | None


class GuidedMediaService:
    """Delete selected bytes while retaining the assessment audit record."""

    def __init__(self, *, media_root: Path):
        self._media_root = Path(media_root)

    def delete_asset_media(
        self,
        db: Session,
        *,
        owner_user_id: int,
        visit_uuid: UUID,
        asset_uuid: UUID,
    ) -> GuidedMediaDeletion:
        visit = db.scalar(
            select(Visit).where(
                Visit.local_uuid == str(visit_uuid),
                Visit.user_id == owner_user_id,
            )
        )
        if visit is None:
            raise GuidedMediaNotFound("Owner-scoped visit was not found")
        asset = db.scalar(
            select(CaptureAsset).where(
                CaptureAsset.visit_id == visit.id,
                CaptureAsset.asset_uuid == str(asset_uuid),
            )
        )
        if asset is None:
            raise GuidedMediaNotFound("Owner-scoped asset was not found")
        if asset.local_path is None and asset.sync_state == "media_deleted":
            return GuidedMediaDeletion(
                asset_uuid=asset_uuid,
                status="already_deleted",
                server_id=asset.id,
                server_object_id=asset.server_object_id,
            )
        if asset.server_acknowledged_at is None or asset.sync_state != "synced":
            raise GuidedMediaConflict(
                "Asset upload is not acknowledged; deletion is blocked "
                "until durable receipt is confirmed"
            )
        if asset.local_path is None:
            asset.sync_state = "media_deleted"
            db.commit()
            return GuidedMediaDeletion(
                asset_uuid=asset_uuid,
                status="already_deleted",
                server_id=asset.id,
                server_object_id=asset.server_object_id,
            )

        path = Path(asset.local_path)
        root = self._media_root.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise GuidedMediaConflict(
                "Asset path is outside guided media storage"
            )
        if path.exists():
            path.unlink()
        asset.local_path = None
        asset.sync_state = "media_deleted"
        if all(candidate.local_path is None for candidate in visit.capture_assets):
            visit.media_deleted_at = datetime.now(timezone.utc).replace(
                tzinfo=None
            )
        db.commit()
        return GuidedMediaDeletion(
            asset_uuid=asset_uuid,
            status="deleted",
            server_id=asset.id,
            server_object_id=asset.server_object_id,
        )
