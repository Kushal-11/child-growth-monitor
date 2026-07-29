"""Owner-scoped media deletion without loss of assessment history."""

from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models.camera_result import CameraResult
from app.models.capture_asset import CaptureAsset
from app.models.measured_detail_revision import MeasuredDetailRevision
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from tests.guided_sync_support import (
    build_context,
    camera_payload,
    put_required_assets,
    put_visit,
    revision_payload,
)


@pytest.fixture()
def ctx(tmp_path):
    context = build_context(tmp_path)
    yield context
    context.close()


def test_media_delete_requires_auth_and_hides_cross_owner_asset(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    asset_uuid = "20000000-0000-0000-0000-000000000001"
    assert put_visit(ctx, visit_uuid).status_code == 200
    assert all(r.status_code == 200 for r in put_required_assets(ctx, visit_uuid))
    path = f"/api/v1/guided/visits/{visit_uuid}/media/{asset_uuid}"

    assert ctx.client.delete(path).status_code == 401
    assert (
        ctx.client.delete(path, headers=ctx.other_headers).status_code == 404
    )


def test_pending_asset_deletion_is_blocked_and_bytes_are_retained(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    asset_uuid = "20000000-0000-0000-0000-000000000001"
    assert put_visit(ctx, visit_uuid).status_code == 200
    path = ctx.service.asset_path(
        ctx.owner_id,
        visit_uuid,
        asset_uuid,
        "image/jpeg",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pending-front")
    db = ctx.session_factory()
    try:
        visit = db.scalar(select(Visit).where(Visit.local_uuid == visit_uuid))
        db.add(
            CaptureAsset(
                asset_uuid=asset_uuid,
                visit_id=visit.id,
                role="front",
                local_path=str(path),
                sync_state="pending",
            )
        )
        db.commit()
    finally:
        db.close()

    response = ctx.client.delete(
        f"/api/v1/guided/visits/{visit_uuid}/media/{asset_uuid}",
        headers=ctx.owner_headers,
    )

    assert response.status_code == 409
    assert "acknowledged" in response.json()["detail"].lower()
    assert path.read_bytes() == b"pending-front"


def test_selected_media_is_tombstoned_idempotently_without_history_loss(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    front_uuid = "20000000-0000-0000-0000-000000000001"
    result_uuid = "30000000-0000-0000-0000-000000000001"
    revision_uuid = "40000000-0000-0000-0000-000000000001"
    assert put_visit(ctx, visit_uuid).status_code == 200
    assert all(r.status_code == 200 for r in put_required_assets(ctx, visit_uuid))
    assert (
        ctx.client.put(
            (
                f"/api/v1/sync/guided/visits/{visit_uuid}"
                f"/camera-results/{result_uuid}"
            ),
            json=camera_payload(visit_uuid, result_uuid),
            headers=ctx.owner_headers,
        ).status_code
        == 200
    )
    assert (
        ctx.client.put(
            (
                f"/api/v1/sync/guided/visits/{visit_uuid}"
                f"/measured-revisions/{revision_uuid}"
            ),
            json=revision_payload(visit_uuid, revision_uuid),
            headers=ctx.owner_headers,
        ).status_code
        == 200
    )
    db = ctx.session_factory()
    try:
        front = db.scalar(
            select(CaptureAsset).where(CaptureAsset.asset_uuid == front_uuid)
        )
        side = db.scalar(
            select(CaptureAsset).where(CaptureAsset.role == "side")
        )
        front_path = Path(front.local_path)
        side_path = Path(side.local_path)
        front_metadata = (
            front.role,
            front.server_object_id,
            front.quality_threshold_version,
        )
    finally:
        db.close()
    path = f"/api/v1/guided/visits/{visit_uuid}/media/{front_uuid}"

    first = ctx.client.delete(path, headers=ctx.owner_headers)
    repeated = ctx.client.delete(path, headers=ctx.owner_headers)

    assert first.status_code == 200
    assert first.json()["status"] == "deleted"
    assert first.json()["history_preserved"] is True
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "already_deleted"
    assert not front_path.exists()
    assert side_path.exists()

    db = ctx.session_factory()
    try:
        front = db.scalar(
            select(CaptureAsset).where(CaptureAsset.asset_uuid == front_uuid)
        )
        assert front.local_path is None
        assert front.sync_state == "media_deleted"
        assert (
            front.role,
            front.server_object_id,
            front.quality_threshold_version,
        ) == front_metadata
        assert db.scalar(select(func.count(CaptureAsset.id))) == 2
        assert db.scalar(select(func.count(Visit.id))) == 1
        assert db.scalar(select(func.count(CameraResult.id))) == 1
        assert db.scalar(select(func.count(MeasurementResult.id))) == 1
        assert db.scalar(select(func.count(MeasuredDetailRevision.id))) == 1
    finally:
        db.close()
