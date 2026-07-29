"""Partial-progress and crash-recovery behavior for guided sync."""

import pytest
from sqlalchemy import func, select

from app.models.measured_detail_revision import MeasuredDetailRevision
from app.models.visit import Visit
from tests.guided_sync_support import (
    asset_payload,
    build_context,
    camera_payload,
    put_visit,
    revision_payload,
)


@pytest.fixture()
def ctx(tmp_path):
    context = build_context(tmp_path)
    yield context
    context.close()


def test_partial_asset_progress_blocks_result_then_resumes(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    front_uuid = "20000000-0000-0000-0000-000000000001"
    side_uuid = "20000000-0000-0000-0000-000000000002"
    result_uuid = "30000000-0000-0000-0000-000000000001"
    assert put_visit(ctx, visit_uuid).status_code == 200
    assert (
        ctx.client.put(
            f"/api/v1/sync/guided/visits/{visit_uuid}/assets/{front_uuid}",
            json=asset_payload(visit_uuid, front_uuid, "front"),
            headers=ctx.owner_headers,
        ).status_code
        == 200
    )

    blocked = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/camera-results/{result_uuid}"
        ),
        json=camera_payload(visit_uuid, result_uuid),
        headers=ctx.owner_headers,
    )
    assert blocked.status_code == 409

    assert (
        ctx.client.put(
            f"/api/v1/sync/guided/visits/{visit_uuid}/assets/{side_uuid}",
            json=asset_payload(visit_uuid, side_uuid, "side"),
            headers=ctx.owner_headers,
        ).status_code
        == 200
    )
    resumed = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/camera-results/{result_uuid}"
        ),
        json=camera_payload(visit_uuid, result_uuid),
        headers=ctx.owner_headers,
    )
    assert resumed.status_code == 200


def test_orphaned_asset_bytes_are_recovered_without_duplication(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    asset_uuid = "20000000-0000-0000-0000-000000000001"
    payload = asset_payload(visit_uuid, asset_uuid, "front")
    assert put_visit(ctx, visit_uuid).status_code == 200
    orphan_path = ctx.service.asset_path(
        ctx.owner_id,
        visit_uuid,
        asset_uuid,
        payload["content_type"],
    )
    orphan_path.parent.mkdir(parents=True, exist_ok=True)
    orphan_path.write_bytes(b"front-asset-bytes")

    response = ctx.client.put(
        f"/api/v1/sync/guided/visits/{visit_uuid}/assets/{asset_uuid}",
        json=payload,
        headers=ctx.owner_headers,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert orphan_path.read_bytes() == b"front-asset-bytes"


def test_measured_revision_can_arrive_before_camera_result(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    revision_uuid = "40000000-0000-0000-0000-000000000001"
    assert put_visit(ctx, visit_uuid).status_code == 200

    response = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/measured-revisions/{revision_uuid}"
        ),
        json=revision_payload(visit_uuid, revision_uuid),
        headers=ctx.owner_headers,
    )

    assert response.status_code == 200
    db = ctx.session_factory()
    try:
        visit = db.scalar(select(Visit).where(Visit.local_uuid == visit_uuid))
        assert visit.capture_state == "measured_report"
    finally:
        db.close()


def test_duplicate_revision_number_with_different_uuid_conflicts(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    first_uuid = "40000000-0000-0000-0000-000000000001"
    second_uuid = "40000000-0000-0000-0000-000000000002"
    assert put_visit(ctx, visit_uuid).status_code == 200
    first = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/measured-revisions/{first_uuid}"
        ),
        json=revision_payload(visit_uuid, first_uuid),
        headers=ctx.owner_headers,
    )
    conflict = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/measured-revisions/{second_uuid}"
        ),
        json=revision_payload(
            visit_uuid,
            second_uuid,
            revision_number=1,
            height_cm=84,
        ),
        headers=ctx.owner_headers,
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    db = ctx.session_factory()
    try:
        assert db.scalar(select(func.count(MeasuredDetailRevision.id))) == 1
    finally:
        db.close()


def test_media_deletion_is_idempotent(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    asset_uuid = "20000000-0000-0000-0000-000000000001"
    assert put_visit(ctx, visit_uuid).status_code == 200
    assert (
        ctx.client.put(
            f"/api/v1/sync/guided/visits/{visit_uuid}/assets/{asset_uuid}",
            json=asset_payload(visit_uuid, asset_uuid, "front"),
            headers=ctx.owner_headers,
        ).status_code
        == 200
    )
    path = f"/api/v1/sync/guided/visits/{visit_uuid}/media/{asset_uuid}"

    first = ctx.client.delete(path, headers=ctx.owner_headers)
    repeated = ctx.client.delete(path, headers=ctx.owner_headers)

    assert first.json()["status"] == "accepted"
    assert repeated.json()["status"] == "already_accepted"
