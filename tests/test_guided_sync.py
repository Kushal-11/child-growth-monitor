"""Happy-path and idempotency coverage for guided sync."""

import pytest
from sqlalchemy import func, select

from app.models.camera_result import CameraResult
from app.models.capture_asset import CaptureAsset
from app.models.measured_detail_revision import MeasuredDetailRevision
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from tests.guided_sync_support import (
    asset_payload,
    build_context,
    camera_payload,
    put_required_assets,
    revision_payload,
    visit_payload,
)


@pytest.fixture()
def ctx(tmp_path):
    context = build_context(tmp_path)
    yield context
    context.close()


def test_each_guided_endpoint_requires_auth(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    cases = [
        (
            "put",
            f"/api/v1/sync/guided/visits/{visit_uuid}",
            visit_payload(ctx.child_id, visit_uuid),
        ),
        (
            "put",
            (
                f"/api/v1/sync/guided/visits/{visit_uuid}/assets/"
                "20000000-0000-0000-0000-000000000001"
            ),
            asset_payload(
                visit_uuid,
                "20000000-0000-0000-0000-000000000001",
                "front",
            ),
        ),
        (
            "put",
            (
                f"/api/v1/sync/guided/visits/{visit_uuid}/camera-results/"
                "30000000-0000-0000-0000-000000000001"
            ),
            camera_payload(
                visit_uuid,
                "30000000-0000-0000-0000-000000000001",
            ),
        ),
        (
            "put",
            (
                f"/api/v1/sync/guided/visits/{visit_uuid}/measured-revisions/"
                "40000000-0000-0000-0000-000000000001"
            ),
            revision_payload(
                visit_uuid,
                "40000000-0000-0000-0000-000000000001",
            ),
        ),
        (
            "delete",
            (
                f"/api/v1/sync/guided/visits/{visit_uuid}/media/"
                "20000000-0000-0000-0000-000000000001"
            ),
            None,
        ),
    ]

    for method, path, body in cases:
        response = (
            ctx.client.delete(path)
            if method == "delete"
            else getattr(ctx.client, method)(path, json=body)
        )
        assert response.status_code == 401


def test_full_guided_sync_is_idempotent_and_server_verified(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    result_uuid = "30000000-0000-0000-0000-000000000001"
    revision_uuid = "40000000-0000-0000-0000-000000000001"

    visit_body = visit_payload(ctx.child_id, visit_uuid)
    first_visit = ctx.client.put(
        f"/api/v1/sync/guided/visits/{visit_uuid}",
        json=visit_body,
        headers=ctx.owner_headers,
    )
    repeated_visit = ctx.client.put(
        f"/api/v1/sync/guided/visits/{visit_uuid}",
        json=visit_body,
        headers=ctx.owner_headers,
    )
    assert first_visit.status_code == 200
    assert first_visit.json()["status"] == "accepted"
    assert repeated_visit.json()["status"] == "already_accepted"
    assert repeated_visit.json()["server_id"] == first_visit.json()["server_id"]

    assets = put_required_assets(ctx, visit_uuid)
    assert [response.status_code for response in assets] == [200, 200]
    repeated_front = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}/assets/"
            "20000000-0000-0000-0000-000000000001"
        ),
        json=asset_payload(
            visit_uuid,
            "20000000-0000-0000-0000-000000000001",
            "front",
        ),
        headers=ctx.owner_headers,
    )
    assert repeated_front.json()["status"] == "already_accepted"
    assert repeated_front.json()["server_object_id"]

    result_body = camera_payload(visit_uuid, result_uuid)
    result = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/camera-results/{result_uuid}"
        ),
        json=result_body,
        headers=ctx.owner_headers,
    )
    repeated_result = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/camera-results/{result_uuid}"
        ),
        json=result_body,
        headers=ctx.owner_headers,
    )
    assert result.status_code == 200
    assert repeated_result.json()["status"] == "already_accepted"
    assert repeated_result.json()["server_id"] == result.json()["server_id"]

    revision_body = revision_payload(visit_uuid, revision_uuid)
    revision = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/measured-revisions/{revision_uuid}"
        ),
        json=revision_body,
        headers=ctx.owner_headers,
    )
    repeated_revision = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/measured-revisions/{revision_uuid}"
        ),
        json=revision_body,
        headers=ctx.owner_headers,
    )
    assert revision.status_code == 200
    assert repeated_revision.json()["status"] == "already_accepted"
    assert repeated_revision.json()["server_id"] == revision.json()["server_id"]

    db = ctx.session_factory()
    try:
        visit = db.scalar(select(Visit).where(Visit.local_uuid == visit_uuid))
        measurement = db.scalar(
            select(MeasurementResult).where(
                MeasurementResult.visit_id == visit.id
            )
        )
        assert db.scalar(select(func.count(Visit.id))) == 1
        assert db.scalar(select(func.count(CaptureAsset.id))) == 2
        assert db.scalar(select(func.count(CameraResult.id))) == 1
        assert db.scalar(select(func.count(MeasuredDetailRevision.id))) == 1
        assert visit.capture_state == "measured_report"
        assert measurement.manual_height_cm == 83.58
        assert measurement.haz_status != "Normal"
        assert measurement.poshan_status == "Indeterminate"
        assert db.scalar(select(CameraResult)).non_clinical is True
    finally:
        db.close()


def test_same_uuid_with_changed_immutable_payload_returns_409(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    body = visit_payload(ctx.child_id, visit_uuid)
    assert (
        ctx.client.put(
            f"/api/v1/sync/guided/visits/{visit_uuid}",
            json=body,
            headers=ctx.owner_headers,
        ).status_code
        == 200
    )
    changed = dict(body)
    changed["visit_date"] = "2026-07-28T00:00:00"

    conflict = ctx.client.put(
        f"/api/v1/sync/guided/visits/{visit_uuid}",
        json=changed,
        headers=ctx.owner_headers,
    )

    assert conflict.status_code == 409
