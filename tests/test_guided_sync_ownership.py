"""Owner isolation for every guided-sync entity endpoint."""

import pytest

from tests.guided_sync_support import (
    asset_payload,
    build_context,
    camera_payload,
    put_required_assets,
    put_visit,
    revision_payload,
    visit_payload,
)


@pytest.fixture()
def ctx(tmp_path):
    context = build_context(tmp_path)
    yield context
    context.close()


def test_cross_owner_access_is_hidden_for_every_entity(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    asset_uuid = "20000000-0000-0000-0000-000000000001"
    result_uuid = "30000000-0000-0000-0000-000000000001"
    revision_uuid = "40000000-0000-0000-0000-000000000001"
    assert put_visit(ctx, visit_uuid).status_code == 200
    assert all(r.status_code == 200 for r in put_required_assets(ctx, visit_uuid))

    cases = [
        (
            "put",
            f"/api/v1/sync/guided/visits/{visit_uuid}/assets/{asset_uuid}",
            asset_payload(visit_uuid, asset_uuid, "front"),
        ),
        (
            "put",
            (
                f"/api/v1/sync/guided/visits/{visit_uuid}"
                f"/camera-results/{result_uuid}"
            ),
            camera_payload(visit_uuid, result_uuid),
        ),
        (
            "put",
            (
                f"/api/v1/sync/guided/visits/{visit_uuid}"
                f"/measured-revisions/{revision_uuid}"
            ),
            revision_payload(visit_uuid, revision_uuid),
        ),
        (
            "delete",
            f"/api/v1/sync/guided/visits/{visit_uuid}/media/{asset_uuid}",
            None,
        ),
    ]

    for method, path, body in cases:
        response = (
            ctx.client.delete(path, headers=ctx.other_headers)
            if method == "delete"
            else getattr(ctx.client, method)(
                path,
                json=body,
                headers=ctx.other_headers,
            )
        )
        assert response.status_code == 404


def test_visit_cannot_bind_another_owners_child(ctx):
    visit_uuid = "10000000-0000-0000-0000-000000000009"
    body = visit_payload(ctx.other_child_id, visit_uuid)
    response = ctx.client.put(
        f"/api/v1/sync/guided/visits/{visit_uuid}",
        json=body,
        headers=ctx.owner_headers,
    )
    assert response.status_code == 404
