"""De-identified guided-capture research export contract."""

import json

import pytest

from scripts.export_guided_capture_dataset import export_guided_capture_dataset
from tests.guided_sync_support import (
    asset_payload,
    build_context,
    camera_payload,
    revision_payload,
    visit_payload,
)


@pytest.fixture()
def ctx(tmp_path):
    context = build_context(tmp_path / "server-media")
    yield context
    context.close()


def _put_visit_assets(
    ctx,
    *,
    child_id: int,
    headers: dict[str, str],
    visit_uuid: str,
    asset_prefix: str,
):
    response = ctx.client.put(
        f"/api/v1/sync/guided/visits/{visit_uuid}",
        json=visit_payload(child_id, visit_uuid),
        headers=headers,
    )
    assert response.status_code == 200
    for index, role in enumerate(("front", "side"), start=1):
        asset_uuid = f"{asset_prefix[:-1]}{index}"
        response = ctx.client.put(
            f"/api/v1/sync/guided/visits/{visit_uuid}/assets/{asset_uuid}",
            json=asset_payload(visit_uuid, asset_uuid, role),
            headers=headers,
        )
        assert response.status_code == 200


def test_export_is_deidentified_child_split_and_pairs_same_visit_measurements(
    ctx,
    tmp_path,
):
    first_visit = "10000000-0000-0000-0000-000000000001"
    second_visit = "10000000-0000-0000-0000-000000000002"
    other_visit = "10000000-0000-0000-0000-000000000003"
    _put_visit_assets(
        ctx,
        child_id=ctx.child_id,
        headers=ctx.owner_headers,
        visit_uuid=first_visit,
        asset_prefix="20000000-0000-0000-0000-000000000001",
    )
    _put_visit_assets(
        ctx,
        child_id=ctx.child_id,
        headers=ctx.owner_headers,
        visit_uuid=second_visit,
        asset_prefix="21000000-0000-0000-0000-000000000001",
    )
    _put_visit_assets(
        ctx,
        child_id=ctx.other_child_id,
        headers=ctx.other_headers,
        visit_uuid=other_visit,
        asset_prefix="22000000-0000-0000-0000-000000000001",
    )
    result_uuid = "30000000-0000-0000-0000-000000000001"
    revision_uuid = "40000000-0000-0000-0000-000000000001"
    assert (
        ctx.client.put(
            (
                f"/api/v1/sync/guided/visits/{first_visit}"
                f"/camera-results/{result_uuid}"
            ),
            json=camera_payload(first_visit, result_uuid),
            headers=ctx.owner_headers,
        ).status_code
        == 200
    )
    assert (
        ctx.client.put(
            (
                f"/api/v1/sync/guided/visits/{first_visit}"
                f"/measured-revisions/{revision_uuid}"
            ),
            json=revision_payload(first_visit, revision_uuid),
            headers=ctx.owner_headers,
        ).status_code
        == 200
    )

    output = tmp_path / "export"
    db = ctx.session_factory()
    try:
        manifest = export_guided_capture_dataset(
            db,
            output,
            pseudonym_secret=b"stable-test-secret",
            source_media_root=ctx.media_root,
        )
    finally:
        db.close()

    records = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text().splitlines()
    ]
    assert manifest["schema_version"] == "guided_capture_export_v1"
    assert manifest["record_count"] == 6
    assert manifest["child_count"] == 2
    assert manifest["source_model_versions"] == ["camera-v1"]
    assert manifest["quality_threshold_versions"] == [
        "guided_capture_quality_v1"
    ]
    assert len({row["pseudonymous_child_id"] for row in records}) == 2
    owner_rows = [row for row in records if row["visit_uuid"] != other_visit]
    assert len({row["split"] for row in owner_rows}) == 1
    assert len({row["pseudonymous_child_id"] for row in owner_rows}) == 1
    measured = [row for row in records if row["visit_uuid"] == first_visit]
    unmeasured = [row for row in records if row["visit_uuid"] == second_visit]
    assert all(row["measured"]["height_cm"] == 83.58 for row in measured)
    assert all(row["measured"] is None for row in unmeasured)
    assert all(row["asset"]["server_object_id"] for row in records)
    assert all(
        (output / row["asset"]["export_relative_path"]).is_file()
        for row in records
    )
    serialized = "\n".join(
        [
            (output / "records.jsonl").read_text(),
            (output / "manifest.json").read_text(),
            (output / "splits.json").read_text(),
        ]
    )
    for forbidden in (
        "Child 001",
        "Child 002",
        "Sync Owner",
        "Sync Other",
        str(ctx.service.asset_path(
            ctx.owner_id,
            first_visit,
            "20000000-0000-0000-0000-000000000001",
            "image/jpeg",
        ).parent),
    ):
        assert forbidden not in serialized


def test_export_refuses_non_empty_output_and_pseudonym_is_stable(ctx, tmp_path):
    visit_uuid = "10000000-0000-0000-0000-000000000001"
    _put_visit_assets(
        ctx,
        child_id=ctx.child_id,
        headers=ctx.owner_headers,
        visit_uuid=visit_uuid,
        asset_prefix="20000000-0000-0000-0000-000000000001",
    )
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    db = ctx.session_factory()
    try:
        export_guided_capture_dataset(
            db,
            first_output,
            pseudonym_secret=b"stable-test-secret",
            source_media_root=ctx.media_root,
        )
        export_guided_capture_dataset(
            db,
            second_output,
            pseudonym_secret=b"stable-test-secret",
            source_media_root=ctx.media_root,
        )
        with pytest.raises(ValueError, match="non-empty"):
            export_guided_capture_dataset(
                db,
                first_output,
                pseudonym_secret=b"stable-test-secret",
                source_media_root=ctx.media_root,
            )
    finally:
        db.close()
    first = json.loads(
        (first_output / "records.jsonl").read_text().splitlines()[0]
    )
    second = json.loads(
        (second_output / "records.jsonl").read_text().splitlines()[0]
    )
    assert first["pseudonymous_child_id"] == second["pseudonymous_child_id"]
    assert first["split"] == second["split"]
