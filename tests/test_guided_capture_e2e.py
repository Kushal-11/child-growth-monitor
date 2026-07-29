"""End-to-end guided workflow using real persistence and WHO workbooks."""

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from app.models.camera_result import CameraResult
from app.models.measurement import MeasurementResult
from app.models.visit import Visit
from app.services.who_data_service import WHODataService
from tests.guided_sync_support import (
    build_context,
    camera_payload,
    put_required_assets,
    put_visit,
    revision_payload,
)


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "contracts"
    / "guided_capture_v1.json"
)


@pytest.fixture()
def ctx(tmp_path):
    context = build_context(tmp_path)
    yield context
    context.close()


def test_complete_guided_workflow_recomputes_measured_report_and_isolates_owner(
    ctx,
):
    contract = json.loads(CONTRACT_PATH.read_text())
    who = WHODataService()
    who.load_all()
    assert who.get_haz_lms("F", 30) is not None

    visit_uuid = "10000000-0000-0000-0000-000000000001"
    result_uuid = "30000000-0000-0000-0000-000000000001"
    revision_uuid = "40000000-0000-0000-0000-000000000001"
    assert put_visit(ctx, visit_uuid).status_code == 200
    asset_responses = put_required_assets(ctx, visit_uuid)
    assert [response.status_code for response in asset_responses] == [200, 200]
    assert {
        response.json()["entity_type"] for response in asset_responses
    } == {"capture_asset"}

    camera_body = camera_payload(visit_uuid, result_uuid)
    camera_response = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/camera-results/{result_uuid}"
        ),
        json=camera_body,
        headers=ctx.owner_headers,
    )
    revision_body = revision_payload(visit_uuid, revision_uuid)
    revision_body["after"]["haz_status"] = "Normal"
    revision_body["after"]["poshan_status"] = "Normal"
    revision_response = ctx.client.put(
        (
            f"/api/v1/sync/guided/visits/{visit_uuid}"
            f"/measured-revisions/{revision_uuid}"
        ),
        json=revision_body,
        headers=ctx.owner_headers,
    )

    assert camera_response.status_code == 200
    assert revision_response.status_code == 200
    assert camera_response.json()["entity_uuid"] == result_uuid
    assert revision_response.json()["entity_uuid"] == revision_uuid

    detail = ctx.client.get(
        f"/api/v1/children/{ctx.child_id}",
        headers=ctx.owner_headers,
    )
    assert detail.status_code == 200
    visit = detail.json()["visits"][0]
    assert visit["capture_state"] == "measured_report"
    assert visit["capture_state"] in contract["visit_states"]
    assert visit["required_assets_acknowledged"] is True
    assert visit["required_asset_acknowledgement"] == {
        "front": "acknowledged",
        "side": "acknowledged",
    }
    assert visit["camera_result_summary"]["result_uuid"] == result_uuid
    assert visit["camera_result_summary"]["non_clinical"] is True
    assert visit["camera_result_summary"]["method"] == contract["camera_method"]
    assert visit["measurement"]["manual_height_cm"] == 83.58
    assert visit["measurement"]["haz_status"] != "Normal"
    assert visit["measurement"]["poshan_status"] == "Indeterminate"

    assert (
        ctx.client.get(
            f"/api/v1/children/{ctx.child_id}",
            headers=ctx.other_headers,
        ).status_code
        == 404
    )

    db = ctx.session_factory()
    try:
        stored_visit = db.scalar(
            select(Visit).where(Visit.local_uuid == visit_uuid)
        )
        stored_camera = db.scalar(
            select(CameraResult).where(CameraResult.result_uuid == result_uuid)
        )
        measured = db.scalar(
            select(MeasurementResult).where(
                MeasurementResult.visit_id == stored_visit.id
            )
        )
        assert stored_camera.estimated_height_cm == camera_body[
            "estimated_height_cm"
        ]
        assert stored_camera.estimated_weight_kg == camera_body[
            "estimated_weight_kg"
        ]
        assert stored_camera.model_version == camera_body["model_version"]
        assert measured.manual_height_cm == 83.58
        assert measured.editor_user_id == ctx.owner_id
    finally:
        db.close()
