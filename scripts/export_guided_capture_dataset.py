#!/usr/bin/env python3
"""Export guided-capture media and paired measurements without direct IDs."""

import argparse
import hashlib
import hmac
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.visit import Visit
from app.models.database import SessionLocal
from config import GUIDED_CAPTURE_MEDIA_DIR, JWT_SECRET


SCHEMA_VERSION = "guided_capture_export_v1"


def _child_identity(child_id: int, secret: bytes) -> tuple[str, str]:
    digest = hmac.new(
        secret,
        f"guided-child:{child_id}".encode(),
        hashlib.sha256,
    ).digest()
    pseudonym = f"c_{digest.hex()[:20]}"
    bucket = int.from_bytes(digest[:4], "big") % 100
    split = "train" if bucket < 70 else "validation" if bucket < 90 else "test"
    return pseudonym, split


def _safe_extension(path: Path) -> str:
    extension = path.suffix.lower()
    return extension if extension in {".jpg", ".jpeg", ".png", ".webp"} else ".bin"


def _asset_record(asset, export_relative_path: str | None) -> dict[str, Any]:
    return {
        "asset_uuid": asset.asset_uuid,
        "role": asset.role,
        "server_object_id": asset.server_object_id,
        "export_relative_path": export_relative_path,
        "quality": {
            "pose_score": asset.pose_score,
            "coverage_score": asset.coverage_score,
            "orientation_score": asset.orientation_score,
            "sharpness_score": asset.sharpness_score,
            "lighting_score": asset.lighting_score,
            "overall_score": asset.overall_score,
            "verdict": asset.quality_verdict,
            "threshold_version": asset.quality_threshold_version,
        },
    }


def _camera_record(result) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "result_uuid": result.result_uuid,
        "version": result.version,
        "estimated_height_cm": result.estimated_height_cm,
        "estimated_weight_kg": result.estimated_weight_kg,
        "height_source": result.height_source,
        "weight_source": result.weight_source,
        "estimated_haz": result.estimated_haz,
        "estimated_whz": result.estimated_whz,
        "estimated_stunting_status": result.estimated_stunting_status,
        "estimated_wasting_status": result.estimated_wasting_status,
        "experimental_overall_category": result.experimental_overall_category,
        "component_probabilities": result.component_probabilities_json or {},
        "body_proportion_features": result.body_proportion_features_json or {},
        "capture_quality_summary": result.capture_quality_summary_json or {},
        "method": result.method,
        "model_version": result.model_version,
        "manifest_checksum": result.manifest_checksum,
        "training_data_label": result.training_data_label,
        "non_clinical": result.non_clinical,
    }


def _measured_record(measurement) -> dict[str, Any] | None:
    if measurement is None:
        return None
    has_measured_value = any(
        value is not None
        for value in (
            measurement.manual_height_cm,
            measurement.manual_weight_kg,
            (
                measurement.muac_cm
                if measurement.muac_method in {"manual", "tape"}
                else None
            ),
        )
    )
    if not has_measured_value:
        return None
    return {
        "height_cm": measurement.manual_height_cm,
        "weight_kg": measurement.manual_weight_kg,
        "muac_cm": (
            measurement.muac_cm
            if measurement.muac_method in {"manual", "tape"}
            else None
        ),
        "muac_method": measurement.muac_method,
        "measurement_mode": measurement.measurement_mode,
        "oedema": measurement.oedema,
        "haz_zscore": measurement.haz_zscore,
        "whz_zscore": measurement.whz_zscore,
        "bmi": measurement.bmi,
        "haz_status": measurement.haz_status,
        "who_acute_status": measurement.who_acute_status,
        "poshan_status": measurement.poshan_status,
        "classification_method": measurement.classification_method,
    }


def export_guided_capture_dataset(
    db: Session,
    output_dir: Path,
    *,
    pseudonym_secret: bytes,
    source_media_root: Path,
) -> dict[str, Any]:
    """Write a new de-identified export and return its manifest."""
    if not pseudonym_secret:
        raise ValueError("A non-empty pseudonym secret is required")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Refusing to overwrite a non-empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir()
    resolved_media_root = Path(source_media_root).resolve()

    visits = list(
        db.scalars(
            select(Visit)
            .where(Visit.entry_method == "guided_capture")
            .order_by(Visit.child_id, Visit.id)
        )
    )
    records: list[dict[str, Any]] = []
    child_splits: dict[str, str] = {}
    model_versions: set[str] = set()
    quality_versions: set[str] = set()
    visit_uuids: set[str] = set()

    for visit in visits:
        if visit.local_uuid is None or not visit.capture_assets:
            continue
        pseudonym, split = _child_identity(visit.child_id, pseudonym_secret)
        child_splits[pseudonym] = split
        visit_uuids.add(visit.local_uuid)
        camera = (
            max(visit.camera_results, key=lambda result: result.version)
            if visit.camera_results
            else None
        )
        if camera is not None:
            model_versions.add(camera.model_version)
        measured = _measured_record(visit.measurement)
        for asset in visit.capture_assets:
            if asset.quality_threshold_version:
                quality_versions.add(asset.quality_threshold_version)
            exported_path = None
            if asset.local_path:
                source = Path(asset.local_path)
                if source.is_file():
                    resolved_source = source.resolve()
                    if not resolved_source.is_relative_to(resolved_media_root):
                        raise ValueError(
                            "Refusing to export media outside guided storage"
                        )
                    relative = (
                        Path("media")
                        / pseudonym
                        / visit.local_uuid
                        / f"{asset.asset_uuid}{_safe_extension(source)}"
                    )
                    destination = output_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(resolved_source, destination)
                    exported_path = relative.as_posix()
            records.append(
                {
                    "pseudonymous_child_id": pseudonym,
                    "split": split,
                    "sex": visit.child.sex,
                    "age_months": visit.age_months,
                    "visit_uuid": visit.local_uuid,
                    "capture_state": visit.capture_state,
                    "asset": _asset_record(asset, exported_path),
                    "camera_estimate": _camera_record(camera),
                    "measured": measured,
                }
            )

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output_dir / "splits.json").write_text(
        json.dumps(child_splits, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_counts = Counter(child_splits.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "child_count": len(child_splits),
        "visit_count": len(visit_uuids),
        "child_split_counts": dict(sorted(split_counts.items())),
        "source_model_versions": sorted(model_versions),
        "quality_threshold_versions": sorted(quality_versions),
        "records_file": "records.jsonl",
        "splits_file": "splits.json",
        "media_directory": "media",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the de-identified guided-capture research dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty destination directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    with SessionLocal() as db:
        manifest = export_guided_capture_dataset(
            db,
            args.output_dir,
            pseudonym_secret=JWT_SECRET.encode(),
            source_media_root=GUIDED_CAPTURE_MEDIA_DIR,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
