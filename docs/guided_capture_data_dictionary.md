# Guided capture research export data dictionary

Schema version: `guided_capture_export_v1`

The exporter writes `records.jsonl`, `manifest.json`, `splits.json`, and copied
media under `media/`. One JSONL record represents one retained capture asset.
All records for a child use the same deterministic child-level split.

## Privacy boundary

`pseudonymous_child_id` is an HMAC-derived stable identifier. The HMAC secret is
the configured `CGM_JWT_SECRET`; changing that secret changes future
pseudonyms. The export excludes child and guardian names, date of birth,
profile image, location free text, user/operator names, consent operator
identifiers, device identifiers, and source filesystem paths.

`visit_uuid`, `asset_uuid`, and `server_object_id` are retained linkage keys.
They are not names, but the exported directory must still be treated as
sensitive research data because child images remain identifiable.

## Record fields

| Field | Meaning | Provenance |
|---|---|---|
| `pseudonymous_child_id` | Stable research-only child linkage key | HMAC-derived |
| `split` | `train`, `validation`, or `test`; assigned once per child | HMAC-derived |
| `sex` | Recorded sex used for WHO standards | recorded demographic |
| `age_months` | Visit age derived from recorded DOB and visit date | calculated-from-measured |
| `visit_uuid` | Idempotent visit linkage key | recorded |
| `capture_state` | Durable guided-capture workflow state | recorded |
| `asset.asset_uuid` | Idempotent asset linkage key | recorded |
| `asset.role` | Front, side, or optional asset role | recorded |
| `asset.server_object_id` | Server-side durable object identifier | recorded |
| `asset.export_relative_path` | De-identified relative copied-media path, or null after deletion | exported |
| `asset.quality.*_score` | Pose, coverage, orientation, sharpness, lighting, and overall image scores | derived-estimated |
| `asset.quality.verdict` | Capture accept/reject outcome | derived-estimated |
| `asset.quality.threshold_version` | Quality-rule version | model metadata |
| `camera_estimate.result_uuid` | Immutable camera-result linkage key | recorded |
| `camera_estimate.version` | Per-visit immutable result revision number | recorded numeric metadata |
| `camera_estimate.estimated_height_cm` | Camera/estimator height | camera-estimated |
| `camera_estimate.estimated_weight_kg` | Camera/ML/WHO-derived weight estimate | camera-estimated |
| `camera_estimate.estimated_haz` | HAZ calculated from an estimated height | derived-estimated |
| `camera_estimate.estimated_whz` | WHZ calculated from estimated anthropometry | derived-estimated |
| `camera_estimate.component_probabilities.*` | Experimental model class probabilities | derived-estimated |
| `camera_estimate.body_proportion_features.*` | Numeric image/body features | derived-estimated |
| `camera_estimate.capture_quality_summary.*` | Aggregate quality values and used roles | derived-estimated |
| `camera_estimate.*_source` | Estimator method identifiers | model metadata |
| `camera_estimate.*_status` | Experimental categories from estimated values | derived-estimated |
| `camera_estimate.method` | Camera pipeline identifier | model metadata |
| `camera_estimate.model_version` | Inference model version | model metadata |
| `camera_estimate.manifest_checksum` | Model-manifest checksum | model metadata |
| `camera_estimate.training_data_label` | Model governance/training label | model metadata |
| `camera_estimate.non_clinical` | Must remain true for camera output | governance metadata |
| `measured.height_cm` | Board/tape measured height or length | measured |
| `measured.weight_kg` | Scale measured weight | measured |
| `measured.muac_cm` | Direct MUAC value when method is manual/tape | tape |
| `measured.muac_method` | Direct MUAC provenance | recorded |
| `measured.measurement_mode` | Standing height or recumbent length | recorded |
| `measured.oedema` | Recorded oedema observation | measured |
| `measured.haz_zscore` | WHO HAZ from measured height/length | calculated-from-measured |
| `measured.whz_zscore` | WHO WHZ/WLZ from measured height and weight | calculated-from-measured |
| `measured.bmi` | BMI from measured height and weight | calculated-from-measured |
| `measured.haz_status` | WHO height-for-age category | calculated-from-measured |
| `measured.who_acute_status` | WHO acute category from eligible measured inputs | calculated-from-measured |
| `measured.poshan_status` | Poshan Setu category from eligible measured inputs | calculated-from-measured |
| `measured.classification_method` | Authoritative classification protocol identifier | model metadata |

`camera_estimate` is null when inference is absent. `measured` is null unless
the same visit has at least one authoritative measured height, weight, or
direct MUAC value. Estimated values are never promoted into `measured`.

## Manifest fields

The manifest records the schema version, UTC generation time, counts, split
counts, and every model and quality-threshold version represented in the
export. It never records a database path or source media path.

## Running the export

Use a new or empty output directory:

```bash
PYTHONPATH=. .venv/bin/python scripts/export_guided_capture_dataset.py \
  --output-dir /secure/research/guided-capture-2026-07-29
```

The command refuses to overwrite a non-empty directory. Store the result only
in an approved encrypted research location with access logging.
