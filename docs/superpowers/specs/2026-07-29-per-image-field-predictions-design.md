# Per-Image Field Predictions and Area Metadata — Design

**Date:** 2026-07-29

**Status:** Approved design, pending implementation plan

**Scope:** Add a dedicated offline command that writes one accountable prediction row for every field photo, keyed by the child-folder ID and enriched with collection area from `field_data/ground_truth.csv`.

## Context

The field dataset is organized under `field_data/raw/<child_id>/` and may contain multiple front, side, back, arm, extra, and video files for a child. The existing `scripts/batch_assess.py` is designed around one assessment per child: it selects a front image and an optional side image, then writes one combined comparison row. That behavior remains useful for the child-level validation study but does not produce one row per original photo.

The new workflow must:

- account for every photo rather than only the selected front/side pair;
- use the same `child_id` as `field_data/ground_truth.csv`;
- add an `area` column to the ground-truth contract;
- preserve a row even when a photo cannot be assessed;
- avoid forcing full-body inference on arm close-ups or unusable images;
- keep predictions separate from manually entered actual values;
- remain explicitly non-clinical until compared with real measurements.

## Goals

1. Write exactly one deterministic CSV row per discovered field photo.
2. Join photo rows to child metadata through the containing folder's `child_id`.
3. Copy the child's `area`, sex, DOB-derived age, and measurement date into each prediction row.
4. Run the existing pose/measurement/ML pipeline on every usable full-body photo.
5. Record explicit skip or failure reasons for every photo that cannot produce a prediction.
6. Leave `batch_assess.py` and its one-row-per-child comparison behavior intact.
7. Permit predictions to run before manual height, weight, MUAC, oedema, or field category have been entered.
8. Preserve strict full ground-truth validation before formal evaluation.

## Non-goals

- Producing meaningful full-body measurements from arm close-ups.
- Treating every file as a front view.
- Selecting a single best photo per child.
- Combining several views into one prediction row.
- Replacing the existing child-level comparison and study-report pipeline.
- Ingesting videos as individual frame rows in the first release.
- Writing prediction values back into `ground_truth.csv`.
- Treating estimated labels as measured or clinically authoritative results.

## Chosen approach

Add a dedicated `scripts/predict_field_images.py` command. It reuses shared pose, measurement, ML, date, and provenance helpers but owns its own enumeration and output contract.

This is preferred over:

- changing `batch_assess.py` to default to per-image output, which would break its established child-level contract;
- adding a large `--per-image` branch to `batch_assess.py`, which would mix two different row meanings in one command;
- blindly running inference on every file, which would create misleading predictions from partial-body views.

## Input layout

```text
field_data/
  raw/
    001/
      front.jpg
      front_2.jpg
      side.jpg
      back.jpg
      arm.jpg
      extra_01.jpg
      rotate.mp4
    002/
      ...
  ground_truth.csv
```

The first path component below `field_data/raw/` is the canonical `child_id`. Names must not be inferred from image filenames or embedded metadata.

The first release enumerates supported image extensions recursively below each child folder. Videos are excluded from `image_predictions.csv`; existing video-view extraction remains a separate workflow. Non-image files are ignored and reported in the run summary, not represented as photo rows.

## Ground-truth contract

The canonical header becomes:

```text
child_id,area,sex,date_of_birth,measurement_date,actual_height_cm,actual_weight_kg,muac_cm,oedema,field_category,notes
```

### Area rules

- `area` is required for every child row.
- It stores a consistent collection-area name or study code, not a household address or child name.
- Leading and trailing whitespace is removed.
- Empty or whitespace-only area is a metadata error.
- Area is preserved as entered after trimming so operators can use existing programme/location codes.
- Documentation must instruct operators to standardize spelling before data entry.
- Research exports may include area for subgroup and external-location evaluation, while direct identifiers remain excluded.

### Two validation levels

The same CSV supports two distinct gates:

1. **Prediction metadata validation**
   - exact canonical header;
   - at least one child row;
   - unique `child_id`;
   - required `child_id`, `area`, `sex`, `date_of_birth`, and `measurement_date`;
   - valid sex, dates, age range, and no future measurement date;
   - actual height, weight, MUAC, oedema, and `field_category` may be blank;
   - any supplied manual value must still be syntactically valid and within the existing broad typo/unit ranges.

2. **Full evaluation validation**
   - runs before comparison metrics and study reports;
   - retains the strict manual-data and category rules required by the field study;
   - a prediction CSV does not make a header-only or incomplete ground-truth file safe for evaluation.

The implementation should expose these as explicit validator modes or separate named functions. It must not weaken the full-evaluation gate to make prediction runs convenient.

## Photo enumeration and identity

Enumeration is deterministic:

1. sort child folders by `child_id`;
2. recursively find supported image files;
3. sort by path relative to the child folder;
4. emit one row for every discovered photo.

Each row stores:

- `child_id`;
- `area`;
- `image_file`;
- `image_relative_path`, relative to `field_data/raw`;
- `image_sha256`;
- byte size;
- detected or filename-hinted role.

`image_relative_path` is the human-readable row identifier. `image_sha256` detects duplicates, renamed copies, and changes between runs. Two files with identical bytes still receive separate rows because the requirement is one row per stored photo.

## Role detection

Use filename hints first:

- `front*` → `front`;
- `side*` → `side`;
- `back*` or `rear*` → `back`;
- `arm*` or `muac*` → `arm`;
- everything else → `unknown`.

For non-arm images, pose/orientation analysis may refine `front`, `side`, `back`, or `unknown`. Preserve both the filename hint and detected role in the output so disagreements can be audited.

Arm-hinted images are never sent to the full-body measurement pipeline in the first release. They receive `prediction_status=skipped_unsupported_role` with a reason explaining that an arm-specific model is not available.

## Per-image processing

For each photo:

1. Resolve the containing `child_id`.
2. Join its metadata row from `ground_truth.csv`.
3. Compute age at `measurement_date`.
4. Read and hash the image.
5. Determine filename role hint.
6. Run photo QC and pose/orientation detection when appropriate.
7. If the photo contains a usable full-body pose, run the existing measurement and ML inference path using that photo as the primary image.
8. Write the complete result row.

No other photo is paired as the side-view input for a per-image row. This keeps each row's prediction attributable to exactly one source photo. The child-level `batch_assess.py` workflow remains responsible for multi-view combined assessments.

Full-body front, side, back, and unknown-role photos may be processed when they pass the same minimum pose/coverage checks. Their role stays in the output so view-specific performance can be measured. Passing QC does not imply that all roles are equally accurate.

## Prediction status

Every photo has exactly one status:

- `predicted`: usable metadata, readable photo, accepted full-body pose, inference completed;
- `skipped_unsupported_role`: known arm/partial-body role without a compatible model;
- `skipped_qc`: readable image but failed pose, coverage, blur, orientation, or full-body requirements;
- `metadata_error`: child row missing or required metadata invalid;
- `unreadable_image`: file cannot be decoded as an image;
- `inference_error`: preprocessing succeeded but measurement/ML inference raised an error.

Skipped and error rows keep all identity, area, file, hash, role, and available quality fields. Prediction fields remain blank. A failure for one photo never stops other photos from being processed.

Global structural errors still fail before processing:

- invalid or duplicate ground-truth header;
- duplicate `child_id`;
- no child data rows;
- missing raw directory;
- output path resolving inside the immutable raw archive.

## Output CSV

Default path:

```text
field_data/reports/image_predictions.csv
```

Columns, in stable order:

```text
child_id
area
image_file
image_relative_path
image_sha256
image_size_bytes
filename_role_hint
detected_role
prediction_status
skip_or_error_reason
sex
date_of_birth
measurement_date
age_months
pred_height_cm
pred_weight_ml_kg
pred_haz_z
pred_whz_z
pred_stunting_status
pred_wasting_status
ml_wasting_status
sam_probability
mam_probability
normal_probability
risk_probability
overweight_probability
pose_confidence
capture_quality_score
estimation_method
effective_height_source
model_version
model_training_data
model_manifest_sha256
non_clinical
annotated_image
feat_shoulder_width_cm
feat_hip_width_cm
feat_torso_length_cm
feat_upper_arm_length_cm
feat_shoulder_height_ratio
feat_hip_height_ratio
feat_body_build_score
```

All model-generated statuses are labelled by column name as predictions and every populated prediction row records `non_clinical=true`.

The output does not copy actual height, weight, MUAC, oedema, or `field_category`. Formal analysis joins `image_predictions.csv` to `ground_truth.csv` using `child_id`. This prevents predicted and actual values from being manually edited in the same file.

## Output integrity

- Write to a temporary file under `field_data/reports/`, flush and close it, then atomically replace the final CSV.
- A successful run replaces the complete predictions snapshot; it does not append duplicate rows.
- Output ordering is stable across identical runs.
- Include every discovered photo in coverage counts.
- The number of output rows must equal the number of discovered supported image files.
- Summaries report totals by status, role, child, and area.
- Raw images are never renamed, edited, moved, or deleted.
- Annotated derivatives, when produced, are written outside `field_data/raw`.

## Command-line interface

Primary command:

```bash
PYTHONPATH=. .venv/bin/python scripts/predict_field_images.py \
  --images field_data/raw \
  --metadata field_data/ground_truth.csv \
  --output field_data/reports/image_predictions.csv
```

Options:

- `--images`: per-child raw-image root;
- `--metadata`: canonical field metadata/ground-truth CSV;
- `--output`: predictions CSV;
- `--quiet`: suppress per-image progress while retaining final summary.

The command refuses to treat a flat directory of unrelated images as per-child input because it cannot derive a reliable `child_id`.

## Relationship to existing workflows

- `scripts/intake_check.py` continues to report missing files and child rows.
- `scripts/clean_media.py` continues to select best front/side images for the child-level pipeline.
- `scripts/extract_scan_views.py` continues to derive candidate stills from videos.
- `scripts/batch_assess.py` continues to produce one combined row per child.
- `scripts/analyze_results.py` continues to generate the formal child-level study report.
- The new per-image CSV supports view robustness, duplicate-image checks, QC analysis, failure analysis, and later per-image model evaluation.

Shared parsing and inference helpers should be factored only where necessary to prevent calculation drift. The implementation must not perform an unrelated rewrite of `batch_assess.py`.

## Area-aware analysis

Area is copied into every prediction row for:

- coverage counts by collection site;
- QC and failure rates by site;
- device/operator/environment investigation where available;
- child-level train/validation/test splitting by area;
- external-area performance checks.

Model training and evaluation must still split by child first. Multiple photos of the same child must never appear across train and test partitions, even when image rows are the immediate input.

## Privacy

- Keep zero-padded `child_id` values as spreadsheet Text.
- Do not add child name, guardian name, address, phone number, or household identifiers.
- `area` must be a collection site/region label rather than a precise home address.
- `field_data/` remains gitignored.
- Prediction output and annotated images remain inside the protected field-data tree.

## Testing

### Metadata and area

- canonical header includes `area` in the exact position;
- missing/blank area fails prediction metadata validation;
- area is trimmed and copied to all rows for the child;
- duplicate child IDs fail before inference;
- actual measurement and field-category blanks are accepted in prediction mode;
- the same blanks remain rejected or reported as incomplete under full evaluation rules;
- day-first Indian dates and legacy ISO dates retain existing behavior.

### Enumeration and accounting

- every supported image creates exactly one row;
- nested image paths are stable and deterministic;
- identical bytes under different paths create separate rows with the same hash;
- videos and non-image files are excluded from row counts;
- empty child folders produce no photo rows but remain visible through intake checks;
- output row count equals discovered-photo count.

### Role and QC

- front, side, back/rear, arm/MUAC, and unknown filename hints;
- arm images are skipped without invoking full-body inference;
- unreadable images create `unreadable_image` rows;
- no-pose, partial-pose, blur, and coverage failures create `skipped_qc` rows;
- usable full-body views invoke inference once per image.

### Prediction and failures

- prediction fields and provenance map correctly from known mocked outputs;
- side or back roles remain visible on predicted rows;
- missing child metadata creates a row-level `metadata_error`;
- one inference error does not abort subsequent images;
- predicted statuses never populate actual/manual fields;
- output write is deterministic and atomic;
- output cannot be written inside `field_data/raw`.

### Integration

- use real WHO files for an end-to-end metadata/WHO calculation test;
- mock MediaPipe for broad unit coverage;
- run a small real-model smoke test on representative full-body and arm images when test assets are available;
- verify that the existing child-level batch and analysis tests remain unchanged in meaning.

## Rollout and run sequence

1. Update the canonical ground-truth template and guide with required `area`.
2. Fill `child_id`, `area`, sex, DOB, and measurement date for children whose photos will be predicted.
3. Run prediction metadata validation.
4. Copy field photos into `field_data/raw/<child_id>/` without modifying the originals.
5. Run `predict_field_images.py`.
6. Review status counts and failure reasons.
7. Enter actual measurements manually in `ground_truth.csv`.
8. Run full ground-truth validation.
9. Run the existing child-level cleaning, batch assessment, and study analysis.

## Acceptance criteria

1. `ground_truth.csv` has the exact approved header with required `area`.
2. Every supported photo under a child folder produces exactly one prediction-ledger row.
3. Every row contains the folder-derived `child_id`, joined `area`, and stable image identity.
4. Usable full-body photos produce predictions with model and non-clinical provenance.
5. Arm, unreadable, failed-QC, metadata-error, and inference-error photos remain visible with explicit statuses.
6. Actual measurement fields remain exclusively in `ground_truth.csv`.
7. Prediction mode can run before actual values are entered, without weakening full evaluation validation.
8. Existing one-row-per-child assessment and study-report behavior remains intact.
9. Raw field media is never changed.
10. Identical runs produce the same row order and complete replacement CSV.
