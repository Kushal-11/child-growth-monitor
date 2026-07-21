# Field Data Pipeline & 200-Child Comparison Study — Design

**Date:** 2026-07-21
**Status:** Approved design, pending implementation plan
**Scope:** Organize real field data (photos/videos + paper measurements), clean it, and run an honest baseline evaluation of the current model against manual measurements for 200+ children. Model improvement is explicitly out of scope (follow-up spec).

## Goal

Produce a defensible comparison-study report: for every gathered child, the app pipeline's estimates (height, weight, WHZ, wasting status) versus same-day manual measurements (height, weight, MUAC, DOB, sex from paper forms), with agreement statistics suitable for a medical screening tool.

Decisions locked during brainstorming:

- **Evaluate first, improve later.** The current (synthetic-trained) model is frozen until the baseline report exists. Fine-tuning uses this data only afterwards, with a proper held-out split, in a separate spec.
- **Staged file-based pipeline**, extending existing scripts (`batch_assess.py`, `extract_best_frame.py`) — not backend/DB ingest, not notebooks.
- Raw data arrives as **one folder per child**, organized manually by the user; front + side photos typical, videos for some.
- Ground truth is **digitized from paper forms** into one master CSV; forms record height, weight, DOB, sex, and MUAC. Photos and measurements are same-day.

## Data layout, IDs, privacy

```
field_data/                 <- gitignored entirely (photos + medical data never enter git)
  raw/                      <- user-organized per-child folders, never modified by the pipeline
    001/  002/  ...         <- folder name = anonymous child ID
  cleaned/                  <- pipeline-selected best front.jpg / side.jpg per child
  ground_truth.csv          <- master CSV digitized from paper forms
  reports/                  <- QC report, batch results, final study report
```

- Folder name is an **anonymous numeric ID**; the name-to-ID mapping stays on paper or a private sheet outside the repo.
- Photos should be named with `front`/`side` prefixes where known; unlabeled photos are auto-classified by pose orientation and the guess is flagged for human confirmation in the QC report.
- `field_data/` is added to `.gitignore` as part of implementation.

## Stage 1 — Intake check (`scripts/intake_check.py`)

Read-only scan of `field_data/raw/`, re-runnable at any time during manual gathering.

- Per child: front photo present, side photo present, video available, ground-truth row present, anomalies (empty folder, duplicate ID, unreadable file).
- Output: manifest CSV in `field_data/reports/` + console summary ("137 children: 112 complete, 19 missing side photo, 6 missing ground truth").
- Serves as the progress dashboard for gathering; nothing expensive runs here.

## Stage 2 — Cleaning (`scripts/clean_media.py`)

Scores every photo per child using the five criteria already in `extract_best_frame.py`: pose-detection confidence, full-body coverage (head-to-heel), upright landmark ordering, frontal-vs-side orientation, sharpness (Laplacian blur rejection).

1. Classify each photo front/side — filename prefix wins, else pose orientation score.
2. Select best front + best side per child → copy to `field_data/cleaned/<id>/front.jpg`, `side.jpg`, with a provenance record (source file, scores).
3. Fallback: if no photo passes the quality bar and a video exists, extract the best video frame as the front image.
4. QC report CSV: chosen sources, all scores, verdict `ok` / `usable_no_side` / `failed` + reason. Failed children form the recapture list.

Properties: never modifies `raw/`; idempotent (skips already-cleaned children unless `--force`); safe to re-run as folders arrive.

Quality thresholds (min pose confidence, min body coverage, blur cutoff) are named constants with defaults taken from `extract_best_frame.py`'s scoring; they are expected to be tuned once on the first real batch and then frozen for the study.

## Stage 3 — Ground-truth digitization

Master CSV, one row per child:

```
child_id, sex, date_of_birth, measurement_date, actual_height_cm, actual_weight_kg, muac_cm, oedema, notes
```

- `measurement_date` is required: age-in-months is computed from it, not from the date the script runs. (Fixes a real bug: `batch_assess.py` currently uses today's date, which shifts every z-score when assessment happens after the field visit.)
- `oedema` (yes/no/blank): bilateral pitting oedema is an independent WHO SAM trigger; recorded if the form has it.
- `scripts/validate_ground_truth.py` gates assessment. Rejects: height outside 40–130 cm, weight outside 2–30 kg, MUAC outside 8–20 cm, age outside 0–60 months, measurement date before DOB or in the future, duplicate/unknown child IDs, invalid sex values.
- Entry practice (process, not code): after typing in the forms, re-check a random 10–15% of rows against paper.
- `batch_assess.py` is extended to read this master CSV keyed by folder ID in per-child layout (no scattered `values.csv` files).

## Stage 4 — Assessment + analysis

**Assessment:** existing `batch_assess.py` over `field_data/cleaned/`, extended with:

- age from `measurement_date` (bug fix above);
- master-CSV ground-truth lookup for per-child layout;
- actual gold-standard status computed by the WHO OR-rule — SAM if manual MUAC < 11.5 cm OR manual WHZ < −3 OR oedema present (MAM analogously) — matching how the app itself combines triggers, not WHZ alone.

**Analysis:** new `scripts/analyze_results.py` reads the results CSV and writes `field_data/reports/study_report.md`:

- **Height & weight agreement:** Bland–Altman mean bias + 95% limits of agreement (plus MAE), judged against published yardsticks — 1.4 cm SMART height tolerance, 0.7 cm WHO TEM.
- **Status agreement:** sensitivity, specificity, PPV, NPV for SAM (headline metric — missed SAM is the fatal error direction) and SAM+MAM combined, with 95% CIs (Wilson); weighted κ over the ordered SAM → MAM → Normal scale.
- **Subgroups:** the above by sex and age band (6–23 vs 24–59 months).
- **Coverage accounting:** every child lands in exactly one bucket — assessed / QC-failed / missing-data — and bucket counts must sum to the total. No silent drops.

## Out of scope: model improvement (follow-up spec)

Frozen until the baseline report exists. The follow-up will cover: stratified fine-tune/held-out split of the children, real+synthetic blended retraining, promotion gated on real-data SAM recall ≥ 0.80. This pipeline already produces its inputs (`finetune_label` column) and its evaluation harness (the analysis script doubles as the promotion gate).

## Error handling

- One bad child never aborts a run: failures become error rows with reasons; processing continues.
- No silent failures anywhere: every skipped/failed child appears in a report with a reason.

## Testing

- QC scoring + front/side classifier: pytest against known-good and known-bad sample images (repo `sample images/` where usable).
- Ground-truth validation: fixture CSV of bad rows, assert each rejection.
- Statistics (Bland–Altman, weighted κ, sensitivity/specificity + CIs): unit tests against small hand-computed examples — a formula bug must not be able to misreport the study.
- MediaPipe mocked in unit tests per project convention; an integration smoke test runs the real model on 1–2 sample images.

## Success criteria

1. Intake manifest accounts for every child folder with zero unexplained entries.
2. Cleaning produces a best-front (and side where available) for every passing child, and an actionable recapture list for the rest.
3. Ground truth passes validation with zero impossible values.
4. `study_report.md` reports the full metric set above for 200+ children, with coverage buckets summing to the total.
