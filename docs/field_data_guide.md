# Field Data Organization Guide

How to arrange each child's photos, videos and paper measurements so the
pipeline can process them. Follow this while gathering; run the intake check
(Stage 1) at any time to see what is still missing.

## Layout

    field_data/                  <- created by you, ignored by git
      raw/                       <- you fill this, one folder per child
        001/
          front.jpg              <- frontal photo (best if named like this)
          side.jpg               <- side photo
          back.jpg               <- rear view, kept for future body-shape work
          arm.jpg                <- MUAC/upper-arm close-up, kept for arm model
          extra_01.jpg           <- any other photos: keep them, pipeline picks best
          rotate.mp4             <- optional rotating/standing video
        002/
          ...
      ground_truth.csv           <- one row per child, typed from the paper forms
      cleaned/                   <- pipeline output. Never edit by hand.
      derived/                   <- pipeline output for video views/features.
      reports/                   <- pipeline output. Never edit by hand.

## Child IDs

- Folder name = child ID. Use zero-padded numbers: `001`, `002`, ... `250`.
- NEVER put the child's name in a folder name, file name, or the CSV.
- Keep the name-to-ID mapping on paper or a private sheet OUTSIDE this
  project folder. The repo and pipeline only ever see the numeric ID.
- Write the same ID on the child's paper form at measurement time.

## What each child folder should contain

Aim for, in order of importance:

1. **One frontal photo** — child standing straight, facing the camera,
   full body visible head to feet, arms slightly away from the body.
2. **One side photo** — child turned 90°, again full body.
3. Optional but useful: **arm/MUAC close-ups** named `arm...` or `muac...`.
4. Optional: back shots, extra shots, rotating video clips. Keep everything;
   the cleaner scores whole-body photos and preserves non-measurement views for
   later model training.

Name files `front...` / `side...` when you know which is which
(`front.jpg`, `front_2.jpg`, `side_a.jpg`). Name rear and arm photos
`back...` / `rear...` and `arm...` / `muac...` so they are archived instead of
mistaken for whole-body measurement photos. If a whole-body photo is unnamed
the pipeline guesses the orientation from the pose and flags the guess for
your confirmation in the QC report.

## Photo quality basics (saves recapture trips)

- Whole child in frame: head AND feet, nothing cropped.
- Camera at roughly the child's waist height, phone held vertically.
- Good light, child not in shadow; plain background if possible.
- Exactly one person in frame (no siblings/adults behind).
- Hold the phone still — motion blur is the top rejection reason.

## Ground truth CSV

`field_data/ground_truth.csv`, one row per child:

    child_id,sex,date_of_birth,measurement_date,actual_height_cm,actual_weight_kg,muac_cm,oedema,notes
    001,M,2023-04-12,2026-07-15,82.5,10.4,13.2,no,
    002,F,2024-01-30,2026-07-15,74.0,8.1,12.1,,left early

- `sex`: `M` or `F`.
- Dates: `YYYY-MM-DD`. `measurement_date` = the day height/weight/MUAC
  were taken (photos must be same-day — this drives the age used for
  z-scores). A value that doesn't parse fails the row outright rather
  than falling back to the run date: age selects the WHO reference table
  and drives every z-score, so a wrong one produces a clean-looking but
  wrong verdict. The results CSV records where each age came from in
  `measurement_date_source` (`supplied` / `today_fallback` /
  `unparseable`).
- Height in cm, weight in kg, MUAC in cm. Decimal point, never a comma.
- `oedema`: `yes` / `no` / blank if not checked.
- `muac_cm`: WHO's MUAC cutoffs are defined for ages 6–59 months only.
  Record the tape reading for younger infants anyway, but it is left out
  of the gold standard rather than classified against cutoffs that don't
  apply at that age.
- Leave a value blank if it truly wasn't measured — never guess.
- After typing everything in, re-check a random 10–15% of rows against
  the paper forms, and run the validator (Stage 3) before any assessment.
- Never add a `child_name` (or any name/ID-beyond-`child_id`) column — this
  is the only ground-truth CSV shape this pipeline reads. It is unrelated
  to `data/ground_truth_template.csv` / `batch_assess.py --template`, which
  is a separate, `image_file`-keyed template for standalone flat-layout use
  of `batch_assess.py` outside `field_data/` — do not use that template as
  your `field_data/ground_truth.csv`, and vice versa. `scripts/validate_
  ground_truth.py` hard-fails on any column that doesn't match this shape
  exactly, so using the wrong template here is caught immediately rather
  than silently breaking coverage accounting.

## Rules

- Treat `raw/` as an archive: after dropping files in, don't rename,
  edit, or delete them. The pipeline never modifies `raw/` either.
- Don't commit `field_data/` — it is gitignored; `git status` must never
  show it.
- Videos welcome but photos preferred: a sharp photo beats a video frame.

## Workflow while gathering

1. Measure the child, fill the paper form, assign the next free ID.
2. Same day: photos (front, side, extras) into `field_data/raw/<id>/`.
3. Type the form into `ground_truth.csv` (or batch it, but soon —
   backlogs breed typos).
4. Any time: run the intake check to list gaps (missing side photo,
   missing CSV row, empty folders). Fix gaps while you still have field
   access to the child.

## Runbook — commands in order

All commands from the project root.

    # 0. One-time: create the ground-truth template
    PYTHONPATH=. .venv/bin/python scripts/validate_ground_truth.py --template

    # 1. While gathering: what's still missing?
    PYTHONPATH=. .venv/bin/python scripts/intake_check.py

    # 2. Validate the typed-in measurements (must pass before assessing)
    PYTHONPATH=. .venv/bin/python scripts/validate_ground_truth.py

    # 3. Clean: pick best front/side per child, get the recapture list
    PYTHONPATH=. .venv/bin/python scripts/clean_media.py

    # 3b. Optional: split rotating videos into best front/side candidate frames
    PYTHONPATH=. .venv/bin/python scripts/extract_scan_views.py

    # 4. Assess every cleaned child against ground truth
    PYTHONPATH=. .venv/bin/python scripts/batch_assess.py \
        --images field_data/cleaned \
        --ground-truth field_data/ground_truth.csv \
        --output field_data/reports/batch_results.csv

    # 5. Generate the study report
    PYTHONPATH=. .venv/bin/python scripts/analyze_results.py

    # Read: field_data/reports/study_report.md

Re-run any stage at any time; stages never modify `raw/` and cleaning
skips already-cleaned children (add `--force` to redo them).

Stage 2's validator (step 2 above) is not just a courtesy check: when
`batch_assess.py` is given `--images field_data/cleaned` (the per-child
layout), it re-runs `validate_rows` on the master ground-truth CSV itself
before assessing a single child, and refuses to run at all if that CSV has
any errors. Running step 2 first just means you find out about a bad row
in seconds instead of after a full batch-assess pass aborts partway
through.
