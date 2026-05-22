# Claude Config Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up CLAUDE.md, AGENTS.md, and hooks in settings.json for the Child Growth Monitor project.

**Architecture:** Three config files at project root / `.claude/` directory. CLAUDE.md defines conventions, AGENTS.md defines 5 specialized agent personas, settings.json hooks run lightweight checks on file edits.

**Tech Stack:** Claude Code configuration (CLAUDE.md, AGENTS.md, settings.json)

---

### Task 1: Create CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Create CLAUDE.md at project root**

```markdown
# Child Growth Monitor

## Project Overview
Medical-grade child growth monitoring system: FastAPI backend + Flutter mobile app + ML wasting detection.
Safety-critical: false negatives for SAM/MAM can endanger lives.

## Stack
- **Mobile**: Flutter (Dart) with Riverpod state management
- **Backend**: FastAPI + SQLAlchemy (SQLite) + Jinja2 web UI
- **ML**: TensorFlow/Keras + scikit-learn, TFLite for mobile inference
- **Pose**: MediaPipe PoseLandmarker (heavy model)
- **Data**: WHO 2006 growth standards (Excel LMS files are authoritative)

## Run Commands
- Backend: `PYTHONPATH=. .venv/bin/python main.py`
- Tests (Python): `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
- Tests (Flutter): `cd flutter_app && flutter test`
- Flutter run: `cd flutter_app && flutter run`
- Flutter analyze: `cd flutter_app && flutter analyze`
- ML train: `PYTHONPATH=. .venv/bin/python ml/train.py`
- ML evaluate: `PYTHONPATH=. .venv/bin/python ml/evaluate.py`
- Generate data: `PYTHONPATH=. .venv/bin/python ml/generate_synthetic_data.py`

## Flutter Conventions
- State management: Riverpod (riverpod + flutter_riverpod + riverpod_annotation)
- Architecture: feature-first folders under `lib/` (e.g., `lib/features/assessment/`)
- Models are immutable with `freezed` or manual `copyWith`
- Use `AsyncValue` for loading/error/data states
- All API calls go through a repository layer, never directly from widgets
- Prefer `ConsumerWidget` / `ConsumerStatefulWidget` over raw StatefulWidget
- Offline-first: local persistence with drift or shared_preferences, sync when online
- Camera workflow: image_picker for now, camera package when custom UI needed
- No hardcoded API URLs — use environment config or runtime settings

## Python Conventions
- Always use `.venv/` — prefix: `PYTHONPATH=. .venv/bin/python`
- Type hints on all function signatures
- Services are stateless classes; use dependency injection via FastAPI `Depends()`
- WHO data: Excel LMS files are the source of truth, never the CSV fallbacks
- Weight priority: manual > ML estimate > WHO median (with body build adjustment)

## ML Conventions
- SAM recall >= 0.80 is a hard floor — never merge a model that drops below this
- Log all experiments: architecture, hyperparams, metrics (especially per-class recall)
- Body width proportions from Snyder et al. 1975 — label as non-WHO in code
- Always export TFLite alongside Keras models
- 14-feature input vector: do not change feature order without updating scaler + all consumers

## Safety Rules
- Never skip WHO z-score validation — all growth assessments must compute HAZ and WHZ
- Manual measurements always take priority over estimated ones
- ML weight estimates must fall within 45-180% of WHO median to be accepted
- MUAC thresholds are fixed by WHO: <11.5 SAM, 11.5-12.5 MAM, >=12.5 Normal
- No silent failures in assessment pipeline — surface all errors to the user

## Testing
- Python: pytest, use TestClient for API tests, mock MediaPipe for unit tests
- Flutter: widget tests for all screens, unit tests for providers/repositories
- ML: evaluate.py after every retrain, check SAM recall before committing models
- Never commit a model file without updated evaluate.py output in the commit message

## Git
- Commit messages: imperative mood, concise
- Don't commit: .env, credentials, SQLite databases, uploaded images, __pycache__
- Large model files (pose_landmarker_heavy.task): tracked via .gitignore, not in repo
```

- [ ] **Step 2: Verify CLAUDE.md is readable**

Run: `head -5 CLAUDE.md`
Expected:
```
# Child Growth Monitor

## Project Overview
Medical-grade child growth monitoring system: FastAPI backend + Flutter mobile app + ML wasting detection.
Safety-critical: false negatives for SAM/MAM can endanger lives.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Add CLAUDE.md with project conventions"
```

---

### Task 2: Create AGENTS.md

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Create AGENTS.md at project root**

```markdown
# Agents

## flutter-dev
**Description**: Flutter mobile app developer specializing in Riverpod architecture and offline-first medical apps.
**Scope**: `flutter_app/`
**Instructions**:
- Follow Riverpod patterns: providers in `lib/providers/`, repositories in `lib/repositories/`
- Migrate existing StatefulWidget code to ConsumerWidget + StateNotifier/AsyncNotifier
- Use feature-first folder structure: `lib/features/<feature>/` with screens, widgets, providers
- All network calls go through repository layer with proper error handling
- Design for offline-first: cache assessments locally, sync when connected
- Follow Material 3 design guidelines
- Reference MOBILE_APP_SPEC.md for feature requirements and API contracts
- Run `flutter analyze` before considering work complete
- Write widget tests for every new screen

## ml-researcher
**Description**: ML engineer focused on architecture exploration and model improvement for wasting detection.
**Scope**: `ml/`, `data/`, `notebooks/`
**Instructions**:
- Current baseline: 70.2% accuracy (5-class), 0.886 SAM recall, 0.403 kg weight MAE
- SAM recall >= 0.80 is a hard safety floor — all experiments must report this metric
- Explore: ensemble methods, feature engineering, augmentation, alternative architectures (XGBoost, LightGBM, deeper networks, attention mechanisms)
- Always compare against baseline — never assume an architecture is better without evaluation
- Document experiments in `notebooks/` with clear methodology and results
- Maintain the 14-feature interface — if adding features, ensure backward compatibility
- Export TFLite alongside Keras — mobile inference is a hard requirement
- Synthetic data generation in `ml/generate_synthetic_data.py` can be improved (distributions, noise, augmentation)
- Run `ml/evaluate.py` after every change and include metrics in commit messages
- Consider model size constraints: current TFLite models are 7 KB + 17 KB, keep mobile-friendly

## backend-dev
**Description**: FastAPI backend developer for the growth monitoring API and services.
**Scope**: `app/`, `config.py`, `main.py`, `scripts/`
**Instructions**:
- Maintain service layer pattern: routes -> services -> models
- WHO Excel LMS files are authoritative for z-score computation — never fall back to CSVs
- Weight priority chain: manual -> ML estimate (if 45-180% of WHO median) -> WHO median with body build adjustment
- Use Pydantic for all request/response schemas
- SQLAlchemy ORM for database operations, SQLite for development
- Keep assessment pipeline deterministic: same inputs must produce same outputs
- Add proper error responses with HTTP status codes and descriptive messages
- Reference object detection (yellow packet) is optional — gracefully degrade without it
- Side-view processing is optional — impute depth from Snyder 1975 ratios when absent

## code-reviewer
**Description**: Cross-cutting code reviewer focused on safety, consistency, and quality.
**Scope**: entire repository
**Instructions**:
- Safety is paramount: check that SAM/MAM detection paths never silently fail
- Verify WHO data usage: LMS method from Excel, not deprecated CSV files
- Check that manual measurements always override estimated values
- Ensure ML weight estimates are validated against WHO median bounds (45-180%)
- Review for OWASP top 10 in API endpoints (input validation, SQL injection, file upload security)
- Verify test coverage for new code — especially edge cases in z-score computation
- Check cross-language consistency: Dart models must match Python schemas
- Flag any hardcoded magic numbers — they should be in config.py or constants
- Verify that assessment responses include confidence scores and method indicators

## test-engineer
**Description**: Test strategist covering Python backend tests and Flutter widget/unit tests.
**Scope**: `tests/`, `flutter_app/test/`
**Instructions**:
- Python: pytest with TestClient for API, mock MediaPipe for unit tests
- Flutter: widget tests for screens, unit tests for providers and repositories
- Prioritize testing the assessment pipeline end-to-end: image -> pose -> measurements -> z-scores -> result
- Test WHO z-score edge cases: boundary ages (0, 24, 60 months), extreme heights, sex-specific tables
- Test ML inference with known inputs — verify deterministic outputs
- Test offline scenarios in Flutter: no network, partial sync, conflict resolution
- MUAC edge cases: age outside 6-59 month range, missing WHZ for estimation
- Never mock the WHO data files — use real data for integration tests
- Aim for: API route coverage 100%, service method coverage >= 90%, Flutter screen coverage >= 80%
```

- [ ] **Step 2: Verify AGENTS.md is readable**

Run: `head -5 AGENTS.md`
Expected:
```
# Agents

## flutter-dev
**Description**: Flutter mobile app developer specializing in Riverpod architecture and offline-first medical apps.
**Scope**: `flutter_app/`
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "Add AGENTS.md with 5 specialized agent personas"
```

---

### Task 3: Add hooks to settings.local.json

**Files:**
- Modify: `.claude/settings.local.json`

- [ ] **Step 1: Add hooks section to existing settings.local.json**

The file currently has only a `permissions` key. Add a `hooks` key at the top level alongside it. The final file should be:

```json
{
  "permissions": {
    "allow": [
      "Bash(wc:*)",
      "Bash(pip3 list:*)",
      "Bash(python3:*)",
      "Bash(source .venv/bin/activate)",
      "Bash(pip install:*)",
      "Bash(python -m pytest:*)",
      "Bash(pip uninstall:*)",
      "Bash(python:*)",
      "Bash(PYTHONPATH=\"/home/kushal/Documents/child-growth-monitor/child-growth-monitor-main\" python -m pytest:*)",
      "Bash(pip3 show tensorflow)",
      "Bash(pip3 show scikit-learn numpy)",
      "Bash(.venv/bin/pip install -q -r requirements.txt)",
      "Bash(.venv/bin/python -c \"import tensorflow; print\\(''TF'', tensorflow.__version__\\); import sklearn; print\\(''sklearn'', sklearn.__version__\\)\")",
      "Bash(PYTHONPATH=. .venv/bin/python ml/train.py)",
      "Bash(PYTHONPATH=. .venv/bin/python ml/evaluate.py)",
      "Bash(PYTHONPATH=. .venv/bin/python -m pytest tests/ -v --tb=short)",
      "Bash(git add README.md requirements.txt config.py scripts/fix_who_data.py app/schemas/assessment.py app/services/assessment_service.py app/services/measurement_service.py app/services/ml_service.py ml/__init__.py ml/generate_synthetic_data.py ml/models.py ml/train.py ml/evaluate.py ml/inference.py data/who_wfh_0_59m.csv data/who_whz_reference.csv tests/test_who_data_service.py)",
      "Bash(git add .gitignore)",
      "Bash(git commit:*)",
      "Bash(git push origin main)",
      "Bash(PYTHONPATH=. .venv/bin/python scripts/batch_assess.py --template)",
      "Bash(PYTHONPATH=. .venv/bin/python -c \":*)",
      "Bash(PYTHONPATH=. .venv/bin/python scripts/extract_best_frame.py --help)"
    ]
  },
  "hooks": {
    "afterEdit": [
      {
        "matcher": "flutter_app/**/*.dart",
        "command": "cd flutter_app && flutter analyze --no-pub 2>&1 | tail -5"
      },
      {
        "matcher": "**/*.py",
        "command": "PYTHONPATH=. .venv/bin/python -m py_compile $FILE 2>&1"
      }
    ]
  }
}
```

- [ ] **Step 2: Verify JSON is valid**

Run: `python3 -c "import json; json.load(open('.claude/settings.local.json')); print('valid')"`
Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add .claude/settings.local.json
git commit -m "Add afterEdit hooks for flutter analyze and python syntax check"
```

---

### Task 4: Final verification

- [ ] **Step 1: Verify all three files exist**

Run: `ls -la CLAUDE.md AGENTS.md .claude/settings.local.json`
Expected: all three files listed with recent timestamps.

- [ ] **Step 2: Verify git status is clean**

Run: `git status`
Expected: no uncommitted changes to CLAUDE.md, AGENTS.md, or .claude/settings.local.json.
