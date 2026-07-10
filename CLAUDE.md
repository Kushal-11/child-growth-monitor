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

## Pushing Commits
- Author all commits as: `Kushal-11 <kushaltherokar1010@gmail.com>` (GitHub: Kushal-11)
- NEVER add `Co-Authored-By: Claude` or any AI/Claude co-author trailer
- NEVER add "Generated with Claude Code", "🤖 Generated", or any AI-attribution line to commit messages or PR bodies
- Commit messages must contain only the human-authored description — no tool/assistant attribution of any kind
- Ensure author identity before pushing (matches `git config user.name`/`user.email` above); if unset, configure with:
  `git config user.name "Kushal-11"` and `git config user.email "kushaltherokar1010@gmail.com"`
