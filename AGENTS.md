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
