# CLAUDE.md + AGENTS.md + Hooks Configuration Design

**Date**: 2026-04-05
**Status**: Approved
**Scope**: Project-level Claude Code configuration for Child Growth Monitor

## Summary

Set up three configuration layers for the Child Growth Monitor project:

1. **CLAUDE.md** — Project conventions and rules (Flutter-first, Riverpod, Python backend, ML safety constraints)
2. **AGENTS.md** — Five specialized agent personas scoped to different parts of the codebase
3. **Hooks in settings.json** — Lightweight auto-checks on file edits (flutter analyze, python syntax)

## Context

The project is a medical-grade child growth monitoring system with:
- FastAPI + SQLAlchemy backend (Python)
- Flutter mobile app (currently basic, migrating to Riverpod)
- ML pipeline for wasting detection (TensorFlow/Keras, TFLite export)
- WHO 2006 growth standards for z-score computation
- Safety-critical: false negatives for SAM/MAM can endanger lives

The user's primary focus is Flutter app development, with ML architecture exploration as secondary.

## Design

### CLAUDE.md

Project-root file covering:
- **Project overview**: medical-grade, safety-critical context
- **Stack**: Flutter (Riverpod) + FastAPI + TensorFlow/Keras + MediaPipe + WHO data
- **Run commands**: all common dev commands with proper venv prefixes
- **Flutter conventions**: Riverpod, feature-first folders, ConsumerWidget, offline-first, repository pattern
- **Python conventions**: venv usage, type hints, service pattern, WHO data source of truth
- **ML conventions**: SAM recall floor, experiment logging, TFLite export, 14-feature interface
- **Safety rules**: WHO validation, manual override priority, ML weight bounds, MUAC thresholds
- **Testing**: pytest + flutter test, coverage targets, evaluate.py gating
- **Git**: commit style, ignore rules

### AGENTS.md

Five specialized agents:

| Agent | Scope | Focus |
|-------|-------|-------|
| flutter-dev | `flutter_app/` | Riverpod architecture, offline-first, Material 3, MOBILE_APP_SPEC.md |
| ml-researcher | `ml/`, `data/`, `notebooks/` | Architecture exploration, experiment tracking, SAM recall floor |
| backend-dev | `app/`, `config.py`, `main.py`, `scripts/` | FastAPI services, WHO data, assessment pipeline |
| code-reviewer | entire repo | Safety checks, cross-language consistency, OWASP, test coverage |
| test-engineer | `tests/`, `flutter_app/test/` | Test strategy, WHO edge cases, offline scenarios, coverage targets |

### Hooks (settings.json)

Two lightweight `afterEdit` hooks:
1. `flutter_app/**/*.dart` — runs `flutter analyze --no-pub` (tail last 5 lines)
2. `**/*.py` — runs `py_compile` on the edited file

Added to `.claude/settings.local.json` alongside existing bash permission allowlists.

## Decisions

- **Riverpod** chosen over BLoC/Provider for state management — modern, compile-safe, good fit for offline-first medical app
- **Flutter-first** priority — CLAUDE.md emphasizes Dart/Flutter conventions; Python/ML are secondary
- **Architecture exploration** for ML — agent is research-oriented, not locked to current TF/Keras stack
- **Lightweight hooks** — no full test suites on edit, just syntax/analyze checks to keep iteration fast
- **5 agents** — covers all active workstreams without overlap; code-reviewer is cross-cutting

## Files to Create/Modify

1. `CLAUDE.md` (new) — project root
2. `AGENTS.md` (new) — project root
3. `.claude/settings.local.json` (modify) — add hooks section
