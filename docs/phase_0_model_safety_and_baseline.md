# Phase 0B/0C: provenance safety and reproducible mobile baseline

## Clinical boundary (Phase 0B)

Camera outputs are research screening evidence. They do not become effective
anthropometric measurements merely because a model returned a plausible value.

- Effective height: manual stadiometer/length-board entry, or an image estimate
  explicitly calibrated by a detected reference object.
- Effective weight: manual entry or a calibrated digital-scale source.
- Effective MUAC: direct tape measurement. Landmark and WHZ-derived MUAC remain
  experimental evidence requiring confirmation.
- HAZ, WHZ, SAM/MAM/normal, stunting, wasting, and growth-chart points use only
  eligible effective measurements. WHO medians never substitute for a child
  measurement.
- Image-derived height/weight and model probabilities remain available with
  their provenance and non-clinical status for model development.

## Frozen evaluation contract (Phase 0C)

`python ml/generate_synthetic_data.py` creates stable `child_id` values and a
70/15/15 train/calibration/test manifest. All trainers and evaluators reuse that
manifest. Splits are by child, never frame, so repeated observations of one child
cannot cross partitions.

The release baseline is produced by:

```bash
python ml/evaluate_tflite.py --json
```

This command verifies every file checksum in the Flutter model manifest and
runs the exact shipped `.tflite` models with the shipped scaler on the locked
test children. It reports five-class accuracy, SAM/MAM recall and precision,
weight MAE/median/p95 absolute error, and expected calibration error. Keras
validation metrics are development diagnostics, not the release baseline.

The shared `shared/ml_parity_cases.json` fixture locks the 14-feature ordering,
depth imputation, and explicit depth-ratio behavior across Python and Dart.

## Historical bundled-model baseline

The currently bundled synthetic research model was measured on the historical
60,000-row generator output (generator commit
`b8785b57f777364817de46b4bcf363e6c287e11f`; model assets introduced in
`58fef5055fbb1ad45996ebaf00566673c31f34a1`):

| Metric | Exact shipped TFLite result |
|---|---:|
| Test children | 8,931 |
| Five-class accuracy | 0.7211 |
| SAM recall / precision | 0.8835 / 0.2364 |
| MAM recall / precision | 0.5012 / 0.4185 |
| Weight MAE | 0.4019 kg |
| Weight median absolute error | 0.3211 kg |
| Weight p95 absolute error | 1.0491 kg |
| Expected calibration error | 0.0680 |

These numbers are synthetic-data engineering baselines, not evidence of field
or clinical accuracy. Any regenerated dataset receives a different fingerprint
and must publish a fresh baseline rather than reusing this table.
