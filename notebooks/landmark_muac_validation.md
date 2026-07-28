# Landmark MUAC paired-tape validation

## Clinical gate and current conclusion

The landmark ratio estimator **has not yet been validated on a real paired-tape
cohort in this repository**. The two illustrative rows in
`data/ground_truth_template.csv` are a data-entry example, not an eligible
validation sample. Consequently MAE, bias, sensitivity, specificity, and SAM
recall are **not reportable**, and the safety gate remains closed. Landmark
MUAC must request tape confirmation and must not make an autonomous definitive
clinical call. The release gate is SAM recall **>= 0.80** on the held-out paired
cohort; confidence intervals and sample counts must be reported alongside the
point estimate.

This explicit non-validation result prevents synthetic or template data from
being represented as clinical evidence.

## Prospective methodology

1. Enrol consecutive children aged 6–59 months across community and referral
   settings. Record site, age band (6–11, 12–23, 24–35, 36–47, 48–59 months),
   sex, oedema, skin tone, clothing, device/camera, and nutritional stratum.
2. Obtain duplicate left-arm MUAC tape measurements by trained assessors blinded
   to the image estimate. Adjudicate readings differing by more than 0.2 cm with
   a third reading; use the median as reference. Capture the image independently.
3. Freeze model and calibration versions before evaluation. Split by child and
   site; calibration subjects must never enter the held-out validation set.
4. Report sample composition and missingness, overall and stratified MAE,
   median absolute error, signed bias (estimate minus tape), 95% limits of
   agreement, and bootstrap 95% confidence intervals.
5. Against tape thresholds `<11.5 cm` (SAM) and `<12.5 cm` (acute malnutrition),
   report confusion matrices, sensitivity, specificity, PPV, NPV, and threshold
   sensitivity analyses at 0.1 cm increments from 11.0–13.0 cm. Report SAM
   recall as `TP / (TP + FN)` and its exact binomial 95% interval.
6. Examine performance by every recorded subgroup and landmark-visibility band.
   Prespecify exclusion criteria; count exclusions rather than silently dropping
   them. Use uncertainty intervals when evaluating threshold crossings.

## Required results record

| Version | Validation N (SAM N) | Sample composition | MAE (cm) | Bias (cm) | SAM sensitivity/recall | SAM specificity | <12.5 sensitivity | <12.5 specificity | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| landmark-ratio-v1 / unvalidated-paired-tape-v0 | 0 (0) | No eligible real paired cohort available | N/R | N/R | N/R | N/R | N/R | N/R | **Blocked** |

`N/R` means not reportable. Autonomous calls may only be enabled after this
record is replaced with reproducible held-out results demonstrating SAM recall
of at least 0.80 and the calibration version is advanced.
