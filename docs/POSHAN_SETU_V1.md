# Poshan Setu v1 classification contract

This document is the implementation contract for the classification described
in the historical `Poshan Setu Formula.docx`.

## Formula

```text
BMI = weight_kg / (height_cm / 100)^2
```

BMI classification:

| Sex | SAM | MAM | Normal |
| --- | --- | --- | --- |
| Male | BMI < 13.0 | 13.0 <= BMI < 13.7 | BMI >= 13.7 |
| Female | BMI < 12.8 | 12.8 <= BMI < 13.5 | BMI >= 13.5 |

MUAC classification applies from the sixth monthly birthday until, but not
including, the fifth birthday (`6.0 <= age_months < 60.0`). In calendar terms,
this covers children aged 6 through 59 completed months:

| Status | MUAC |
| --- | --- |
| SAM | < 11.5 cm |
| MAM | 11.5 cm through < 12.5 cm |
| Normal | >= 12.5 cm |

The final classification is the most severe result from BMI and MUAC. SAM
always takes precedence over MAM, and MAM takes precedence over Normal.

## Missing or estimated inputs

Poshan Setu is a measurement-based classification. A WHO population median,
fixed fallback, WHZ-derived MUAC, or unvalidated ML estimate must not certify a
child as Normal.

- BMI is eligible only when weight and height are manual measurements or come
  from a separately validated measurement method.
- MUAC is eligible only when it is a tape/manual measurement and the child is
  at least 6 months old and has not yet reached the fifth birthday.
- If either eligible component is SAM, the final result is SAM even if the
  other component is unavailable.
- If both eligible components are available, the final result is their
  severity maximum.
- Otherwise the final result is `Indeterminate`. Any known MAM signal is
  retained in `triggered_by` and the rationale, but a missing component is
  never silently treated as Normal.

WHO HAZ/WHZ and ML predictions remain separate screening outputs. They do not
override the Poshan Setu result.

## Canonical wire values

Component and final statuses use:

- `SAM`
- `MAM`
- `Normal`
- `Indeterminate`

The classification method is `poshan_setu_v1`. Every stored assessment records
the effective values and their sources:

- `manual`
- `reference_object`
- `ml_estimated`
- `who_statistical`
- `whz_derived`
- `landmark_estimated`
- `unavailable`

Only sources explicitly marked as eligible by the classifier participate in
the final result.
