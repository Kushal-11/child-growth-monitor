# Poshan Setu v1 classification contract

Poshan Setu is the final programme classification. WHO HAZ/WHZ and ML outputs
remain separate screening evidence and do not override it.

Eligible BMI requires measured height and weight whose source is `manual` or
`reference_object`. BMI thresholds are:

| Sex | SAM | MAM | Normal |
| --- | --- | --- | --- |
| M | `< 13.0` | `13.0 to < 13.7` | `>= 13.7` |
| F | `< 12.8` | `12.8 to < 13.5` | `>= 13.5` |

Eligible MUAC requires a manual/tape value and age from 6 months through
59.999 months. MUAC `< 11.5 cm` is SAM, `11.5 to < 12.5 cm` is MAM, and
`>= 12.5 cm` is Normal.

Any eligible SAM component finalizes SAM. Otherwise both BMI and MUAC must be
eligible: the more severe component finalizes MAM or both Normal finalizes
Normal. Missing or estimated evidence produces `Indeterminate`.

Canonical final values are `SAM`, `MAM`, `Normal`, and `Indeterminate`.
`classification_method` is `poshan_setu_v1`. Sync uploads contain raw evidence;
the backend recomputes the programme result and never trusts a client verdict.
