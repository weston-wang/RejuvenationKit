# Phase 1: longitudinal quality control

`BaselineLongitudinalQC` is the dependency-light reference pipeline. It accepts a validated
`Study` and an immutable `QCConfig`, then returns a `QCReport` without changing the study.

## Implemented checks

| Check | Finding code | Default severity |
|---|---|---|
| Non-finite numeric value | `nonfinite_value` | Error |
| Unexpected measurement unit | `unexpected_unit` | Error |
| Value outside a configured range | `out_of_range` | Error |
| Required feature absent for subjects | `required_feature_missing` | Warning or error |
| Input rows not in chronological order | `timestamp_out_of_order` | Warning |
| Technical replicates exceed tolerance | `replicate_disagreement` | Warning |
| Batch mean differs from other batches | `batch_mean_shift` | Warning |

Batch detection uses a standardized mean difference based on pooled within-group variance. It is
a screening diagnostic, not proof of a batch effect. Confounding between batch, treatment, time,
and cohort must be assessed before correction.

## Configuration

```python
from rejuvenationkit import BaselineLongitudinalQC, FeatureRule, QCConfig

config = QCConfig(
    feature_rules=(
        FeatureRule(
            feature="body_mass",
            expected_unit="g",
            minimum=10,
            maximum=60,
            required=True,
        ),
    ),
    replicate_relative_tolerance=0.20,
    batch_z_threshold=3.0,
    minimum_batch_size=3,
)

report = BaselineLongitudinalQC(config).run(study)
print(report.summary())
```

Every finding includes a stable code, severity, message, affected subject identifiers,
observation indices, and check-specific context. Downstream workflows should branch on codes,
not human-readable messages.

## Interpretation limits

- Missingness currently means a required feature is absent for a subject across the study. Visit-
  level schedules will be added when the study schema gains explicit expected visits.
- Replicate checks compare measurements with the same subject, timestamp, modality, feature, and
  batch but distinct replicate identifiers.
- Batch screening requires the configured minimum number of observations both inside and outside
  the evaluated batch.
- A passing report establishes only that configured checks found no errors.

