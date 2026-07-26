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
| Entire expected visit absent | `expected_visit_missing` | Warning or error |
| Required feature absent at a partially completed visit | `visit_feature_missing` | Warning or error |
| Scheduled visit selects no study subjects | `expected_visit_has_no_subjects` | Warning |
| Measurement outside all applicable visit windows | `observation_outside_visit_window` | Warning |
| Input rows not in chronological order | `timestamp_out_of_order` | Warning |
| Technical replicates exceed tolerance | `replicate_disagreement` | Warning |
| Batch mean differs from other batches | `batch_mean_shift` | Warning |

Batch detection uses a standardized mean difference based on pooled within-group variance. It is
a screening diagnostic, not proof of a batch effect. Confounding between batch, treatment, time,
and cohort must be assessed before correction.

## Configuration

```python
from datetime import UTC, datetime, timedelta

from rejuvenationkit import (
    BaselineLongitudinalQC,
    ExpectedVisit,
    FeatureRule,
    QCConfig,
    VisitFeature,
)

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
    expected_visits=(
        ExpectedVisit(
            visit_id="week-4",
            scheduled_at=datetime(2026, 2, 1, tzinfo=UTC),
            window_before=timedelta(days=2),
            window_after=timedelta(days=2),
            required_features=(
                VisitFeature(feature="body_mass"),
                VisitFeature(feature="heart_rate"),
            ),
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

## Expected-visit semantics

Visit windows are inclusive. By default, a visit applies to every study subject. Set
`subject_ids`, `cohorts`, or both to narrow eligibility; when both are present, their union is
used.

For each eligible subject:

1. A finite required measurement within the window satisfies that feature.
2. No required measurements within the window produces `expected_visit_missing`.
3. At least one, but not all, required measurements produces `visit_feature_missing`.
4. A matching measurement outside every applicable window produces
   `observation_outside_visit_window`.

Missingness findings use the existing warning and error fraction thresholds. Observations outside
visit windows can be disabled with `check_observations_outside_visit_windows=False` when a study
permits unscheduled measurements.

## Interpretation limits

- Global `FeatureRule(required=True)` still checks whether a subject has the feature anywhere in
  the study. `ExpectedVisit` adds protocol-specific visit-level completeness.
- Schedules currently use absolute timestamps. Subject-relative schedules based on enrollment or
  intervention time require an explicit anchor in a future schema extension.
- Overlapping visit windows are allowed; a measurement may satisfy more than one visit.
- Replicate checks compare measurements with the same subject, timestamp, modality, feature, and
  batch but distinct replicate identifiers.
- Batch screening requires the configured minimum number of observations both inside and outside
  the evaluated batch.
- A passing report establishes only that configured checks found no errors.
