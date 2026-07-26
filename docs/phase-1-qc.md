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
| Subject lacks an anchor required by a relative visit | `visit_anchor_missing` | Error |
| Input rows not in chronological order | `timestamp_out_of_order` | Warning |
| Technical replicates exceed tolerance | `replicate_disagreement` | Warning |
| Batch mean differs from other batches | `batch_mean_shift` | Warning |
| Cohort or intervention is strongly associated with batch | `batch_assignment_confounding` | Error |

Batch detection uses a standardized mean difference based on pooled within-group variance. It is
a screening diagnostic, not proof of a batch effect. Confounding between batch, treatment, time,
and cohort is screened separately using Cramér's V.

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

### Subject-relative visits

Store timezone-aware protocol anchors on each subject, then define one relative visit policy:

```python
subject = Subject(
    subject_id="dog-001",
    cohort="rapamycin",
    interventions=("rapamycin",),
    anchors={"first_dose": datetime(2026, 1, 8, tzinfo=UTC)},
)

month_one = ExpectedVisit(
    visit_id="month-1-after-dose",
    anchor_id="first_dose",
    offset=timedelta(days=28),
    window_before=timedelta(days=3),
    window_after=timedelta(days=3),
    required_features=(VisitFeature(feature="body_mass"),),
)
```

Each expected visit must use exactly one schedule mode: `scheduled_at` for a common calendar date,
or `anchor_id` plus `offset` for subject-relative timing. A missing anchor produces
`visit_anchor_missing`; it is not misclassified as a missed clinic visit.

## Treatment/batch confounding

For each modality and feature with batch identifiers, the baseline pipeline builds contingency
tables for cohort and every named intervention. It reports `batch_assignment_confounding` when:

- there are at least two groups and two batches;
- every comparison group has at least `minimum_confounding_group_size` subjects; and
- Cramér's V meets `batch_confounding_threshold`, which defaults to 0.80.

This catches designs such as every rapamycin sample being processed in batch A and every control
sample in batch B. It does not claim that batch caused the measured outcome; it says the treatment
effect cannot be cleanly separated from batch for that feature. Disable the diagnostic with
`check_batch_confounding=False` only when another documented analysis handles the issue.

## Interpretation limits

- Global `FeatureRule(required=True)` still checks whether a subject has the feature anywhere in
  the study. `ExpectedVisit` adds protocol-specific visit-level completeness.
- Overlapping visit windows are allowed; a measurement may satisfy more than one visit.
- Replicate checks compare measurements with the same subject, timestamp, modality, feature, and
  batch but distinct replicate identifiers.
- Batch screening requires the configured minimum number of observations both inside and outside
  the evaluated batch.
- Confounding screening currently covers cohort and binary exposure to each named intervention.
  Visit/timepoint and site confounding remain future extensions.
- A passing report establishes only that configured checks found no errors.
