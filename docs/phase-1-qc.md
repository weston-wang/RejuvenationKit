# Phase 1: longitudinal quality control

`BaselineLongitudinalQC` is the dependency-light reference pipeline. It accepts a validated
`Study` and an immutable `QCConfig`, then returns a `QCReport` without changing the study.

`StudyProfiler` uses that same configuration to quantify analysis readiness rather than only
flagging failures. It reports feature coverage at each visit, complete-case retention between
consecutive visits, and the number of subjects available for paired feature analyses.
It also summarizes visit-level distributions, identifies robust Tukey-IQR outliers, and compares
baseline values between retained and missing-follow-up subjects.

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
| Cohort or intervention is associated with site, clinic, plate, run, operator, lot, manufacturing batch, or sequencing lane | `experimental_assignment_confounding` | Warning or error |
| Expected visit/timepoint is associated with an experimental factor | `timepoint_factor_confounding` | Warning or error |

Batch detection uses a standardized mean difference based on pooled within-group variance. It is
a screening diagnostic, not proof of a batch effect. Confounding between experimental factors,
treatment, time, and cohort is screened separately using Cramér's V.

## Configuration

```python
from datetime import UTC, datetime, timedelta

from rejuvenationkit import (
    BaselineLongitudinalQC,
    ExpectedVisit,
    FeatureRule,
    QCConfig,
    StudyProfiler,
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
    confounding_warning_threshold=0.50,
    batch_confounding_threshold=0.80,
)

report = BaselineLongitudinalQC(config).run(study)
print(report.summary())

profile = StudyProfiler(config).profile(study)
print(profile.coverage_frame())
print(profile.retention_frame())
print(profile.paired_readiness_frame())
print(profile.distributions_frame())
print(profile.attrition_bias_frame())
```

Every finding includes a stable code, severity, message, affected subject identifiers,
observation indices, and check-specific context. Downstream workflows should branch on codes,
not human-readable messages.

Configured range bounds and observation standard errors must be finite. Non-finite measured values
remain representable at ingestion so QC can report them explicitly, but they never enter visit-level
analysis vectors.

## Analysis-readiness profiling

The profile contains typed readiness tables plus the structured visit-alignment exclusions that
explain every omitted profile value:

| Table | Question answered |
|---|---|
| `visit_coverage` | How many eligible subjects have each required feature at each visit? |
| `visit_retention` | How many complete cases remain complete at the next scheduled visit? |
| `paired_readiness` | How many subjects can support a paired comparison for each shared feature? |
| `feature_distributions` | What are the visit-level quantiles, spread, and robust outliers? |
| `attrition_bias` | Do retained and missing-follow-up subjects differ at baseline? |
| `differential_attrition` | Do treatment/cohort arms lose complete cases at different rates? |
| `longitudinal_exclusions` | Why was a subject/visit/channel omitted from profiling? |

Every table includes an `all` cohort summary and separate rows for each actual cohort. Coverage
uses finite measurements inside the configured inclusive visit window. Complete-case retention
requires every feature specified for both visits; paired readiness is feature-specific and is
therefore often less restrictive.

Because `all` names the aggregate, a study that combines a literal `all` cohort with other cohort
labels is rejected rather than silently overwriting that summary. Serialized rows revalidate
count/fraction arithmetic, quantile ordering, outlier counts, and attrition contrasts before a
profile can be loaded into an audit.

Distribution rows aggregate repeated matching observations within a subject before calculating
the mean, sample standard deviation, quartiles, extrema, and Tukey fences. Values below
`Q1 - 1.5 × IQR` or above `Q3 + 1.5 × IQR` are reported by subject identifier. The multiplier is
configurable on `StudyProfiler`; outliers are diagnostic flags, not automatic exclusions.

Attrition diagnostics are feature-specific. For every consecutive pair of visits, subjects with a
finite baseline value are divided according to whether that feature is present at follow-up. The
profile reports both baseline means and their standardized mean difference (retained minus
attrited, divided by pooled within-group standard deviation). It returns no standardized estimate
when either group has fewer than two observations or pooled variance is zero.

Profiles quantify usable data but do not test treatment effects. In a cross-sectional study where
different subjects are collected at each age, visit coverage remains useful while paired
longitudinal readiness is not scientifically applicable.

## One-command audit and analysis gate

`run_phase1_audit` can include prespecified held-out pairwise detection, held-out sequential
detection, and randomized treatment-effect inference in the same integrity-tracked bundle. A
`SequentialDetectionAuditPlan` declares at least three ordered visit identifiers, the detector
configuration, and disjoint reference and evaluation subject identifiers. Sequential outputs
include subject summaries, transition-level trajectories, and (when visualization is enabled)
trajectory, classification, and modality-evidence figures.

Inferential and detection plans do **not** run when the QC report contains an error. The audit is
still published, records `analysis_blocked_by_qc=true`, retains each requested plan, and explains
the gate in `summary.md`. If a protocol owner has a documented reason to continue, set
`Phase1AuditConfig(allow_analysis_with_qc_errors=True)`. The serialized report then records
`analysis_override_applied=true`; this is an audit trail, not a claim that the underlying error is
harmless.

All visit-aligned analyses use the shared exact-channel contract: feature, modality, unit,
within-window aggregation policy, selected source-row indices, and effective observed timestamp.
The bundle always includes `longitudinal_exclusions.csv`, even when it contains only its header.
Every omitted value, incomplete vector, or excluded trajectory is represented by a structured
reason, including omissions produced while building the readiness profile, and `audit.json`
includes counts by analysis and reason. Identical exclusions are deduplicated within an analysis;
the same omission remains separately attributed when it affects profiling and an optional
inferential analysis. Detection plots label exact units and aggregation policies; sequential
trajectory plots use selected observation timestamps rather than nominal visit spacing.

Bundle files are first rendered in a staging directory. Only a complete staged bundle is
published, and artifacts listed by a previous manifest but absent from the new run are removed.
Files in the output directory that were not managed by the previous manifest are preserved.

## Randomized treatment effects

After QC and readiness profiling pass, `RandomizedTreatmentEffectEvaluator` compares prespecified
treated and control groups across one or more follow-up visits. It produces signed,
feature-specific differences in change from baseline, bootstrap confidence intervals, a
covariance-aware permutation test, and subject-level scores calibrated entirely from out-of-fold
controls. Its configuration requires an explicit randomized-assignment declaration and rejects
observational assignment rather than presenting confounded comparisons as randomized inference.
See [Randomized treatment-effect inference](randomized-treatment-effects.md) for the workflow and
interpretation limits.

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

## Experimental confounding

For each modality and feature, the baseline pipeline builds contingency tables for cohort and
every named intervention against available experimental nuisance factors. The default factors
are:

- `batch_id`;
- `site` and `clinic`;
- `plate` and `assay_run`;
- `operator`;
- `vector_lot` and `manufacturing_batch`; and
- `sequencing_lane`.

The named factors are read from `Observation.attributes`, with fallback to
`Subject.attributes`. This permits site to remain a subject-level assignment while plate and
assay run vary by sample.

A diagnostic is evaluated when:

- there are at least two assignment groups and two nuisance levels;
- every comparison group has at least `minimum_confounding_group_size` subjects; and
- Cramér's V meets `confounding_warning_threshold`, which defaults to 0.50.

Associations below `batch_confounding_threshold` are warnings; associations at or above that
threshold are errors. The legacy `batch_assignment_confounding` code is preserved for batch
assignment errors. Other factors use `experimental_assignment_confounding`.

Assignment-versus-factor screening uses one independent subject as the unit of analysis and only
includes subjects whose factor level is stable for that assay/feature. Subjects observed across
multiple plates, lots, or sites are counted in `excluded_changing_factor_subjects` rather than
pseudo-replicated. Timepoint screening uses one unique subject-visit record, excludes records with
multiple factor levels, reports them in `excluded_changing_factor_records`, and labels the unit in
the finding context. Cramér's V remains a descriptive screening statistic, not a clustered
hypothesis test.

When expected visits are configured, the same engine tests whether visit identity is associated
with each factor. This catches longitudinal designs in which all baseline samples use one plate or
assay version and all follow-up samples use another. Such designs cannot distinguish biological
change from processing change.

This does not claim that the nuisance factor caused the outcome. It says the intended biological
contrast cannot be cleanly separated from experimental handling. Disable all generalized checks
with `check_experimental_confounding=False`; `check_batch_confounding=False` disables only the
legacy batch factor.

Custom factors can be configured explicitly:

```python
from rejuvenationkit import ExperimentalFactor, QCConfig

config = QCConfig(
    experimental_factors=(
        ExperimentalFactor(name="site", attribute="site"),
        ExperimentalFactor(name="capsid_lot", attribute="capsid_lot"),
        ExperimentalFactor(name="histology_scanner", attribute="scanner_id"),
    )
)
```

## Interpretation limits

- Global `FeatureRule(required=True)` still checks whether a subject has the feature anywhere in
  the study. `ExpectedVisit` adds protocol-specific visit-level completeness.
- Overlapping visit windows are allowed only when distinct measurements resolve to the requested
  visits. One source observation cannot satisfy multiple visits for the same subject and channel:
  QC emits `observation_reused_across_expected_visits`, and longitudinal extraction excludes the
  reused visit-channel before paired or sequential inference.
- Exact and wildcard requests for the same feature cannot coexist when their modalities overlap,
  including inside `ExpectedVisit.required_features`. They could otherwise resolve to the same
  channel and count one measurement twice. Exact requests for the same feature remain valid when
  they declare distinct modalities. Serialized extractions bind every value to the exact declared
  subject, visit, channel index, and channel identity; visit-vector collections require unique
  subject/visit keys and one shared exact channel axis.
- Within each subject and channel, selected observation times must increase in the declared visit
  order. A later-declared visit selected at the same or an earlier time is excluded with
  `nonchronological_observed_time` rather than being converted into a backward change score.
- Replicate checks compare measurements with the same subject, timestamp, modality, feature, and
  batch but distinct replicate identifiers.
- Batch screening requires the configured minimum number of observations both inside and outside
  the evaluated batch.
- Confounding screening covers cohort and binary exposure to each named intervention. Encode dose,
  construct, or indication groups as cohorts until arbitrary assignment-factor policies are added.
- A passing report establishes only that configured checks found no errors.
- Retention is calculated only between consecutive visits in configuration order.
- A zero denominator produces a fraction of zero rather than an undefined or infinite value.
- Attrition standardized differences are descriptive screening metrics, not hypothesis tests or
  evidence that missingness is causal.
