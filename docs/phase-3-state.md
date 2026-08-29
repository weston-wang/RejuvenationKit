# Phase 3: prespecified longitudinal state estimation

Phase 3 estimates an uncertain latent trajectory from repeated, partially observed measurements.
The implemented baseline is a continuous-time linear-Gaussian model with Kalman filtering,
Rauch--Tung--Striebel (RTS) smoothing, future-state forecasting, and held-out innovation
diagnostics.

This is a **state-estimation framework**, not an algorithm that discovers biological age. The
researcher must define the state, measurement relationships, dynamics, units, and uncertainty
before evaluating a study. A latent coordinate called `inflammatory_burden`, for example, has
only the meaning supplied by its prespecified and independently justified loadings.

## Model contract

`LinearGaussianStateConfig` declares the continuous-time process

\[
dx(t) = \left(Ax(t) + d\right)dt + L\,dW(t), \qquad LL^\mathsf{T}=Q_c,
\]

and every `StateChannel` declares one observation equation

\[
y_j(t) = b_j + H_jx(t) + \epsilon_j(t).
\]

The required configuration is fully typed and immutable:

| Field | Meaning |
|---|---|
| `state_names` | Ordered, prespecified latent coordinates. |
| `channels` | Exact modality, feature, unit, loading, offset, and assay variance for each measurement. |
| `continuous_dynamics` | State-transition-rate matrix \(A\). |
| `continuous_process_covariance` | Continuous process covariance \(Q_c\). |
| `continuous_drift` | Constant drift rate \(d\). |
| `initial_mean`, `initial_covariance` | Prior state distribution at a subject's first observed time. |
| `time_unit_days` | Wall-clock days represented by one model time unit. |
| `smooth` | Whether `estimate()` returns retrospective RTS-smoothed rather than filtered states. |

Dimensions, finiteness, unique channel identities, and positive-semidefinite covariances are
validated up front. Channel matching is exact on `(modality, feature, unit)`. A familiar feature
with the wrong unit fails rather than being silently converted. Duplicate values for one channel
at one subject timestamp also fail because their correlation and aggregation rule would otherwise
be ambiguous.

`fit()` does not learn the dynamics, loadings, noise, or state definition from the evaluation
subjects. It validates the declared model against a study and records provenance metadata. Model
selection and parameter estimation therefore remain an upstream validation responsibility.

## Artifact provenance and lineage

Phase 3 attaches deterministic SHA-256 identities at each boundary:

| Artifact | What its identity binds |
|---|---|
| Study | The study ID plus complete subjects, observations, and metadata. Subject and observation ordering does not change the identity. |
| Model configuration | Every state name, channel identity and loading, dynamic, covariance, drift, prior, time scale, and smoothing default. |
| State trajectory | All estimates, coverage, observation exclusions, study/config identities, and any parent-trajectory identity. |
| State report | The selected and excluded subject partition, fit-reference IDs, smoothing choice, complete trajectories, study identity, and model identity. |
| Forecast calibration or change-point report | Study/model identities, exact reference and evaluation partitions, hashes of both diagnostic collections, calibration settings, and results. |

`fit()` stores the study and model-configuration identities. Estimation, diagnostics,
calibration, and change-point methods recompute both and reject a changed study ID, changed study
content, or changed configuration. A `StateEstimationReport` additionally verifies that its
trajectories carry the same study and model identities, use the declared state names and estimate
kind, and—together with explicit exclusions—exactly partition the requested subjects.

Each report exposes a computed `artifact_hash` over its complete serialized content. Forecast
trajectories preserve the source study and model hashes and store the input trajectory's
`artifact_hash` as `source_trajectory_artifact_hash`, creating an explicit parent-child lineage.
Forecast calibration and innovation change-point reports separately hash the reference and
held-out diagnostic collections, so a partition or diagnostic change yields a different result
identity.

The explicit Phase 3-to-4 bridge and the integrated workflow use that same
`StateEstimationReport.artifact_hash`; they do not create a second wrapper-specific identity for
the report. Serialized reports also reject duplicate trajectory subjects, duplicate exclusions,
or any subject represented in both partitions. Serialized forecast-calibration and innovation
reports reject overlapping reference/evaluation subjects; innovation detections must also agree
with the recorded per-innovation threshold.

These are content-addressed provenance checks, not digital signatures. Detecting later alteration
requires retaining and comparing the expected hash; a matching hash does not establish biological
validity, causal identification, or independent calibration.

## Irregular visits and partial channels

Elapsed wall-clock time is converted through `time_unit_days`. Matrix exponentials and the Van
Loan construction discretize the transition, constant drift, and integrated process covariance
for each actual interval. A 17-day gap and a 73-day gap therefore do not receive the same update.

At a time point with only some configured channels, the filter constructs the observation update
from the available rows. It does not impute absent assays. A row-level `Observation.standard_error`
is squared and added to the channel's prespecified measurement variance. `StateCoverage` reports
observed and missing channel-time combinations, including per-channel standard-error coverage.

Two boundaries matter:

- Time points are inferred from timestamps having at least one configured observation. A visit at
  which **all** configured measurements are absent is not invented by the state estimator; detect
  it with Phase 1 `ExpectedVisit` QC before state estimation.
- Observations that match no configured channel are retained as explicit
  `ObservationExclusion` records. A subject with no configured observations is returned in
  `excluded_subjects`, not silently omitted.

## Filtering, smoothing, and uncertainty

`filter()` uses information available through each timestamp. `smooth()` performs a retrospective
RTS pass, so an earlier estimate can use later measurements from the same subject. Filtered and
smoothed means always include the full covariance matrix, not only a point estimate. The Joseph
covariance update and numerical positive-semidefinite checks protect the uncertainty calculation
from common floating-point failure modes.

Use filtering for prospective monitoring. Use smoothing for a completed longitudinal analysis
when retrospective use of later observations matches the estimand. Do not present a smoothed
trajectory as if it had been available in real time.

`forecast()` propagates the terminal posterior to strictly increasing, timezone-aware future
timestamps. Process uncertainty accumulates over the actual forecast horizon. Forecasts are model
conditional: a narrow covariance is not proof that the biological model is valid.

## Held-out forecast calibration

`one_step_diagnostics()` records predictions immediately before each follow-up update. It returns
the observed and predicted channel values, innovation covariance, marginal standardized
innovations, normalized innovation squared (NIS), degrees of freedom, and the contribution from
reported standard errors.

`calibrate_forecasts()` requires nonempty, **disjoint** reference and evaluation subject sets. It
selects an empirical absolute-standardized-innovation threshold using reference subjects only and
reports coverage, bias, and RMSE on held-out subjects. The empirical threshold uses a conservative
higher quantile without interpolation.

This result checks whether declared predictive uncertainty transfers to held-out subjects. It does
not repair a misspecified model and does not establish treatment efficacy.

## Innovation change points

`detect_innovation_change_points()` also requires disjoint reference and evaluation subjects. For
each follow-up it scores `NIS / observed_channels`, sets a threshold from reference innovations at
the declared false-alarm rate, and reports a finite-sample empirical tail probability. Partial
channel visits naturally carry their own degrees of freedom. The declared rate and tail
probability apply to each innovation comparison. They do not control the familywise false-alarm
probability across all follow-ups for one subject or across a complete evaluation cohort. The
machine-readable report records this as
`false_alarm_scope="per_innovation_no_trajectory_multiplicity_control"`.

A detection means the new measurement was surprising under the prespecified reference dynamics.
It does **not** determine whether the cause was treatment response, adverse effect, assay drift,
infection, changed medication, or another event. Phase 1 QC and experimental metadata remain
necessary for interpretation.

## Minimal workflow

```python
from rejuvenationkit.state import (
    LinearGaussianStateConfig,
    LinearGaussianStateEstimator,
    StateChannel,
)

config = LinearGaussianStateConfig(
    state_names=("inflammatory_burden", "functional_reserve"),
    channels=(
        StateChannel(
            name="inflammation-score",
            modality="transcriptomics",
            feature="inflammation_score",
            unit="z_score",
            loadings=(1.0, 0.0),
            measurement_variance=0.20,
        ),
        StateChannel(
            name="activity-score",
            modality="wearable",
            feature="activity_score",
            unit="z_score",
            loadings=(0.0, 1.0),
            measurement_variance=0.15,
        ),
    ),
    continuous_dynamics=((0.0, 0.0), (0.0, 0.0)),
    continuous_process_covariance=((0.03, 0.0), (0.0, 0.025)),
    continuous_drift=(0.05, -0.04),
    initial_mean=(0.0, 0.0),
    initial_covariance=((0.5, 0.0), (0.0, 0.5)),
    time_unit_days=30.0,
    smooth=True,
)

estimator = LinearGaussianStateEstimator(config).fit(
    study,
    reference_subject_ids=reference_subject_ids,
)
report = estimator.estimate_report(study, subject_ids=evaluation_subject_ids)
calibration = estimator.calibrate_forecasts(
    study,
    reference_subject_ids=reference_subject_ids,
    evaluation_subject_ids=evaluation_subject_ids,
    interval_level=0.90,
)
changes = estimator.detect_innovation_change_points(
    study,
    reference_subject_ids=reference_subject_ids,
    evaluation_subject_ids=evaluation_subject_ids,
    false_alarm_rate=0.05,
)
```

See
[`phase3_longitudinal_state.py`](https://github.com/weston-wang/RejuvenationKit/blob/main/examples/phase3_longitudinal_state.py)
for a reproducible synthetic example with irregular visit spacing, a missing wearable channel,
one deliberately injected trajectory shift, retrospective smoothing, calibration, and forecasting.

## Phase boundaries

### Phase 2 to Phase 3

The estimator consumes timestamped scalar `Observation` rows in a `Study`. A Phase 2 fusion result
must not be treated as a time series merely because its estimand contains a textual time contrast.
`TimedEvidenceMeasurement` and `TimedEvidenceBatch` provide the explicit bridge: every value must
declare one subject, aware timestamp, feature, modality, unit, standard error, exact estimand, and
fusion-eligible calibration reference. `TimedEvidenceBatch.to_study()` preserves those calibration
and source-artifact identities in the generated observations. Rows mapped to the same Phase 3
channel must share one complete estimand, species, tissue, assay identifier, and calibration
reference; mixing any of those meanings (including present versus missing assay identity) under
one feature and unit is rejected. Evidence IDs are unique within a batch, so one Phase 2 estimate
cannot be injected at multiple Phase 3 times. Subject and measurement row order is canonicalized,
so merely reordering an otherwise identical batch does not change its artifact identity or
generated study.

High-dimensional expression, methylation, or proteomic matrices should remain in their Phase 2
contracts until a prespecified signature or independently calibrated target produces such a
subject-level scalar measurement. Database enrichment and interaction-network context are not
state observations.

### Phase 3 to Phase 4

Phase 4 factorial inference operates on exactly one declared `SubjectEndpoint` per independent
subject. A state trajectory is not automatically an endpoint. The researcher prespecifies the
state coordinate, exact endpoint estimand, whether a baseline covariate is required, and minimum
trajectory length in `StateEndpointConfig`. `state_report_to_endpoints()` then selects each
subject's terminal state, carries its marginal standard error, records the first state as the
baseline covariate, and audits excluded trajectories in a provenance-bound `SubjectEndpointBatch`.
It deliberately refuses to invent a change-score standard error because marginal covariances do
not identify cross-time covariance.

This explicit boundary prevents choosing the most favorable state, time point, or transformation
after looking at treatment assignments.

## Interpretation limits and guardrails

- The latent state is only as meaningful as the externally justified observation and dynamics
  model. A convenient name is not biological validation.
- Linear-Gaussian dynamics do not represent every aging process. Nonlinearity, censoring, assay
  limits, and discrete events may require a different model.
- State coordinates must be observable through the declared channel loadings. Passing schema
  validation alone does not demonstrate practical identifiability.
- Measurement variances, process covariance, and loadings should be estimated or validated
  outside the subjects used for confirmatory evaluation.
- Empirical calibration needs enough independent reference subjects and follow-ups. Repeated
  innovations within a subject are not automatically independent biological replicates.
- Per-innovation change-point calibration does not provide subject-level, trajectory-level, or
  cohort-level familywise false-alarm control across repeated testing.
- Change points are anomaly signals, not causal effects or safety classifications.
- Forecast intervals are conditional on the model and currently do not include uncertainty from
  estimating model parameters upstream.
- All timestamps must be timezone-aware, and units are never converted implicitly.
- Synthetic examples test software behavior; they do not estimate real rejuvenation or rapamycin
  effects.
