# Phase 4: combination-therapy analysis

Phase 4 asks whether the response to a declared combination differs from the response expected
under an additive model on one prespecified outcome scale. It operates on exactly one endpoint per
independent biological subject and preserves the endpoint definition, assignment mechanism,
uncertainty policy, exclusions, analysis-study identity, and analysis configuration in a
hash-bound report.

It does **not** infer a rejuvenation mechanism, establish efficacy, or label a combination
"synergistic" from the sign of one coefficient.

## Analysis boundary

```text
Phase 1 raw measurements                Phase 3 latent trajectories
  exact modality/feature/unit             prespecified latent state
  protocol visit windows                  filtered/smoothed uncertainty
             |                                      |
             | study_feature_endpoints()            | state_report_to_endpoints()
             v                                      v
                 SubjectEndpointBatch
           one outcome per independent subject
           follow-up + optional baseline + SE
                              |
                              v
                 FactorialCombinationAnalysis
      assignment cells -> factorial model -> interaction family
                              |
                              v
                  CombinationAnalysisReport
       estimates + HC3 uncertainty + cells + exclusions + hash
```

`SubjectEndpointBatch` is the deliberate boundary between longitudinal measurement processing and
factorial inference. Every endpoint shares one exact `Estimand`, and subject identifiers must be
unique. The batch hash binds the endpoints, exclusions, and upstream source artifact so a result
cannot silently drift away from its input.

For a latent-state endpoint batch, `source_artifact_hash` is exactly the source
`StateEstimationReport.artifact_hash`. `CombinationAnalysisReport` reconstructs the declared design
columns, complete factorial cell set, counts, degrees of freedom, coefficient statistics,
interaction family, assignment/covariance policies, multiplicity adjustment, confidence
intervals, warnings, and estimand identities when serialized data are loaded. Internally
contradictory report payloads fail validation even if each isolated field has the right type.
The report also records the canonical hash of the `Study` used for intervention assignments and
covariates; matching a study ID alone is not accepted as reproducible provenance.

## Creating subject endpoints

### From Phase 1 or other raw study data

`study_feature_endpoints()` uses the shared visit-alignment machinery. The caller supplies the
exact modality, feature, unit, baseline visit, endpoint visit, and visit windows. A dog missing
either selected visit is recorded in `SubjectEndpointBatch.excluded`; measurements from another
modality or unit are never silently substituted.

The raw bridge stores the follow-up value as `estimate` and the baseline value as
`baseline_estimate`. Raw observations usually lack a defensible endpoint standard error, so the
bridge marks that limitation and the factorial analysis should use unweighted endpoints unless a
validated upstream uncertainty model is added.

### From a Phase 3 latent-state report

`state_report_to_endpoints()` selects a prespecified state, places its terminal estimate in
`estimate`, its first estimate in `baseline_estimate`, and carries the terminal marginal standard
error into the endpoint. It excludes trajectories with too few time points.

The bridge deliberately does not invent a standard error for a change score: two marginal state
covariances do not provide the cross-time covariance needed for that calculation. Use the terminal
state as the outcome with `include_baseline_covariate=True`, or provide a separately validated
change-score uncertainty calculation.

## Prespecifying a 2 x 2 analysis

For interventions $A$ and $B$, the subject-level model is

\[
y_i = \beta_0 + \beta_A A_i + \beta_B B_i + \beta_{AB} A_iB_i
      + \gamma y_{i,\mathrm{baseline}} + \mathbf{x}_i^\mathsf{T}\boldsymbol{\delta}
      + \epsilon_i.
\]

The interaction coefficient is the cell contrast

\[
\beta_{AB}=\mu_{11}-\mu_{10}-\mu_{01}+\mu_{00}.
\]

It is therefore a **departure from additivity on the declared outcome scale**. Its biological
meaning changes with the endpoint direction and transform. A negative interaction can be favorable
for a frailty score where lower is better, unfavorable for a functional score where higher is
better, or simply an artifact of the chosen nonlinear scale. Call it synergy or antagonism only
when a scientific definition, causal design, and scale-specific interpretation justify that claim.

```python
from rejuvenationkit.combinations import (
    AssignmentMechanism,
    CovarianceEstimator,
    EndpointWeighting,
    FactorialCombinationAnalysis,
    FactorialCombinationConfig,
    MultiplicityMethod,
)

analysis = FactorialCombinationAnalysis(
    FactorialCombinationConfig(
        interventions=("rapamycin", "senolytic"),
        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        covariance_estimator=CovarianceEstimator.HC3,
        multiplicity_method=MultiplicityMethod.BENJAMINI_HOCHBERG,
        minimum_cell_size=16,
        include_baseline_covariate=True,
        endpoint_weighting=EndpointWeighting.INVERSE_VARIANCE_REQUIRED,
    )
)
report = analysis.analyze(study, endpoints=endpoint_batch)
```

The implementation fits individual-subject factorial regression. `HC3` is the default covariance
estimator and is useful when endpoint variance differs across cells, although it does not repair
clustering, informative missingness, poor randomization, or a misspecified mean model.

## Assignment and causal interpretation

`assignment_mechanism` has no default: the analysis must declare it. Set
`assignment_mechanism=RANDOMIZED` only for actual randomized assignment recorded in
`Subject.interventions`. A randomized label does not by itself guarantee causal validity: protocol
adherence, attrition, interference, endpoint construction, and the randomization unit still matter.

For nonrandom exposure, set `OBSERVATIONAL`. The report and every interaction then carry
`observational_assignment_noncausal`. Covariate adjustment may reduce measured imbalance, but the
tool does not convert observational comparisons into randomized evidence or claim control of
unmeasured confounding.

## Cell, identifiability, and missingness checks

All $2^k$ assignment cells are required for $k$ declared interventions. The analysis fails
closed when any cell has fewer than `minimum_cell_size` analyzable subjects, the design matrix is
rank deficient, it has no residual degrees of freedom, or its condition number exceeds the
configured maximum. The report records:

- assigned and analyzable subjects, mean endpoint, standard error, and exclusions for every cell;
- observed versus required cells and the smallest and largest analyzable cell sizes;
- design columns, rank, residual degrees of freedom, and condition number;
- all subjects excluded from the fitted model, the endpoint-batch artifact hash, and the exact
  source-artifact hash carried by that endpoint batch.

Endpoint and endpoint-exclusion subject IDs must all belong to the exact input `Study`. Every
excluded study subject is assigned to one declared factorial cell for auditing, including subjects
excluded for undeclared co-interventions, and the cell exclusion partitions must match the design
diagnostics exactly. Unknown or diagnostic-only exclusion IDs are rejected.

`MissingEndpointPolicy.ERROR` is appropriate when the analysis must stop on any missing endpoint.
`EXCLUDE` retains an explicit exclusion audit and warning, but does not correct attrition bias.
Likewise, undeclared co-interventions can either stop the analysis or be explicitly excluded; they
are never folded into a declared cell silently.

## Endpoint uncertainty and weighting

The endpoint standard error describes uncertainty in a subject's endpoint estimate; it is not the
between-subject residual standard deviation. Choose the policy before inspecting results:

- `UNWEIGHTED` gives every analyzable subject equal regression weight and warns when supplied
  endpoint standard errors are not propagated.
- `INVERSE_VARIANCE_REQUIRED` requires a positive standard error for every endpoint and fits with
  normalized inverse-variance weights.
- `INVERSE_VARIANCE_IF_COMPLETE` uses those weights only when every endpoint has an error;
  otherwise it fits unweighted and records a warning.

Precision weighting is appropriate only when standard errors are comparable, calibrated, and do
not encode treatment-dependent selection. A highly precise but biased endpoint should not receive
more influence. HC3 coefficient covariance and endpoint weighting solve different problems: HC3
addresses regression residual heteroskedasticity, while weighting uses declared per-subject
measurement precision.

## Interaction multiplicity

The multiplicity family contains every fitted term of order two or higher, through
`maximum_interaction_order`. With two interventions there is one interaction, so adjustment leaves
its p-value unchanged. With three or more interventions, use Benjamini-Hochberg or Bonferroni for
the prespecified family, or choose `NONE` explicitly. Main effects are reported as model
coefficients but are not part of this local interaction family.

Multiplicity adjustment does not protect against outcome shopping, trying many transforms, or
selecting models after seeing the data. Those choices require a broader prespecified analysis plan.

## Design recommendation

`recommend_two_by_two_design()` provides an approximate equal-allocation sample size for a
continuous Gaussian 2 x 2 interaction contrast. Inputs are target interaction, residual standard
deviation, alpha, power, number of multiplicity tests, and expected dropout. It reports analyzable
and enrollment counts per cell plus the expected interaction standard error.

This calculator assumes independent subjects, common residual variance, equal cell allocation,
and a two-sided normal approximation. It is not a planner for survival outcomes, repeated-measures
models, clustered clinics or litters, adaptive allocation, or uncertain variance estimates. Use
simulation or a design-specific method when those conditions matter.

## Synthetic canine example

[`phase4_factorial_combinations.py`](https://github.com/weston-wang/RejuvenationKit/blob/main/examples/phase4_factorial_combinations.py)
creates 72 entirely synthetic dogs in a balanced rapamycin-by-senolytic design. It demonstrates:

- deterministic, auditable dog-level randomization with 18 dogs per cell;
- a month-6 frailty endpoint adjusted for baseline;
- required endpoint standard errors and inverse-variance weighting;
- HC3 coefficient uncertainty and Benjamini-Hochberg interaction multiplicity;
- cell, rank, exclusion, and identifiability diagnostics;
- an approximate enrollment recommendation and a hash-bound report.

Run it from a development checkout:

```bash
python examples/phase4_factorial_combinations.py
```

The generated effect is a software fixture, not a rapamycin, senolytic, canine-aging, safety, or
efficacy result.
