# Randomized treatment-effect inference

`RandomizedTreatmentEffectEvaluator` closes the gap between detecting an unusual animal and
estimating whether a randomized treatment group changed more than its controls.

## What it estimates

For every configured follow-up visit, the evaluator reports:

- the treated and control mean change from baseline for each channel;
- the difference in those changes, with a bootstrap standard error and Welch t interval;
- a covariance-aware omnibus statistic across all channels;
- a randomization-test p-value obtained by permuting treatment labels, plus a Holm-adjusted
  p-value across the configured follow-up visits; and
- an out-of-fold unusual-trajectory score for every complete subject.

The feature-level estimate is the treated change from baseline minus the control change from
baseline. This removes stable baseline differences. Random assignment remains the basis for a
causal interpretation.

The estimand is the effect among subjects with complete channel vectors at that visit
(completers). Under the sharp null that treatment changes no outcome and no dropout, the
permutation test is exact for completers. The difference in means is causal for the full
randomized cohort only if completion is unrelated to treatment and outcome.

The configuration therefore requires an explicit
`assignment_mechanism=AssignmentMechanism.RANDOMIZED` declaration. The evaluator rejects
`OBSERVATIONAL` assignment before analysis: unrestricted treatment-label permutation is not a
valid substitute for a prespecified observational estimand and confounding model.

## Inference details

- **Omnibus statistic.** \(d^T S_W^{-1} d\), where \(d\) is the treated-minus-control mean
  change and \(S_W\) is the arm-centered, pooled within-group covariance (shrunk and ridge
  stabilized). It is recomputed for every permutation. Pooling without centering each arm would let
  a real effect inflate the covariance and hide itself.
- **Randomization strata.** Pass `randomization_strata={subject_id: stratum}` when treatment was
  randomized within blocks or strata. Labels are then permuted only within strata, matching the
  actual randomization distribution. In a simulated 8-per-arm stratified trial with a
  1.5 SD stratum difference, unrestricted permutation had a 1% type I error and 4% power. The
  stratified test had 5.3% and 25%. The strata are recorded in the report provenance.
- **Feature intervals.** Within-group deviations are inflated by \(\sqrt{n/(n-1)}\) before
  bootstrap resampling, so the bootstrap SE matches the unbiased \(\sqrt{s_t^2/n_t + s_c^2/n_c}\).
  The interval is estimate ± Welch–Satterthwaite t × SE. The previous percentile interval covered
  90% at nominal 95% with 8 subjects per arm; the corrected interval covered 95.5%. Feature
  intervals ignore strata, so they are conservative under stratified randomization with strong
  stratum effects.
- **Multiplicity.** `holm_adjusted_permutation_p_value` controls the family-wise error rate across
  follow-up visits. Channel-level intervals are not multiplicity-adjusted.

## Leakage-safe calibration

Subjects are assigned deterministically to cross-validation folds. For each fold:

1. The control mean and covariance are fitted using control animals outside the fold.
2. Held-out control animals produce out-of-fold null scores.
3. The detection threshold and empirical tail probabilities are calibrated from the combined
   out-of-fold control scores.
4. Treated and control animals are scored using the model for their own held-out fold.

Consequently, no animal contributes to the nuisance model used to score that animal, and no
in-sample control score is used as the empirical null.

```python
from rejuvenationkit import (
    AssignmentMechanism,
    RandomizedTreatmentEffectEvaluator,
    TreatmentEffectConfig,
)

evaluator = RandomizedTreatmentEffectEvaluator(
    TreatmentEffectConfig(
        features=features,
        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        cross_validation_folds=5,
        permutations=999,
        bootstrap_samples=999,
    )
)
report = evaluator.evaluate(
    study,
    baseline=baseline,
    follow_ups=(month_1, month_3, month_6),
    treated_subject_ids=treated_ids,
    control_subject_ids=control_ids,
    treated_label="rapamycin",
    control_label="placebo",
)
print(report.effects_frame())
print(report.scores_frame())
```

The runnable `examples/randomized_rapamycin_effect.py` demonstrates a synthetic 60-dog trial
with correlated inflammatory and frailty channels.

## Interpretation limits

- Group membership must be prespecified. The required assignment declaration records the design;
  the evaluator does not infer or verify randomization from cohort names, intervention metadata,
  or outcome patterns.
- Observational groups are rejected rather than relabeled as randomized. They require a separately
  justified observational analysis with an explicit estimand, covariate strategy, and sensitivity
  analysis.
- Channel-level confidence intervals are not multiplicity-adjusted confirmatory intervals.
- The omnibus permutation test preserves correlation among channels and tests each follow-up
  separately. Use the Holm-adjusted p-values for a claim at any follow-up.
- Missing observations are excluded visit by visit and reported. The estimator does not impute
  outcomes or correct informative dropout.
- Cross-validated trajectory detection measures departure from control behavior. It does not
  determine whether the direction is beneficial; inspect the signed channel effects.
- Covariate adjustment and repeated-measures mixed models require a prespecified statistical
  analysis plan beyond this baseline.
