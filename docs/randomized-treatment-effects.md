# Randomized treatment-effect inference

`RandomizedTreatmentEffectEvaluator` closes the gap between detecting an unusual animal and
estimating whether a randomized treatment group changed more than its controls.

## What it estimates

For every configured follow-up visit, the evaluator reports:

- the treated and control mean change from baseline for each channel;
- the difference in those changes, with a subject-bootstrap confidence interval;
- a covariance-aware omnibus statistic across all channels;
- a randomization-test p-value obtained by permuting treatment labels; and
- an out-of-fold unusual-trajectory score for every complete subject.

The feature-level estimate is the treated change from baseline minus the control change from
baseline. This removes stable baseline differences. Random assignment remains the basis for a
causal interpretation.

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
from rejuvenationkit import RandomizedTreatmentEffectEvaluator, TreatmentEffectConfig

evaluator = RandomizedTreatmentEffectEvaluator(
    TreatmentEffectConfig(
        features=features,
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

- Group membership must be prespecified. The evaluator does not infer randomization from cohort
  names or intervention metadata.
- Confidence intervals are nonparametric subject-bootstrap intervals, not multiplicity-adjusted
  confirmatory intervals.
- The omnibus permutation test preserves correlation among channels but currently tests each
  follow-up separately.
- Missing observations are excluded visit by visit and reported. The estimator does not impute
  outcomes or correct informative dropout.
- Cross-validated trajectory detection measures departure from control behavior. It does not
  determine whether the direction is beneficial; inspect the signed channel effects.
- Stratified or blocked randomization, covariate adjustment, repeated-measures mixed models, and
  multiplicity policies require a prespecified statistical analysis plan beyond this Phase 1
  baseline.
