# Surrogate-endpoint validation

The largest obstacle to faster aging trials is the lack of a validated surrogate endpoint. Such a
biomarker would let a months-long trial stand in for a years-long lifespan or healthspan study.
Epigenetic clocks, proteomic ages and frailty indices are all candidates. For a candidate to
count, the property that matters is:

> an intervention's effect on the biomarker predicts its effect on the outcome.

Biomarkers are usually validated differently: they are shown to correlate with age or mortality
across individuals. That does not establish the property above. A clock can track population
mortality risk yet shift under an intervention for reasons unrelated to how that intervention
changes risk. `rejuvenationkit.surrogates` implements the meta-analytic surrogate-evaluation
framework, which tests the property directly.

## Units

The analysis needs several independent **units**, each with its own treated and control groups:

- separate trials;
- sites within a multi-site trial;
- cohorts; or
- intervention arms that each have their own controls.

For every unit it needs the treatment effect on the surrogate and on the outcome, with their
standard errors and within-unit covariance. `unit_effects_from_subjects` computes these from a
subject-level table.

## Diagnostics

| Diagnostic | Question | Function |
|---|---|---|
| Trial-level R² | Across units, how much of the variation in outcome effects do surrogate effects explain, after removing each unit's sampling error? | `fit_trial_level`, `validate_surrogate` |
| Individual-level R² | Within arms, how strongly do the two endpoints correlate across subjects? | `individual_level_association` |
| Surrogate threshold effect (STE) | How large must a new trial's surrogate effect be before its predicted outcome effect is significantly nonzero? | `validate_surrogate` |
| Leave-one-unit-out prediction | Predicting each unit's outcome effect from a model fitted without it, how often does the interval cover the observed effect, and is the sign right? | `validate_surrogate` |

### Trial-level model

Let \((\alpha_i, \beta_i)\) be the true treatment effects on the surrogate and the outcome in unit
\(i\), with estimates \((\hat\alpha_i, \hat\beta_i)\) and sampling covariance \(W_i\). The
between-unit covariance of true effects is estimated by the method of moments:

\[
\hat\Sigma_b = \operatorname{Cov}(\hat\alpha, \hat\beta) - \overline{W}.
\]

Negative eigenvalues are clipped to zero, and the clipping is flagged in the report. From this
estimate:

- \(R^2_\text{trial} = \Sigma_{b,12}^2 / (\Sigma_{b,11}\Sigma_{b,22})\);
- slope \(= \Sigma_{b,12}/\Sigma_{b,11}\); and
- residual between-unit variance \(= \Sigma_{b,22} - \Sigma_{b,12}^2/\Sigma_{b,11}\).

Predictions for a new unit shrink its observed surrogate effect by the surrogate's reliability.
Their variance adds the residual between-unit variance, the surrogate's measurement error and the
parameter uncertainty.

Intervals come from a **parametric bootstrap**: units are simulated from the fitted model and the
model is refitted. A nonparametric bootstrap over units undercovered badly with few units,
because the moment estimator piles up at |correlation| = 1. It covered 61% at nominal 95% with 6
units. The parametric bootstrap covered 92–100% across 6–30 units. The R² interval is derived
from the interval for the signed correlation, so it includes 0 whenever the correlation interval
does.

## Validation by simulation

The harness was evaluated with units drawn from a known trial-level correlation, 20 subjects per
arm and 40 replicates per scenario:

| Units | True R²_trial | True R²_ind | Median R²_trial | R² interval coverage | Held-out prediction coverage |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.81 | 0.25 | 0.83 | 95% | 98% |
| 12 | 0.81 | 0.25 | 0.82 | 93% | 97% |
| 12 | 0.00 | 0.81 | 0.06 | 95% | 93% |
| 30 | 0.49 | 0.09 | 0.45 | 100% | 96% |

The third row is the **surrogate paradox**. The biomarker correlates strongly with the outcome
across individuals (R²_ind = 0.81), but its treatment effect predicts nothing about the outcome's
treatment effect. Individual-level evidence alone would endorse this biomarker. The trial-level
interval correctly includes zero.

## How many independent trials does validation need?

`simulate_trial_level_precision` answers this question before data are collected. With 80% of each
unit's observed effect variance being real signal:

| Units | True R²_trial | Median interval width | Probability the interval excludes R² ≤ 0.5 |
|---:|---:|---:|---:|
| 5 | 0.81 | 1.00 | 7% |
| 10 | 0.81 | 0.83 | 15% |
| 20 | 0.81 | 0.62 | 33% |
| 40 | 0.81 | 0.43 | 66% |
| 40 | 0.49 | 0.64 | 5% |

Even an excellent surrogate needs about 40 independent units before a trial-level R² above 0.5
is likely to be demonstrated. Aging research has only a handful of interventions with both
biomarker data and hard outcomes measured in the same animals. So claims that a clock is a
*validated* intervention surrogate are premature on statistical grounds alone. Pooling across
intervention arms of large multi-arm studies, each arm with its own controls, is the most
realistic route to enough units.

## Example

```bash
python examples/surrogate_validation.py
```

The example simulates twelve canine cohorts with epigenetic-age and frailty-index change as the
surrogate and outcome. It runs a valid surrogate and a paradoxical one. For the paradoxical
surrogate the diagnostics report:

- individual-level R² 0.82;
- trial-level R² 0.35, with an interval of 0.00–0.78;
- held-out sign agreement 75%; and
- no surrogate threshold effect within range.

## Limitations

- **Units must be independent.** Arms that share a control group violate this assumption, because
  their effect estimates share control noise. See the
  [concordance reanalysis](gse131754-concordance-reanalysis.md) for how large that artifact can be.
- **The method-of-moments fit is transparent but not efficient.** A full bivariate random-effects
  likelihood would be more precise when units differ greatly in size.
- **Linearity is assumed.** A surrogate that predicts benefit only above a threshold needs a
  different model.
- **Survival outcomes need care.** For survival, supply per-unit log hazard ratios and standard
  errors. The harness does not fit survival models.
- **Out-of-scope interventions.** Trial-level validation holds only for interventions resembling
  those in the validation set. A surrogate validated on dietary interventions may fail for a
  senolytic.
