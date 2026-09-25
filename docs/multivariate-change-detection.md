# Multivariate longitudinal change detection

`MultivariateChangeDetector` is the first DSP-oriented capability in RejuvenationKit. It asks
whether a subject's *joint* change across correlated biomarkers is unusual relative to a
reference population.

For a baseline vector \(x_{i,0}\) and follow-up vector \(x_{i,1}\), the detector forms

\[
\Delta x_i = x_{i,1} - x_{i,0}.
\]

It estimates the reference mean change \(\mu_\Delta\) and covariance \(\Sigma_\Delta\), shrinks
off-diagonal covariance toward a diagonal model, and scores a new innovation with squared
Mahalanobis distance:

\[
D_i^2 = (\Delta x_i-\mu_\Delta)^T
\Sigma_\Delta^{-1}
(\Delta x_i-\mu_\Delta).
\]

This is analogous to whitening a correlated sensor innovation before thresholding its energy.
The reported `whitened_innovation` shows each subject in those normalized coordinates.

## Calibration

The threshold is an empirical reference-score quantile selected by `false_alarm_rate`. Each
reference subject is scored against a mean and covariance fitted **without** that subject
(`reference_score_method="leave_one_out"`). In-sample Mahalanobis distances are biased low because
every subject helped fit the model it is scored against. With 20 reference subjects and four
channels, an in-sample threshold produced a 17% false-alarm rate on new subjects at a nominal 5%.
Leave-one-out scores are approximately exchangeable with held-out scores, so the realized rate stays
at or slightly below nominal. Fitting requires at least `feature count + 2` complete reference
subjects. Tail probabilities also use these reference scores, with a one-count correction.
Covariance shrinkage and a small ridge stabilize inversion when channels are correlated.

```python
detector = MultivariateChangeDetector(
    ChangeDetectionConfig(
        features=(
            VisitFeature(feature="albumin", modality=Modality.CLINICAL),
            VisitFeature(feature="creatinine", modality=Modality.CLINICAL),
        ),
        covariance_shrinkage=0.20,
        false_alarm_rate=0.05,
        minimum_reference_subjects=100,
    )
)

detector.fit(
    study,
    baseline=baseline_visit,
    follow_up=follow_up_visit,
    reference_subject_ids=calibration_ids,
)
report = detector.score(
    study,
    baseline=baseline_visit,
    follow_up=follow_up_visit,
    subject_ids=evaluation_ids,
)
```

## Serialized-model integrity and reconstruction

A fitted `ChangeDetectionModel` contains the exact fit configuration and visits, resolved
modality/feature/unit/aggregation channels, requested and complete reference-subject IDs, the
reference-input artifact hash, and the sorted reference-score distribution. The serialized
`threshold_quantile_method` is `higher`; validation recomputes the threshold from the stored
scores and `false_alarm_rate` rather than trusting the stored threshold alone.

`model_artifact_hash` is a deterministic SHA-256 digest over every other serialized model field.
`ChangeDetectionModel.model_validate(...)` and `model_validate_json(...)` reject incomplete fitted
provenance, mismatches between duplicated config/channel metadata, an inconsistent empirical
threshold, and a stale model hash. `MultivariateChangeDetector.from_model(...)` performs the same
validation before rebuilding its inverse covariance, Cholesky factor, and empirical reference
scores. It refuses lightweight report-only models that lack complete fitted provenance.

When the reconstructed detector scores a `Study`, it also recomputes the reference-input artifact
from the current labeled reference changes and compares it with the fit-time value. This detects
changed reference values, identities, visits, or channel resolution. These hashes are
content-integrity and reproducibility identifiers, not keyed signatures and not proof that the
reference data or model is scientifically valid.

## Interpretation limits

- A detection means the joint change is unusual under the fitted reference distribution. It does
  not mean the subject improved, deteriorated, or responded to treatment.
- Feature direction remains in `change` and `innovation`; Mahalanobis distance itself is
  directionless.
- Calibration subjects should represent the intended null or normal-aging population. Scoring
  fails closed if any requested evaluation subject was part of the fitted reference set.
- The empirical threshold controls false alarms only to the extent that calibration and
  evaluation data are exchangeable.
- Missing any configured channel excludes that subject rather than imputing it.
- Treatment-versus-control inference requires a randomized design and a separate group-level
  model.
