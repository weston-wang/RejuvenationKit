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

The threshold is an empirical reference-score quantile selected by `false_alarm_rate`. Tail
probabilities also use the reference scores, with a one-count correction. Covariance shrinkage
and a small ridge stabilize inversion when channels are correlated.

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

## Interpretation limits

- A detection means the joint change is unusual under the fitted reference distribution. It does
  not mean the subject improved, deteriorated, or responded to treatment.
- Feature direction remains in `change` and `innovation`; Mahalanobis distance itself is
  directionless.
- Calibration subjects should represent the intended null or normal-aging population and should
  be independent of evaluation subjects.
- The empirical threshold controls false alarms only to the extent that calibration and
  evaluation data are exchangeable.
- Missing any configured channel excludes that subject rather than imputing it.
- Treatment-versus-control inference requires a randomized design and a separate group-level
  model.
