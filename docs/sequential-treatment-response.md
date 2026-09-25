# Sequential multimodal response detection

`SequentialTreatmentResponseDetector` monitors whether a subject's trajectory persistently
departs from placebo or other reference-cohort aging dynamics. It extends the two-visit detector
to three or more expected visits, irregular spacing, missing visits, and repeated monitoring.

For elapsed time \(\Delta t\), it estimates reference drift \(\mu\) and forms a time-normalized
innovation:

\[
e_t = \frac{x_t-x_{t-1}-\mu\Delta t}{\sqrt{\Delta t}}.
\]

The reference innovation covariance is shrunk, regularized, and Cholesky-whitened. Cumulative
evidence after \(n\) observed transitions is the whitened energy of
\(\sum_t e_t / \sqrt{n}\). The threshold is calibrated from the maximum cumulative score reached
by each reference subject, so it addresses repeated looks across visits rather than calibrating
each visit independently. Each reference subject's maximum is computed against drift and
covariance refitted without that subject (`reference_score_method="leave_one_out"`), because
in-sample scores are biased low and inflate the false-alarm rate for new subjects (about 10%
at a nominal 5% with 30 reference subjects and four channels).

The \(\sqrt{\Delta t}\) normalization assumes random-walk dynamics. When measurement error
around a stable trait dominates, a subject with skipped visits spans a longer interval, gets a
smaller normalized innovation, and so loses sensitivity.

The report includes:

- first threshold-crossing visit;
- transient versus persistent crossing status;
- interval and cumulative evidence at each observed transition;
- empirical reference-tail probability;
- peak evidence calculated separately within each modality; and
- subjects excluded because fewer than two complete visits were available.

Missing intermediate visits are allowed. The detector spans the actual elapsed interval and
normalizes it explicitly; it does not silently impute the missing measurement.

```python
detector = SequentialTreatmentResponseDetector(
    SequentialDetectionConfig(
        features=features,
        false_alarm_rate=0.05,
        persistence_crossings=2,
        minimum_reference_subjects=75,
    )
).fit(study, visits=visits, reference_subject_ids=placebo_ids)

report = detector.score(study, visits=visits, subject_ids=treated_ids)
```

## Serialized-model integrity and reconstruction

A fitted `SequentialDetectionModel` records the full configuration, ordered visit definitions,
resolved modality/feature/unit/aggregation channels, requested and complete reference-subject
IDs, the reference-input artifact hash, and the sorted distribution of per-subject maximum
cumulative scores. Its `threshold_quantile_method` is explicitly `higher`. Validation recomputes
`maximum_cumulative_score_threshold` from that distribution and the stored `false_alarm_rate`.

The deterministic `model_artifact_hash` covers every other serialized model field.
`SequentialDetectionModel.model_validate(...)` and `model_validate_json(...)` reject incomplete
fit provenance, inconsistent config or channel summaries, impossible reference counts, a changed
threshold, and a stale model hash. `SequentialTreatmentResponseDetector.from_model(...)`
revalidates before reconstructing drift, covariance, Cholesky, and empirical reference state; it
does not accept a lightweight report-only model without complete fitted provenance.

At scoring time, the detector reconstructs the exact reference trajectories from the supplied
`Study` and verifies the fit-time reference-input artifact. Changes to subject identity, observed
values or times, visit definitions, or resolved channels therefore fail closed. The SHA-256
identifiers support content integrity and reproducibility; they are not keyed signatures or
evidence that the reference cohort is scientifically appropriate.

## Interpretation and limits

- Prefer a concurrent randomized placebo group. An observational reference cohort supports
  anomaly monitoring, not a treatment-effect claim.
- Evaluation IDs must be disjoint from the fitted reference IDs; overlap is rejected.
- Detection is directionless: it identifies an unusual joint trajectory, not improvement.
- `persistent=True` means the configured number of final consecutive scores exceeded the
  threshold. `transient=True` requires a later fall below threshold. A final crossing without
  enough subsequent visits is detected but remains unclassified. These are monitoring rules, not
  clinical endpoints.
- Modality evidence identifies where a departure is expressed; correlated modality scores are not
  additive causal contributions.
- The current threshold is fit in-sample on reference maxima. External or cross-validated
  calibration is needed before confirmatory use.
- A later group-level layer must estimate average treatment effects, heterogeneity, and confidence
  intervals under the study's randomization and missing-data assumptions.

Run the public observational demonstration with:

```bash
python examples/public_dog_sequential.py \
  --output-dir analysis/dog-sequential \
  --plots
```

It uses three Dog Aging Project Precision waves of clinical chemistry and metabolomics. Its
detections are unusual held-out trajectories, not evidence of rejuvenation or treatment response.

For a treatment-oriented demonstration without inventing public results:

```bash
python examples/triad_like_rapamycin_sequential.py \
  --output-dir analysis/triad-like \
  --plots
```

This generates 580 synthetic dogs, seven six-month visits, placebo-calibrated normal aging, and
several injected response phenotypes across echocardiography, activity, cognition, and clinical
laboratory channels. The schedule and domains resemble the public TRIAD design, but every
measurement and response label is simulated.
